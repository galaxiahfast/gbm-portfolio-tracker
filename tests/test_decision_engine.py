from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from portfolio_tracker.analytics.expected_value import calculate_expectation
from portfolio_tracker.analytics.horizon_selector import select_best_horizon
from portfolio_tracker.analytics.technical_probability import TechnicalSignal
from portfolio_tracker.db import Database
from portfolio_tracker.models import TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.decision_engine import generate_decision
from portfolio_tracker.services.pdf_report import build_executive_report
from portfolio_tracker.services.position_sizing import (
    PortfolioRiskContext,
    calculate_position_size,
)
from portfolio_tracker.services.price_zones import build_zone_snapshot
from tests.test_pdf_report import _analysis


def repository(tmp_path):
    result = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    result.database.initialize()
    result.ensure_initial_capital()
    return result


def test_expected_value_uses_only_calibrated_directional_probability():
    result = calculate_expectation(
        "SMCI", 100, 99, 103, bullish_score=60,
        brier_touch=0.90, brier_close=0.90,
        calibrated_probability=0.60,
    )
    assert result.adjusted_probability == 0.60
    assert result.reward_risk == 3.0
    assert result.expected_value_per_share == 1.4
    assert result.brier_touch is None and result.brier_close is None


def test_horizon_selector_requires_full_calibration_and_baseline_skill():
    analysis = _analysis()
    preliminary = {item.label: item for item in analysis.horizon_projections}
    assert select_best_horizon(preliminary) is None
    validated = {
        item.label: replace(
            item, probability_up=80, probability_range=10, probability_down=10,
            probability_status="Probabilidad empíricamente calibrada",
            calibration_samples=500, calibration_holdout_samples=100,
            brier_score=0.20, baseline_brier_score=0.30,
        )
        for item in analysis.horizon_projections
    }
    assert select_best_horizon(validated) is not None


def test_horizon_selector_rejects_range_dominance_and_weak_baseline():
    analysis = _analysis()
    range_dominant = tuple(
        replace(
            item, probability_up=30, probability_range=45, probability_down=25,
            probability_status="Probabilidad empíricamente calibrada",
            calibration_samples=500, calibration_holdout_samples=100,
            brier_score=0.20, baseline_brier_score=0.30,
        )
        for item in analysis.horizon_projections
    )
    assert select_best_horizon(range_dominant) is None
    no_skill = tuple(
        replace(item, probability_up=80, probability_range=10, probability_down=10,
                brier_score=0.35, baseline_brier_score=0.30)
        for item in range_dominant
    )
    assert select_best_horizon(no_skill) is None


def test_position_size_is_capped_by_risk_cash_and_concentration():
    context = PortfolioRiskContext("SMCI", None, 0, 1000, 1000, 0, 0, 0)
    result = calculate_position_size(context, 40, 38)
    assert result.risk_budget == 20
    assert result.shares == 7  # 30% concentration cap: floor(300 / 40)
    assert result.monetary_risk == 14
    assert result.concentration_after <= 0.30


def test_veto_always_waits_and_missing_evidence_is_disclosed(tmp_path):
    repo = repository(tmp_path)
    analysis = replace(_analysis(), risk_veto=True, fundamental_risk_veto=True)
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
    )
    assert decision.action == "ESPERAR"
    assert decision.position_size == 0
    assert decision.adjusted_win_probability is None
    assert "Veto" in " ".join(decision.reasons)


def test_buy_requires_every_gate_and_uses_real_account_capital(tmp_path):
    repo = repository(tmp_path)
    analysis = _analysis()
    horizons = tuple(
        replace(
            item, probability_up=80, probability_range=10, probability_down=10,
            probability_status="Probabilidad empíricamente calibrada",
            calibration_samples=500, calibration_holdout_samples=100,
            brier_score=0.20, baseline_brier_score=0.30,
        )
        for item in analysis.horizon_projections
    )
    analysis = replace(
        analysis, horizon_projections=horizons, signal=TechnicalSignal.BUY,
        activation_trigger_met=True, operation_probability=80,
        risk_veto=False, fundamental_risk_veto=False, signal_rejected=False,
        long_entry_blocked=False, macro_permission="LONG_ONLY", position_state="FLAT",
    )
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
    )
    assert decision.action == "COMPRAR"
    assert decision.position_size > 0
    assert Decimal(str(decision.monetary_risk)) <= Decimal("921.05") * Decimal("0.02")
    assert decision.reward_risk >= 1.5
    assert decision.expected_value_per_share > 0
    assert decision.brier_touch is None and decision.brier_close is None


def test_preliminary_decision_has_no_probability_or_expected_value(tmp_path):
    repo = repository(tmp_path)
    analysis = _analysis()
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
    )
    assert decision.action == "ESPERAR"
    assert decision.horizon == "Sin horizonte validado"
    assert decision.adjusted_win_probability is None
    assert decision.expected_value_per_share is None
    assert decision.expected_value_total is None
    assert "N/D" in decision.explanation


def test_existing_position_is_sold_when_persistent_take_profit_is_reached(tmp_path):
    repo = repository(tmp_path)
    repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("5"),
        price_usd=Decimal("40"), commission_usd=Decimal("0"),
        executed_at=datetime.now(timezone.utc),
    ))
    base = _analysis()
    price = float(base.buy_levels.take_profit_1) + 0.10
    analysis = replace(
        base, last_price=price, risk_veto=False,
        fundamental_risk_veto=False, position_state="LONG_ACTIVE",
    )
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
    )
    assert decision.action == "VENDER"
    assert decision.current_shares == 5
    assert "take profit" in " ".join(decision.reasons).lower()


def test_system_decision_renders_in_ui_and_pdf(tmp_path):
    repo = repository(tmp_path)
    analysis = _analysis()
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
    )
    app = AppTest.from_string('''
from portfolio_tracker.ui.system_decision import render_system_decision
from tests.test_decision_engine import repository
from tests.test_pdf_report import _analysis
from portfolio_tracker.services.price_zones import build_zone_snapshot
from portfolio_tracker.services.decision_engine import generate_decision
from pathlib import Path
import tempfile
a = _analysis()
r = repository(Path(tempfile.mkdtemp()))
s = build_zone_snapshot(a, now="2026-09-04T16:25:00Z")
render_system_decision(generate_decision("SMCI", analysis=a, repository=r, zone_snapshot=s))
''').run(timeout=30)
    assert not app.exception
    assert "DECISIÓN DEL SISTEMA" in "\n".join(item.value for item in app.markdown)
    assert "Condiciones de activación LONG" in "\n".join(item.value for item in app.markdown)

    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(build_executive_report(
            analysis, zone_snapshot=snapshot, system_decision=decision,
        ))).pages
    )
    assert "DECISIÓN DEL SISTEMA" in text
    assert decision.action in text
    assert "Recomendacion" in text and "ejecucion manual" in text
