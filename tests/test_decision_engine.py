from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from portfolio_tracker.analytics.expected_value import calculate_expectation
from portfolio_tracker.analytics.horizon_models import (
    FEATURE_NAMES, MODEL_FEATURE_VERSION, _sha, model_feature_contract,
)
from portfolio_tracker.analytics.operational_target import TARGET_VERSION
from portfolio_tracker.analytics.horizon_selector import select_best_horizon
from portfolio_tracker.analytics.technical_probability import TechnicalSignal
from portfolio_tracker.db import Database
from portfolio_tracker.models import TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.decision_engine import generate_decision
from portfolio_tracker.services.model_execution_record import build_replay_snapshot, prediction_snapshot
from portfolio_tracker.services.pdf_report import build_executive_report
from portfolio_tracker.services.position_sizing import (
    PortfolioRiskContext,
    calculate_position_size,
)
from portfolio_tracker.services.price_zones import build_zone_snapshot
from tests.test_pdf_report import _analysis


ACTIONABLE_AT = datetime(2026, 9, 3, 15, 0, 21, tzinfo=timezone.utc)


def _at_validated_cut(analysis):
    return replace(analysis, as_of=pd.Timestamp("2026-09-03T14:55:00Z"))


def repository(tmp_path):
    result = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    result.database.initialize()
    result.ensure_initial_capital()
    return result


def _approved_operational_result(horizon, probabilities):
    temperature = 1.5
    weights = np.zeros((len(FEATURE_NAMES) + 1, 3))
    weights[0] = np.log(probabilities) * temperature
    result = {
        "horizon": horizon,
        "status": "APPROVED_SEALED_HOLDOUT_CALIBRATED",
        "promotable": True,
        "target": TARGET_VERSION,
        "resolved_samples": 330,
        "minimum_samples_required": 300,
        "population": {"executable_entries": {"n": 330}},
        "execution_evidence": {
            "source": "ELIGIBLE_LONG_DEVELOPMENT_ONLY",
            "target": TARGET_VERSION,
            "observed_sl_samples": 30,
            "gap_samples": 1,
            "gross_loss_multiples": [1.0] * 29 + [1.5],
            "semantics": "OBSERVED_EXIT_VS_FROZEN_STOP_DISTANCE_BEFORE_FILL_COSTS",
        },
        "score_semantics": "HISTORICAL_OOS_CALIBRATED_PRELIMINARY",
        "feature_names": list(FEATURE_NAMES),
        "feature_version": MODEL_FEATURE_VERSION,
        "available_features": list(FEATURE_NAMES),
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "final_holdout": {
            "status": "OPENED_ONCE_AFTER_PROTOCOL_FREEZE",
            "metrics": {
                "samples": 80, "brier": 0.2, "baseline_brier": 0.4,
                "raw_brier": 0.3, "log_loss": 0.4,
                "raw_log_loss": 0.5, "baseline_log_loss": 0.6,
            },
        },
        "calibration": {
            "approved": True,
            "validation_metrics": {
                "raw": {"brier": 0.3, "log_loss": 0.5},
                "calibrated": {"brier": 0.2, "log_loss": 0.4},
                "baseline": {"brier": 0.4, "log_loss": 0.6},
            },
        },
        "model": {
            "calibration": {"method": "MULTICLASS_TEMPERATURE_SCALING_V1", "temperature": temperature},
            "intercept_and_coefficients": weights.tolist(),
            "imputation_medians": [0.0] * len(FEATURE_NAMES),
            "standardization_means": [0.0] * len(FEATURE_NAMES),
            "standardization_scales": [1.0] * len(FEATURE_NAMES),
        },
    }
    result["result_sha256"] = _sha(result)
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
    analysis = _at_validated_cut(replace(_analysis(), risk_veto=True, fundamental_risk_veto=True))
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
        inference_at=ACTIONABLE_AT,
    )
    assert decision.action == "ESPERAR"
    assert decision.position_size == 0
    assert decision.adjusted_win_probability is None
    assert "Veto" in " ".join(decision.reasons)


def test_directional_calibration_alone_does_not_authorize_buy(tmp_path):
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
    analysis = _at_validated_cut(replace(
        analysis, horizon_projections=horizons, signal=TechnicalSignal.BUY,
        activation_trigger_met=True, operation_probability=80,
        risk_veto=False, fundamental_risk_veto=False, signal_rejected=False,
        long_entry_blocked=False, macro_permission="LONG_ONLY", position_state="FLAT",
    ))
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
        inference_at=ACTIONABLE_AT,
    )
    assert decision.action == "ESPERAR"
    assert decision.position_size == 0
    assert decision.expected_value_per_share is None
    assert "TP/SL/timeout" in decision.explanation
    assert decision.brier_touch is None and decision.brier_close is None
    assert decision.waiting_cause == "FALTA_EVIDENCIA"
    assert "sin evidencia de entradas ejecutables" in decision.explanation.lower()


def test_hypothetical_only_model_is_wait_for_evidence_not_trigger(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    analysis = _at_validated_cut(replace(
        base, signal=TechnicalSignal.BUY, activation_trigger_met=True,
        execution_plan_conditional=False, risk_veto=False,
        fundamental_risk_veto=False, signal_rejected=False,
        macro_permission="LONG_ONLY", position_state="FLAT",
    ))
    rejected = deepcopy(_approved_operational_result("1 Hora", (0.85, 0.10, 0.05)))
    rejected["population"]["executable_entries"]["n"] = 0
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo,
        operational_models={"1 Hora": rejected}, inference_at=ACTIONABLE_AT,
    )
    assert decision.action == "ESPERAR"
    assert decision.waiting_cause == "FALTA_EVIDENCIA"
    assert decision.position_size == 0
    assert "sin evidencia de entradas ejecutables" in decision.explanation.lower()

    app = AppTest.from_string('''
from portfolio_tracker.ui.system_decision import render_system_decision
from tests.test_decision_engine import repository, _approved_operational_result, _at_validated_cut, ACTIONABLE_AT
from tests.test_pdf_report import _analysis
from portfolio_tracker.services.decision_engine import generate_decision
from dataclasses import replace
from pathlib import Path
import tempfile
a = _at_validated_cut(replace(_analysis(), risk_veto=False, fundamental_risk_veto=False))
r = repository(Path(tempfile.mkdtemp()))
render_system_decision(generate_decision("SMCI", analysis=a, repository=r, operational_models={}, inference_at=ACTIONABLE_AT))
''').run(timeout=30)
    assert not app.exception
    assert any("FALTA DE EVIDENCIA" in item.value for item in app.warning)
    assert any("sin evidencia de entradas ejecutables" in item.value.lower() for item in app.warning)


def test_approved_artifact_with_299_forward_entries_cannot_buy(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    analysis = _at_validated_cut(replace(
        base, execution_levels=replace(base.execution_levels, take_profit_1=47.0),
        signal=TechnicalSignal.BUY, activation_trigger_met=True,
        execution_plan_conditional=False, risk_veto=False,
        fundamental_risk_veto=False, signal_rejected=False,
        macro_permission="LONG_ONLY", position_state="FLAT",
    ))
    model = _approved_operational_result("1 Semana", (0.85, 0.10, 0.05))
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo,
        operational_models={"1 Semana": model},
        validation_counts={"1 Semana": {"eligible": 299, "resolved": 299}},
        inference_at=ACTIONABLE_AT,
    )
    assert decision.action == "ESPERAR"
    assert decision.position_size == 0
    assert decision.recommendation_mode == "CONSERVADOR"
    assert decision.waiting_cause == "FALTA_EVIDENCIA"
    assert "n=299/300, modelo no aprobado" in decision.explanation


def test_buy_selects_highest_net_first_hit_ev_not_highest_directional_score(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    horizons = tuple(
        replace(item, probability_up=(90 if item.label == "1 Hora" else 60),
                probability_range=5, probability_down=(5 if item.label == "1 Hora" else 35))
        for item in base.horizon_projections
    )
    analysis = _at_validated_cut(replace(
        base, horizon_projections=horizons,
        execution_levels=replace(base.execution_levels, take_profit_1=47.0),
        signal=TechnicalSignal.BUY, activation_trigger_met=True,
        execution_plan_conditional=False, risk_veto=False,
        fundamental_risk_veto=False, signal_rejected=False,
        macro_permission="LONG_ONLY", position_state="FLAT",
    ))
    models = {
        "1 Hora": _approved_operational_result("1 Hora", (0.60, 0.25, 0.15)),
        "1 Semana": _approved_operational_result("1 Semana", (0.85, 0.10, 0.05)),
    }
    snapshot = build_replay_snapshot(
        analysis, observed_at=ACTIONABLE_AT, protocol="TEST_READ_ONLY",
    )
    for label, model in models.items():
        projection = next(item for item in analysis.horizon_projections if item.label == label)
        manifest = model_feature_contract(
            {"feature_snapshot": snapshot},
            {"prediction": prediction_snapshot(projection),
             "operational_contract": {
                 "entry_price": analysis.last_price,
                 "stop_loss": analysis.execution_levels.stop_loss,
                 "take_profit": analysis.execution_levels.take_profit_1,
             }},
        )
        model["available_features"] = manifest["available_features"]
        model["result_sha256"] = _sha({key: value for key, value in model.items() if key != "result_sha256"})
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, operational_models=models,
        inference_at=ACTIONABLE_AT,
        validation_counts={label: {"eligible": 330} for label in models},
    )
    assert decision.action == "COMPRAR"
    assert decision.horizon == "1 Semana"
    assert decision.position_size > 0
    assert Decimal(str(decision.monetary_risk)) <= Decimal("921.05") * Decimal("0.02")
    assert decision.reward_risk >= 1.5
    assert decision.expected_value_per_share > 0
    assert decision.expected_value_total > 0
    assert decision.adjusted_win_probability == 0.85
    assert decision.sl_first_probability == pytest.approx(0.10)
    assert decision.timeout_probability == pytest.approx(0.05)
    assert decision.directional_brier is None
    assert decision.operational_brier == 0.2


def test_both_reduced_caps_order_size_before_buy(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    analysis = _at_validated_cut(replace(
        base, execution_levels=replace(base.execution_levels, take_profit_1=47.0),
        signal=TechnicalSignal.BUY, activation_trigger_met=True,
        execution_plan_conditional=False, risk_veto=False,
        fundamental_risk_veto=False, signal_rejected=False,
        macro_permission="LONG_ONLY", exposure_factor=1.0, position_state="FLAT",
    ))

    def decide(current):
        model = _approved_operational_result("1 Semana", (0.85, 0.10, 0.05))
        projection = next(item for item in current.horizon_projections if item.label == "1 Semana")
        snapshot = build_replay_snapshot(current, observed_at=ACTIONABLE_AT, protocol="TEST_READ_ONLY")
        manifest = model_feature_contract(
            {"feature_snapshot": snapshot},
            {"prediction": prediction_snapshot(projection),
             "operational_contract": {
                 "entry_price": current.last_price,
                 "stop_loss": current.execution_levels.stop_loss,
                 "take_profit": current.execution_levels.take_profit_1,
             }},
        )
        model["available_features"] = manifest["available_features"]
        model["result_sha256"] = _sha({k: v for k, v in model.items() if k != "result_sha256"})
        return generate_decision(
            "SMCI", analysis=current, repository=repo,
            operational_models={"1 Semana": model}, inference_at=ACTIONABLE_AT,
            validation_counts={"1 Semana": {"eligible": 330}},
        )

    unrestricted = decide(analysis)
    reduced = decide(replace(analysis, macro_permission="BOTH_REDUCED", exposure_factor=0.25))
    assert unrestricted.action == reduced.action == "COMPRAR"
    assert reduced.position_size == unrestricted.position_size // 4
    assert reduced.exposure_factor_applied == 0.25
    assert reduced.monetary_risk < unrestricted.monetary_risk
    assert reduced.expected_value_total == pytest.approx(
        reduced.expected_value_per_share * reduced.position_size
    )


def test_injected_signed_model_that_loses_to_baseline_cannot_buy(tmp_path):
    repo = repository(tmp_path)
    base = _analysis()
    analysis = _at_validated_cut(replace(
        base, execution_levels=replace(base.execution_levels, take_profit_1=47.0),
        activation_trigger_met=True, execution_plan_conditional=False,
        risk_veto=False, fundamental_risk_veto=False, signal_rejected=False,
        macro_permission="LONG_ONLY", position_state="FLAT",
    ))
    model = deepcopy(_approved_operational_result("1 Semana", (0.85, 0.10, 0.05)))
    model["final_holdout"]["metrics"]["brier"] = 0.5
    model["result_sha256"] = _sha({key: value for key, value in model.items() if key != "result_sha256"})
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo,
        operational_models={"1 Semana": model},
        inference_at=ACTIONABLE_AT,
    )
    assert decision.action == "ESPERAR"
    assert decision.position_size == 0
    assert decision.expected_value_per_share is None


def test_preliminary_decision_has_no_probability_or_expected_value(tmp_path):
    repo = repository(tmp_path)
    analysis = _at_validated_cut(_analysis())
    snapshot = build_zone_snapshot(analysis, now="2026-09-04T16:25:00Z")
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo, zone_snapshot=snapshot,
        inference_at=ACTIONABLE_AT,
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
    labels = [item.label for item in app.metric]
    assert "EV neta observada/realista · por acción" in labels
    assert "EV neta teórica · por acción" in labels

    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(build_executive_report(
            analysis, zone_snapshot=snapshot, system_decision=decision,
        ))).pages
    )
    assert "DECISIÓN DEL SISTEMA" in text
    assert "EV neta teórica / acción" in text
    assert "EV neta observada/realista / acción" in text
    assert decision.action in text
    assert "Recomendacion" in text and "ejecucion manual" in text
