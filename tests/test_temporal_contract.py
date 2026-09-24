"""One causal XNYS anchor for directional, first-hit, replay and decisions."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_tracker.analytics.historical_replay import replay_target_close
from portfolio_tracker.analytics.operational_target import make_operational_contract
from portfolio_tracker.analytics.temporal_contract import (
    ANCHOR_VERSION, first_evaluable_open, is_actionable_emission,
)
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.directional_collection import (
    cut_forecasts, record_fixed_directional,
)
from portfolio_tracker.services.model_observations import (
    POLICY, PREVIOUS_POLICY, VERSION, forecast_digest, maturity,
)
from portfolio_tracker.services.decision_engine import generate_decision
from tests.test_model_execution_record import _analysis as synthetic_analysis
from tests.test_pdf_report import _analysis as full_analysis


CUT = datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc)  # 11:00 NY
EMITTED = CUT + timedelta(seconds=21)
DUE = pd.Timestamp("2026-09-03T16:05:00Z")


def repository(tmp_path):
    repo = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repo.database.initialize()
    return repo


def test_three_targets_share_one_hour_expiry_after_1100_21_emission():
    analysis = synthetic_analysis()
    horizon = analysis.horizon_projections[0]
    contract = make_operational_contract("SMCI", horizon, 60, analysis, EMITTED, {})
    forecasts = cut_forecasts(analysis, {}, EMITTED)
    one_hour = next(row for row in forecasts if row["horizon_minutes"] == 60)
    assert is_actionable_emission(EMITTED, CUT)
    assert first_evaluable_open(EMITTED) == pd.Timestamp("2026-09-03T15:05:00Z")
    assert maturity(EMITTED, 60) == replay_target_close(EMITTED, 60) == DUE
    assert contract["evaluation_starts_at"] == "2026-09-03T15:05:00+00:00"
    assert contract["timeout_at"] == DUE.isoformat()
    assert contract["model"]["anchor_version"] == ANCHOR_VERSION
    assert contract["eligible_at_emission"] is True
    assert one_hour["horizon_minutes"] == 60


def test_off_cut_cannot_be_collected_as_actionable_training_sample():
    late = CUT + timedelta(minutes=5, seconds=21)
    analysis = synthetic_analysis(source="2026-09-03T15:05:00Z")
    assert not is_actionable_emission(late, analysis.source_bar_closed_at)
    with pytest.raises(ValueError, match="Corte no accionable"):
        cut_forecasts(analysis, {}, late)


def test_decision_marks_off_cut_inference_no_accionable(tmp_path):
    repo = repository(tmp_path)
    analysis = replace(full_analysis(), as_of=pd.Timestamp("2026-09-03T15:00:00Z"))
    decision = generate_decision(
        "SMCI", analysis=analysis, repository=repo,
        inference_at=CUT + timedelta(minutes=5, seconds=21),
    )
    assert decision.action == "NO_ACCIONABLE"
    assert not decision.actionable_at_emission
    assert decision.position_size == 0
    assert decision.anchor_version == ANCHOR_VERSION
    assert "Fuera del corte temporal" in decision.explanation


def test_anchor_version_is_stored_signed_and_tampering_is_rejected(tmp_path):
    repo = repository(tmp_path)
    assert record_fixed_directional(repo, synthetic_analysis(), {}, EMITTED) == 6
    with repo.database.connect() as connection:
        rows = connection.execute("SELECT * FROM live_model_observations").fetchall()
    assert {row["anchor_version"] for row in rows} == {ANCHOR_VERSION}
    assert {row["horizon_policy"] for row in rows} == {POLICY}
    assert {row["integrity_version"] for row in rows} == {VERSION}
    assert next(row for row in rows if row["horizon_minutes"] == 60)["available_at"] == DUE.isoformat()
    with repo.database.transaction() as connection:
        connection.execute("DROP TRIGGER live_forecast_immutable")
        connection.execute("UPDATE live_model_observations SET anchor_version='altered' WHERE horizon_minutes=60")
    assert repo.verify_live_model_observations() == (5, (1,))


def test_existing_v3_hash_remains_verifiable_after_anchor_migration(tmp_path):
    repo = repository(tmp_path)
    row = repo._live_observation_row(
        symbol="SMCI", observed_at=EMITTED, source_bar_at=CUT,
        reference_price=Decimal("100"), raw_probability_up=Decimal("0.6"),
        parameters_json="{}", horizon_minutes=60,
    )
    row.update(
        integrity_version=3, horizon_policy=PREVIOUS_POLICY,
        available_at=maturity(EMITTED, 60, policy=PREVIOUS_POLICY).isoformat(),
        anchor_version=None,
    )
    row["observation_sha256"] = forecast_digest(row)
    with repo.database.transaction() as connection:
        fields = tuple(row)
        connection.execute(
            f"INSERT INTO live_model_observations({','.join(fields)}) "
            f"VALUES ({','.join('?' for _ in fields)})",
            tuple(row[field] for field in fields),
        )
    assert repo.verify_live_model_observations() == (1, ())
