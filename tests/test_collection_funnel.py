"""The coverage report is read-only and counts one scheduled cut, not six wins."""
from datetime import datetime, timezone

from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.collection_funnel import collection_funnel
from portfolio_tracker.services.directional_collection import record_fixed_directional
from tests.test_model_execution_record import _analysis, CUT


def test_funnel_exposes_invalid_plan_and_next_missing_session(tmp_path):
    repository = PortfolioRepository(Database(tmp_path / "isolated.db"))
    repository.database.initialize()
    analysis = _analysis()
    analysis.execution_levels.take_profit_1 = 99.0
    assert record_fixed_directional(repository, analysis, {}, CUT) == 6
    with repository.database.connect() as connection:
        before = connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0]
    report = collection_funnel(
        repository, ("SMCI",), now=datetime(2026, 9, 8, 13, 5, tzinfo=timezone.utc),
        sessions=2,
    )
    assert report["summary"] == {
        "expected_after_enrollment": 2, "signed": 1, "missing": 1,
        "invalid": 0, "eligible_at_emission": 0,
        "simulated_resolved_fills": 0,
    }
    assert report["rows"][0]["reason"] == "NO_VALID_FIRST_PASSAGE_PLAN"
    assert report["rows"][0]["directional_resolved_horizons"] == 0
    assert report["rows"][1]["status"] == "MISSING_CUT"
    with repository.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0] == before


def test_funnel_does_not_call_six_horizons_six_eligible_trades(tmp_path):
    repository = PortfolioRepository(Database(tmp_path / "isolated.db"))
    repository.database.initialize()
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    report = collection_funnel(
        repository, ("SMCI",), now=datetime(2026, 9, 3, 15, 6, tzinfo=timezone.utc),
        sessions=1,
    )
    assert report["summary"]["signed"] == 1
    assert report["summary"]["eligible_at_emission"] == 1
    assert report["summary"]["simulated_resolved_fills"] == 0
