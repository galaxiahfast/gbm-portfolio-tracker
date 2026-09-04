"""Prospective contracts only; synthetic bars never enter production evidence."""
from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository, ZonePrediction
from portfolio_tracker.services.zone_forward import (
    digest, log_snapshot, model_version_hash, session_bounds, validation_data,
)
from portfolio_tracker.services.price_zones import DisplayZone, ZoneSnapshot
from portfolio_tracker.analytics.zone_reach import ReachEstimate


@pytest.fixture
def repo(tmp_path):
    repository = PortfolioRepository(Database(tmp_path / "evidence.db"))
    repository.ensure_zone_forward_schema()
    return repository


def prediction(**kwargs):
    fields = dict(
        symbol="SMCI", timestamp_prediction="2026-08-31T15:00:00+00:00",
        source_bar_closed_at="2026-08-31T15:00:00+00:00", zone_key="TP1", zone_type="TP1",
        zone_low=101.0, zone_high=101.0, reference_price=100.0,
        predicted_touch_probability=0.71, predicted_close_probability=0.40,
        close_direction="ABOVE", model_version_hash="a"*64, model_name="test-only",
    )
    return ZonePrediction(**(fields | kwargs))


def emit(repo, item):
    return repo.save_prediction(item, now=item.timestamp_prediction)


def market(day="2026-08-31", touched=True):
    bounds = session_bounds(day+"T15:00:00Z")
    index = pd.date_range(bounds[1], bounds[2], freq="5min", inclusive="left")
    bars = pd.DataFrame(dict(Open=100., High=100.5, Low=99.5, Close=100., Volume=1000.), index=index)
    if touched:
        bars.loc[bars.index[-2], "High"] = 102.
    daily = pd.DataFrame(dict(Open=[100.], High=[bars.High.max()], Low=[99.5],
                              Close=[100.], Volume=[bars.Volume.sum()]), index=pd.to_datetime([day]))
    return bars, daily


def test_schema_only_adds_analytic_tables_and_is_idempotent(repo):
    repo.ensure_zone_forward_schema()
    with repo.database.connect() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {"zone_prediction_log", "zone_market_evidence", "zone_daily_validation"}


def test_no_future_backfill_stale_or_holiday_emissions(repo):
    with pytest.raises(ValueError):
        repo.save_prediction(prediction(), now="2026-09-01T15:00:00Z")
    for item in (
        prediction(timestamp_prediction="2026-08-31T15:06:00Z"),
        prediction(timestamp_prediction="2026-09-07T15:00:00Z", source_bar_closed_at="2026-09-07T15:00:00Z"),
        prediction(timestamp_prediction="2026-08-31T15:00:00"),
        prediction(predicted_touch_probability=71),
        prediction(zone_low=float("nan")),
    ):
        with pytest.raises(ValueError):
            emit(repo, item)


def test_duplicate_first_emission_wins_and_forecast_is_immutable(repo):
    assert emit(repo, prediction())
    assert not emit(repo, prediction(predicted_touch_probability=.20))
    assert repo.zone_predictions()[0]["predicted_touch_probability"] == .71
    with pytest.raises(sqlite3.IntegrityError):
        with repo.database.transaction() as conn:
            conn.execute("UPDATE zone_prediction_log SET zone_price=105")
    with pytest.raises(sqlite3.IntegrityError):
        with repo.database.transaction() as conn:
            conn.execute("DELETE FROM zone_prediction_log")
    assert repo.zone_predictions()[0]["integrity_ok"]


def test_exact_session_closed_plus15_and_brier(repo):
    emit(repo, prediction())
    calls = []
    def provider(symbol, day):
        calls.append(day)
        return market(day)
    assert repo.resolve_predictions(provider, now="2026-08-31T20:14:59Z")["resolved"] == 0
    assert not calls
    assert repo.resolve_predictions(provider, now="2026-09-01T14:00:00Z")["resolved"] == 1
    assert calls == ["2026-08-31"]
    row = repo.zone_predictions()[0]
    assert row["actual_touch_occurred"] == 1 and row["actual_close_relation"] == "BELOW"
    assert row["integrity_ok"]
    scored = validation_data([row])
    assert scored[0]["brier"] == pytest.approx((.71-1)**2)
    assert scored[1]["brier"] == pytest.approx(.4**2)
    with repo.database.connect() as conn:
        stored = conn.execute("SELECT payload_json,sha256 FROM zone_daily_validation").fetchone()
    import json
    assert digest(json.loads(stored[0])) == stored[1]
    assert repo.resolve_predictions(provider, now="2026-09-01T15:00:00Z")["resolved"] == 0


def test_early_close_and_dst(repo):
    assert session_bounds("2026-11-27T16:00:00Z")[2] == pd.Timestamp("2026-11-27T18:00:00Z")
    assert session_bounds("2026-12-01T16:00:00Z")[2] == pd.Timestamp("2026-12-01T21:00:00Z")


@pytest.mark.parametrize("fault", ["missing", "null", "timezone", "duplicate"])
def test_incomplete_or_disagreeing_market_remains_pending(repo, fault):
    emit(repo, prediction())
    bars, daily = market()
    if fault == "missing":
        bars = bars.iloc[1:]
    elif fault == "null":
        bars.iloc[1, 0] = np.nan
    elif fault == "timezone":
        bars.index = bars.index.tz_localize(None)
    else:
        bars = pd.concat([bars, bars.iloc[:1]])
    result = repo.resolve_predictions(lambda *_: (bars, daily), now="2026-08-31T20:15:00Z")
    assert result["pending"] == 1 and result["errors"]
    assert repo.zone_predictions()[0]["resolved_at"] is None


def test_daily_official_close_resolves_and_records_intraday_discrepancy(repo):
    emit(repo, prediction())
    bars, daily = market()
    daily["Close"] = 100.05

    result = repo.resolve_predictions(lambda *_: (bars, daily), now="2026-08-31T20:15:00Z")

    assert result["resolved"] == 1 and result["pending"] == 0
    assert len(result["warnings"]) == 1
    assert "cierre 1D 100.050000" in result["warnings"][0]
    row = repo.zone_predictions()[0]
    assert row["actual_close_price"] == pytest.approx(100.05)
    assert row["actual_touch_occurred"] == 1
    assert "ADVERTENCIA" in row["resolution_note"]
    assert row["integrity_ok"]


def test_pre_prediction_touch_does_not_count(repo):
    emit(repo, prediction())
    bars, daily = market(touched=False)
    bars.iloc[0, bars.columns.get_loc("High")] = 102.
    repo.resolve_predictions(lambda *_: (bars, daily), now="2026-08-31T20:15:00Z")
    assert repo.zone_predictions()[0]["actual_touch_occurred"] == 0


def test_partial_bar_ambiguous_excluded_not_false(repo):
    emit(repo, prediction(timestamp_prediction="2026-08-31T15:00:03Z"))
    bars, daily = market(touched=False)
    bars.loc[pd.Timestamp("2026-08-31T15:00:00Z"), "High"] = 102.
    repo.resolve_predictions(lambda *_: (bars, daily), now="2026-08-31T20:15:00Z")
    row = repo.zone_predictions()[0]
    assert row["actual_touch_occurred"] is None and row["resolved_at"]
    assert "AMBIGUA" in row["resolution_note"]
    assert {r["event"] for r in validation_data([row])} == {"Cierre"}


def test_resolution_tampering_excluded(repo):
    emit(repo, prediction())
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        with repo.database.transaction() as conn:
            conn.execute("UPDATE zone_prediction_log SET actual_touch_occurred=0")
    # Simulate malicious/out-of-band write with trigger bypass, NOT production.
    with repo.database.transaction() as conn:
        conn.execute("DROP TRIGGER zone_resolution_immutable")
        conn.execute("UPDATE zone_prediction_log SET actual_close_price=200")
    rows = repo.zone_predictions()
    assert not rows[0]["integrity_ok"]
    assert validation_data(rows) == []


def test_market_evidence_tampering_excluded(repo):
    emit(repo, prediction())
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    with repo.database.transaction() as conn:
        conn.execute("DROP TRIGGER zone_market_evidence_no_update")
        conn.execute("UPDATE zone_market_evidence SET payload_json='{}'")
    assert not repo.zone_predictions()[0]["integrity_ok"]


def test_snapshot_logs_six_levels_ui_probabilities_no_fake_stop(repo):
    zone = DisplayZone("test", 99., 99., "test", 70, "alcista")
    sale = replace(zone, low=101., high=101.)
    estimate = ReachEstimate(probability=71., samples=21, lower=50., upper=90.,
                             status="preliminar", close_probability=40.)
    snapshot = ZoneSnapshot(pd.Timestamp("2026-08-31T15:00:00Z"), (zone,)*3, (sale,)*3, (estimate,)*6)
    analysis = SimpleNamespace(symbol="SMCI", last_price=100.,
                               source_bar_closed_at=pd.Timestamp("2026-08-31T15:00:00Z"))
    result = log_snapshot(repo, analysis, snapshot, now=snapshot.evaluated_at)
    assert result["saved"] == 6
    assert log_snapshot(repo, analysis, snapshot, now=snapshot.evaluated_at)["saved"] == 0
    rows = repo.zone_predictions()
    assert len(rows) == 6 and all(r["integrity_ok"] for r in rows)
    assert {r["predicted_touch_probability"] for r in rows} == {.71}
    assert {r["zone_key"] for r in rows} == {"ENTRY1", "ENTRY2", "ENTRY3", "TP1", "TP2", "R3"}


def test_primary_cohort_not_six_sessions_or_first_success(repo):
    emit(repo, prediction())
    emit(repo, prediction(timestamp_prediction="2026-08-31T15:05:00Z", source_bar_closed_at="2026-08-31T15:05:00Z"))
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    scored = validation_data(repo.zone_predictions())
    assert len(scored) == 2  # one touch + one close, not two emissions
    assert len({r["session_date"] for r in scored}) == 1


def test_known_zone_not_evidence_of_predictive_success(repo):
    emit(repo, prediction(zone_low=99., zone_high=100.))
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    row = repo.zone_predictions()[0]
    assert row["touch_eligible"] == 0
    assert {r["event"] for r in validation_data([row])} == {"Cierre"}


def test_close_below_is_not_inverted_above_for_interval(repo):
    item = prediction(zone_low=99., zone_high=101., close_direction="BELOW")
    emit(repo, item)
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    row = repo.zone_predictions()[0]
    assert row["predicted_close_above_probability"] is None
    assert row["actual_close_relation"] == "INSIDE"


def test_model_hash_is_sha256():
    assert len(model_version_hash()) == 64


def test_daily_cache_and_closed_context(tmp_path, monkeypatch):
    from portfolio_tracker.services import forward_market as module
    index = pd.bdate_range("2024-01-02", "2026-08-31")
    close = np.linspace(40, 100, len(index))
    frame = pd.DataFrame(dict(Open=close, High=close+1, Low=close-1, Close=close, Volume=1000), index=index)
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return frame
    monkeypatch.setattr(module, "_download", download)
    first, meta = module.download_daily_history(now="2026-08-31T20:15:00Z", cache_dir=tmp_path)
    second, cached = module.download_daily_history(now="2026-08-31T20:16:00Z", cache_dir=tmp_path)
    assert len(calls) == 1 and calls[0]["start"] == "2024-01-01"
    assert meta == cached and len(first) == len(second)
    assert meta["context"]["Diario"]["EMA200"] is not None
    assert meta["context"]["Mensual"]["EMA50"] is None
    assert meta["context"]["Diario"]["ADX14"] is not None

def test_validation_panel_empty_and_resolved_without_network(repo):
    from streamlit.testing.v1 import AppTest
    code = """
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.db import Database
from portfolio_tracker.ui.forward_validation import validation_panel
r = PortfolioRepository(Database(PATH))
validation_panel(r, {"errors": []})
""".replace("PATH", repr(str(repo.database.path)))
    at = AppTest.from_string(code).run(timeout=15)
    assert not at.exception
    assert any("Aún no hay" in item.value for item in at.info)
    emit(repo, prediction())
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    at = AppTest.from_string(code).run(timeout=15)
    assert not at.exception
    assert any("Brier acumulado" in metric.label for metric in at.metric)


def test_touch_side_follows_reference_price_not_name(repo):
    # An old ENTRY plan can sit ABOVE the current price; touch then means rise.
    item = prediction(zone_key="ENTRY1", zone_type="ENTRY", close_direction="BELOW")
    emit(repo, item)
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    assert repo.zone_predictions()[0]["actual_touch_occurred"] == 1


def test_pending_fields_cannot_be_smuggled_into_evidence(repo):
    emit(repo, prediction())
    with repo.database.transaction() as conn:
        conn.execute("UPDATE zone_prediction_log SET actual_close_price=200")
    assert not repo.zone_predictions()[0]["integrity_ok"]


def test_invalid_first_emission_is_not_replaced_by_later_success(repo):
    emit(repo, prediction())
    emit(repo, prediction(timestamp_prediction="2026-08-31T15:05:00Z",
                          source_bar_closed_at="2026-08-31T15:05:00Z"))
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    rows = repo.zone_predictions()
    rows[0]["integrity_ok"] = False
    assert validation_data(rows) == []
