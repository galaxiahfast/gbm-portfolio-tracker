"""One immutable, inspectable execution across six forecast horizons.

All persistence in this module is confined to a temporary database/cache.
The tests never emit a live prediction or touch the accounting database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.directional_collection import (
    COLLECTION_PROTOCOL,
    HORIZON_MINUTES,
    LEGACY_COLLECTION_PROTOCOL,
    record_fixed_directional,
)
from portfolio_tracker.services.model_execution_record import (
    build_replay_snapshot,
    execution_id,
    prediction_snapshot,
)
from portfolio_tracker.services.scenario_calibration import make_scenario_contract
from scripts.autopilot_market_cache import MarketCache


CUT = datetime(2026, 9, 3, 15, 0, 10, tzinfo=timezone.utc)  # 11:00:10 NY


@pytest.fixture
def repository(tmp_path):
    repo = PortfolioRepository(Database(tmp_path / "isolated.db"))
    repo.database.initialize()
    repo.ensure_initial_capital()
    return repo


def _analysis(symbol="SMCI", source="2026-09-03T15:00:00Z"):
    horizons = tuple(
        SimpleNamespace(
            label=label, engine_name="synthetic-test", bias="Alcista",
            probability_up=60.0, probability_range=30.0,
            probability_down=10.0, bullish_target=103.0,
            range_low=99.0, range_high=101.0,
            bearish_target=97.0, atr_value=1.0,
            local_support=98.0, local_resistance=102.0,
            probability_status="Score heurístico preliminar",
            calibration_samples=0, brier_score=None,
        ) for label in HORIZON_MINUTES
    )
    indicators = pd.DataFrame(
        {"RSI": [48.5], "ADX": [27.0], "Volume": [1200.0]},
        index=pd.DatetimeIndex(["2026-09-03T14:55:00Z"]),
    )
    return SimpleNamespace(
        symbol=symbol, last_price=100.0,
        source_bar_closed_at=pd.Timestamp(source),
        horizon_projections=horizons,
        probability_up=60.0, probability_down=10.0,
        adx=27.0, volume_ratio=1.35, weekly_trend="Alcista",
        macro_permission="LONG_ONLY",
        market_regime="TREND",
        cross_asset_context={"peer_symbol": "NVDA", "correlation": 0.72},
        intraday_indicators=indicators,
        execution_levels=SimpleNamespace(
            direction="LONG", stop_loss=95.0, take_profit_1=105.0,
        ), activation_trigger_met=True, execution_plan_conditional=False,
        risk_veto=False, signal_rejected=False,
    )


def _rows(repo):
    with repo.database.connect() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM live_model_observations ORDER BY horizon_minutes"
        ).fetchall()]


def _operational_row(repo, horizon_minutes=1_440):
    """Return one child target without relying on presentation-layer parsing."""
    with repo.database.connect() as connection:
        row = connection.execute(
            """SELECT target.* FROM operational_model_outcomes AS target
               JOIN live_model_observations AS live
                 ON live.id=target.observation_id
               WHERE live.symbol='SMCI' AND live.horizon_minutes=?""",
            (horizon_minutes,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _flat_remainder_of_observation_session():
    """Every fully prospective 5m candle after the 11:00 NY emission."""
    index = pd.date_range(
        "2026-09-03T15:05:00Z", "2026-09-03T19:55:00Z", freq="5min",
    )
    return pd.DataFrame(
        {
            "Open": 100.0, "High": 101.0, "Low": 99.0,
            "Close": 100.25, "Volume": 1_000.0,
        },
        index=index,
    )


def _insert_legacy_execution(repo):
    """Create an authentic signed V1 shape without retrofitting V2 fields."""
    analysis = _analysis()
    session = "2026-09-03"
    replay = build_replay_snapshot(
        analysis, observed_at=CUT, protocol=LEGACY_COLLECTION_PROTOCOL,
    )
    parameters = {"observation_protocol": LEGACY_COLLECTION_PROTOCOL}
    horizons = {item.label: item for item in analysis.horizon_projections}
    rows = []
    for label, minutes in HORIZON_MINUTES.items():
        horizon = horizons[label]
        contract = make_scenario_contract("SMCI", horizon, minutes, parameters)
        metadata = {
            **parameters,
            "engine": horizon.engine_name,
            "feedback_version": 3,
            "scenario_contract": contract,
            "collection_protocol": LEGACY_COLLECTION_PROTOCOL,
            "scheduled_cut_ny": "11:00",
            "session_date": session,
            "replay": replay,
            "prediction_snapshot": prediction_snapshot(horizon),
        }
        rows.append(repo._live_observation_row(
            symbol="SMCI", observed_at=CUT,
            source_bar_at=analysis.source_bar_closed_at,
            reference_price=Decimal("100"),
            raw_probability_up=Decimal(str(contract["probabilities"][0])),
            parameters_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
            horizon_minutes=minutes,
        ))
    with repo.database.transaction() as connection:
        for row in rows:
            names = tuple(row)
            connection.execute(
                f"INSERT INTO live_model_observations({','.join(names)}) "
                f"VALUES ({','.join('?' for _ in names)})",
                tuple(row[name] for name in names),
            )
    return execution_id("SMCI", session, LEGACY_COLLECTION_PROTOCOL)


def test_legacy_v1_execution_remains_directional_without_fake_operational_target(repository):
    run_id = _insert_legacy_execution(repository)

    record = repository.live_model_execution_record(run_id)

    assert record is not None
    assert record["collection_protocol"] == LEGACY_COLLECTION_PROTOCOL
    assert len(record["predictions"]) == 6
    assert all("operational_target" not in item for item in record["predictions"])
    assert all("operational_result" not in item for item in record["predictions"])
    assert all(item["model_id"] for item in record["predictions"])
    assert repository.verify_live_model_observations() == (6, ())


def test_v2_execution_requires_every_signed_operational_child(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    with repository.database.transaction() as connection:
        connection.execute("DROP TRIGGER operational_model_no_delete")
        connection.execute(
            "DELETE FROM operational_model_outcomes WHERE observation_id IN "
            "(SELECT id FROM live_model_observations WHERE horizon_minutes=60)"
        )

    assert repository.live_model_execution_record(run_id) is None


def test_six_forecasts_share_signed_execution_and_replay_features(repository):
    cash_before = repository.cash_balance_usd()
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    rows = _rows(repository)
    assert len(rows) == 6
    assert repository.verify_live_model_observations() == (6, ())

    snapshots = [json.loads(row["parameters_json"])["replay"] for row in rows]
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    assert {snapshot["run_id"] for snapshot in snapshots} == {run_id}
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert snapshots[0]["features"]["adx"] == 27.0
    assert snapshots[0]["features"]["cross_asset_context"]["peer_symbol"] == "NVDA"
    assert snapshots[0]["indicator_tail"]["intraday_indicators"]["values"]["RSI"] == 48.5
    assert snapshots[0]["replay_level"] == "FEATURE_SNAPSHOT_ONLY"
    assert all(len(value) == 64 for value in snapshots[0]["code_sha256"].values())

    record = repository.live_model_execution_record(run_id)
    assert record is not None
    assert record["run_id"] == run_id
    assert record["features"]["last_price"] == 100.0
    assert len(record["predictions"]) == 6
    assert {item["horizon_minutes"] for item in record["predictions"]} == set(HORIZON_MINUTES.values())
    assert all(item["resolution_status"] == "PENDING" for item in record["predictions"])

    retry = datetime(2026, 9, 3, 15, 5, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Corte no accionable"):
        record_fixed_directional(
            repository, _analysis(source="2026-09-03T15:05:00Z"), {}, retry,
        )
    assert len(_rows(repository)) == 6
    assert repository.cash_balance_usd() == cash_before
    with repository.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_exact_outcome_is_attached_to_same_verified_execution(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    one_hour = next(item for item in repository.live_model_execution_record(run_id)["predictions"]
                    if item["horizon_minutes"] == 60)
    assert one_hour["resolution_status"] == "PENDING"
    candle = pd.DataFrame(
        {"Open": [100.0], "High": [102.0], "Low": [99.0],
         "Close": [101.5], "Volume": [1500.0]},
        index=pd.DatetimeIndex(["2026-09-03T16:00:00Z"]),
    )
    assert repository.resolve_live_model_observations(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 16, 5, tzinfo=timezone.utc),
        historical_bars=candle,
    ) == 1
    record = repository.live_model_execution_record(run_id)
    assert record is not None
    resolved = next(item for item in record["predictions"] if item["horizon_minutes"] == 60)
    assert resolved["resolution_status"] == "RESOLVED"
    assert float(resolved["outcome_price"]) == 101.5
    assert resolved["outcome_bar_at"] == "2026-09-03T16:05:00+00:00"
    assert resolved["outcome_source"] == "yfinance:5m:raw-close"
    assert resolved["resolution_sha256"]
    assert sum(item["resolution_status"] == "PENDING" for item in record["predictions"]) == 5
    assert repository.verify_live_model_observations() == (6, ())


def test_mutated_features_fail_closed_and_do_not_enter_verified_record(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with repository.database.transaction() as connection:
            connection.execute(
                "UPDATE live_model_observations SET parameters_json=json_set(parameters_json, "
                "'$.replay.features.adx', 99) WHERE horizon_minutes=60"
            )
    assert repository.live_model_execution_record(run_id) is not None

    # Simulate direct file tampering outside the application, bypassing the
    # immutable trigger. The original SHA must then invalidate the whole run.
    with repository.database.transaction() as connection:
        connection.execute("DROP TRIGGER live_forecast_immutable")
        connection.execute(
            "UPDATE live_model_observations SET parameters_json=json_set(parameters_json, "
            "'$.replay.features.adx', 99) WHERE horizon_minutes=60"
        )
    assert repository.verify_live_model_observations()[0] == 5
    assert repository.live_model_execution_record(run_id) is None


def test_incomplete_cohort_is_not_presented_as_execution(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    with repository.database.transaction() as connection:
        connection.execute("DROP TRIGGER operational_model_no_delete")
        connection.execute("DROP TRIGGER live_observation_no_delete")
        connection.execute("DELETE FROM operational_model_outcomes WHERE observation_id IN "
                           "(SELECT id FROM live_model_observations WHERE horizon_minutes=60)")
        connection.execute("DELETE FROM live_model_observations WHERE horizon_minutes=60")
    assert len(_rows(repository)) == 5
    assert repository.live_model_execution_record(run_id) is None


def test_unserializable_feature_fails_before_any_write(repository):
    analysis = _analysis()
    analysis.cross_asset_context = {"peer": object()}
    with pytest.raises(ValueError, match="serializable"):
        record_fixed_directional(repository, analysis, {}, CUT)
    assert _rows(repository) == []


def test_archive_references_are_verifiable_and_fail_on_tamper(tmp_path, repository):
    cache = MarketCache(tmp_path)
    intraday = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0],
         "Close": [100.5], "Volume": [1000.0]},
        index=pd.DatetimeIndex(["2026-09-03T14:55:00Z"]),
    )
    daily = intraday.copy()
    daily.index = pd.DatetimeIndex(["2026-09-02"])
    cache._write(tmp_path / "SMCI" / "5m.json", intraday, CUT, {"interval": "5m"})
    cache._write(tmp_path / "SMCI" / "daily.json", daily, CUT, {"interval": "1d"})
    refs = cache.artifact_refs("SMCI")
    assert set(refs) == {"5m", "1d"}
    for ref in refs.values():
        assert len(ref["sha256"]) == 64
        assert (tmp_path / "SMCI" / ref["archive_path"]).is_file()
    assert cache.verify_artifact_refs("SMCI", refs)
    assert record_fixed_directional(
        repository, _analysis(), {}, CUT, input_artifacts=refs,
    ) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    record = repository.live_model_execution_record(run_id)
    assert record["input_artifacts"] == refs
    assert record["replay_level"] == "ARCHIVED_PRIMARY_INPUTS_WITH_PEER_FEATURE_SNAPSHOT"

    # A later collection advances only the current pointer. Older signed
    # execution references must still verify against the immutable archive.
    revised = intraday.copy()
    revised["Close"] = 100.6
    cache._write(tmp_path / "SMCI" / "5m.json", revised, CUT, {"interval": "5m"})
    assert cache.verify_artifact_refs("SMCI", record["input_artifacts"])

    archive = tmp_path / "SMCI" / refs["5m"]["archive_path"]
    archive.write_text("{}", encoding="utf-8")
    # Current pointer now targets the revised file, but the historical signed
    # run must reject its damaged original archive.
    assert not cache.verify_artifact_refs("SMCI", refs)


def test_operational_tp_first_resolves_early_and_is_joined_to_execution(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    pending = repository.live_model_execution_record(run_id)
    assert pending is not None
    assert all(item["operational_target"]["operational_probabilities"] is None
               for item in pending["predictions"])
    assert all(item["operational_result"]["resolution_status"] == "PENDING"
               for item in pending["predictions"])

    # First fully prospective candle opens at 11:05 NY and closes at 11:10.
    bars = pd.DataFrame(
        {"Open": [100.0], "High": [106.0], "Low": [99.0],
         "Close": [104.0], "Volume": [2_000.0]},
        index=pd.DatetimeIndex(["2026-09-03T15:05:00Z"]),
    )
    assert repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 15, 10, tzinfo=timezone.utc),
        historical_bars=bars,
    ) == 6
    record = repository.live_model_execution_record(run_id)
    results = [item["operational_result"] for item in record["predictions"]]
    assert {item["outcome"] for item in results} == {"TP_FIRST"}
    assert {float(item["exit_price"]) for item in results} == {105.0}
    assert {item["exit_at"] for item in results} == {"2026-09-03T15:10:00+00:00"}
    assert all(item["outcome_sha256"] for item in results)
    # The same candle also touched the limit price. OHLC cannot order its
    # entry and TP, so these barrier wins are not executable training wins.
    assert {item["evidence"]["execution_assessment"]["status"] for item in results} == {
        "AMBIGUOUS_NO_TRADE"
    }
    assert repository.verify_operational_model_outcomes() == (6, ())
    coverage = repository.operational_event_coverage("SMCI")
    assert coverage["horizon_resolutions"] == 6
    assert coverage["independent_events"] == 1
    assert coverage["resolved_events"] == 1
    assert coverage["tp_first_events"] == 1
    assert coverage["tp_first_rate"] == 1.0
    per_horizon = repository.operational_validation_counts("SMCI")
    assert all(item["resolved"] == 1 for item in per_horizon.values())
    assert all(item["eligible"] <= item["resolved"] for item in per_horizon.values())
    assert all(item["eligible"] == 0 for item in per_horizon.values())


def test_forward_possible_fill_precedes_target_and_counts_once_per_horizon(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    bars = pd.DataFrame(
        {"Open": [100.0, 100.0], "High": [101.0, 106.0],
         "Low": [99.0, 99.0], "Close": [100.0, 104.0],
         "Volume": [2_000.0, 2_000.0]},
        index=pd.DatetimeIndex(["2026-09-03T15:05:00Z", "2026-09-03T15:10:00Z"]),
    )
    assert repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 15, 15, tzinfo=timezone.utc),
        historical_bars=bars,
    ) == 6
    record = repository.live_model_execution_record(
        execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    )
    assert {row["operational_result"]["evidence"]["execution_assessment"]["status"]
            for row in record["predictions"]} == {"SIMULATED_RESOLVED"}
    assert all(row["eligible"] == 1 for row in
               repository.operational_validation_counts("SMCI").values())
    assert repository.verify_operational_model_outcomes() == (6, ())


def test_operational_timeout_requires_exact_horizon_close(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    index = pd.date_range("2026-09-03T15:05:00Z", periods=12, freq="5min")
    bars = pd.DataFrame(
        {"Open": 100.0, "High": 102.0, "Low": 98.0,
         "Close": 101.25, "Volume": 1_000.0}, index=index,
    )
    assert repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 16, 5, tzinfo=timezone.utc),
        historical_bars=bars,
    ) == 1
    run_id = execution_id("SMCI", "2026-09-03", COLLECTION_PROTOCOL)
    predictions = repository.live_model_execution_record(run_id)["predictions"]
    one_hour = next(item for item in predictions if item["horizon_minutes"] == 60)
    assert one_hour["operational_result"]["outcome"] == "TIMEOUT"
    assert one_hour["operational_result"]["exit_at"] == "2026-09-03T16:05:00+00:00"
    assert float(one_hour["operational_result"]["exit_price"]) == 101.25
    assert sum(item["operational_result"]["resolution_status"] == "PENDING"
               for item in predictions) == 5


def test_operational_child_rejects_partial_insert_and_is_immutable(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    with repository.database.connect() as connection:
        parent = connection.execute(
            "SELECT id FROM live_model_observations ORDER BY id LIMIT 1"
        ).fetchone()
    with pytest.raises(sqlite3.IntegrityError, match="insert_incomplete"):
        with repository.database.transaction() as connection:
            connection.execute(
                """INSERT INTO operational_model_outcomes(
                       observation_id,target_version,contract_sha256,
                       resolution_status,outcome,created_at
                   ) VALUES (999,'TP_FIRST_SL_FIRST_TIMEOUT_V1',?,'PENDING','TP_FIRST',?)""",
                ("a" * 64, CUT.isoformat()),
            )

    bars = pd.DataFrame(
        {"Open": [100.0], "High": [106.0], "Low": [99.0],
         "Close": [104.0], "Volume": [2_000.0]},
        index=pd.DatetimeIndex(["2026-09-03T15:05:00Z"]),
    )
    assert repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 15, 10, tzinfo=timezone.utc),
        historical_bars=bars,
    ) == 6
    with pytest.raises(sqlite3.IntegrityError, match="resolution_immutable"):
        with repository.database.transaction() as connection:
            connection.execute(
                "UPDATE operational_model_outcomes SET exit_price='999' WHERE observation_id=?",
                (parent["id"],),
            )
    assert repository.verify_operational_model_outcomes() == (6, ())


def test_operational_checkpoint_resumes_with_only_new_session_bars_after_restart(repository):
    """A pending path must not require redownloading already verified 5m bars."""
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    first_session = _flat_remainder_of_observation_session()

    # No barrier is touched.  The one-hour target may time out, while the
    # one-day target must remain pending with the completed session checkpoint.
    repository.resolve_operational_model_outcomes(
        symbol="SMCI",
        current_as_of=datetime(2026, 9, 3, 20, 0, tzinfo=timezone.utc),
        historical_bars=first_session,
    )
    checkpoint = _operational_row(repository)
    assert checkpoint["resolution_status"] == "PENDING"
    assert checkpoint["scanned_through"] == "2026-09-03T20:00:00+00:00"
    assert int(checkpoint["scan_evidence_count"]) == len(first_session)
    assert len(checkpoint["scan_evidence_sha256"]) == 64
    assert checkpoint["scan_evidence_json"]
    assert len(checkpoint["checkpoint_sha256"]) == 64
    assert checkpoint["checkpoint_updated_at"]
    pending_execution = json.loads(checkpoint["scan_evidence_json"])["execution_assessment"]
    assert pending_execution["status"] == "FILLED_PENDING"
    assert pending_execution["checkpoint_version"] == "OHLC_EXECUTION_STATE_CHECKPOINT_V1"
    assert pending_execution["fill_price"] == 100.0
    assert pending_execution["scanned_through"] == checkpoint["scanned_through"]
    assert repository.verify_operational_model_outcomes() == (6, ())

    # Simulate a fresh process.  Only the first new candle of the following
    # XNYS session is supplied; the persisted checkpoint carries prior proof.
    restarted = PortfolioRepository(Database(repository.database.path))
    new_only = pd.DataFrame(
        {
            "Open": [100.25], "High": [106.0], "Low": [99.5],
            "Close": [105.25], "Volume": [2_000.0],
        },
        index=pd.DatetimeIndex(["2026-09-04T13:30:00Z"]),
    )
    assert restarted.resolve_operational_model_outcomes(
        symbol="SMCI",
        current_as_of=datetime(2026, 9, 4, 13, 35, tzinfo=timezone.utc),
        historical_bars=new_only,
    ) >= 1
    resolved = _operational_row(restarted)
    assert resolved["resolution_status"] == "RESOLVED"
    assert resolved["outcome"] == "TP_FIRST"
    assert resolved["exit_at"] == "2026-09-04T13:35:00+00:00"
    assert int(resolved["scan_evidence_count"]) == len(first_session) + 1
    assert resolved["scanned_through"] == "2026-09-04T13:35:00+00:00"
    assert resolved["checkpoint_sha256"]
    resumed_execution = json.loads(resolved["evidence_json"])["execution_assessment"]
    assert resumed_execution["status"] == "SIMULATED_RESOLVED"
    assert resumed_execution["fill_at"] == "2026-09-03T15:10:00+00:00"
    assert resumed_execution["outcome"] == "TP_FIRST"
    assert repository.operational_validation_counts("SMCI")["1 Día"]["eligible"] == 1
    assert restarted.verify_operational_model_outcomes() == (6, ())

    # Retrying the same worker payload is a no-op and preserves the signed row.
    before = resolved
    assert restarted.resolve_operational_model_outcomes(
        symbol="SMCI",
        current_as_of=datetime(2026, 9, 4, 13, 35, tzinfo=timezone.utc),
        historical_bars=new_only,
    ) == 0
    assert _operational_row(restarted) == before


def test_checkpoint_without_possible_fill_cannot_turn_later_tp_into_trade(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    source = _flat_remainder_of_observation_session().copy()
    source["Open"] = 101.0
    source["High"] = 102.0
    source["Low"] = 100.5
    source["Close"] = 101.0
    repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 20, tzinfo=timezone.utc),
        historical_bars=source,
    )
    checkpoint = _operational_row(repository)
    assert json.loads(checkpoint["scan_evidence_json"])["execution_assessment"]["status"] == "NO_FILL"
    restarted = PortfolioRepository(Database(repository.database.path))
    new_only = pd.DataFrame(
        {"Open": [101.0], "High": [106.0], "Low": [100.5],
         "Close": [105.25], "Volume": [2_000.0]},
        index=pd.DatetimeIndex(["2026-09-04T13:30:00Z"]),
    )
    assert restarted.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 4, 13, 35, tzinfo=timezone.utc),
        historical_bars=new_only,
    ) >= 1
    resolved = _operational_row(restarted)
    assert resolved["outcome"] == "TP_FIRST"  # hypothetical barrier only
    assert json.loads(resolved["evidence_json"])["execution_assessment"]["status"] == "NO_FILL"
    assert restarted.operational_validation_counts("SMCI")["1 Día"]["eligible"] == 0


def test_checkpointed_fill_keeps_gap_open_loss_after_restart(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    repository.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 3, 20, tzinfo=timezone.utc),
        historical_bars=_flat_remainder_of_observation_session(),
    )
    restarted = PortfolioRepository(Database(repository.database.path))
    gap = pd.DataFrame(
        {"Open": [90.0], "High": [92.0], "Low": [88.0],
         "Close": [90.5], "Volume": [2_000.0]},
        index=pd.DatetimeIndex(["2026-09-04T13:30:00Z"]),
    )
    assert restarted.resolve_operational_model_outcomes(
        symbol="SMCI", current_as_of=datetime(2026, 9, 4, 13, 35, tzinfo=timezone.utc),
        historical_bars=gap,
    ) >= 1
    state = json.loads(_operational_row(restarted)["evidence_json"])["execution_assessment"]
    assert state["status"] == "SIMULATED_RESOLVED"
    assert state["outcome"] == "SL_FIRST"
    assert state["exit_price"] == 90.0  # stop was 95, but the gap opened at 90
    assert state["net_pnl_per_share"] < -10


def test_operational_checkpoint_tamper_fails_closed(repository):
    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    repository.resolve_operational_model_outcomes(
        symbol="SMCI",
        current_as_of=datetime(2026, 9, 3, 20, 0, tzinfo=timezone.utc),
        historical_bars=_flat_remainder_of_observation_session(),
    )
    stored = _operational_row(repository)
    observation_id = int(stored["observation_id"])

    # Either SQLite rejects the mutation immediately or the SHA verifier must
    # quarantine it.  Both are fail-closed; neither may resolve altered state.
    try:
        with repository.database.transaction() as connection:
            connection.execute(
                """UPDATE operational_model_outcomes
                   SET scan_evidence_count=scan_evidence_count+1
                   WHERE observation_id=?""",
                (observation_id,),
            )
    except sqlite3.IntegrityError:
        assert repository.verify_operational_model_outcomes() == (6, ())
        return

    valid, invalid = repository.verify_operational_model_outcomes()
    assert valid == 5
    assert observation_id in invalid
    new_only = pd.DataFrame(
        {
            "Open": [100.25], "High": [106.0], "Low": [99.5],
            "Close": [105.25], "Volume": [2_000.0],
        },
        index=pd.DatetimeIndex(["2026-09-04T13:30:00Z"]),
    )
    with pytest.raises(ValueError, match="integridad inválida"):
        repository.resolve_operational_model_outcomes(
            symbol="SMCI",
            current_as_of=datetime(2026, 9, 4, 13, 35, tzinfo=timezone.utc),
            historical_bars=new_only,
        )
    assert _operational_row(repository)["resolution_status"] == "PENDING"


def test_operational_labeler_does_not_hold_sqlite_write_transaction(
    repository, monkeypatch,
):
    """Slow first-passage calculation must not monopolize SQLite's writer."""
    from portfolio_tracker.analytics import operational_target

    assert record_fixed_directional(repository, _analysis(), {}, CUT) == 6
    with repository.database.transaction() as connection:
        connection.execute(
            "CREATE TABLE labeler_write_probe(id INTEGER PRIMARY KEY, writes INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO labeler_write_probe VALUES (1, 0)")

    real_labeler = operational_target.scan_operational_outcome

    def labeler_with_concurrent_writer(*args, **kwargs):
        # A BEGIN IMMEDIATE held by the resolver would make this deterministic
        # independent connection fail with `database is locked`.
        probe = sqlite3.connect(repository.database.path, timeout=0.2)
        try:
            probe.execute("PRAGMA busy_timeout=200")
            probe.execute(
                "UPDATE labeler_write_probe SET writes=writes+1 WHERE id=1"
            )
            probe.commit()
        finally:
            probe.close()
        return real_labeler(*args, **kwargs)

    monkeypatch.setattr(
        operational_target, "scan_operational_outcome",
        labeler_with_concurrent_writer,
    )
    hit = pd.DataFrame(
        {
            "Open": [100.0], "High": [106.0], "Low": [99.0],
            "Close": [104.0], "Volume": [2_000.0],
        },
        index=pd.DatetimeIndex(["2026-09-03T15:05:00Z"]),
    )
    assert repository.resolve_operational_model_outcomes(
        symbol="SMCI",
        current_as_of=datetime(2026, 9, 3, 15, 10, tzinfo=timezone.utc),
        historical_bars=hit,
    ) == 6
    with repository.database.connect() as connection:
        writes = connection.execute(
            "SELECT writes FROM labeler_write_probe WHERE id=1"
        ).fetchone()[0]
    assert writes == 6
