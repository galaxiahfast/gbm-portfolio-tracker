"""Historical causal replay: point-in-time features, future-only labels."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from portfolio_tracker.analytics.closed_bars import _calendar
from portfolio_tracker.analytics.historical_replay import (
    HISTORICAL_REPLAY_CONTRACT,
    build_historical_replay,
    validate_historical_replay,
    write_historical_replay,
)
from portfolio_tracker.analytics.replay import ReplayDataset


def _frames():
    schedule = _calendar(2024, 2026).schedule.loc["2025-01-01":"2026-09-18"]
    daily_schedule = schedule.tail(320)
    daily_prices = [70 + index * 0.04 + math.sin(index / 9) for index in range(len(daily_schedule))]
    daily = pd.DataFrame(
        {
            "Open": [value - 0.1 for value in daily_prices],
            "High": [value + 1.0 for value in daily_prices],
            "Low": [value - 1.0 for value in daily_prices],
            "Close": daily_prices,
            "Volume": [2_000_000 + index * 100 for index in range(len(daily_schedule))],
        },
        index=daily_schedule.index,
    )
    pieces = []
    counter = 0
    for session in daily_schedule.tail(12).itertuples():
        index = pd.date_range(session.open, session.close, freq="5min", inclusive="left")
        prices = [82 + counter * 0.002 + math.sin((counter + offset) / 5) * 0.35 for offset in range(len(index))]
        counter += len(index)
        pieces.append(pd.DataFrame(
            {
                "Open": [value - 0.03 for value in prices],
                "High": [value + 0.22 for value in prices],
                "Low": [value - 0.22 for value in prices],
                "Close": prices,
                "Volume": [50_000 + (offset % 17) * 700 for offset in range(len(index))],
            },
            index=index,
        ))
    intraday = pd.concat(pieces)
    as_of = pd.Timestamp(daily_schedule.iloc[-1].close)
    return intraday, daily, as_of


@pytest.fixture(scope="module")
def replay_payload():
    intraday, daily, as_of = _frames()
    dataset = ReplayDataset(intraday, daily, as_of)
    day = str(as_of.tz_convert("America/New_York").date())
    return build_historical_replay(
        "SMCI",
        dataset,
        start=day,
        end=day,
        max_cuts=1,
    )


def test_historical_replay_builds_one_signed_six_horizon_cut(replay_payload):
    assert replay_payload["contract"] == HISTORICAL_REPLAY_CONTRACT
    assert replay_payload["separation_policy"] == "REPLAY_NEVER_COUNTS_AS_LIVE_OOS"
    assert replay_payload["candidate_cuts"] == 1
    assert replay_payload["rejected"] == {}
    assert validate_historical_replay(replay_payload)
    cut = replay_payload["observations"][0]
    assert len(cut["horizons"]) == 6
    assert cut["observed_at"].endswith("+00:00")
    assert cut["source_bar_closed_at"] == cut["observed_at"]
    assert "HISTORICAL_REPLAY_NOT_LIVE_OOS" in cut["limitations"]
    one_hour = next(item for item in cut["horizons"] if item["label"] == "1 Hora")
    assert one_hour["scenario_result"]["status"] == "RESOLVED"
    assert one_hour["operational_result"]["status"] == "RESOLVED"
    assert one_hour["operational_result"]["outcome"] in {"TP_FIRST", "SL_FIRST", "TIMEOUT"}
    assert one_hour["execution_result"]["version"] == "SESSION_LIMIT_POSSIBLE_FILL_V1"
    six_months = next(item for item in cut["horizons"] if item["label"] == "6 Meses")
    assert six_months["scenario_result"]["status"] == "RIGHT_CENSORED"


def test_future_prices_change_labels_not_point_in_time_features(replay_payload):
    intraday, daily, as_of = _frames()
    changed = intraday.copy()
    cut = pd.Timestamp(replay_payload["observations"][0]["observed_at"])
    future = changed.index >= cut
    changed.loc[future, "Open"] *= 1.25
    changed.loc[future, "Close"] *= 1.25
    changed.loc[future, "High"] = changed.loc[future, ["Open", "Close"]].max(axis=1) + 0.5
    changed.loc[future, "Low"] = changed.loc[future, ["Open", "Close"]].min(axis=1) - 0.5
    dataset = ReplayDataset(changed, daily, as_of)
    day = str(as_of.tz_convert("America/New_York").date())
    rerun = build_historical_replay("SMCI", dataset, start=day, end=day, max_cuts=1)
    original_cut = replay_payload["observations"][0]
    changed_cut = rerun["observations"][0]
    assert changed_cut["feature_snapshot"]["features"] == original_cut["feature_snapshot"]["features"]
    assert [item["prediction"] for item in changed_cut["horizons"]] == [
        item["prediction"] for item in original_cut["horizons"]
    ]
    assert changed_cut["dataset_sha256"] != original_cut["dataset_sha256"]


def test_historical_replay_artifact_is_atomic_and_tamper_evident(tmp_path, replay_payload):
    destination = write_historical_replay(replay_payload, tmp_path / "replay.json")
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert validate_historical_replay(stored)
    stored["observations"][0]["horizons"][0]["prediction"]["probability_up"] = 99.0
    with pytest.raises(ValueError, match="Firma de corte"):
        validate_historical_replay(stored)


def test_execution_assessment_is_inside_signed_replay(replay_payload):
    from copy import deepcopy
    altered = deepcopy(replay_payload)
    altered["observations"][0]["horizons"][0]["execution_result"]["status"] = "SIMULATED_RESOLVED"
    with pytest.raises(ValueError, match="Firma de corte"):
        validate_historical_replay(altered)
