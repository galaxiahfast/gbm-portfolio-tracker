"""Research measurements do not authorize trades or replace execution evidence."""
import pandas as pd
import pytest

from portfolio_tracker.analytics.causal_core import trigger_momentum
from scripts.measure_trigger_selectivity import (
    CONFIRMATIONS, confirmation_frame, cost_components, lag_measurement,
    summarize_lags, volume_summary,
)


def confirmations():
    index = pd.date_range("2026-09-24 10:40", periods=5, freq="5min", tz="America/New_York")
    return pd.DataFrame(False, index=index, columns=CONFIRMATIONS)


def test_lag_reports_span_and_freshness_without_future():
    frame = confirmations()
    frame.loc[frame.index[0], "stoch_cross"] = True  # closes 10:45, age 3
    frame.loc[frame.index[1], "volume"] = True       # closes 10:50, age 2
    frame.loc[frame.index[2], "macd"] = True         # closes 10:55, age 1
    frame.loc[frame.index[3], "previous_bar_break"] = True  # closes 11:00
    frame.iloc[-1] = True  # closes 11:05: prohibited future observation
    result = lag_measurement(frame, pd.Timestamp("2026-09-24 11:00", tz="America/New_York"))
    assert result["span_first_to_last_bars"] == 3
    assert result["ages_in_bars"]["stoch_cross"] == 3
    assert result["micro_shadow_by_tolerance"] == {"0": False, "1": False, "2": False, "3": True}


def test_compact_stale_sequence_does_not_pass_tolerance():
    frame = confirmations()
    frame.iloc[0] = True
    result = lag_measurement(frame, pd.Timestamp("2026-09-24 11:00", tz="America/New_York"))
    assert result["span_first_to_last_bars"] == 0
    assert result["oldest_confirmation_age_bars"] == 3
    assert not result["micro_shadow_by_tolerance"]["2"]


def test_missing_confirmation_is_censored_not_zero():
    frame = confirmations()
    frame.iloc[3, :3] = True
    result = lag_measurement(frame, pd.Timestamp("2026-09-24 11:00", tz="America/New_York"))
    assert result["span_first_to_last_bars"] is None
    assert result["missing_in_lookback"] == ["previous_bar_break"]
    assert not any(result["micro_shadow_by_tolerance"].values())


def test_prior_session_cannot_supply_confirmation():
    index = pd.DatetimeIndex(["2026-09-23 15:55", "2026-09-24 09:55"], tz="America/New_York")
    frame = pd.DataFrame(False, index=index, columns=CONFIRMATIONS)
    frame.iloc[0] = True
    result = lag_measurement(frame, pd.Timestamp("2026-09-24 10:00", tz="America/New_York"), lookback=1000)
    assert set(result["missing_in_lookback"]) == set(CONFIRMATIONS)


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_confirmation_frame_zero_tolerance_matches_shared_momentum(side):
    index = pd.date_range("2026-09-24 09:10", periods=22, freq="5min", tz="America/New_York")
    bars = pd.DataFrame({"Open": 100., "High": 100., "Low": 100., "Close": 100.,
                         "Volume": 100., "StochRSI_K": 30. if side == "LONG" else 50.,
                         "StochRSI_D": 40., "MACD": 0., "MACD_signal": 0.}, index=index)
    bars.iloc[-1, bars.columns.get_loc("Volume")] = 200
    bars.iloc[-1, bars.columns.get_loc("StochRSI_K")] = 50. if side == "LONG" else 30.
    frame = confirmation_frame(bars, side)
    assert not frame.iloc[:21].to_numpy().any()
    assert bool(frame.iloc[-1].all()) == trigger_momentum(bars)[0 if side == "LONG" else 1]
    assert frame.iloc[-1].all()


def test_volume_threshold_is_strict_and_quantiles_are_reported():
    result = volume_summary([.5, 1., 1.5, 2.])
    assert result["above1"] == 2
    assert result["at_or_below1"] == 2
    assert result["threshold1_empirical_percentile"] == 50.
    assert result["median"] == 1.25
    with pytest.raises(ValueError):
        volume_summary([float("nan")])


def test_costs_are_explicit_assumptions_not_observed_spread():
    result = cost_components(100, 110, {"commission_bps_per_side": 25., "spread_bps": 4., "slippage_bps_per_side": 5.})
    assert result["commission_roundtrip_assumed"] == pytest.approx(.525)
    assert result["spread_roundtrip_assumed_NOT_observed"] == pytest.approx(.042)
    assert result["slippage_roundtrip_assumed_NOT_observed"] == pytest.approx(.105)


def test_shadow_cut_counts_are_not_reported_as_fills():
    lag = {"span_first_to_last_bars": 1, "missing_in_lookback": [],
           "micro_shadow_by_tolerance": {"0": False, "1": True, "2": True, "3": True}}
    row = {"failed_original_trigger": True, "selected_side_lag": lag,
           "full_shadow_by_tolerance": {"0": False, "1": False, "2": True, "3": True}, "selected_side": "LONG"}
    result = summarize_lags([row])
    assert result["micro_shadow_all_cuts"]["1"] == 1
    assert result["full_shadow_all_cuts"]["1"] == 0
    assert result["additional_failed_cuts_full_shadow"]["2"] == 1
    assert "possible_fills" not in result
