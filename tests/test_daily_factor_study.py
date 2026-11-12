"""Causality, cost and estimator checks for the daily factor study (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics import daily_factor_study as study

CONFIG = study.StudyConfig(bootstrap_draws=50)


def _days(n, start="2010-01-01"):
    return pd.bdate_range(start, periods=n)


def test_decision_never_earns_its_own_or_next_day_return():
    days = _days(6)
    returns = pd.DataFrame({"A": [0, 0, 0.10, 0.05, 0, 0]}, index=days, dtype=float)
    targets = pd.DataFrame({"A": [np.nan] * 6}, index=days)
    targets.iloc[1] = 1.0  # decided at close of day 1
    book = study.simulate(targets, returns, pd.Series(0.0, index=days), study.StudyConfig(cost_bps_per_side=0))
    # executes at close of day 2 (lag 1): the +10% of day 2 is NOT earned, day 3 is.
    assert book["net"].iloc[2] == pytest.approx(0.0)
    assert book["net"].iloc[3] == pytest.approx(0.05)


def test_buy_and_hold_drifts_and_pays_cost_once():
    days = _days(5)
    returns = pd.DataFrame({"A": [0, 0.1, -0.05, 0.2, 0.0]}, index=days)
    targets = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan]}, index=days)
    config = study.StudyConfig(cost_bps_per_side=30, execution_lag_days=0)
    book = study.simulate(targets, returns, pd.Series(0.0, index=days), config)
    assert book["turnover"].sum() == pytest.approx(1.0)
    wealth = (1 + book["net"]).prod()
    assert wealth == pytest.approx((1 - 0.003) * 1.1 * 0.95 * 1.2)


def test_two_asset_book_drifts_between_rebalances():
    days = _days(4)
    returns = pd.DataFrame({"A": [0, 1.0, 0, 0], "B": [0, 0, 0, 0]}, index=days, dtype=float)
    targets = pd.DataFrame(np.nan, index=days, columns=["A", "B"])
    targets.iloc[0] = [0.5, 0.5]
    config = study.StudyConfig(cost_bps_per_side=0, execution_lag_days=0)
    book = study.simulate(targets, returns, pd.Series(0.0, index=days), config)
    assert book["net"].iloc[1] == pytest.approx(0.5)
    # No silent rebalancing afterwards: no turnover after the first trade.
    assert book["turnover"].iloc[1:].sum() == 0


def test_leverage_and_shorts_are_refused():
    days = _days(3)
    returns = pd.DataFrame({"A": [0, 0, 0]}, index=days, dtype=float)
    with pytest.raises(ValueError):
        study.simulate(pd.DataFrame({"A": [1.5, np.nan, np.nan]}, index=days), returns,
                       pd.Series(0.0, index=days), CONFIG)
    with pytest.raises(ValueError):
        study.simulate(pd.DataFrame({"A": [-0.5, np.nan, np.nan]}, index=days), returns,
                       pd.Series(0.0, index=days), CONFIG)


def test_momentum_skips_the_most_recent_month():
    days = _days(320)
    n = len(days)
    # A rises steadily but stalls in the last month; B is flat then jumps last month.
    a = np.r_[np.linspace(100, 200, n - 21), np.full(21, 200.0)]
    b = np.r_[np.full(n - 21, 100.0), np.linspace(100, 300, 21)]
    flat = np.full(n, 100.0)
    closes = pd.DataFrame({"A": a, "B": b, "C": flat, "D": flat * 0.9, "E": flat * 1.1, "F": flat}, index=days)
    config = study.StudyConfig(momentum_min_names=1, momentum_quantile=1 / 6)
    targets = study.momentum_targets(closes, config)
    last = targets.dropna(how="all").iloc[-1]
    assert last["A"] == 1.0 and last["B"] == 0.0


def test_momentum_uses_only_past_prices():
    days = _days(320)
    rng = np.random.default_rng(1)
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (320, 8)), axis=0)),
                          index=days, columns=list("ABCDEFGH"))
    config = study.StudyConfig(momentum_min_names=2)
    base = study.momentum_targets(closes, config)
    future = closes.copy()
    future.iloc[-5:] *= 5  # change only the last five days
    changed = study.momentum_targets(future, config)
    cutoff = days[-6]
    pd.testing.assert_frame_equal(base.loc[:cutoff], changed.loc[:cutoff])


def test_trend_is_cash_below_average_and_decides_on_month_ends():
    days = _days(300)
    prices = pd.Series(np.r_[np.linspace(100, 200, 250), np.linspace(200, 120, 50)], index=days)
    signal = study.trend_targets(prices, study.StudyConfig(trend_window=50))
    decisions = signal.dropna()
    assert study.month_end_flags(days)[decisions.index].all()
    assert decisions.iloc[-1] == 0.0
    assert decisions.loc[decisions.index < days[250]].iloc[-1] == 1.0


def test_vol_target_never_levers():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0, 0.002, 400), index=_days(400))
    exposure = study.vol_target_exposure(returns, CONFIG).dropna()
    assert (exposure <= 1.0).all() and (exposure >= 0).all()
    assert exposure.iloc[-1] == pytest.approx(1.0)  # calm market -> capped at 1x


def test_garch_recovers_parameters_and_forecasts_are_causal():
    rng = np.random.default_rng(3)
    alpha, beta, omega = 0.08, 0.90, 0.02 * (0.01 ** 2)
    n, variance, values = 3000, 0.01 ** 2, []
    for _ in range(n):
        value = rng.normal() * np.sqrt(variance)
        values.append(value)
        variance = omega + alpha * value * value + beta * variance
    fit = study.fit_garch(np.array(values))
    assert abs(fit["alpha"] - alpha) < 0.04 and abs(fit["beta"] - beta) < 0.05
    series = pd.Series(values[:1400], index=_days(1400))
    base = study.garch_forecasts(series, 5, refit_every=200, minimum_history=1000)
    tampered = series.copy()
    tampered.iloc[-10:] = 0.2
    changed = study.garch_forecasts(tampered, 5, refit_every=200, minimum_history=1000)
    pd.testing.assert_series_equal(base.iloc[:-10], changed.iloc[:-10])


def test_realized_target_is_strictly_future():
    returns = pd.Series([0.0, 0.1, 0.0, 0.0, 0.2, 0.0], index=_days(6))
    target = study.realized_forward_variance(returns, 2)
    assert target.iloc[0] == pytest.approx(0.01)       # days 1-2
    assert target.iloc[2] == pytest.approx(0.04)       # days 3-4
    assert np.isnan(target.iloc[-1])


def test_clean_closes_removes_bad_prints_without_filling():
    days = _days(4)
    raw = pd.DataFrame({"A": [10.0, -1.0, np.nan, 12.0]}, index=days)
    clean, report = study.clean_closes(raw)
    assert report["invalid_prices_removed"] == 1
    assert clean["A"].isna().sum() == 2


def test_identical_series_have_zero_bootstrap_difference():
    rng = np.random.default_rng(4)
    series = pd.Series(rng.normal(0.0003, 0.01, 600), index=_days(600))
    result = study.paired_bootstrap(series, series, pd.Series(0.0, index=series.index), CONFIG)
    assert result["annual_return_difference_95"] == [0.0, 0.0]


def test_risk_sized_position_is_capped_by_capital():
    assert study.risk_sized_position(1000, 0.04) == pytest.approx(1000 * 0.01 / (2.33 * 0.04))
    assert study.risk_sized_position(1000, 0.0001) == 1000


def test_full_study_runs_on_synthetic_market():
    rng = np.random.default_rng(5)
    days = _days(900)
    closes = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (900, 12)), axis=0)),
                          index=days, columns=[f"S{i}" for i in range(11)] + ["BENCH"])
    result = study.run_strategy_study(closes, "BENCH", pd.Series(0.0, index=days),
                                      study.StudyConfig(bootstrap_draws=20, holdout_start="2012-06-01"))
    full = result["periods"]["FULL"]["strategies"]
    assert set(full) == {"BUY_HOLD_BENCHMARK", "TREND_200", "VOL_TARGET",
                         "EQUAL_WEIGHT_UNIVERSE", "MOMENTUM_12_1", "MOMENTUM_12_1_TREND"}
    assert full["TREND_200"]["average_exposure"] <= 1.0
    assert full["BUY_HOLD_BENCHMARK"]["turnover_per_year"] < 0.5


def test_documented_price_break_is_neutralized_only_where_listed():
    days = _days(4)
    closes = pd.DataFrame({"A": [40.0, 41.0, 66.0, 67.0], "B": [10.0, 11.0, 12.0, 13.0]}, index=days)
    fixed, applied = study.neutralize_price_breaks(closes, [("A", days[2], "spin-off")])
    returns = fixed.pct_change()
    assert returns["A"].iloc[2] == pytest.approx(0.0)
    assert returns["A"].iloc[1] == pytest.approx(41 / 40 - 1)
    pd.testing.assert_series_equal(fixed["B"], closes["B"])
    assert applied[0]["symbol"] == "A"
