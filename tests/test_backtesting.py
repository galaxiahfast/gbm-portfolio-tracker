import math

import numpy as np
import pandas as pd

from portfolio_tracker.analytics.backtesting import (
    BacktestConfig,
    PerformanceMetrics,
    batch_to_payload,
    evaluate_capital_preservation,
    payload_sha256,
    run_backtest_batch,
    run_symbol_backtest,
)


def synthetic_ohlcv(periods: int = 1_600, seed: int = 7) -> pd.DataFrame:
    index = pd.bdate_range("2019-01-02", periods=periods)
    rng = np.random.default_rng(seed)
    returns = (
        0.0002
        + 0.018 * np.sin(np.arange(periods) / 13)
        + rng.normal(0, 0.012, periods)
    )
    close = 40 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, periods))
    span = np.maximum(close * 0.012, np.abs(close - open_))
    high = np.maximum(open_, close) + span * rng.uniform(0.2, 1.0, periods)
    low = np.minimum(open_, close) - span * rng.uniform(0.2, 1.0, periods)
    volume = 1_000_000 * (
        1 + 1.5 * np.abs(returns) / 0.03 + rng.uniform(0, 0.4, periods)
    )
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_out_of_sample_split_is_chronological_and_costs_are_applied() -> None:
    config = BacktestConfig(minimum_probability=0.50)
    result = run_symbol_backtest(
        "TEST", synthetic_ohlcv(), config, starting_capital_usd=10_000
    )

    assert result.training_trades
    assert result.validation_trades
    assert all(trade.exit_date < result.split_date for trade in result.training_trades)
    assert all(trade.signal_date >= result.split_date for trade in result.validation_trades)
    assert all(trade.stop_price > 0 and trade.target_price > 0 for trade in result.validation_trades)
    assert all(math.isclose(trade.reward_risk, 2.0, abs_tol=1e-4) for trade in result.validation_trades)
    assert all(
        math.isclose(
            trade.net_pnl_usd,
            trade.gross_pnl_usd - trade.costs_usd,
            abs_tol=2e-5,
        )
        for trade in result.validation_trades
    )
    assert result.validation.costs_usd > 0


def test_future_validation_prices_do_not_change_training_results() -> None:
    frame = synthetic_ohlcv()
    config = BacktestConfig(minimum_probability=0.50)
    baseline = run_symbol_backtest(
        "TEST", frame, config, starting_capital_usd=10_000
    )
    changed = frame.copy()
    mutation_start = int(len(changed) * 0.80)
    changed.iloc[mutation_start:, changed.columns.get_loc("Close")] *= 1.35
    changed.iloc[mutation_start:, changed.columns.get_loc("Open")] *= 1.35
    changed.iloc[mutation_start:, changed.columns.get_loc("High")] *= 1.35
    changed.iloc[mutation_start:, changed.columns.get_loc("Low")] *= 1.35
    rerun = run_symbol_backtest(
        "TEST", changed, config, starting_capital_usd=10_000
    )

    assert baseline.split_date == rerun.split_date
    assert baseline.training == rerun.training
    assert baseline.training_trades == rerun.training_trades


def test_capital_preservation_rejects_weak_out_of_sample_metrics() -> None:
    weak = PerformanceMetrics(
        setups=30,
        trades=10,
        rejected=20,
        wins=3,
        losses=7,
        win_rate=0.30,
        win_rate_lower_bound=0.15,
        profit_factor=0.65,
        maximum_drawdown_pct=18.0,
        net_return_pct=-4.0,
        gross_profit_usd=300,
        gross_loss_usd=-460,
        costs_usd=80,
        brier_score=0.31,
    )

    decision, reasons = evaluate_capital_preservation(weak, BacktestConfig())

    assert decision == "RECHAZADO"
    assert any("Profit factor" in reason for reason in reasons)
    assert any("Drawdown" in reason for reason in reasons)


def test_batch_payload_is_deterministic_and_hashable() -> None:
    batch = run_backtest_batch(
        {"AAA": synthetic_ohlcv(seed=11), "BBB": synthetic_ohlcv(seed=12)},
        BacktestConfig(minimum_probability=0.50),
        starting_capital_usd=5_000,
    )

    payload = batch_to_payload(batch)
    assert payload == batch_to_payload(batch)
    assert len(payload_sha256(payload)) == 64
    assert len(batch.results) == 2
