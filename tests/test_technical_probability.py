import math
from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest
from tests.market_fixtures import intraday_index as session_index

from portfolio_tracker.analytics.technical_probability import (
    CandlePattern,
    CloudPosition,
    DailyTrend,
    FibonacciLevels,
    MacroTrend,
    MomentumState,
    ObvState,
    TechnicalSignal,
    _phase2_probability,
    _detect_candlestick,
    _fibonacci_levels,
    _fibonacci_score,
    _monthly_execution_policy,
    add_daily_indicators,
    add_intraday_indicators,
    analyze_probability,
    preliminary_probability,
    validate_probability_analysis,
)


def _ohlcv(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [price - 0.08 for price in prices],
            "High": [price + 0.45 for price in prices],
            "Low": [price - 0.45 for price in prices],
            "Close": prices,
            "Volume": [1000 + (item % 23) * 70 for item in range(len(prices))],
        },
        index=index,
    )


def test_preliminary_probability_rewards_aligned_signal_and_volume() -> None:
    probability_up, _, warnings = preliminary_probability(
        TechnicalSignal.BUY, DailyTrend.BULLISH, True
    )
    assert probability_up == 70.0
    assert not warnings

    probability_up, verdict, warnings = preliminary_probability(
        TechnicalSignal.BUY, DailyTrend.BEARISH, True
    )
    assert probability_up == 35.0
    assert "no vale la pena" in verdict
    assert warnings


def test_analysis_builds_indicators_pivots_and_complementary_probabilities() -> None:
    intraday_index = session_index(180)
    intraday_prices = [
        40 + item * 0.015 + math.sin(item / 4) * 1.2
        for item in range(len(intraday_index))
    ]
    daily_index = pd.date_range("2026-05-01", periods=90, freq="B")
    daily_prices = [
        32 + item * 0.12 + math.sin(item / 6) * 1.5
        for item in range(len(daily_index))
    ]

    analysis = analyze_probability(
        "SMCI",
        _ohlcv(intraday_index, intraday_prices),
        _ohlcv(daily_index, daily_prices),
    )

    assert analysis.symbol == "SMCI"
    assert analysis.probability_up + analysis.probability_down == 100.0
    assert 0 <= analysis.stochastic_k <= 100
    assert analysis.bollinger_lower < analysis.bollinger_upper
    assert analysis.pivots.s2 < analysis.pivots.s1 < analysis.pivots.r1 < analysis.pivots.r2
    assert len(analysis.intraday_indicators) > 0
    assert analysis.ichimoku_5m in CloudPosition
    assert analysis.ichimoku_daily in CloudPosition
    assert analysis.fibonacci.low < analysis.fibonacci.level_618
    assert analysis.fibonacci.level_618 < analysis.fibonacci.level_500
    assert analysis.fibonacci.level_500 < analysis.fibonacci.level_382
    assert analysis.fibonacci.level_382 < analysis.fibonacci.high
    phase3_filters = {
        "Fibonacci mensual",
        "Ichimoku 5 min",
        "Ichimoku diario",
        "Vela japonesa 5 min",
    }
    assert phase3_filters.issubset({item.name for item in analysis.score_breakdown})
    assert analysis.weekly_support < analysis.weekly_resistance
    assert 0 < analysis.execution_levels.stop_loss < analysis.execution_levels.entry_low
    validate_probability_analysis(analysis)
    with pytest.raises(ValueError, match="no suman 100"):
        validate_probability_analysis(
            replace(analysis, probability_down=analysis.probability_down + 1)
        )
    assert analysis.execution_levels.entry_high <= analysis.last_price
    assert analysis.last_price <= analysis.execution_levels.take_profit_1
    assert analysis.execution_levels.take_profit_1 <= analysis.execution_levels.take_profit_2
    assert analysis.activation_trigger
    assert analysis.activation_trigger.startswith(("Activar LONG", "Activar SHORT"))


def test_extreme_oversold_at_support_enables_long_rebound_watch() -> None:
    intraday_index = session_index(220)
    intraday_prices = [
        40 + item * 0.015 + math.sin(item / 4) * 1.2
        for item in range(len(intraday_index))
    ]
    daily_index = pd.date_range("2021-01-01", periods=1400, freq="B")
    daily_prices = [
        30 + item * 0.025 + math.sin(item / 12) * 1.8
        for item in range(len(daily_index))
    ]

    analysis = analyze_probability(
        "GENERIC",
        _ohlcv(intraday_index, intraday_prices),
        _ohlcv(daily_index, daily_prices),
    )

    assert analysis.stochastic_k == 0.0
    assert analysis.stoch_oversold_extreme
    assert analysis.rebound_watch_active
    assert analysis.support_interaction
    assert analysis.signal is TechnicalSignal.WATCH_BUY
    assert analysis.execution_levels.direction == "LONG"
    assert "Activar LONG" in analysis.activation_trigger
    assert not analysis.activation_trigger_met


def test_analysis_ignores_provisional_zero_volume_bar() -> None:
    intraday_index = session_index(180)
    intraday_prices = [
        40 + item * 0.015 + math.sin(item / 4) * 1.2
        for item in range(len(intraday_index))
    ]
    daily_index = pd.date_range("2026-04-01", periods=110, freq="B")
    daily_prices = [
        32 + item * 0.12 + math.sin(item / 6) * 1.5
        for item in range(len(daily_index))
    ]
    intraday = _ohlcv(intraday_index, intraday_prices)
    daily = _ohlcv(daily_index, daily_prices)
    cutoff = intraday.index[-1] + timedelta(minutes=5)
    baseline = analyze_probability("SMCI", intraday, daily, as_of_time=cutoff)

    provisional_time = intraday.index[-1] + timedelta(minutes=5)
    provisional = pd.DataFrame(
        {
            "Open": [999.0],
            "High": [1000.0],
            "Low": [998.0],
            "Close": [999.5],
            "Volume": [0],
        },
        index=pd.DatetimeIndex([provisional_time]),
    )
    with_provisional = analyze_probability(
        "SMCI", pd.concat([intraday, provisional]), daily, as_of_time=cutoff
    )

    assert with_provisional.as_of == baseline.as_of
    assert with_provisional.last_price == baseline.last_price
    assert with_provisional.signal is baseline.signal


def test_phase2_indicators_are_finite_and_vwap_resets_each_session() -> None:
    index = pd.date_range("2026-08-20 13:30", periods=220, freq="5min", tz="UTC")
    prices = [40 + item * 0.01 + math.sin(item / 5) for item in range(len(index))]
    intraday = add_intraday_indicators(_ohlcv(index, prices))
    latest = intraday.dropna(subset=["MACD_signal", "VWAP", "ADX14"]).iloc[-1]

    assert all(
        math.isfinite(float(latest[column]))
        for column in ("MACD", "MACD_signal", "VWAP", "ADX14", "OBV")
    )
    assert 0 <= float(latest["ADX14"]) <= 100

    two_sessions = _ohlcv(
        pd.DatetimeIndex([
            "2026-08-20 19:55:00+00:00",
            "2026-08-21 13:30:00+00:00",
        ]),
        [40.0, 50.0],
    )
    with_vwap = add_intraday_indicators(two_sessions)
    second_typical_price = (
        with_vwap.iloc[1]["High"]
        + with_vwap.iloc[1]["Low"]
        + with_vwap.iloc[1]["Close"]
    ) / 3
    assert math.isclose(with_vwap.iloc[1]["VWAP"], second_typical_price)

    daily = add_daily_indicators(_ohlcv(pd.date_range("2026-01-01", periods=80), prices[:80]))
    assert math.isfinite(float(daily.dropna(subset=["MACD_signal"]).iloc[-1]["MACD"]))


def test_phase2_macd_vetoes_conflicting_buy_signal() -> None:
    probability, verdict, warnings, breakdown, rejected = _phase2_probability(
        TechnicalSignal.BUY,
        DailyTrend.BULLISH,
        True,
        MomentumState.BEARISH,
        MomentumState.BULLISH,
        True,
        28.0,
        ObvState.CONFIRMING_UP,
    )

    assert rejected
    assert probability <= 42.0
    assert "invalidada" in verdict.lower()
    assert any("MACD" in warning for warning in warnings)
    assert any(item.name == "Veto MACD intradía" for item in breakdown)


def test_monthly_strong_bullish_makes_short_tactical_and_reduces_exposure() -> None:
    tactical, exposure, required_volume = _monthly_execution_policy(
        "SHORT",
        MacroTrend.STRONG_BULLISH,
    )
    normal = _monthly_execution_policy("LONG", MacroTrend.STRONG_BULLISH)

    assert tactical
    assert exposure == 0.50
    assert required_volume == 1.20
    assert normal == (False, 1.0, 1.0)


def test_phase2_adx_range_reduces_distance_from_neutral() -> None:
    strong, *_ = _phase2_probability(
        TechnicalSignal.BUY,
        DailyTrend.BULLISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BULLISH,
        True,
        28.0,
        ObvState.CONFIRMING_UP,
    )
    lateral, _, warnings, _, _ = _phase2_probability(
        TechnicalSignal.BUY,
        DailyTrend.BULLISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BULLISH,
        True,
        15.0,
        ObvState.CONFIRMING_UP,
    )

    assert abs(lateral - 50) < abs(strong - 50)
    assert any("Lateral" in warning for warning in warnings)


def test_stochastic_extreme_penalizes_bullish_bias_by_18_points() -> None:
    kwargs = dict(
        signal=TechnicalSignal.NEUTRAL,
        trend=DailyTrend.NEUTRAL,
        volume_confirmed=False,
        macd_5m=MomentumState.NEUTRAL,
        macd_daily=MomentumState.NEUTRAL,
        above_vwap=True,
        adx=22.0,
        obv_state=ObvState.NEUTRAL,
    )
    baseline, *_ = _phase2_probability(**kwargs)
    penalized, _, warnings, breakdown, _ = _phase2_probability(
        **kwargs,
        stoch_overbought_extreme=True,
    )

    assert penalized == baseline - 18.0
    assert any(
        item.name == "Bloqueo Estocástico RSI extremo"
        and item.impact_points == -18.0
        for item in breakdown
    )
    assert any("LONG bloqueada" in warning for warning in warnings)


def test_phase2_context_cannot_create_entry_without_primary_signal() -> None:
    probability, verdict, _, breakdown, rejected = _phase2_probability(
        TechnicalSignal.NEUTRAL,
        DailyTrend.BULLISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BULLISH,
        True,
        30.0,
        ObvState.CONFIRMING_UP,
    )

    assert not rejected
    assert probability == 74.0
    assert "sin gatillo" in verdict.lower()
    assert any(item.name == "Ausencia de disparador" for item in breakdown)


def test_phase2_daily_macd_conflict_applies_visible_penalty() -> None:
    aligned, *_ = _phase2_probability(
        TechnicalSignal.BUY,
        DailyTrend.BULLISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BULLISH,
        True,
        22.0,
        ObvState.CONFIRMING_UP,
    )
    conflicted, _, warnings, breakdown, rejected = _phase2_probability(
        TechnicalSignal.BUY,
        DailyTrend.BULLISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BEARISH,
        True,
        22.0,
        ObvState.CONFIRMING_UP,
    )

    assert not rejected
    assert conflicted < aligned
    assert any("8 puntos" in warning for warning in warnings)
    assert any(item.name == "Penalización MACD diario" for item in breakdown)


def test_short_term_momentum_overrides_slow_bullish_context_symmetrically() -> None:
    bearish, _, _, bearish_breakdown, _ = _phase2_probability(
        TechnicalSignal.NEUTRAL,
        DailyTrend.BULLISH,
        True,
        MomentumState.BEARISH,
        MomentumState.BULLISH,
        False,
        24.0,
        ObvState.NEUTRAL,
        candle_impact=-3.0,
        short_term_return_pct=-0.8,
        price_vs_vwap_pct=-0.6,
        macd_histogram_delta=-0.01,
    )
    bullish, _, _, bullish_breakdown, _ = _phase2_probability(
        TechnicalSignal.NEUTRAL,
        DailyTrend.BEARISH,
        True,
        MomentumState.BULLISH,
        MomentumState.BEARISH,
        True,
        24.0,
        ObvState.NEUTRAL,
        candle_impact=3.0,
        short_term_return_pct=0.8,
        price_vs_vwap_pct=0.6,
        macd_histogram_delta=0.01,
    )

    assert bearish < 50.0 < bullish
    assert any(item.name == "Retorno reciente 5 min" for item in bearish_breakdown)
    assert any(item.name == "Aceleración MACD 5 min" for item in bullish_breakdown)


def test_fibonacci_uses_previous_22_complete_daily_sessions() -> None:
    index = pd.date_range("2026-07-01", periods=30, freq="B")
    prices = [30.0 + item for item in range(len(index))]
    daily = _ohlcv(index, prices)
    session_date = (index[-1] + timedelta(days=1)).date()
    levels_price = 50.0
    levels = _fibonacci_levels(daily, session_date, price=levels_price)
    eligible = daily.tail(22)
    expected_high = float(eligible["High"].max())
    expected_low = float(eligible["Low"].min())

    assert levels.high == expected_high
    assert levels.low == expected_low
    assert math.isclose(
        levels.level_382,
        expected_high - (expected_high - expected_low) * 0.382,
    )
    assert levels.nearest_level == min(
        (levels.level_382, levels.level_500, levels.level_618),
        key=lambda value: abs(levels_price - value),
    )


def test_fibonacci_bonus_requires_signal_and_matching_zone_role() -> None:
    levels = FibonacciLevels(
        high=50.0,
        low=40.0,
        level_382=46.18,
        level_500=45.0,
        level_618=43.82,
        nearest_level=45.0,
        nearest_ratio="0.500",
        distance_pct=0.10,
        near_zone=True,
        role="SOPORTE",
        source_start="2026-07-01",
        source_end="2026-08-01",
    )

    bullish_impact, _ = _fibonacci_score(TechnicalSignal.BUY, levels)
    neutral_impact, _ = _fibonacci_score(TechnicalSignal.NEUTRAL, levels)
    bearish_impact, _ = _fibonacci_score(TechnicalSignal.SELL, levels)

    assert bullish_impact == 4.0
    assert neutral_impact == 0.0
    assert bearish_impact > 0  # Soporte cercano penaliza el escenario bajista.


def test_ichimoku_columns_use_rolling_windows_and_displacement() -> None:
    index = pd.date_range("2026-01-01", periods=100, freq="B")
    prices = [30 + item * 0.2 for item in range(len(index))]
    result = add_daily_indicators(_ohlcv(index, prices))
    latest = result.iloc[-1]

    assert all(
        math.isfinite(float(latest[column]))
        for column in (
            "Ichimoku_Tenkan",
            "Ichimoku_Kijun",
            "Ichimoku_Senkou_A",
            "Ichimoku_Senkou_B",
        )
    )
    expected_tenkan = (
        result["High"].iloc[-9:].max() + result["Low"].iloc[-9:].min()
    ) / 2
    assert math.isclose(float(latest["Ichimoku_Tenkan"]), expected_tenkan)


def test_candlestick_detects_hammer_and_bullish_engulfing() -> None:
    index = pd.date_range("2026-08-20 14:00", periods=2, freq="5min", tz="UTC")
    hammer = pd.DataFrame(
        {
            "Open": [10.0, 10.0],
            "High": [10.3, 10.25],
            "Low": [9.8, 9.0],
            "Close": [10.1, 10.2],
            "Volume": [1000, 1500],
        },
        index=index,
    )
    pattern, impact, _ = _detect_candlestick(hammer)
    assert pattern is CandlePattern.HAMMER
    assert impact > 0

    engulfing = pd.DataFrame(
        {
            "Open": [10.8, 9.7],
            "High": [11.0, 11.2],
            "Low": [9.8, 9.6],
            "Close": [10.0, 11.1],
            "Volume": [1000, 1800],
        },
        index=index,
    )
    pattern, impact, _ = _detect_candlestick(engulfing)
    assert pattern is CandlePattern.BULLISH_ENGULFING
    assert impact == 4.0
