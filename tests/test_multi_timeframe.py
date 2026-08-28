import math
from datetime import date

import pandas as pd

from portfolio_tracker.analytics.multi_timeframe import (
    MacroTrend,
    add_context_indicators,
    calculate_execution_levels,
    calculate_15_day_projection,
    calculate_horizon_projections,
    calculate_liquidity_zones,
    classify_macro_trend,
    resample_ohlcv,
    strict_confluence_gate,
)


def _ohlcv(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [price - 0.10 for price in prices],
            "High": [price + 0.30 for price in prices],
            "Low": [price - 0.30 for price in prices],
            "Close": prices,
            "Volume": [1000 + item * 10 for item in range(len(prices))],
        },
        index=index,
    )


def test_resample_builds_hourly_ohlcv_without_network_calls() -> None:
    index = pd.date_range("2026-08-27 09:30", periods=24, freq="5min")
    source = _ohlcv(index, [40 + item * 0.1 for item in range(24)])
    hourly = resample_ohlcv(source, "60min")

    assert len(hourly) == 2
    assert hourly.iloc[0]["Open"] == source.iloc[0]["Open"]
    assert hourly.iloc[0]["Close"] == source.iloc[11]["Close"]
    assert hourly.iloc[0]["High"] == source.iloc[:12]["High"].max()
    assert hourly.iloc[0]["Volume"] == source.iloc[:12]["Volume"].sum()


def test_macro_classifier_requires_price_ema_and_macd_alignment() -> None:
    index = pd.date_range("2022-01-07", periods=220, freq="W-FRI")
    rising = [20 + item * 0.25 + math.sin(item / 9) for item in range(len(index))]
    falling = list(reversed(rising))

    assert classify_macro_trend(add_context_indicators(_ohlcv(index, rising))) is MacroTrend.STRONG_BULLISH
    assert classify_macro_trend(add_context_indicators(_ohlcv(index, falling))) is MacroTrend.STRONG_BEARISH


def test_liquidity_zones_are_ranked_and_bounded() -> None:
    index = pd.date_range("2025-01-01", periods=252, freq="B")
    prices = [35 + (item % 40) * 0.2 for item in range(len(index))]
    zones = calculate_liquidity_zones(_ohlcv(index, prices))

    assert len(zones) == 3
    assert all(zone.lower < zone.center < zone.upper for zone in zones)
    assert all(0 < zone.volume_share_pct < 100 for zone in zones)


def test_strict_gate_caps_buy_when_weekly_macro_is_against() -> None:
    decision = strict_confluence_gate(
        probability_up=72.0,
        signal="BUY",
        weekly_trend=MacroTrend.STRONG_BEARISH,
        monthly_trend=MacroTrend.NEUTRAL,
        adx=28.0,
        overextended_unconfirmed=False,
    )

    assert decision.risk_veto
    assert decision.operation_probability == 39.0
    assert decision.probability_up == 72.0
    assert decision.probability_down == 28.0
    assert decision.alert == "⚠️ RIESGO ALTO / TENDENCIA MACRO EN CONTRA: EVITAR OPERACIÓN"


def test_strict_gate_leaves_monthly_countertrend_short_to_tactical_policy() -> None:
    decision = strict_confluence_gate(
        probability_up=25.0,
        signal="SELL",
        weekly_trend=MacroTrend.NEUTRAL,
        monthly_trend=MacroTrend.STRONG_BULLISH,
        adx=30.0,
        overextended_unconfirmed=False,
    )

    assert not decision.risk_veto
    assert decision.operation_probability == 75.0
    assert decision.probability_down == 75.0
    assert decision.probability_up == 25.0


def test_strict_gate_vetoes_lateral_or_overextended_operation() -> None:
    lateral = strict_confluence_gate(
        70.0, "BUY", MacroTrend.BULLISH, MacroTrend.BULLISH, 18.0, False
    )
    overextended = strict_confluence_gate(
        70.0, "BUY", MacroTrend.BULLISH, MacroTrend.BULLISH, 25.0, True
    )
    no_signal = strict_confluence_gate(
        60.0, "NEUTRAL", MacroTrend.STRONG_BEARISH, MacroTrend.STRONG_BEARISH, 10.0, True
    )

    assert lateral.risk_veto and lateral.operation_probability < 40
    assert overextended.risk_veto and overextended.operation_probability < 40
    assert not no_signal.risk_veto
    assert no_signal.scenario.startswith("SIN OPERACIÓN")


def test_strict_gate_forces_conditional_wait_when_stochastic_is_extreme() -> None:
    decision = strict_confluence_gate(
        62.0,
        "BUY",
        MacroTrend.BULLISH,
        MacroTrend.BULLISH,
        28.0,
        False,
        conditional_block_reason="Estocástico RSI sobre 80.",
    )

    assert decision.risk_veto
    assert decision.operation_probability == 0.0
    assert decision.alert == "ESPERAR CORRECCIÓN / ENTRADA LONG BLOQUEADA"
    assert decision.scenario.startswith("PLAN CONDICIONAL")


def test_horizon_projections_cover_requested_periods_and_sum_to_100() -> None:
    hourly = add_context_indicators(_ohlcv(pd.date_range("2026-01-01", periods=220, freq="h"), [20 + item * 0.05 for item in range(220)]))
    daily = add_context_indicators(_ohlcv(pd.date_range("2025-01-01", periods=320, freq="B"), [25 + item * 0.08 for item in range(320)]))
    weekly = resample_ohlcv(daily, "W-FRI")
    weekly = add_context_indicators(weekly)
    monthly = resample_ohlcv(daily, "ME")
    monthly = add_context_indicators(monthly)

    last_price = float(daily.iloc[-1]["Close"])
    projections = calculate_horizon_projections(
        72.0, False, last_price, hourly, hourly, daily, weekly, monthly
    )

    assert [item.label for item in projections] == [
        "1 Hora", "6 Horas", "1 Día", "1 Semana", "1 Mes", "6 Meses"
    ]
    assert all(math.isclose(item.probability_up + item.probability_range + item.probability_down, 100.0) for item in projections)
    assert all(0 <= value <= 100 for item in projections for value in (item.probability_up, item.probability_range, item.probability_down))
    assert all(item.bullish_target >= last_price for item in projections)
    assert all(0 < item.bearish_target <= last_price for item in projections)
    assert all(0 < item.range_low <= last_price <= item.range_high for item in projections)
    assert all(item.atr_value > 0 for item in projections)
    assert all(
        math.isfinite(value)
        for item in projections
        for value in (
            item.bullish_target,
            item.range_low,
            item.range_high,
            item.bearish_target,
            item.local_support,
            item.local_resistance,
        )
    )
    one_day = projections[2]
    assert one_day.bullish_target - last_price <= one_day.atr_value * 1.6 + 0.01
    assert last_price - one_day.bearish_target <= one_day.atr_value * 1.6 + 0.01


def test_risk_veto_increases_short_term_uncertainty() -> None:
    frame = add_context_indicators(_ohlcv(pd.date_range("2025-01-01", periods=260, freq="B"), [30 + item * 0.1 for item in range(260)]))
    last_price = float(frame.iloc[-1]["Close"])
    normal = calculate_horizon_projections(75.0, False, last_price, frame, frame, frame, frame, frame)
    vetoed = calculate_horizon_projections(75.0, True, last_price, frame, frame, frame, frame, frame)

    assert vetoed[0].probability_range > normal[0].probability_range
    assert vetoed[2].probability_range > normal[2].probability_range
    assert vetoed[-1].probability_range == normal[-1].probability_range


def test_execution_levels_are_positive_and_strictly_ordered() -> None:
    frame = add_context_indicators(_ohlcv(pd.date_range("2025-01-01", periods=300, freq="B"), [30 + item * 0.03 for item in range(300)]))
    last_price = float(frame.iloc[-1]["Close"])
    projections = calculate_horizon_projections(68.0, False, last_price, frame, frame, frame, frame, frame)
    levels = calculate_execution_levels(
        last_price,
        projections,
        (last_price - 1.25,),
        intraday_atr=0.42,
    )

    assert 0 < levels.stop_loss < levels.entry_low <= levels.entry_high <= last_price
    assert last_price <= levels.take_profit_1 <= levels.take_profit_2
    assert levels.reference_support > 0
    assert 2.0 <= levels.stop_atr_multiple <= 2.5
    assert levels.stop_loss <= round(
        levels.entry_low - levels.stop_atr_multiple * levels.atr_5m,
        2,
    )


def test_double_bottom_target_replaces_nearby_intraday_tp1() -> None:
    frame = add_context_indicators(_ohlcv(pd.date_range("2025-01-01", periods=300, freq="B"), [30 + item * 0.03 for item in range(300)]))
    last_price = float(frame.iloc[-1]["Close"])
    projections = calculate_horizon_projections(68.0, False, last_price, frame, frame, frame, frame, frame)
    pattern_target = last_price + 2.35
    levels = calculate_execution_levels(
        last_price,
        projections,
        additional_supports=(last_price - 1.25,),
        probability_up=68.0,
        signal="BUY",
        intraday_atr=0.42,
        confirmed_pattern_target=pattern_target,
        confirmed_pattern_label="Doble suelo 5m",
    )

    assert levels.take_profit_1 == round(pattern_target, 2)
    assert levels.take_profit_2 >= levels.take_profit_1
    assert levels.pattern_target_applied
    assert levels.pattern_target_label == "Doble suelo 5m"


def test_execution_levels_reverse_cleanly_for_bearish_bias() -> None:
    frame = add_context_indicators(_ohlcv(pd.date_range("2025-01-01", periods=300, freq="B"), [50 - item * 0.03 for item in range(300)]))
    last_price = float(frame.iloc[-1]["Close"])
    projections = calculate_horizon_projections(32.0, False, last_price, frame, frame, frame, frame, frame)
    levels = calculate_execution_levels(
        last_price,
        projections,
        additional_resistances=(last_price + 1.0,),
        probability_up=32.0,
        signal="SELL",
    )

    assert levels.direction == "SHORT"
    assert last_price <= levels.entry_high < levels.stop_loss
    assert 0 < levels.take_profit_2 <= levels.take_profit_1 <= last_price


def test_15_day_projection_has_business_dates_and_positive_daily_envelopes() -> None:
    frame = add_context_indicators(
        _ohlcv(
            pd.date_range("2025-01-01", periods=320, freq="B"),
            [30 + item * 0.04 + math.sin(item / 11) for item in range(320)],
        )
    )
    last_price = float(frame.iloc[-1]["Close"])
    points = calculate_15_day_projection(
        last_price=last_price,
        daily=frame,
        probability_up=63.0,
        risk_veto=False,
        as_of=date(2026, 8, 27),
    )

    assert len(points) == 15
    assert [item.day_number for item in points] == list(range(1, 16))
    assert all(item.session_date.weekday() < 5 for item in points)
    assert all(left.session_date < right.session_date for left, right in zip(points, points[1:]))
    assert all(0 < item.daily_floor <= item.expected_close <= item.daily_ceiling for item in points)
    assert all(item.atr_value > 0 for item in points)
    assert points[-1].daily_ceiling - points[-1].daily_floor > points[0].daily_ceiling - points[0].daily_floor


def test_15_day_projection_is_reproducible_and_has_historical_fluctuations() -> None:
    index = pd.date_range("2025-01-01", periods=320, freq="B")
    prices = [40 + item * 0.015 + math.sin(item * 1.7) * 1.1 + math.sin(item / 5) * 0.7 for item in range(320)]
    frame = add_context_indicators(_ohlcv(index, prices))
    kwargs = dict(
        last_price=float(frame.iloc[-1]["Close"]),
        daily=frame,
        probability_up=47.0,
        risk_veto=False,
        as_of=date(2026, 8, 27),
    )
    first = calculate_15_day_projection(**kwargs)
    second = calculate_15_day_projection(**kwargs)
    closes = [point.expected_close for point in first]
    changes = [right - left for left, right in zip(closes, closes[1:])]

    assert first == second
    assert len({round(change, 2) for change in changes}) >= 5
    assert any(change > 0 for change in changes)
    assert any(change < 0 for change in changes)
