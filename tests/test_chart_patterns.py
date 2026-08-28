import math
from pathlib import Path

import pandas as pd

from portfolio_tracker.analytics.chart_patterns import (
    ChartPattern,
    ChartPatternType,
    PatternDirection,
    detect_chart_patterns,
    evaluate_pattern_influence,
    scan_multi_timeframe_patterns,
)


def _interpolated_prices(length: int, anchors: list[tuple[int, float]]) -> list[float]:
    prices = [0.0] * length
    for (left_index, left_price), (right_index, right_price) in zip(
        anchors, anchors[1:]
    ):
        span = right_index - left_index
        for index in range(left_index, right_index + 1):
            ratio = (index - left_index) / span
            prices[index] = left_price + (right_price - left_price) * ratio
    return prices


def _ohlcv(
    prices: list[float],
    *,
    breakout_index: int | None = None,
    breakout_volume: float = 3_000.0,
    frequency: str = "5min",
) -> pd.DataFrame:
    index = pd.date_range("2026-01-02 14:30", periods=len(prices), freq=frequency)
    volumes = [1_000.0 + (position % 5) * 15 for position in range(len(prices))]
    if breakout_index is not None:
        volumes[breakout_index] = breakout_volume
    return pd.DataFrame(
        {
            "Open": [price - 0.06 for price in prices],
            "High": [price + 0.35 for price in prices],
            "Low": [price - 0.35 for price in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=index,
    )


def _double_bottom_frame(*, breakout_volume: float = 3_000.0) -> pd.DataFrame:
    prices = _interpolated_prices(
        100,
        [
            (0, 101.0),
            (20, 105.0),
            (45, 94.0),
            (58, 103.0),
            (72, 94.10),
            (80, 104.0),
            (99, 106.0),
        ],
    )
    return _ohlcv(prices, breakout_index=80, breakout_volume=breakout_volume)


def _double_top_frame() -> pd.DataFrame:
    prices = _interpolated_prices(
        100,
        [
            (0, 99.0),
            (20, 95.0),
            (45, 106.0),
            (58, 97.0),
            (72, 105.90),
            (80, 96.0),
            (99, 94.0),
        ],
    )
    return _ohlcv(prices, breakout_index=80, breakout_volume=3_100.0)


def test_detects_confirmed_double_bottom_with_objective_evidence() -> None:
    patterns = detect_chart_patterns(_double_bottom_frame(), timeframe="5m")
    pattern = next(
        item for item in patterns if item.pattern_type is ChartPatternType.DOUBLE_BOTTOM
    )

    assert pattern.valid
    assert pattern.direction is PatternDirection.BULLISH
    assert pattern.confidence > 75.0
    assert pattern.volume_ratio > 1.0
    assert pattern.rsi_divergence or pattern.macd_divergence
    assert 102.0 < pattern.neckline < 104.0
    assert pattern.target_price > pattern.neckline
    expected_target = pattern.neckline + (
        pattern.neckline - min(pattern.pivot_prices)
    )
    assert math.isclose(pattern.target_price, expected_target, abs_tol=0.001)


def test_detects_confirmed_double_top_symmetrically() -> None:
    patterns = detect_chart_patterns(_double_top_frame(), timeframe="5m")
    pattern = next(
        item for item in patterns if item.pattern_type is ChartPatternType.DOUBLE_TOP
    )

    assert pattern.valid
    assert pattern.direction is PatternDirection.BEARISH
    assert pattern.confidence > 75.0
    assert pattern.volume_ratio > 1.0
    assert pattern.rsi_divergence or pattern.macd_divergence
    assert pattern.target_price < pattern.neckline


def test_false_double_bottom_without_volume_confirmation_never_becomes_valid() -> None:
    patterns = detect_chart_patterns(
        _double_bottom_frame(breakout_volume=650.0), timeframe="5m"
    )
    candidates = [
        item
        for item in patterns
        if item.pattern_type is ChartPatternType.DOUBLE_BOTTOM
    ]

    assert candidates
    assert not candidates[0].valid
    assert not candidates[0].confirmed
    assert candidates[0].volume_ratio <= 1.0


def test_range_breakout_requires_more_than_1_2x_volume() -> None:
    base = [50.0 + math.sin(position / 3) * 0.15 for position in range(59)]
    strong = _ohlcv(base + [51.4], breakout_index=59, breakout_volume=2_600.0)
    weak = _ohlcv(base + [51.4], breakout_index=59, breakout_volume=1_050.0)

    strong_pattern = next(
        item
        for item in detect_chart_patterns(strong, timeframe="5m")
        if item.pattern_type is ChartPatternType.RANGE_BREAKOUT_UP
    )
    weak_pattern = next(
        item
        for item in detect_chart_patterns(weak, timeframe="5m")
        if item.pattern_type is ChartPatternType.RANGE_BREAKOUT_UP
    )

    assert strong_pattern.valid
    assert strong_pattern.volume_ratio > 1.2
    assert not weak_pattern.confirmed
    assert not weak_pattern.valid


def test_confirmed_opposite_intraday_pattern_activates_veto() -> None:
    bearish = ChartPattern(
        pattern_type=ChartPatternType.DOUBLE_TOP,
        direction=PatternDirection.BEARISH,
        timeframe="5m",
        confidence=88.0,
        neckline=98.0,
        target_price=92.0,
        confirmed=True,
        volume_ratio=1.8,
        rsi_divergence=True,
        macd_divergence=True,
        pivot_timestamps=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")),
        pivot_prices=(105.0, 105.1),
        detected_at=pd.Timestamp("2026-01-02"),
        detail="Estructura bajista confirmada.",
    )

    long_influence = evaluate_pattern_influence((bearish,), signal="BUY")
    short_influence = evaluate_pattern_influence((bearish,), signal="SELL")

    assert long_influence.veto
    assert long_influence.impact_points < 0
    assert "Doble techo" in long_influence.veto_reason
    assert not short_influence.veto


def test_multi_timeframe_scan_keeps_5m_and_daily_contracts() -> None:
    intraday = _double_bottom_frame()
    daily = _double_top_frame().copy()
    daily.index = pd.date_range("2025-01-01", periods=len(daily), freq="B")
    patterns = scan_multi_timeframe_patterns(intraday, daily)

    assert {item.timeframe for item in patterns} == {"5m", "1D"}
    assert any(item.valid for item in patterns if item.timeframe == "5m")
    assert any(item.valid for item in patterns if item.timeframe == "1D")


def test_real_smci_daily_snapshot_is_processed_without_optimistic_defaults() -> None:
    fixture = Path(__file__).parent / "fixtures" / "smci_daily_2026_sample.csv"
    frame = pd.read_csv(fixture, parse_dates=["Date"], index_col="Date")
    patterns = detect_chart_patterns(frame, timeframe="1D")

    assert all(0.0 <= item.confidence <= 100.0 for item in patterns)
    assert all(item.neckline > 0 and item.target_price > 0 for item in patterns)
    assert all(
        item.volume_ratio > 1.2
        for item in patterns
        if item.valid
        and item.pattern_type
        in {
            ChartPatternType.RANGE_BREAKOUT_UP,
            ChartPatternType.RANGE_BREAKOUT_DOWN,
        }
    )
