from dataclasses import replace

from portfolio_tracker.services.projection_chart import (
    HORIZON_ORDER,
    PROJECTION_DAYS,
    build_15_day_projection_figure,
    build_15_day_projection_pdf_drawing,
    ordered_daily_projection,
    ordered_horizon_projections,
)

from test_pdf_report import _analysis


def test_projection_order_is_chronological_even_when_input_is_reversed() -> None:
    analysis = _analysis()
    ordered = ordered_horizon_projections(tuple(reversed(analysis.horizon_projections)))

    assert tuple(item.label for item in ordered) == HORIZON_ORDER


def test_daily_projection_order_is_chronological_even_when_input_is_reversed() -> None:
    analysis = _analysis()
    ordered = ordered_daily_projection(tuple(reversed(analysis.daily_projection)))

    assert [item.day_number for item in ordered] == list(range(1, PROJECTION_DAYS + 1))
    assert all(left.session_date < right.session_date for left, right in zip(ordered, ordered[1:]))


def test_plotly_projection_contains_15_sequential_business_days() -> None:
    analysis = _analysis()
    figure = build_15_day_projection_figure(
        tuple(reversed(analysis.daily_projection)),
        analysis.last_price,
        analysis.daily_indicators,
    )

    expected_dates = tuple(item.session_date for item in analysis.daily_projection)
    assert figure.layout.xaxis.type == "date"
    history = next(trace for trace in figure.data if trace.name == "Histórico real · 30 sesiones")
    candles = next(trace for trace in figure.data if trace.name == "Velas proyectadas")
    assert len(history.close) == 30
    assert tuple(candles.x) == expected_dates
    assert len(candles.close) == PROJECTION_DAYS
    assert all(value > 0 for value in candles.open)
    assert all(low <= min(open_, close) for low, open_, close in zip(candles.low, candles.open, candles.close))
    assert all(high >= max(open_, close) for high, open_, close in zip(candles.high, candles.open, candles.close))
    assert all(
        high - low <= point.atr_value * 1.35 + 0.03
        for high, low, point in zip(candles.high, candles.low, analysis.daily_projection)
    )
    assert len(figure.data) == 2
    assert figure.layout.shapes
    assert figure.layout.yaxis.range is not None


def test_plotly_defensively_clips_legacy_extreme_projection_wicks() -> None:
    analysis = _analysis()
    extreme = tuple(
        replace(
            point,
            daily_floor=max(0.01, point.expected_close * 0.20),
            daily_ceiling=point.expected_close * 3.0,
        )
        for point in analysis.daily_projection
    )

    figure = build_15_day_projection_figure(
        extreme,
        analysis.last_price,
        analysis.daily_indicators,
    )
    candles = next(trace for trace in figure.data if trace.name == "Velas proyectadas")

    assert all(
        high - low <= point.atr_value * 1.35 + 0.03
        for high, low, point in zip(candles.high, candles.low, extreme)
    )


def test_projection_rejects_incoherent_or_short_history() -> None:
    analysis = _analysis()
    short_history = analysis.daily_indicators.tail(29)

    try:
        build_15_day_projection_figure(
            analysis.daily_projection,
            analysis.last_price,
            short_history,
        )
    except ValueError as exc:
        assert "30 sesiones" in str(exc)
    else:
        raise AssertionError("Se aceptó un histórico menor a 30 sesiones")


def test_reportlab_projection_drawing_is_positive_and_compact() -> None:
    analysis = _analysis()
    drawing = build_15_day_projection_pdf_drawing(
        analysis.daily_projection,
        analysis.last_price,
        width=493,
    )

    assert drawing.width == 493
    assert 150 <= drawing.height <= 220
    assert len(drawing.contents) > 30
