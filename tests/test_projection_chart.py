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
    )

    expected_dates = tuple(item.session_date for item in analysis.daily_projection)
    assert figure.layout.xaxis.type == "date"
    assert all(tuple(trace.x) == expected_dates for trace in figure.data)
    central = next(trace for trace in figure.data if trace.name == "Cierre esperado")
    assert len(central.y) == PROJECTION_DAYS
    assert all(value > 0 for value in central.y)
    assert len(figure.layout.xaxis.ticktext) == PROJECTION_DAYS


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
