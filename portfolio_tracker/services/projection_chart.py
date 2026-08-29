"""Visualizacion compartida para horizontes y proyeccion diaria ejecutiva."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, PolyLine, String
from reportlab.lib import colors

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from portfolio_tracker.analytics.multi_timeframe import (
        DailyProjectionPoint,
        HorizonProjection,
    )


HORIZON_ORDER = (
    "1 Hora",
    "6 Horas",
    "1 Día",
    "1 Semana",
    "1 Mes",
    "6 Meses",
)
PROJECTION_DAYS = 15


def ordered_horizon_projections(
    projections: Sequence["HorizonProjection"],
) -> tuple["HorizonProjection", ...]:
    """Ordena el mapa tabular de probabilidades; no alimenta la grafica diaria."""

    by_label = {item.label: item for item in projections}
    missing = [label for label in HORIZON_ORDER if label not in by_label]
    duplicated = len(by_label) != len(projections)
    unexpected = [label for label in by_label if label not in HORIZON_ORDER]
    if missing or duplicated or unexpected:
        details = []
        if missing:
            details.append(f"faltan: {', '.join(missing)}")
        if duplicated:
            details.append("hay horizontes duplicados")
        if unexpected:
            details.append(f"no reconocidos: {', '.join(unexpected)}")
        raise ValueError(f"Mapa de horizontes incompleto ({'; '.join(details)}).")
    return tuple(by_label[label] for label in HORIZON_ORDER)


def ordered_daily_projection(
    points: Sequence["DailyProjectionPoint"],
) -> tuple["DailyProjectionPoint", ...]:
    """Valida y ordena los 15 puntos diarios por fecha y numero de sesion."""

    ordered = tuple(sorted(points, key=lambda item: (item.session_date, item.day_number)))
    expected_numbers = tuple(range(1, PROJECTION_DAYS + 1))
    if len(ordered) != PROJECTION_DAYS or tuple(item.day_number for item in ordered) != expected_numbers:
        raise ValueError("La trayectoria ejecutiva debe contener exactamente Dia 1 a Dia 15.")
    if len({item.session_date for item in ordered}) != PROJECTION_DAYS:
        raise ValueError("La trayectoria ejecutiva contiene fechas duplicadas.")
    return ordered


def build_15_day_projection_figure(
    points: Sequence["DailyProjectionPoint"],
    current_price: float,
    historical_daily=None,
) -> "Figure":
    """Une 30 sesiones OHLC reales con 15 velas estadísticas futuras."""

    import plotly.graph_objects as go
    import pandas as pd

    ordered = ordered_daily_projection(points)
    projection_dates = [item.session_date for item in ordered]
    expected = [item.expected_close for item in ordered]
    floors = [item.daily_floor for item in ordered]
    ceilings = [item.daily_ceiling for item in ordered]
    opens = [current_price, *expected[:-1]]
    highs = [
        max(open_price, close_price, ceiling)
        for open_price, close_price, ceiling in zip(opens, expected, ceilings)
    ]
    lows = [
        min(open_price, close_price, floor)
        for open_price, close_price, floor in zip(opens, expected, floors)
    ]
    if historical_daily is None or historical_daily.empty:
        raise ValueError("Se requieren velas históricas para enlazar la proyección.")
    required = ("Open", "High", "Low", "Close")
    missing = [column for column in required if column not in historical_daily.columns]
    if missing:
        raise ValueError("Faltan columnas históricas OHLC: " + ", ".join(missing) + ".")
    history = historical_daily.loc[:, list(required)].copy()
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    history = history[~history.index.duplicated(keep="last")].sort_index()
    for column in required:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna().tail(30)
    if len(history) != 30:
        raise ValueError("Se requieren exactamente 30 sesiones históricas utilizables.")
    invalid = (
        (history[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (history["High"] < history[["Open", "Close", "Low"]].max(axis=1))
        | (history["Low"] > history[["Open", "Close", "High"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError("El histórico contiene velas OHLC matemáticamente inválidas.")
    history_dates = list(history.index)
    if pd.Timestamp(history_dates[-1]).date() >= projection_dates[0]:
        raise ValueError("La proyección debe comenzar después del último dato histórico.")

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=history_dates,
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            increasing={"line": {"color": "#10B981"}, "fillcolor": "#10B981"},
            decreasing={"line": {"color": "#F43F5E"}, "fillcolor": "#F43F5E"},
            name="Histórico real · 30 sesiones",
            hovertext=["Sesión observada"] * len(history),
        )
    )
    figure.add_trace(
        go.Candlestick(
            x=projection_dates,
            open=opens,
            high=highs,
            low=lows,
            close=expected,
            increasing={"line": {"color": "#22C55E"}, "fillcolor": "#22C55E"},
            decreasing={"line": {"color": "#EF4444"}, "fillcolor": "#EF4444"},
            name="Velas proyectadas",
            customdata=[item.day_number for item in ordered],
            hovertext=[
                f"Día {item.day_number} · escenario estadístico"
                for item in ordered
            ],
        )
    )
    boundary = datetime.combine(projection_dates[0], time.min) - timedelta(hours=12)
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 16, "b": 22},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={
            "title": None,
            "type": "date",
            "tickformat": "%d/%m",
            "dtick": "D5",
            "rangebreaks": [{"bounds": ["sat", "mon"]}],
            "showgrid": False,
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Precio histórico y proyectado (USD)",
            "tickprefix": "$",
            "tickformat": ",.2f",
            "gridcolor": "rgba(148, 163, 184, 0.18)",
            "zeroline": False,
        },
    )
    figure.add_vline(
        x=boundary,
        line_width=1.5,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Inicio de proyección",
        annotation_position="top right",
    )
    return figure


def build_15_day_projection_pdf_drawing(
    points: Sequence["DailyProjectionPoint"],
    current_price: float,
    *,
    width: float,
    height: float = 178,
) -> Drawing:
    """Dibuja la misma proyeccion diaria como vector nativo de ReportLab."""

    ordered = ordered_daily_projection(points)
    expected = [item.expected_close for item in ordered]
    floors = [item.daily_floor for item in ordered]
    ceilings = [item.daily_ceiling for item in ordered]
    all_values = expected + floors + ceilings + [current_price]
    value_min, value_max = min(all_values), max(all_values)
    padding = max((value_max - value_min) * 0.10, current_price * 0.003, 0.10)
    value_min = max(0.0, value_min - padding)
    value_max += padding

    left, right, bottom, top = 42.0, 10.0, 32.0, 24.0
    plot_width = width - left - right
    plot_height = height - bottom - top

    def x_at(index: int) -> float:
        return left + plot_width * index / (PROJECTION_DAYS - 1)

    def y_at(value: float) -> float:
        return bottom + (value - value_min) / (value_max - value_min) * plot_height

    drawing = Drawing(width, height)
    drawing.add(String(left, height - 9, "Cierre esperado", fontName="Helvetica-Bold", fontSize=7.5, fillColor=colors.HexColor("#2563EB")))
    drawing.add(String(left + 86, height - 9, "Rango diario ATR", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#2563EB")))
    drawing.add(String(left + 178, height - 9, "Precio actual", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#64748B")))

    for step in range(5):
        value = value_min + (value_max - value_min) * step / 4
        y = y_at(value)
        drawing.add(Line(left, y, width - right, y, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.5))
        drawing.add(String(left - 5, y - 2.5, f"${value:.2f}", fontName="Helvetica", fontSize=6.2, fillColor=colors.HexColor("#64748B"), textAnchor="end"))

    band_points: list[float] = []
    for index, value in enumerate(ceilings):
        band_points.extend((x_at(index), y_at(value)))
    for index in reversed(range(PROJECTION_DAYS)):
        band_points.extend((x_at(index), y_at(floors[index])))
    drawing.add(Polygon(band_points, fillColor=colors.HexColor("#DBEAFE"), strokeColor=None))

    current_y = y_at(current_price)
    current_line = Line(left, current_y, width - right, current_y, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.8)
    current_line.strokeDashArray = [3, 2]
    drawing.add(current_line)

    points_flat: list[float] = []
    for index, value in enumerate(expected):
        points_flat.extend((x_at(index), y_at(value)))
    drawing.add(PolyLine(points_flat, strokeColor=colors.HexColor("#2563EB"), strokeWidth=2.4, fillColor=None))
    for index, value in enumerate(expected):
        drawing.add(Circle(x_at(index), y_at(value), 2.2, fillColor=colors.HexColor("#2563EB"), strokeColor=colors.white, strokeWidth=0.5))

    for index, item in enumerate(ordered):
        if index % 2 == 0 or index == PROJECTION_DAYS - 1:
            drawing.add(String(x_at(index), 17, f"D{item.day_number}", fontName="Helvetica-Bold", fontSize=6.2, fillColor=colors.HexColor("#475569"), textAnchor="middle"))
            drawing.add(String(x_at(index), 8, f"{item.session_date:%d/%m}", fontName="Helvetica", fontSize=5.8, fillColor=colors.HexColor("#64748B"), textAnchor="middle"))
    drawing.add(String(width - right, current_y + 3, f"Actual ${current_price:.2f}", fontName="Helvetica", fontSize=6.2, fillColor=colors.HexColor("#64748B"), textAnchor="end"))
    drawing.add(Line(left, bottom, width - right, bottom, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.7))
    return drawing
