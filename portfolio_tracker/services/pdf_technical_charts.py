"""Graficas tecnicas vectoriales para el reporte cuantitativo en PDF."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors

if TYPE_CHECKING:
    from portfolio_tracker.analytics.technical_probability import ProbabilityAnalysis


@dataclass(frozen=True, slots=True)
class TechnicalPdfChart:
    section: str
    title: str
    caption: str
    drawing: Drawing
    observation_count: int


def _finite_values(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for column in columns:
        for raw in frame[column].tolist():
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
    return values


def _technical_line_drawing(
    frame: pd.DataFrame,
    series: tuple[tuple[str, str, str], ...],
    *,
    width: float,
    height: float = 126,
    value_prefix: str = "",
    reference_lines: tuple[tuple[float, str], ...] = (),
    histogram_column: str | None = None,
) -> Drawing:
    """Renderiza lineas reales del DataFrame sin rasterizar ni crear temporales."""

    columns = tuple(column for column, _, _ in series)
    working = frame.loc[:, list(dict.fromkeys(columns + ((histogram_column,) if histogram_column else ())))].copy()
    working = working.dropna(how="all")
    if len(working) < 2:
        drawing = Drawing(width, height)
        drawing.add(
            Rect(
                42,
                22,
                width - 51,
                height - 44,
                fillColor=colors.HexColor("#F8FAFC"),
                strokeColor=colors.HexColor("#CBD5E1"),
                strokeWidth=0.7,
            )
        )
        drawing.add(
            String(
                width / 2,
                height / 2,
                "Datos insuficientes para completar este indicador",
                fontName="Helvetica",
                fontSize=8,
                fillColor=colors.HexColor("#64748B"),
                textAnchor="middle",
            )
        )
        return drawing

    values = _finite_values(working, columns)
    if histogram_column:
        values.extend(_finite_values(working, (histogram_column,)))
    values.extend(value for value, _ in reference_lines)
    if not values:
        raise ValueError("La grafica tecnica no contiene valores finitos.")
    value_min, value_max = min(values), max(values)
    padding = max((value_max - value_min) * 0.10, abs(value_max) * 0.01, 0.05)
    value_min -= padding
    value_max += padding

    left, right, bottom, top = 42.0, 9.0, 22.0, 22.0
    plot_width = width - left - right
    plot_height = height - bottom - top
    point_count = len(working)

    def x_at(index: int) -> float:
        return left + plot_width * index / (point_count - 1)

    def y_at(value: float) -> float:
        return bottom + (value - value_min) / (value_max - value_min) * plot_height

    drawing = Drawing(width, height)
    for step in range(4):
        value = value_min + (value_max - value_min) * step / 3
        y = y_at(value)
        drawing.add(Line(left, y, width - right, y, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.45))
        drawing.add(String(left - 5, y - 2.3, f"{value_prefix}{value:.2f}", fontName="Helvetica", fontSize=5.8, fillColor=colors.HexColor("#64748B"), textAnchor="end"))

    for reference, label in reference_lines:
        line = Line(left, y_at(reference), width - right, y_at(reference), strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.65)
        line.strokeDashArray = [3, 2]
        drawing.add(line)
        drawing.add(String(width - right, y_at(reference) + 2, label, fontName="Helvetica", fontSize=5.5, fillColor=colors.HexColor("#64748B"), textAnchor="end"))

    if histogram_column:
        zero_y = y_at(0.0)
        bar_width = max(0.6, plot_width / point_count * 0.72)
        for index, raw in enumerate(working[histogram_column].tolist()):
            value = float(raw)
            if not math.isfinite(value):
                continue
            y = y_at(value)
            drawing.add(
                Rect(
                    x_at(index) - bar_width / 2,
                    min(y, zero_y),
                    bar_width,
                    max(abs(y - zero_y), 0.4),
                    fillColor=colors.HexColor("#BBF7D0" if value >= 0 else "#FECACA"),
                    strokeColor=None,
                )
            )

    legend_slot = plot_width / max(len(series), 1)
    for position, (column, label, color_hex) in enumerate(series):
        color = colors.HexColor(color_hex)
        legend_x = left + position * legend_slot
        drawing.add(Line(legend_x, height - 8, legend_x + 12, height - 8, strokeColor=color, strokeWidth=1.8))
        drawing.add(String(legend_x + 15, height - 10.5, label, fontName="Helvetica", fontSize=6.2, fillColor=color))

        segment: list[float] = []
        for index, raw in enumerate(working[column].tolist()):
            value = float(raw)
            if math.isfinite(value):
                segment.extend((x_at(index), y_at(value)))
            elif len(segment) >= 4:
                drawing.add(PolyLine(segment, strokeColor=color, strokeWidth=1.15, fillColor=None))
                segment = []
        if len(segment) >= 4:
            drawing.add(PolyLine(segment, strokeColor=color, strokeWidth=1.15, fillColor=None))

    index_values = list(working.index)
    for position in (0, point_count // 2, point_count - 1):
        timestamp = pd.Timestamp(index_values[position])
        label = timestamp.strftime("%d/%m %H:%M") if timestamp.hour or timestamp.minute else timestamp.strftime("%d/%m/%y")
        drawing.add(String(x_at(position), 8, label, fontName="Helvetica", fontSize=5.8, fillColor=colors.HexColor("#64748B"), textAnchor="middle"))
    drawing.add(Line(left, bottom, width - right, bottom, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.7))
    return drawing


def build_technical_pdf_charts(
    analysis: "ProbabilityAnalysis",
    *,
    width: float,
) -> tuple[TechnicalPdfChart, ...]:
    """Construye todos los paneles graficos disponibles en la vista avanzada."""

    intraday = analysis.intraday_indicators.tail(78)
    daily = analysis.daily_indicators.tail(180)
    intraday_patterns = tuple(
        pattern for pattern in analysis.chart_patterns if pattern.timeframe == "5m"
    )[:2]
    daily_patterns = tuple(
        pattern for pattern in analysis.chart_patterns if pattern.timeframe == "1D"
    )[:2]

    def pattern_caption(patterns) -> str:  # type: ignore[no-untyped-def]
        if not patterns:
            return "No se detectaron figuras recientes que superen el filtro Zig-Zag; el precio se conserva como referencia objetiva."
        return " | ".join(
            f"{pattern.label}: {pattern.confidence:.1f}% ({'confirmado' if pattern.valid else 'sin validar'}), cuello ${pattern.neckline:.2f}"
            for pattern in patterns
        )

    def neckline_references(patterns) -> tuple[tuple[float, str], ...]:  # type: ignore[no-untyped-def]
        seen: set[float] = set()
        references: list[tuple[float, str]] = []
        for pattern in patterns:
            level = round(float(pattern.neckline), 4)
            if level in seen:
                continue
            seen.add(level)
            references.append((level, f"Cuello {pattern.confidence:.0f}%"))
        return tuple(references)

    charts = (
        TechnicalPdfChart(
            section="Intradia",
            title="Bandas de Bollinger y VWAP - 5 min",
            caption="Precio real de cierre, VWAP de sesion y Bandas de Bollinger (20,2) sobre las ultimas 78 velas utilizables.",
            drawing=_technical_line_drawing(
                intraday,
                (
                    ("Close", "Precio", "#0F172A"),
                    ("VWAP", "VWAP", "#7C3AED"),
                    ("BB_upper", "Banda superior", "#2563EB"),
                    ("BB_middle", "Media", "#64748B"),
                    ("BB_lower", "Banda inferior", "#2563EB"),
                ),
                width=width,
                value_prefix="$",
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="Patrones chartistas - 5 min",
            caption=pattern_caption(intraday_patterns),
            drawing=_technical_line_drawing(
                intraday,
                (("Close", "Precio", "#0F172A"),),
                width=width,
                value_prefix="$",
                reference_lines=neckline_references(intraday_patterns),
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="Estocastico RSI - 5 min",
            caption="Oscilador Stoch RSI (14,14,3,3); las guias 80 y 20 delimitan sobrecompra y sobreventa.",
            drawing=_technical_line_drawing(
                intraday,
                (
                    ("StochRSI_K", "%K", "#2563EB"),
                    ("StochRSI_D", "%D", "#D97706"),
                ),
                width=width,
                reference_lines=((80.0, "80"), (20.0, "20")),
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="MACD (12,26,9) - 5 min",
            caption="MACD y linea de senal sobre las ultimas 78 velas; barras verdes/rojas representan el histograma real.",
            drawing=_technical_line_drawing(
                intraday,
                (
                    ("MACD", "MACD", "#2563EB"),
                    ("MACD_signal", "Senal 9", "#D97706"),
                ),
                width=width,
                reference_lines=((0.0, "0"),),
                histogram_column="MACD_histogram",
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="Estructura diaria - EMA 9/21/50/200",
            caption="Cierres diarios reales y medias exponenciales del contexto multi-temporal sobre hasta 180 sesiones.",
            drawing=_technical_line_drawing(
                daily,
                (
                    ("Close", "Cierre", "#0F172A"),
                    ("EMA9", "EMA 9", "#16A34A"),
                    ("EMA21", "EMA 21", "#2563EB"),
                    ("EMA50", "EMA 50", "#D97706"),
                    ("EMA200", "EMA 200", "#DC2626"),
                ),
                width=width,
                value_prefix="$",
            ),
            observation_count=len(daily),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="Patrones chartistas - diario",
            caption=pattern_caption(daily_patterns),
            drawing=_technical_line_drawing(
                daily,
                (("Close", "Precio", "#0F172A"),),
                width=width,
                value_prefix="$",
                reference_lines=neckline_references(daily_patterns),
            ),
            observation_count=len(daily),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="Confirmacion MACD - 1 hora",
            caption="MACD horario agregado localmente desde velas de 5 minutos; valida o contradice el impulso rapido.",
            drawing=_technical_line_drawing(
                analysis.hourly_indicators.tail(40),
                (("MACD", "MACD", "#2563EB"), ("MACD_signal", "Senal 9", "#D97706")),
                width=width,
                reference_lines=((0.0, "0"),),
                histogram_column="MACD_histogram",
            ),
            observation_count=len(analysis.hourly_indicators.tail(40)),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="ADX y direccion - 5 min",
            caption="ADX 14 mide fuerza; +DI y -DI muestran la direccion dominante sobre las ultimas 78 velas.",
            drawing=_technical_line_drawing(
                intraday,
                (("ADX14", "ADX 14", "#7C3AED"), ("Plus_DI14", "+DI", "#16A34A"), ("Minus_DI14", "-DI", "#DC2626")),
                width=width,
                reference_lines=((20.0, "20"), (25.0, "25")),
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Intradia",
            title="Volumen acumulado OBV - 5 min",
            caption="OBV real acumulado para detectar confirmacion, acumulacion o distribucion frente al precio.",
            drawing=_technical_line_drawing(
                intraday,
                (("OBV", "OBV", "#0F766E"),),
                width=width,
            ),
            observation_count=len(intraday),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="Estructura semanal - EMA 9/21/50/200",
            caption="Cierres semanales y medias exponenciales agregados desde las sesiones diarias disponibles.",
            drawing=_technical_line_drawing(
                analysis.weekly_indicators.tail(156),
                (("Close", "Cierre", "#0F172A"), ("EMA9", "EMA 9", "#16A34A"), ("EMA21", "EMA 21", "#2563EB"), ("EMA50", "EMA 50", "#D97706"), ("EMA200", "EMA 200", "#DC2626")),
                width=width,
                value_prefix="$",
            ),
            observation_count=len(analysis.weekly_indicators.tail(156)),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="MACD macro - diario",
            caption="MACD diario con linea de senal e histograma para validar el contexto de swing trading.",
            drawing=_technical_line_drawing(
                analysis.daily_indicators.tail(180),
                (("MACD", "MACD", "#2563EB"), ("MACD_signal", "Senal 9", "#D97706")),
                width=width,
                reference_lines=((0.0, "0"),),
                histogram_column="MACD_histogram",
            ),
            observation_count=len(analysis.daily_indicators.tail(180)),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="MACD macro - semanal",
            caption="MACD semanal agregado localmente para identificar aceleracion o deterioro del impulso macro.",
            drawing=_technical_line_drawing(
                analysis.weekly_indicators.tail(104),
                (("MACD", "MACD", "#2563EB"), ("MACD_signal", "Senal 9", "#D97706")),
                width=width,
                reference_lines=((0.0, "0"),),
                histogram_column="MACD_histogram",
            ),
            observation_count=len(analysis.weekly_indicators.tail(104)),
        ),
        TechnicalPdfChart(
            section="Contexto",
            title="Ichimoku - diario",
            caption="Precio, Tenkan, Kijun y limites de la nube diaria construidos con los mismos datos del motor.",
            drawing=_technical_line_drawing(
                analysis.daily_indicators.tail(180),
                (("Close", "Precio", "#0F172A"), ("Ichimoku_Tenkan", "Tenkan", "#16A34A"), ("Ichimoku_Kijun", "Kijun", "#D97706"), ("Ichimoku_Senkou_A", "Nube A", "#2563EB"), ("Ichimoku_Senkou_B", "Nube B", "#7C3AED")),
                width=width,
                value_prefix="$",
            ),
            observation_count=len(analysis.daily_indicators.tail(180)),
        ),
        TechnicalPdfChart(
            section="Estructural",
            title="Estructura mensual - EMA 9/21/50",
            caption="Cierre mensual y medias exponenciales para el mapa estructural de largo plazo.",
            drawing=_technical_line_drawing(
                analysis.monthly_indicators.tail(60),
                (("Close", "Cierre", "#0F172A"), ("EMA9", "EMA 9", "#16A34A"), ("EMA21", "EMA 21", "#2563EB"), ("EMA50", "EMA 50", "#D97706")),
                width=width,
                value_prefix="$",
            ),
            observation_count=len(analysis.monthly_indicators.tail(60)),
        ),
        TechnicalPdfChart(
            section="Estructural",
            title="MACD estructural - mensual",
            caption="MACD mensual con senal e histograma para observar ciclos de impulso de largo plazo.",
            drawing=_technical_line_drawing(
                analysis.monthly_indicators.tail(60),
                (("MACD", "MACD", "#2563EB"), ("MACD_signal", "Senal 9", "#D97706")),
                width=width,
                reference_lines=((0.0, "0"),),
                histogram_column="MACD_histogram",
            ),
            observation_count=len(analysis.monthly_indicators.tail(60)),
        ),
    )
    section_order = {"Intradia": 0, "Contexto": 1, "Estructural": 2}
    return tuple(sorted(charts, key=lambda chart: section_order[chart.section]))
