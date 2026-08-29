"""Reporte PDF en memoria para el resumen ejecutivo del motor cuantitativo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from portfolio_tracker.services.pdf_technical_charts import build_technical_pdf_charts
from portfolio_tracker.services.projection_chart import (
    build_15_day_projection_pdf_drawing,
    ordered_horizon_projections,
)

if TYPE_CHECKING:
    from portfolio_tracker.analytics.technical_probability import ProbabilityAnalysis


NAVY = colors.HexColor("#0B1220")
BLUE = colors.HexColor("#2563EB")
GREEN = colors.HexColor("#16A34A")
RED = colors.HexColor("#DC2626")
AMBER = colors.HexColor("#D97706")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#F1F5F9")
GRID = colors.HexColor("#CBD5E1")


@dataclass(frozen=True, slots=True)
class ExecutiveDecision:
    label: str
    tone: str
    rationale: str


def executive_decision(analysis: "ProbabilityAnalysis") -> ExecutiveDecision:
    """Traduce la confluencia a una decisión concreta y prudencial."""

    if analysis.stoch_overbought_extreme:
        return ExecutiveDecision(
            "ESPERAR CORRECCIÓN",
            "warning",
            (
                f"Estocástico RSI 5m en sobrecompra extrema (%K {analysis.stochastic_k:.1f} / "
                f"%D {analysis.stochastic_d:.1f}). LONG bloqueado: esperar que %K cruce a la "
                "baja de 80; un rebote exige una nueva señal desde <20."
            ),
        )
    if analysis.risk_veto or analysis.signal_rejected:
        reason = analysis.risk_reasons[0] if analysis.risk_reasons else analysis.verdict
        return ExecutiveDecision("EVITA / ESPERA", "danger", reason)
    if analysis.rebound_watch_active and not analysis.activation_trigger_met:
        return ExecutiveDecision(
            "VIGILAR REBOTE LONG",
            "warning",
            f"Sobreventa extrema en soporte ${analysis.nearest_support:.2f}; el escenario LONG se evalúa, pero solo se activa con el detonante cuantitativo indicado.",
        )
    if analysis.tactical_short and not analysis.activation_trigger_met:
        return ExecutiveDecision(
            "SHORT TÁCTICO - ESPERAR RUPTURA",
            "warning",
            f"El mensual sigue fuertemente alcista. Exposición máxima relativa {analysis.exposure_factor:.2f}x y confirmación reforzada obligatoria.",
        )
    if analysis.signal.value == "BUY" and analysis.operation_probability >= 65:
        return ExecutiveDecision(
            "COMPRA AHORA",
            "success",
            f"Compra validada con {analysis.operation_probability:.1f}% de confluencia; define salida y riesgo antes de ejecutar.",
        )
    if analysis.signal.value == "SELL" and analysis.operation_probability >= 65:
        return ExecutiveDecision(
            "VENDE / PROTEGE CAPITAL",
            "danger",
            f"Venta validada con {analysis.operation_probability:.1f}% de confluencia; evita abrir posiciones largas.",
        )
    if analysis.probability_down >= 55:
        return ExecutiveDecision(
            "EVITA / ESPERA",
            "danger",
            f"Debilidad bajista inmediata: {analysis.probability_down:.1f}% de sesgo a la baja; no abrir largos sin una nueva confirmación.",
        )
    return ExecutiveDecision("EVITA / ESPERA", "warning", analysis.scenario)


def _safe_text(value: object) -> str:
    return str(value).replace("⚠️", "ALERTA:").replace("×", "x").replace("·", "-")


def _page_footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(18 * mm, 8 * mm, "GBM+ Portfolio - Reporte técnico heurístico")
    canvas.drawRightString(192 * mm, 8 * mm, f"Página {document.page}")
    canvas.restoreState()


def _table(data: list[list[object]], widths: list[float], header: bool = True) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands: list[tuple] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def _report_styles():  # type: ignore[no-untyped-def]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], fontSize=9, leading=13, textColor=SLATE, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="ChartTitle", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=NAVY, spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=NAVY))
    styles.add(ParagraphStyle(name="Decision", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.white))
    styles.add(ParagraphStyle(name="DecisionDetail", parent=styles["BodyText"], fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.white))
    return styles


def _document(buffer: BytesIO, analysis: "ProbabilityAnalysis", title: str, subject: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"{title} {analysis.symbol}",
        author="GBM+ Portfolio",
        subject=subject,
    )


def _report_header(analysis: "ProbabilityAnalysis", title: str, styles) -> list[object]:  # type: ignore[no-untyped-def]
    return [
        Paragraph(f"{title} - {analysis.symbol}", styles["ReportTitle"]),
        Paragraph(
            f"Datos hasta {_safe_text(analysis.as_of.isoformat())} | Fuente: Yahoo Finance / yfinance | Generado por el motor Fase 4",
            styles["ReportSubtitle"],
        ),
    ]


def _fundamental_story(analysis: "ProbabilityAnalysis", styles) -> list[object]:  # type: ignore[no-untyped-def]
    reasons = " | ".join(_safe_text(item) for item in analysis.fundamental_reasons[:6])
    return [
        Paragraph("Contexto fundamental y noticias", styles["Section"]),
        _table(
            [
                ["Lectura", "Impacto", "Veto", "Corte SHA-256"],
                [
                    _safe_text(analysis.fundamental_label),
                    f"{analysis.fundamental_score:+.1f} pp",
                    "ACTIVO" if analysis.fundamental_risk_veto else "INACTIVO",
                    analysis.fundamental_snapshot_sha256[:16] + "…"
                    if analysis.fundamental_snapshot_sha256 else "Sin corte",
                ],
            ],
            [43.5 * mm] * 4,
        ),
        Spacer(1, 5),
        Paragraph(reasons or "Fuente no disponible; ponderación neutral.", styles["BodySmall"]),
    ]


def _executive_story(analysis: "ProbabilityAnalysis", styles) -> list[object]:  # type: ignore[no-untyped-def]
    story: list[object] = []

    decision = executive_decision(analysis)
    decision_color = GREEN if decision.tone == "success" else RED if decision.tone == "danger" else AMBER
    decision_box = Table(
        [[Paragraph(decision.label, styles["Decision"])], [Paragraph(_safe_text(decision.rationale), styles["DecisionDetail"])]],
        colWidths=[174 * mm],
    )
    decision_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), decision_color), ("BOX", (0, 0), (-1, -1), 0, decision_color), ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 10), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story.extend([decision_box, Spacer(1, 10), Paragraph("Resumen ejecutivo", styles["Section"])])
    summary_rows = [
        ["Último precio", "Nivel sugerido", "Subida", "Bajada", "Confluencia"],
        [f"${analysis.last_price:,.2f}", f"${analysis.suggested_level:,.2f}", f"{analysis.probability_up:.1f}%", f"{analysis.probability_down:.1f}%", f"{analysis.operation_probability:.1f}%" if analysis.operation_probability else "Detonante pendiente"],
    ]
    story.extend([_table(summary_rows, [34.8 * mm] * 5), Spacer(1, 8), Paragraph(_safe_text(analysis.scenario), styles["BodySmall"])])
    story.extend(_fundamental_story(analysis, styles))
    if analysis.neckline_heat_warning:
        story.extend(
            [
                Spacer(1, 5),
                Paragraph(_safe_text(analysis.neckline_heat_warning), styles["BodySmall"]),
            ]
        )

    levels = analysis.execution_levels
    story.append(Paragraph("Plan de ejecución y riesgo", styles["Section"]))
    story.append(Paragraph(_safe_text(analysis.execution_plan_label), styles["BodySmall"]))
    story.append(
        Paragraph(
            _safe_text(
                f"Detonante: {'CUMPLIDO' if analysis.activation_trigger_met else 'PENDIENTE'} - "
                f"{analysis.activation_trigger} Exposición relativa: {analysis.exposure_factor:.2f}x."
            ),
            styles["BodySmall"],
        )
    )
    if analysis.execution_plan_conditional:
        story.append(
            Paragraph(
                "Niveles informativos: no constituyen una orden activa hasta que desaparezca el bloqueo y exista un gatillo confirmado.",
                styles["BodySmall"],
            )
        )
    bearish_plan = levels.direction == "SHORT"
    execution_rows = [
        [
            "Zona de salida / venta" if bearish_plan else "Zona de entrada",
            "Invalidacion bajista" if bearish_plan else "Stop loss",
            "Objetivo bajista 1" if bearish_plan else "Take profit 1",
            "Objetivo bajista 2" if bearish_plan else "Take profit 2",
        ],
        [
            f"${levels.entry_low:.2f} - ${levels.entry_high:.2f}",
            f"${levels.stop_loss:.2f}",
            f"${levels.take_profit_1:.2f}",
            f"${levels.take_profit_2:.2f}",
        ],
    ]
    story.append(_table(execution_rows, [43.5 * mm] * 4))
    stop_basis = (
        f"Stop dinámico: entrada conservadora ${levels.entry_low:.2f} - "
        f"{levels.stop_atr_multiple:.2f} x ATR(14) 5m ${levels.atr_5m:.4f}"
        f"{' o soporte estructural' if levels.structural_stop_applied else ''}."
        if not bearish_plan
        else f"Invalidación dinámica: techo de entrada ${levels.entry_high:.2f} + {levels.stop_atr_multiple:.2f} x ATR(14) 5m ${levels.atr_5m:.4f}{' o resistencia estructural' if levels.structural_stop_applied else ''}."
    )
    target_basis = (
        f" TP1 alineado con {levels.pattern_target_label}."
        if levels.pattern_target_applied
        else ""
    )
    story.append(Paragraph(_safe_text(stop_basis + target_basis), styles["BodySmall"]))

    story.append(Paragraph("Patrones chartistas activos", styles["Section"]))
    executive_patterns = analysis.chart_patterns[:4]
    if executive_patterns:
        pattern_rows: list[list[object]] = [
            ["Marco", "Patron", "Direccion", "Confianza", "Estado", "Cuello"]
        ]
        pattern_rows.extend(
            [
                pattern.timeframe,
                Paragraph(_safe_text(pattern.label), styles["BodySmall"]),
                pattern.direction.value,
                f"{pattern.confidence:.1f}%",
                "CONFIRMADO" if pattern.valid else "SIN VALIDAR",
                f"${pattern.neckline:.2f}",
            ]
            for pattern in executive_patterns
        )
        story.append(
            _table(pattern_rows, [18 * mm, 43 * mm, 27 * mm, 25 * mm, 36 * mm, 25 * mm])
        )
    else:
        story.append(
            Paragraph(
                "Sin estructuras geometricas recientes que superen el filtro Zig-Zag.",
                styles["BodySmall"],
            )
        )

    # Mantiene la tabla completa en una página y evita una fila huérfana.
    story.append(PageBreak())
    story.append(Paragraph("Probabilidades y precios por horizonte", styles["Section"]))
    ordered_projections = ordered_horizon_projections(analysis.horizon_projections)
    horizon_rows: list[list[object]] = [["Horizonte", "Subida / objetivo", "Rango / precios", "Bajada / objetivo", "Sesgo"]]
    horizon_rows.extend(
        [
            item.label,
            Paragraph(f"{item.probability_up:.1f}%<br/>${item.bullish_target:.2f}", styles["BodySmall"]),
            Paragraph(f"{item.probability_range:.1f}%<br/>${item.range_low:.2f} - ${item.range_high:.2f}", styles["BodySmall"]),
            Paragraph(f"{item.probability_down:.1f}%<br/>${item.bearish_target:.2f}", styles["BodySmall"]),
            item.bias,
        ]
        for item in ordered_projections
    )
    story.append(_table(horizon_rows, [27 * mm, 37 * mm, 45 * mm, 37 * mm, 28 * mm]))
    story.append(PageBreak())
    story.append(
        KeepTogether(
            [
                Paragraph("Trayectoria proyectada - proximos 15 dias habiles", styles["Section"]),
                build_15_day_projection_pdf_drawing(
                    analysis.daily_projection,
                    analysis.last_price,
                    width=174 * mm,
                ),
                Paragraph(
                    "La linea azul es un escenario bootstrap reproducible con choques de la volatilidad historica y reaccion suave a soportes/resistencias. El area azul representa percentiles 15-85 de 320 trayectorias; no son velas futuras observadas.",
                    styles["BodySmall"],
                ),
            ]
        )
    )
    return story


def _technical_story(analysis: "ProbabilityAnalysis", styles) -> list[object]:  # type: ignore[no-untyped-def]
    story: list[object] = []
    decision = executive_decision(analysis)
    levels = analysis.execution_levels
    story.append(Paragraph("Graficas tecnicas reales", styles["Section"]))
    current_section = ""
    for chart in build_technical_pdf_charts(analysis, width=174 * mm):
        if chart.section != current_section:
            if current_section:
                story.append(PageBreak())
            current_section = chart.section
            story.append(Paragraph(f"Panel {chart.section}", styles["Section"]))
        story.append(
            KeepTogether(
                [
                    Paragraph(chart.title, styles["ChartTitle"]),
                    chart.drawing,
                    Paragraph(chart.caption, styles["BodySmall"]),
                    Spacer(1, 5),
                ]
            )
        )
    story.append(PageBreak())

    risk_reasons = " | ".join(_safe_text(item) for item in analysis.risk_reasons) or "Sin veto central activo."
    context_rows = [
        ["Variable", "Lectura"],
        ["Señal operativa", analysis.signal.value],
        ["Tendencia diaria", analysis.daily_trend.value],
        ["Tendencia semanal", analysis.weekly_trend.value],
        ["Tendencia mensual", analysis.monthly_trend.value],
        ["Veto de riesgo", "ACTIVO" if analysis.risk_veto else "INACTIVO"],
        ["Fundamental/noticias", f"{analysis.fundamental_label} ({analysis.fundamental_score:+.1f} pp)"],
        ["Veto fundamental", "ACTIVO" if analysis.fundamental_risk_veto else "INACTIVO"],
        ["Motivos", Paragraph(risk_reasons, styles["BodySmall"])],
    ]
    story.append(
        KeepTogether(
            [
                Paragraph("Contexto y riesgo", styles["Section"]),
                _table(context_rows, [48 * mm, 126 * mm]),
            ]
        )
    )

    story.append(Paragraph("Detalle técnico completo", styles["Section"]))
    technical_rows = [
        ["Indicador", "Valor / lectura"],
        ["Estocástico RSI 5m", f"%K {analysis.stochastic_k:.1f} | %D {analysis.stochastic_d:.1f}"],
        ["Bloqueo LONG", "ACTIVO - ESPERAR CORRECCION" if analysis.long_entry_blocked else "INACTIVO"],
        ["Vigilancia rebote LONG", f"{'ACTIVA' if analysis.rebound_watch_active else 'INACTIVA'} | soporte ${analysis.nearest_support:.2f}"],
        ["Detonante", Paragraph(_safe_text(analysis.activation_trigger), styles["BodySmall"])],
        ["Estado / exposición", f"{'CUMPLIDO' if analysis.activation_trigger_met else 'PENDIENTE'} | {analysis.exposure_factor:.2f}x | {'SHORT tactico' if analysis.tactical_short else 'Normal'}"],
        ["ATR(14) 5m", f"${analysis.atr_5m:.4f} | stop a {analysis.execution_levels.stop_atr_multiple:.1f}x ATR"],
        ["Bollinger 5m", f"Inferior ${analysis.bollinger_lower:.2f} | Media ${analysis.bollinger_middle:.2f} | Superior ${analysis.bollinger_upper:.2f}"],
        ["Volumen", f"{analysis.volume_ratio:.2f}x media 20 | {'Confirmado' if analysis.volume_confirmed else 'Sin confirmar'}"],
        ["MACD 5m", f"{analysis.macd_5m:+.3f} | Señal {analysis.macd_signal_5m:+.3f} | Hist. {analysis.macd_histogram_5m:+.3f}"],
        ["MACD diario", f"{analysis.macd_daily:+.3f} | Señal {analysis.macd_signal_daily:+.3f} | Hist. {analysis.macd_histogram_daily:+.3f}"],
        ["VWAP", f"${analysis.vwap:.2f} | Precio {analysis.price_vs_vwap_pct:+.2f}%"],
        ["ADX 5m", f"{analysis.adx:.1f} | {'Rango / lateral' if analysis.range_market else 'Tendencia activa'}"],
        ["OBV", f"{analysis.obv_state.value} | Precio 12 velas {analysis.obv_price_change_pct:+.2f}%"],
        ["EMA diaria", f"9 ${analysis.ema9:.2f} | 21 ${analysis.ema21:.2f} | 50 ${analysis.ema50:.2f} | 200 ${analysis.ema200:.2f}"],
        ["Fibonacci mensual", f"{analysis.fibonacci.nearest_ratio} en ${analysis.fibonacci.nearest_level:.2f} | {analysis.fibonacci.role}"],
        ["Ichimoku", f"5m {analysis.ichimoku_5m.value} | Diario {analysis.ichimoku_daily.value}"],
        ["Vela japonesa", Paragraph(_safe_text(analysis.candle_detail), styles["BodySmall"])],
        ["Pivotes", f"S2 ${analysis.pivots.s2:.2f} | S1 ${analysis.pivots.s1:.2f} | P ${analysis.pivots.pivot:.2f} | R1 ${analysis.pivots.r1:.2f} | R2 ${analysis.pivots.r2:.2f}"],
    ]
    story.append(_table(technical_rows, [48 * mm, 126 * mm]))

    story.append(Paragraph("Validacion de patrones chartistas", styles["Section"]))
    if analysis.chart_patterns:
        pattern_rows = [
            ["Marco / patron", "Conf.", "Cuello / objetivo", "Evidencia"]
        ]
        pattern_rows.extend(
            [
                Paragraph(
                    _safe_text(f"{pattern.timeframe} - {pattern.label} - {pattern.direction.value}"),
                    styles["BodySmall"],
                ),
                f"{pattern.confidence:.1f}%\n{'OK' if pattern.valid else 'NO'}",
                f"${pattern.neckline:.2f}\n${pattern.target_price:.2f}",
                Paragraph(_safe_text(pattern.detail), styles["BodySmall"]),
            ]
            for pattern in analysis.chart_patterns[:10]
        )
        story.append(_table(pattern_rows, [46 * mm, 20 * mm, 31 * mm, 77 * mm]))
    else:
        story.append(
            Paragraph(
                "No se detectaron patrones recientes utilizables en 5 minutos ni diario.",
                styles["BodySmall"],
            )
        )

    story.append(Paragraph("Lectura del motor", styles["Section"]))
    score_rows: list[list[object]] = [["Filtro", "Impacto", "Justificación"]]
    score_rows.extend([[Paragraph(_safe_text(item.name), styles["BodySmall"]), f"{item.impact_points:+.1f} pp", Paragraph(_safe_text(item.detail), styles["BodySmall"])] for item in analysis.score_breakdown])
    story.append(_table(score_rows, [42 * mm, 24 * mm, 108 * mm]))

    if analysis.liquidity_zones:
        liquidity_rows: list[list[object]] = [["Inferior", "Centro", "Superior", "Volumen anual"]]
        liquidity_rows.extend([[f"${zone.lower:.2f}", f"${zone.center:.2f}", f"${zone.upper:.2f}", f"{zone.volume_share_pct:.1f}%"] for zone in analysis.liquidity_zones])
        story.append(
            KeepTogether(
                [
                    Paragraph("Zonas de liquidez aproximadas", styles["Section"]),
                    _table(liquidity_rows, [43.5 * mm] * 4),
                ]
            )
        )

    story.append(
        KeepTogether(
            [
                Spacer(1, 10),
                Paragraph("Bloque estructurado para análisis por IA", styles["Section"]),
                Paragraph(
                    _safe_text(
                        f"symbol={analysis.symbol}; decision={decision.label}; signal={analysis.signal.value}; "
                        f"price={analysis.last_price:.4f}; suggested_level={analysis.suggested_level:.4f}; "
                        f"probability_up={analysis.probability_up:.1f}; probability_down={analysis.probability_down:.1f}; "
                        f"operation_confluence={analysis.operation_probability:.1f}; risk_veto={analysis.risk_veto}; "
                        f"plan_direction={levels.direction}; "
                        f"plan_conditional={analysis.execution_plan_conditional}; long_blocked={analysis.long_entry_blocked}; "
                        f"rebound_watch={analysis.rebound_watch_active}; trigger_met={analysis.activation_trigger_met}; "
                        f"tactical_short={analysis.tactical_short}; exposure_factor={analysis.exposure_factor:.2f}; "
                        f"entry_low={levels.entry_low:.2f}; entry_high={levels.entry_high:.2f}; "
                        f"stop_loss={levels.stop_loss:.2f}; take_profit_1={levels.take_profit_1:.2f}; "
                        f"take_profit_2={levels.take_profit_2:.2f}; "
                        f"atr_5m={levels.atr_5m:.4f}; stop_atr_multiple={levels.stop_atr_multiple:.1f}; "
                        f"pattern_target_applied={levels.pattern_target_applied}; "
                        f"weekly_trend={analysis.weekly_trend.value}; monthly_trend={analysis.monthly_trend.value}; "
                        f"chart_pattern_impact={analysis.chart_pattern_impact:+.1f}; "
                        f"chart_pattern_veto={analysis.chart_pattern_veto}; patterns="
                        + ",".join(
                            f"{pattern.timeframe}:{pattern.pattern_type.value}:{pattern.confidence:.1f}:{pattern.valid}"
                            for pattern in analysis.chart_patterns[:10]
                        )
                        + "."
                    ),
                    styles["BodySmall"],
                ),
                Spacer(1, 8),
                Paragraph(
                    "AVISO: los porcentajes son puntajes heurísticos no calibrados. Este documento no es asesoría financiera, no garantiza resultados y no sustituye una estrategia de tamaño de posición, stop y pérdida máxima.",
                    styles["BodySmall"],
                ),
            ]
        )
    )
    return story


def _json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _calibration_story(
    context: Mapping[str, object] | None,
    styles,
) -> list[object]:  # type: ignore[no-untyped-def]
    story: list[object] = [
        Paragraph("Calibración y backtesting", styles["Section"]),
        Paragraph(
            "Validación cronológica fuera de muestra. Los parámetros se seleccionan dentro "
            "del entrenamiento y se congelan antes de evaluar el tramo OOS.",
            styles["BodySmall"],
        ),
    ]
    payload = dict(context or {})
    online = _json_mapping(payload.get("online_stats", {}))
    story.extend(
        [
            Spacer(1, 7),
            Paragraph("Realimentación progresiva", styles["ChartTitle"]),
            _table(
                [
                    ["Observaciones resueltas", "Acierto", "Error Brier", "Umbral adaptativo"],
                    [
                        int(online.get("resolved", 0) or 0),
                        f"{float(online.get('accuracy', 0) or 0):.1%}",
                        f"{float(online.get('brier_score', 0.25) or 0):.3f}",
                        f"{float(online.get('adaptive_threshold', 0.55) or 0):.1%}",
                    ],
                ],
                [43.5 * mm] * 4,
            ),
        ]
    )
    run = _json_mapping(payload.get("backtest_run", {}))
    if not run:
        story.extend(
            [
                Spacer(1, 8),
                Paragraph(
                    "Aún no existe una ejecución histórica registrada. El PDF maestro conserva "
                    "esta ausencia explícita para evitar presentar métricas inventadas.",
                    styles["BodySmall"],
                ),
            ]
        )
        return story

    parameters = _json_mapping(run.get("parameters_json", {}))
    result_payload = _json_mapping(run.get("payload_json", {}))
    aggregate = _json_mapping(result_payload.get("aggregate", {}))
    raw_payload = str(run.get("payload_json", ""))
    stored_hash = str(run.get("payload_sha256", ""))
    hash_valid = bool(raw_payload) and hashlib.sha256(raw_payload.encode("utf-8")).hexdigest() == stored_hash
    story.extend(
        [
            Spacer(1, 9),
            Paragraph("Último backtest registrado", styles["ChartTitle"]),
            _table(
                [
                    ["ID", "Estado", "Motor", "Integridad SHA-256"],
                    [
                        run.get("id", "-"),
                        _safe_text(run.get("status", "-")),
                        _safe_text(run.get("engine_version", "-")),
                        "VALIDA" if hash_valid else "ERROR",
                    ],
                ],
                [43.5 * mm] * 4,
            ),
            Spacer(1, 7),
            _table(
                [
                    ["Umbral", "Stop ATR", "Riesgo/op.", "Pruebas de rejilla"],
                    [
                        f"{float(parameters.get('minimum_probability', 0) or 0):.0%}",
                        f"{float(parameters.get('stop_atr_multiple', 0) or 0):.2f}x",
                        f"{float(parameters.get('risk_per_trade_pct', 0) or 0):.2f}%",
                        int(parameters.get("optimization_trials", 0) or 0),
                    ],
                ],
                [43.5 * mm] * 4,
            ),
            Spacer(1, 7),
            _table(
                [
                    ["Trades OOS", "Acierto OOS", "Profit factor", "Drawdown máximo"],
                    [
                        int(aggregate.get("trades", 0) or 0),
                        f"{float(aggregate.get('win_rate', 0) or 0):.1%}",
                        (
                            "Infinito"
                            if aggregate.get("profit_factor") is None
                            else f"{float(aggregate.get('profit_factor', 0) or 0):.2f}"
                        ),
                        f"{float(aggregate.get('maximum_drawdown_pct', 0) or 0):.2f}%",
                    ],
                ],
                [43.5 * mm] * 4,
            ),
            Spacer(1, 7),
            Paragraph(
                _safe_text(
                    f"Dataset SHA-256: {run.get('dataset_sha256', '-')}. "
                    f"Resultado agregado: {result_payload.get('aggregate_decision', run.get('status', '-'))}."
                ),
                styles["BodySmall"],
            ),
        ]
    )
    return story


def _build_report(
    analysis: "ProbabilityAnalysis",
    *,
    include_executive: bool,
    include_technical: bool,
    include_calibration: bool = False,
    calibration_context: Mapping[str, object] | None = None,
    title: str,
    subject: str,
) -> bytes:
    buffer = BytesIO()
    document = _document(buffer, analysis, title, subject)
    styles = _report_styles()
    story = _report_header(analysis, title, styles)
    if include_executive:
        story.extend(_executive_story(analysis, styles))
    if include_technical:
        if include_executive:
            story.append(PageBreak())
        story.extend(_technical_story(analysis, styles))
    if include_calibration:
        if include_executive or include_technical:
            story.append(PageBreak())
        story.extend(_calibration_story(calibration_context, styles))
    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buffer.getvalue()


def build_executive_report(analysis: "ProbabilityAnalysis") -> bytes:
    """Genera exclusivamente la vista ejecutiva y su proyeccion diaria."""

    return _build_report(
        analysis,
        include_executive=True,
        include_technical=False,
        title="Vista ejecutiva",
        subject="Decision, horizontes, niveles de riesgo y proyeccion de 15 sesiones",
    )


def build_technical_report(analysis: "ProbabilityAnalysis") -> bytes:
    """Genera exclusivamente graficas, indicadores y lectura tecnica completa."""

    return _build_report(
        analysis,
        include_executive=False,
        include_technical=True,
        title="Vista tecnica avanzada",
        subject="Panel tecnico multi-temporal completo para auditoria",
    )


def build_probability_report(analysis: "ProbabilityAnalysis") -> bytes:
    """Genera el reporte unificado con la vista ejecutiva y la tecnica."""

    return _build_report(
        analysis,
        include_executive=True,
        include_technical=True,
        title="Reporte cuantitativo completo",
        subject="Resumen ejecutivo y tecnico para revision humana o por IA",
    )


def build_master_report(
    analysis: "ProbabilityAnalysis",
    calibration_context: Mapping[str, object] | None,
) -> bytes:
    """Genera las tres vistas: ejecutiva, técnica y calibración auditada."""

    return _build_report(
        analysis,
        include_executive=True,
        include_technical=True,
        include_calibration=True,
        calibration_context=calibration_context,
        title="Reporte maestro cuantitativo",
        subject="Vista ejecutiva, tecnica avanzada y calibracion fuera de muestra",
    )
