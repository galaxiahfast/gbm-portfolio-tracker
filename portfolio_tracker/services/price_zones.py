"""Read-only presentation adapter for existing levels. Never creates trading targets.

Distance and screen position are display calculations, not execution rules.
Touch estimates use a separate historical model, never directional scores.
"""
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from portfolio_tracker.analytics.zone_reach import estimate_zone_reach
from portfolio_tracker.analytics.conditional_zone_reach import estimate_conditional_reach

if TYPE_CHECKING:
    from portfolio_tracker.analytics.technical_probability import ProbabilityAnalysis


@dataclass(frozen=True)
class DisplayZone:
    label: str
    low: float | None
    high: float | None
    source: str
    context_score: float | None
    context_direction: str


def _positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError):
        return None


def _score(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 <= value <= 100 else None
    except (ValueError, TypeError):
        return None


def build_zone_lists(analysis: "ProbabilityAnalysis"):
    """Select/order existing levels on copies, retaining the fixed active plan."""
    plan = analysis.buy_levels
    if plan is None and analysis.execution_levels.direction == "LONG":
        plan = analysis.execution_levels
    bullish, bearish = _score(analysis.probability_up), _score(analysis.probability_down)
    price = _positive(analysis.last_price)
    entry_low = _positive(plan.entry_low) if plan else None
    entry_high = _positive(plan.entry_high) if plan else None
    if entry_low is None or entry_high is None or entry_low > entry_high:
        entry_low = entry_high = None
    buys = [DisplayZone("Zona 1 · Entrada del plan", entry_low, entry_high,
                        "Plan existente (no se recalcula)", bullish, "alcista")]
    supports = [(analysis.nearest_support, "Soporte cercano"),
                (analysis.structural_support, "Estructura 4h / 1h"),
                (analysis.weekly_support, "Soporte semanal"),
                (analysis.pivots.s1, "Pivote diario S1"), (analysis.pivots.s2, "Pivote diario S2"),
                (analysis.fibonacci.level_382, "Fibonacci 0.382"),
                (analysis.fibonacci.level_500, "Fibonacci 0.500"),
                (analysis.fibonacci.level_618, "Fibonacci 0.618")]
    ceiling = min(v for v in (price, entry_low) if v is not None) if price or entry_low else None
    available = sorted([(float(v), name) for v, name in supports
                        if _positive(v) is not None and ceiling is not None and round(float(v), 2) < round(ceiling, 2)],
                       reverse=True)
    selected = []
    for value, name in available:
        if not selected or round(value, 2) != round(selected[-1][0], 2):
            selected.append((value, name))
    for i, label in enumerate(("Zona 2 · Retroceso moderado", "Zona 3 · Soporte inferior")):
        value, name = selected[i] if len(selected) > i else (None, "Sin soporte distinto disponible")
        buys.append(DisplayZone(label, value, value, name, bearish, "bajista"))
    tp1 = _positive(plan.take_profit_1) if plan else None
    tp2 = _positive(plan.take_profit_2) if plan else None
    # Preserve the original plan even when its two targets coincide.
    tp2_source = "Plan existente"
    if tp1 is not None and tp2 is not None and round(tp2, 2) == round(tp1, 2):
        tp2_source = "Plan existente · coincide con TP1; no es otro nivel de precio"
    elif tp1 is not None and tp2 is not None and tp2 < tp1:
        tp2_source = "Plan existente · objetivo inferior a TP1; revisar el plan"
    sales = [DisplayZone("TP1 · Primer objetivo", tp1, tp1, "Plan existente", bullish, "alcista"),
             DisplayZone("TP2 · Segundo objetivo", tp2, tp2, tp2_source, bullish, "alcista")]
    resistances = [(analysis.structural_resistance, "Estructura 4h / 1h"),
                   (analysis.weekly_resistance, "Resistencia semanal"),
                   (analysis.pivots.r1, "Pivote diario R1"), (analysis.pivots.r2, "Pivote diario R2")]
    floor = max(v for v in (tp1, tp2, price) if v is not None) if tp1 or tp2 or price else None
    higher = sorted((float(v), name) for v, name in resistances
                    if _positive(v) is not None and floor is not None and round(float(v), 2) > round(floor, 2))
    value, name = higher[0] if higher else (None, "Sin resistencia superior disponible")
    sales.append(DisplayZone("Nivel 3 · Resistencia mayor", value, value, name, bullish, "alcista"))
    return tuple(buys), tuple(sales)


def distance_to_zone(price, zone):
    """Signed USD and percent to the nearest boundary; zero means inside."""
    price = _positive(price)
    if price is None or zone.low is None or zone.high is None:
        return None
    edge = zone.low if price < zone.low else zone.high if price > zone.high else price
    distance = edge - price
    return distance, distance / price * 100


def price_location(price, zones):
    """Bounded display coordinate; explicitly disclose prices outside the scale."""
    price = _positive(price)
    edges = [v for z in zones for v in (z.low, z.high) if v is not None]
    if price is None or not edges or min(edges) == max(edges):
        return None
    low, high = min(edges), max(edges)
    ratio = min(1.0, max(0.0, (price - low) / (high - low)))
    status = "Por debajo de las zonas" if price < low else "Por encima de las zonas" if price > high else "Dentro de la escala de zonas"
    return low, high, ratio, status



@dataclass(frozen=True)
class ZoneSnapshot:
    evaluated_at: object
    buys: tuple
    sales: tuple
    estimates: tuple


def build_zone_snapshot(analysis, now=None):
    from portfolio_tracker.analytics.closed_bars import utc
    evaluated_at = utc(now)
    buys, sales = build_zone_lists(analysis)
    estimates = estimate_conditional_reach(analysis.intraday_indicators, analysis.daily_indicators,
                                          analysis.last_price,
                                          [(z.low, z.high, 'BELOW') for z in buys] +
                                          [(z.low, z.high, 'ABOVE') for z in sales], now=evaluated_at,
                                          matching='weighted', min_sessions=12)
    return ZoneSnapshot(evaluated_at, buys, sales, estimates)
