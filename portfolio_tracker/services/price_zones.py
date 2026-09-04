"""Read-only presentation adapter for existing levels. Never creates trading targets.

Distance and screen position are display calculations, not execution rules.
Touch estimates use a separate historical model, never directional scores.
"""
from dataclasses import dataclass, replace
import json
import math
from typing import TYPE_CHECKING

import pandas as pd

from portfolio_tracker.analytics.zone_reach import estimate_zone_reach
from portfolio_tracker.analytics.conditional_zone_reach import (
    calculate_dynamic_visual_zone,
    estimate_conditional_reach,
)
from portfolio_tracker.analytics.zone_reach import ReachEstimate

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


@dataclass(frozen=True)
class ProjectedLevel:
    label: str
    price: float
    source: str
    direction: str

    @property
    def type(self):
        """Stable presentation contract used by UI/PDF serializers."""
        return "soporte_extendido" if self.direction == "BELOW" else "resistencia_extendida"


@dataclass(frozen=True)
class MarketSessionStatus:
    is_open: bool
    message: str
    next_collection_at: object | None = None


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


def market_session_status(now=None):
    """Read-only XNYS status and next regular 11:00 NY collection cut."""
    from portfolio_tracker.analytics.closed_bars import NY, _calendar, utc

    clock = utc(now)
    local_day = pd.Timestamp(clock.tz_convert(NY).date())
    schedule = _calendar(local_day.year - 1, local_day.year + 2).schedule
    if local_day in schedule.index:
        row = schedule.loc[local_day]
        if row["open"] <= clock < row["close"]:
            return MarketSessionStatus(True, "Mercado abierto.")
    future = schedule.loc[schedule["open"] > clock]
    next_collection = None
    if not future.empty:
        next_day = future.index[0]
        next_collection = pd.Timestamp(
            f"{next_day.date()} 11:00:00", tz=NY
        ).tz_convert("UTC")
    suffix = (
        f" Próximo corte: {next_collection.tz_convert(NY):%d/%m/%Y a las 11:00 AM NY}."
        if next_collection is not None else ""
    )
    return MarketSessionStatus(
        False,
        "MERCADO CERRADO. Los niveles mostrados corresponden a la última sesión. "
        "Los nuevos niveles se calcularán en la próxima sesión "
        "(próximo día hábil a las 11:00 AM NY)." + suffix,
        next_collection,
    )


def _latest_intraday_range(frame):
    if frame is None or frame.empty or not {"High", "Low"} <= set(frame.columns):
        return None
    data = frame.loc[:, ["High", "Low"]].copy().sort_index()
    index = pd.DatetimeIndex(data.index)
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")
    data.index = index
    latest_day = data.index[-1].date()
    session = data.loc[data.index.date == latest_day]
    high = pd.to_numeric(session["High"], errors="coerce").max()
    low = pd.to_numeric(session["Low"], errors="coerce").min()
    high, low = _positive(high), _positive(low)
    if high is None or low is None or high <= low:
        return None
    return low, high, high - low


def projected_extended_levels(analysis, snapshot):
    """Visual-only references beyond every original level; never persisted."""
    price = _positive(getattr(analysis, "last_price", None))
    if price is None:
        return ()
    plan = getattr(analysis, "buy_levels", None)
    pivots = getattr(analysis, "pivots", None)
    upside_references = [
        _positive(getattr(plan, "take_profit_1", None)),
        _positive(getattr(plan, "take_profit_2", None)),
        _positive(getattr(pivots, "r1", None)),
    ]
    downside_references = [
        _positive(getattr(plan, "entry_low", None)),
        _positive(getattr(pivots, "s1", None)),
        _positive(getattr(pivots, "s2", None)),
    ]
    upside_references = [value for value in upside_references if value is not None]
    downside_references = [value for value in downside_references if value is not None]
    if not upside_references:
        upside_references = [
            value for zone in snapshot.sales for value in (_positive(zone.low), _positive(zone.high))
            if value is not None
        ]
    if not downside_references:
        downside_references = [
            value for zone in snapshot.buys for value in (_positive(zone.low), _positive(zone.high))
            if value is not None
        ]
    upside = bool(upside_references) and price > max(upside_references)
    downside = bool(downside_references) and price < min(downside_references)
    if not upside and not downside:
        return ()

    session_range = _latest_intraday_range(getattr(analysis, "intraday_indicators", None))
    atr = _positive(getattr(analysis, "atr_5m", None)) or price * 0.005
    if session_range is None:
        low, high, movement = price - 2 * atr, price + 2 * atr, 4 * atr
    else:
        low, high, movement = session_range
    candidates = []
    if upside:
        threshold = max(upside_references)
        for ratio in (1.272, 1.618, 2.0, 2.618):
            candidates.append((low + ratio * movement, f"Fibonacci {ratio:g} del último rango intradía"))
        candidates.extend([
            (price + 1.5 * atr, "Precio actual + 1.5 x ATR(14) 5m"),
            (price + 2.0 * atr, "Precio actual + 2.0 x ATR(14) 5m"),
        ])
        candidates = sorted((value, source) for value, source in candidates if value > max(price, threshold))
        heading = "Objetivo extendido"
        direction = "ABOVE"
    else:
        threshold = min(downside_references)
        for ratio in (1.272, 1.618, 2.0, 2.618):
            candidates.append((high - ratio * movement, f"Fibonacci {ratio:g} del último rango intradía"))
        candidates.extend([
            (price - 1.5 * atr, "Precio actual - 1.5 x ATR(14) 5m"),
            (price - 2.0 * atr, "Precio actual - 2.0 x ATR(14) 5m"),
        ])
        candidates = sorted(
            ((value, source) for value, source in candidates if 0 < value < min(price, threshold)),
            reverse=True,
        )
        heading = "Soporte extendido"
        direction = "BELOW"

    selected = []
    seen = set()
    for value, source in candidates:
        rounded = round(float(value), 2)
        if rounded <= 0 or rounded in seen:
            continue
        seen.add(rounded)
        selected.append(ProjectedLevel(
            f"{heading} {len(selected) + 1} · Proyectado",
            rounded,
            source,
            direction,
        ))
        if len(selected) == 3:
            break
    return tuple(selected)



@dataclass(frozen=True)
class ZoneSnapshot:
    evaluated_at: object
    buys: tuple
    sales: tuple
    estimates: tuple
    extended_levels: tuple = ()


def build_zone_snapshot(analysis, now=None):
    from portfolio_tracker.analytics.closed_bars import utc
    evaluated_at = utc(now)
    buys, sales = build_zone_lists(analysis)
    estimates = estimate_conditional_reach(analysis.intraday_indicators, analysis.daily_indicators,
                                          analysis.last_price,
                                          [(z.low, z.high, 'BELOW') for z in buys] +
                                          [(z.low, z.high, 'ABOVE') for z in sales], now=evaluated_at,
                                          matching='weighted', min_sessions=12)
    snapshot = ZoneSnapshot(evaluated_at, buys, sales, estimates)
    return replace(snapshot, extended_levels=projected_extended_levels(analysis, snapshot))


def _forward_reference_snapshot(repository, analysis, now=None):
    """Return the latest session's first complete, verified 11:00-11:20 cohort."""
    from portfolio_tracker.analytics.closed_bars import NY, utc

    clock = utc(now)
    rows = [
        row for row in repository.zone_predictions()
        if row.get("integrity_ok") and row.get("symbol") == analysis.symbol
    ]
    groups = {}
    for row in rows:
        emitted = utc(row["timestamp_prediction"])
        if emitted > clock:
            continue
        local_time = emitted.tz_convert(NY).time()
        if not (local_time.hour == 11 and local_time.minute < 20):
            continue
        key = (
            str(row.get("session_date") or emitted.tz_convert(NY).date()),
            row["timestamp_prediction"],
            row["source_bar_closed_at"],
            row["model_version_hash"],
        )
        groups.setdefault(key, {})[row["zone_key"]] = row
    order = ("ENTRY1", "ENTRY2", "ENTRY3", "TP1", "TP2", "R3")
    complete = [(key, group) for key, group in groups.items() if set(group) >= set(order)]
    if not complete:
        return None
    latest_session = max(item[0][0] for item in complete)
    key, group = min(
        (item for item in complete if item[0][0] == latest_session),
        key=lambda item: item[0][1],
    )
    zones = []
    estimates = []
    for zone_key in order:
        row = group[zone_key]
        try:
            context = json.loads(row.get("context_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            context = {}
        zones.append(DisplayZone(
            str(context.get("zone_label") or zone_key),
            float(row["zone_low"]),
            float(row["zone_high"]),
            str(context.get("source") or "Pronóstico forward firmado de las 11:00 NY"),
            None,
            "bajista" if row["close_direction"] == "BELOW" else "alcista",
        ))
        close_probability = row.get("predicted_close_probability")
        estimates.append(ReachEstimate(
            float(row["predicted_touch_probability"]) * 100,
            int(context.get("samples") or 0),
            None,
            None,
            str(context.get("status") or "Pronóstico forward firmado"),
            None if close_probability is None else float(close_probability) * 100,
            None,
            None,
            str(row["close_direction"]),
            str(row["model_name"]),
            str(context.get("detail") or ""),
            confidence_available=False,
        ))
    return ZoneSnapshot(utc(key[1]), tuple(zones[:3]), tuple(zones[3:]), tuple(estimates))


def build_visual_zone_snapshot(analysis, *, repository=None, now=None, original_snapshot=None):
    """Build an in-memory live view without writing forward-test evidence."""
    from portfolio_tracker.analytics.closed_bars import utc
    from portfolio_tracker.services.zone_forward import session_bounds

    clock = utc(now)
    original = original_snapshot
    if original is None and repository is not None:
        original = _forward_reference_snapshot(repository, analysis, now=clock)
    if original is None:
        original = build_zone_snapshot(analysis, now=clock)
    bounds = session_bounds(clock)
    if bounds is None or clock < bounds[1] or clock >= bounds[2]:
        frozen = replace(original, evaluated_at=clock)
        return replace(frozen, extended_levels=projected_extended_levels(analysis, frozen))
    pairs = tuple((zone, "BELOW") for zone in original.buys) + tuple(
        (zone, "ABOVE") for zone in original.sales
    )
    assessments = calculate_dynamic_visual_zone(
        analysis.last_price,
        clock,
        bounds[2],
        pairs,
        intraday=analysis.intraday_indicators,
        daily=analysis.daily_indicators,
        original_estimates=original.estimates,
        min_sessions=12,
    )
    zones = tuple(
        replace(zone, label=assessment.label)
        for (zone, _), assessment in zip(pairs, assessments)
    )
    snapshot = ZoneSnapshot(
        clock,
        zones[:3],
        zones[3:],
        tuple(assessment.estimate for assessment in assessments),
    )
    return replace(snapshot, extended_levels=projected_extended_levels(analysis, snapshot))
