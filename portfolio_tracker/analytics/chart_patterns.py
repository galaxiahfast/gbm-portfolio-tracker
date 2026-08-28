"""Deteccion objetiva de patrones chartistas sobre series OHLCV.

El modulo no descarga datos ni infiere figuras visualmente. Convierte precios en
pivotes alternados mediante extremos locales y un filtro Zig-Zag dependiente del
ATR. Cada deteccion conserva sus evidencias para que el motor, la UI y el PDF
puedan auditar exactamente por que fue aceptada o descartada.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
import math

import pandas as pd


class PatternDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ChartPatternType(StrEnum):
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    DOUBLE_TOP = "DOUBLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    RANGE_BREAKOUT_UP = "RANGE_BREAKOUT_UP"
    RANGE_BREAKOUT_DOWN = "RANGE_BREAKOUT_DOWN"
    IMPULSE_3_UP = "IMPULSE_3_UP"
    IMPULSE_3_DOWN = "IMPULSE_3_DOWN"
    ABC_UP = "ABC_UP"
    ABC_DOWN = "ABC_DOWN"


PATTERN_LABELS: dict[ChartPatternType, str] = {
    ChartPatternType.DOUBLE_BOTTOM: "Doble suelo",
    ChartPatternType.DOUBLE_TOP: "Doble techo",
    ChartPatternType.TRIPLE_BOTTOM: "Triple suelo",
    ChartPatternType.TRIPLE_TOP: "Triple techo",
    ChartPatternType.RANGE_BREAKOUT_UP: "Ruptura alcista de rango",
    ChartPatternType.RANGE_BREAKOUT_DOWN: "Ruptura bajista de rango",
    ChartPatternType.IMPULSE_3_UP: "Tres impulsos alcistas",
    ChartPatternType.IMPULSE_3_DOWN: "Tres impulsos bajistas",
    ChartPatternType.ABC_UP: "Correccion ABC alcista",
    ChartPatternType.ABC_DOWN: "Correccion ABC bajista",
}


@dataclass(frozen=True, slots=True)
class PricePivot:
    position: int
    timestamp: pd.Timestamp
    kind: str
    price: float
    atr: float
    rsi: float
    macd: float


@dataclass(frozen=True, slots=True)
class ChartPattern:
    pattern_type: ChartPatternType
    direction: PatternDirection
    timeframe: str
    confidence: float
    neckline: float
    target_price: float
    confirmed: bool
    volume_ratio: float
    rsi_divergence: bool
    macd_divergence: bool
    pivot_timestamps: tuple[pd.Timestamp, ...]
    pivot_prices: tuple[float, ...]
    detected_at: pd.Timestamp
    detail: str

    @property
    def label(self) -> str:
        return PATTERN_LABELS[self.pattern_type]

    @property
    def valid(self) -> bool:
        """Solo figuras confirmadas por encima del umbral afectan al motor."""

        return self.confirmed and self.confidence > 75.0


@dataclass(frozen=True, slots=True)
class PatternInfluence:
    impact_points: float
    detail: str
    veto: bool
    veto_reason: str


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, float("nan"))
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    return result.where(average_loss != 0.0, 100.0)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("Open", "High", "Low", "Close", "Volume")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {', '.join(missing)}")
    result = frame.loc[:, list(required)].copy()
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=list(required)).sort_index()
    result = result[~result.index.duplicated(keep="last")].tail(600)
    if len(result) < 30:
        return result

    previous_close = result["Close"].shift(1)
    true_range = pd.concat(
        [
            result["High"] - result["Low"],
            (result["High"] - previous_close).abs(),
            (result["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    result["RSI14"] = _rsi(result["Close"])
    ema12 = result["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = result["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_signal"] = result["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["Volume_MA20"] = result["Volume"].shift(1).rolling(
        20, min_periods=20
    ).mean()
    return result


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def zigzag_pivots(
    frame: pd.DataFrame,
    *,
    local_window: int = 2,
    minimum_move_atr: float = 0.45,
) -> tuple[PricePivot, ...]:
    """Extrae pivotes alternados y descarta ruido menor a una fraccion del ATR."""

    prepared = _prepare(frame)
    if len(prepared) < max(30, local_window * 2 + 3):
        return ()
    candidates: list[PricePivot] = []
    for position in range(local_window, len(prepared) - local_window):
        row = prepared.iloc[position]
        local = prepared.iloc[position - local_window : position + local_window + 1]
        low = float(row["Low"])
        high = float(row["High"])
        atr = _finite(row["ATR14"], max(high - low, abs(float(row["Close"])) * 0.005))
        common = dict(
            position=position,
            timestamp=pd.Timestamp(prepared.index[position]),
            atr=max(atr, 1e-9),
            rsi=_finite(row["RSI14"], 50.0),
            macd=_finite(row["MACD"]),
        )
        if low <= float(local["Low"].min()):
            candidates.append(PricePivot(kind="LOW", price=low, **common))
        if high >= float(local["High"].max()):
            candidates.append(PricePivot(kind="HIGH", price=high, **common))

    pivots: list[PricePivot] = []
    for candidate in sorted(candidates, key=lambda item: (item.position, item.kind)):
        if not pivots:
            pivots.append(candidate)
            continue
        previous = pivots[-1]
        if candidate.position == previous.position:
            continue
        if candidate.kind == previous.kind:
            more_extreme = (
                candidate.price < previous.price
                if candidate.kind == "LOW"
                else candidate.price > previous.price
            )
            if more_extreme:
                pivots[-1] = candidate
            continue
        threshold = minimum_move_atr * max(previous.atr, candidate.atr)
        if abs(candidate.price - previous.price) >= threshold:
            pivots.append(candidate)
    return tuple(pivots)


def _volume_ratio(row: pd.Series) -> float:
    average = _finite(row.get("Volume_MA20"))
    return _finite(row.get("Volume")) / average if average > 0 else 0.0


def _spacing_quality(selected: tuple[PricePivot, ...]) -> float:
    if len(selected) <= 2:
        return 1.0
    gaps = [right.position - left.position for left, right in zip(selected, selected[1:])]
    largest = max(gaps)
    return min(gaps) / largest if largest > 0 else 0.0


def _reversal_pattern(
    prepared: pd.DataFrame,
    selected: tuple[PricePivot, ...],
    *,
    timeframe: str,
) -> ChartPattern | None:
    kind = selected[0].kind
    count = len(selected)
    mean_atr = sum(item.atr for item in selected) / count
    symmetry_atr = (max(item.price for item in selected) - min(item.price for item in selected)) / max(mean_atr, 1e-9)
    # La especificacion se interpreta como una banda de medio ATR. Expresarla
    # como 0.5% del precio haria que la tolerancia dejara de adaptarse a volatilidad.
    if symmetry_atr > 0.50:
        return None
    first_position, last_position = selected[0].position, selected[-1].position
    between = prepared.iloc[first_position : last_position + 1]
    if kind == "LOW":
        neckline = float(between["High"].max())
        direction = PatternDirection.BULLISH
        pattern_type = (
            ChartPatternType.DOUBLE_BOTTOM
            if count == 2
            else ChartPatternType.TRIPLE_BOTTOM
        )
        breakout_mask = prepared.iloc[last_position + 1 :]["Close"] > neckline
        rsi_divergence = selected[-1].rsi >= selected[0].rsi + 2.0
        macd_divergence = selected[-1].macd > selected[0].macd
        # Objetivo clásico medido desde el mínimo real del suelo, no desde el
        # promedio de pivotes, para no recortar artificialmente la figura.
        height = max(0.0, neckline - min(item.price for item in selected))
        target = neckline + height
    else:
        neckline = float(between["Low"].min())
        direction = PatternDirection.BEARISH
        pattern_type = (
            ChartPatternType.DOUBLE_TOP
            if count == 2
            else ChartPatternType.TRIPLE_TOP
        )
        breakout_mask = prepared.iloc[last_position + 1 :]["Close"] < neckline
        rsi_divergence = selected[-1].rsi <= selected[0].rsi - 2.0
        macd_divergence = selected[-1].macd < selected[0].macd
        height = max(0.0, max(item.price for item in selected) - neckline)
        target = max(0.01, neckline - height)

    breakout_rows = prepared.iloc[last_position + 1 :][breakout_mask]
    breakout = breakout_rows.iloc[0] if not breakout_rows.empty else None
    breakout_timestamp = (
        pd.Timestamp(breakout_rows.index[0])
        if not breakout_rows.empty
        else selected[-1].timestamp
    )
    volume_ratio = _volume_ratio(breakout) if breakout is not None else 0.0
    has_momentum_divergence = rsi_divergence or macd_divergence
    confirmed = breakout is not None and volume_ratio > 1.0 and has_momentum_divergence

    symmetry_score = 25.0 * max(0.0, 1.0 - symmetry_atr / 0.50)
    spacing_score = 10.0 * _spacing_quality(selected)
    confidence = 20.0 + symmetry_score + spacing_score
    if breakout is not None:
        confidence += 15.0
    if volume_ratio > 1.0:
        confidence += min(15.0, 10.0 + (volume_ratio - 1.0) * 12.5)
    if rsi_divergence:
        confidence += 7.5
    if macd_divergence:
        confidence += 7.5
    confidence = round(min(100.0, confidence), 1)
    if not confirmed:
        confidence = min(confidence, 75.0)
    detail = (
        f"Simetria {symmetry_atr:.2f} ATR; cuello {neckline:.2f}; "
        f"volumen {volume_ratio:.2f}x; divergencia RSI "
        f"{'si' if rsi_divergence else 'no'} / MACD "
        f"{'si' if macd_divergence else 'no'}."
    )
    return ChartPattern(
        pattern_type=pattern_type,
        direction=direction,
        timeframe=timeframe,
        confidence=confidence,
        neckline=round(neckline, 4),
        target_price=round(target, 4),
        confirmed=confirmed,
        volume_ratio=round(volume_ratio, 3),
        rsi_divergence=rsi_divergence,
        macd_divergence=macd_divergence,
        pivot_timestamps=tuple(item.timestamp for item in selected),
        pivot_prices=tuple(round(item.price, 4) for item in selected),
        detected_at=breakout_timestamp,
        detail=detail,
    )


def _detect_reversals(
    prepared: pd.DataFrame,
    pivots: tuple[PricePivot, ...],
    timeframe: str,
) -> list[ChartPattern]:
    patterns: list[ChartPattern] = []
    for pivot_kind in ("LOW", "HIGH"):
        same_kind = [item for item in pivots[-18:] if item.kind == pivot_kind]
        for count in (2, 3):
            best: ChartPattern | None = None
            for raw_selected in combinations(same_kind, count):
                selected = tuple(raw_selected)
                if selected[-1].position - selected[0].position > 180:
                    continue
                # Cada valle/techo debe estar separado por el extremo contrario.
                if any(
                    not any(
                        pivot.kind != pivot_kind
                        and left.position < pivot.position < right.position
                        for pivot in pivots
                    )
                    for left, right in zip(selected, selected[1:])
                ):
                    continue
                candidate = _reversal_pattern(
                    prepared, selected, timeframe=timeframe
                )
                if candidate and (
                    best is None
                    or (candidate.valid, candidate.confidence, candidate.detected_at)
                    > (best.valid, best.confidence, best.detected_at)
                ):
                    best = candidate
            if best:
                patterns.append(best)
    return patterns


def _detect_range_breakout(
    prepared: pd.DataFrame,
    timeframe: str,
    lookback: int = 20,
) -> ChartPattern | None:
    if len(prepared) < lookback + 3:
        return None
    best: ChartPattern | None = None
    for position in range(max(lookback, len(prepared) - 3), len(prepared)):
        history = prepared.iloc[position - lookback : position]
        row = prepared.iloc[position]
        atr = max(_finite(row["ATR14"]), 1e-9)
        upper = float(history["High"].max())
        lower = float(history["Low"].min())
        width_atr = (upper - lower) / atr
        close = float(row["Close"])
        if close > upper:
            direction = PatternDirection.BULLISH
            pattern_type = ChartPatternType.RANGE_BREAKOUT_UP
            neckline = upper
            magnitude_atr = (close - upper) / atr
            momentum = _finite(row["RSI14"], 50.0) > 55 and _finite(row["MACD"]) > _finite(row["MACD_signal"])
            target = close + (upper - lower)
        elif close < lower:
            direction = PatternDirection.BEARISH
            pattern_type = ChartPatternType.RANGE_BREAKOUT_DOWN
            neckline = lower
            magnitude_atr = (lower - close) / atr
            momentum = _finite(row["RSI14"], 50.0) < 45 and _finite(row["MACD"]) < _finite(row["MACD_signal"])
            target = max(0.01, close - (upper - lower))
        else:
            continue
        volume_ratio = _volume_ratio(row)
        confidence = 45.0
        confidence += min(20.0, max(0.0, (volume_ratio - 1.0) * 35.0))
        confidence += min(15.0, magnitude_atr * 20.0)
        confidence += 10.0 if momentum else 0.0
        confidence += max(0.0, 10.0 - max(0.0, width_atr - 3.0) * 2.0)
        confidence = round(min(100.0, confidence), 1)
        confirmed = volume_ratio > 1.20 and momentum
        if not confirmed:
            confidence = min(confidence, 75.0)
        timestamp = pd.Timestamp(prepared.index[position])
        candidate = ChartPattern(
            pattern_type=pattern_type,
            direction=direction,
            timeframe=timeframe,
            confidence=confidence,
            neckline=round(neckline, 4),
            target_price=round(target, 4),
            confirmed=confirmed,
            volume_ratio=round(volume_ratio, 3),
            rsi_divergence=False,
            macd_divergence=False,
            pivot_timestamps=(pd.Timestamp(history.index[0]), timestamp),
            pivot_prices=(round(lower, 4), round(upper, 4)),
            detected_at=timestamp,
            detail=(
                f"Rango {width_atr:.2f} ATR; ruptura {magnitude_atr:.2f} ATR; "
                f"volumen {volume_ratio:.2f}x (umbral estricto 1.20x); "
                f"momentum {'alineado' if momentum else 'sin alinear'}."
            ),
        )
        if best is None or (candidate.valid, candidate.confidence) > (
            best.valid,
            best.confidence,
        ):
            best = candidate
    return best


def _wave_pattern(
    selected: tuple[PricePivot, ...],
    *,
    timeframe: str,
    prepared: pd.DataFrame,
) -> ChartPattern | None:
    kinds = tuple(item.kind for item in selected)
    prices = tuple(item.price for item in selected)
    bullish = kinds == ("LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH")
    bearish = kinds == ("HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW")
    if bullish:
        structural = prices[2] > prices[0] and prices[4] > prices[2] and prices[3] > prices[1] and prices[5] > prices[3]
        direction = PatternDirection.BULLISH
        pattern_type = ChartPatternType.IMPULSE_3_UP
    elif bearish:
        structural = prices[2] < prices[0] and prices[4] < prices[2] and prices[3] < prices[1] and prices[5] < prices[3]
        direction = PatternDirection.BEARISH
        pattern_type = ChartPatternType.IMPULSE_3_DOWN
    else:
        return None
    if not structural:
        return None
    impulse_moves = [abs(prices[1] - prices[0]), abs(prices[3] - prices[2]), abs(prices[5] - prices[4])]
    corrections = [abs(prices[2] - prices[1]), abs(prices[4] - prices[3])]
    ratios = [corrections[index] / max(impulse_moves[index], 1e-9) for index in range(2)]
    geometry = sum(0.20 <= ratio <= 0.85 for ratio in ratios) / 2
    row = prepared.iloc[selected[-1].position]
    volume_ratio = _volume_ratio(row)
    momentum = (
        _finite(row["MACD"]) > _finite(row["MACD_signal"])
        if bullish
        else _finite(row["MACD"]) < _finite(row["MACD_signal"])
    )
    confidence = round(min(100.0, 50.0 + 25.0 * geometry + (10.0 if momentum else 0.0) + min(15.0, max(0.0, volume_ratio - 0.8) * 25.0)), 1)
    confirmed = geometry == 1.0 and momentum
    if not confirmed:
        confidence = min(confidence, 75.0)
    neckline = prices[-1]
    target = prices[-1] + (impulse_moves[-1] if bullish else -impulse_moves[-1])
    return ChartPattern(
        pattern_type=pattern_type,
        direction=direction,
        timeframe=timeframe,
        confidence=confidence,
        neckline=round(neckline, 4),
        target_price=round(max(0.01, target), 4),
        confirmed=confirmed,
        volume_ratio=round(volume_ratio, 3),
        rsi_divergence=False,
        macd_divergence=momentum,
        pivot_timestamps=tuple(item.timestamp for item in selected),
        pivot_prices=tuple(round(item.price, 4) for item in selected),
        detected_at=selected[-1].timestamp,
        detail=f"Tres impulsos y dos correcciones; retrocesos {ratios[0]:.2f}/{ratios[1]:.2f}; volumen {volume_ratio:.2f}x.",
    )


def _abc_pattern(
    selected: tuple[PricePivot, ...],
    *,
    timeframe: str,
    prepared: pd.DataFrame,
) -> ChartPattern | None:
    kinds = tuple(item.kind for item in selected)
    prices = tuple(item.price for item in selected)
    bullish = kinds == ("LOW", "HIGH", "LOW", "HIGH") and prices[2] > prices[0] and prices[3] > prices[1]
    bearish = kinds == ("HIGH", "LOW", "HIGH", "LOW") and prices[2] < prices[0] and prices[3] < prices[1]
    if not bullish and not bearish:
        return None
    move_a = abs(prices[1] - prices[0])
    move_b = abs(prices[2] - prices[1])
    move_c = abs(prices[3] - prices[2])
    retracement = move_b / max(move_a, 1e-9)
    extension = move_c / max(move_a, 1e-9)
    geometry = 0.30 <= retracement <= 0.80 and extension >= 0.60
    row = prepared.iloc[selected[-1].position]
    momentum = (
        _finite(row["MACD"]) > _finite(row["MACD_signal"])
        if bullish
        else _finite(row["MACD"]) < _finite(row["MACD_signal"])
    )
    volume_ratio = _volume_ratio(row)
    confidence = round(min(100.0, 45.0 + (30.0 if geometry else 0.0) + (15.0 if momentum else 0.0) + min(10.0, max(0.0, volume_ratio - 0.8) * 20.0)), 1)
    confirmed = geometry and momentum
    if not confirmed:
        confidence = min(confidence, 75.0)
    return ChartPattern(
        pattern_type=ChartPatternType.ABC_UP if bullish else ChartPatternType.ABC_DOWN,
        direction=PatternDirection.BULLISH if bullish else PatternDirection.BEARISH,
        timeframe=timeframe,
        confidence=confidence,
        neckline=round(prices[1], 4),
        target_price=round(max(0.01, prices[3] + (move_c * 0.50 if bullish else -move_c * 0.50)), 4),
        confirmed=confirmed,
        volume_ratio=round(volume_ratio, 3),
        rsi_divergence=False,
        macd_divergence=momentum,
        pivot_timestamps=tuple(item.timestamp for item in selected),
        pivot_prices=tuple(round(item.price, 4) for item in selected),
        detected_at=selected[-1].timestamp,
        detail=f"ABC objetivo: retroceso B {retracement:.2f}; extension C {extension:.2f}; volumen {volume_ratio:.2f}x.",
    )


def detect_chart_patterns(
    frame: pd.DataFrame,
    *,
    timeframe: str,
) -> tuple[ChartPattern, ...]:
    """Detecta y ordena patrones recientes sin producir una señal por defecto."""

    prepared = _prepare(frame)
    if len(prepared) < 30:
        return ()
    local_window = 2 if timeframe.lower() in {"5m", "5 min", "intradia"} else 3
    pivots = zigzag_pivots(
        prepared,
        local_window=local_window,
        minimum_move_atr=0.45,
    )
    patterns = _detect_reversals(prepared, pivots, timeframe)
    breakout = _detect_range_breakout(prepared, timeframe)
    if breakout:
        patterns.append(breakout)
    if len(pivots) >= 6:
        wave = _wave_pattern(
            tuple(pivots[-6:]), timeframe=timeframe, prepared=prepared
        )
        if wave:
            patterns.append(wave)
    if len(pivots) >= 4:
        abc = _abc_pattern(
            tuple(pivots[-4:]), timeframe=timeframe, prepared=prepared
        )
        if abc:
            patterns.append(abc)

    # Una sola lectura por tipo evita que combinaciones solapadas inflen el sesgo.
    unique: dict[ChartPatternType, ChartPattern] = {}
    for pattern in patterns:
        current = unique.get(pattern.pattern_type)
        if current is None or (pattern.valid, pattern.confidence, pattern.detected_at) > (
            current.valid,
            current.confidence,
            current.detected_at,
        ):
            unique[pattern.pattern_type] = pattern
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.valid, item.confirmed, item.confidence, item.detected_at),
            reverse=True,
        )
    )


def evaluate_pattern_influence(
    patterns: tuple[ChartPattern, ...],
    *,
    signal: str,
) -> PatternInfluence:
    """Convierte solo patrones validos en puntos simetricos y veto operativo."""

    valid = [item for item in patterns if item.valid]
    if not valid:
        return PatternInfluence(0.0, "Sin patrón confirmado por encima de 75%.", False, "")

    raw_impact = 0.0
    details: list[str] = []
    for pattern in valid:
        base_weight = 5.0 if pattern.timeframe == "5m" else 3.5
        if pattern.pattern_type in {
            ChartPatternType.RANGE_BREAKOUT_UP,
            ChartPatternType.RANGE_BREAKOUT_DOWN,
        }:
            base_weight += 1.0
        signed = base_weight * pattern.confidence / 100.0
        if pattern.direction is PatternDirection.BEARISH:
            signed = -signed
        raw_impact += signed
        details.append(
            f"{pattern.label} {pattern.timeframe} {pattern.confidence:.1f}% ({signed:+.1f} pp)"
        )
    impact = round(max(-10.0, min(10.0, raw_impact)), 1)

    opposite = PatternDirection.BEARISH if signal == "BUY" else PatternDirection.BULLISH
    veto_candidates = [
        item
        for item in valid
        if signal in {"BUY", "SELL"}
        and item.timeframe == "5m"
        and item.direction is opposite
    ]
    veto_pattern = max(veto_candidates, key=lambda item: item.confidence, default=None)
    veto_reason = (
        f"{veto_pattern.label} confirmado en 5m con {veto_pattern.confidence:.1f}% "
        f"contradice la operación {signal}."
        if veto_pattern
        else ""
    )
    return PatternInfluence(
        impact_points=impact,
        detail="; ".join(details),
        veto=veto_pattern is not None,
        veto_reason=veto_reason,
    )


def scan_multi_timeframe_patterns(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
) -> tuple[ChartPattern, ...]:
    """Analiza 5 minutos y diario manteniendo un único orden auditable."""

    combined = detect_chart_patterns(intraday, timeframe="5m") + detect_chart_patterns(
        daily, timeframe="1D"
    )
    return tuple(
        sorted(
            combined,
            key=lambda item: (item.valid, item.confirmed, item.confidence, item.detected_at),
            reverse=True,
        )
    )
