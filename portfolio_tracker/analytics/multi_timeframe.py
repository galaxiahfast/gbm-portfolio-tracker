"""Contexto multi-temporal, zonas de liquidez y veto central de riesgo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
import hashlib
import math
from statistics import median
import warnings

import numpy as np
import pandas as pd


class MacroTrend(StrEnum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass(frozen=True, slots=True)
class LiquidityZone:
    lower: float
    upper: float
    center: float
    volume_share_pct: float


@dataclass(frozen=True, slots=True)
class StrictConfluenceDecision:
    probability_up: float
    probability_down: float
    operation_probability: float
    risk_veto: bool
    alert: str
    reasons: tuple[str, ...]
    scenario: str


@dataclass(frozen=True, slots=True)
class HorizonProjection:
    """Distribución auditable producida por un motor temporal identificado."""

    label: str
    probability_up: float
    probability_range: float
    probability_down: float
    bias: str
    bullish_target: float
    range_low: float
    range_high: float
    bearish_target: float
    atr_value: float
    local_support: float
    local_resistance: float
    engine_name: str = "Motor no identificado"
    probability_status: str = "Score heurístico preliminar"
    calibration_samples: int = 0
    brier_score: float | None = None
    calibration_training_samples: int = 0
    calibration_fit_samples: int = 0
    calibration_holdout_samples: int = 0
    calibration_excluded: int = 0
    raw_brier_score: float | None = None
    baseline_brier_score: float | None = None
    calibration_detail: str = "Sin evaluación OOS"


@dataclass(frozen=True, slots=True)
class ExecutionLevels:
    """Plan técnico informativo derivado de soporte, ATR y objetivos cortos."""

    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reference_support: float
    direction: str
    atr_5m: float
    stop_atr_multiple: float
    pattern_target_applied: bool
    pattern_target_label: str
    structural_stop_applied: bool
    take_profit_1_reward_risk: float = 0.0
    minimum_reward_risk: float = 1.5


@dataclass(frozen=True, slots=True)
class DailyProjectionPoint:
    """Escenario reproducible de bootstrap para una sesión futura."""

    day_number: int
    session_date: date
    expected_close: float
    daily_floor: float
    daily_ceiling: float
    atr_value: float


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrega OHLCV localmente; no realiza solicitudes de red adicionales."""

    source = frame.copy()
    if isinstance(source.index, pd.DatetimeIndex):
        source.index = source.index.as_unit("ns")
    # Pandas/NumPy pueden emitir una advertencia interna por índices sintéticos
    # con resolución "generic"; se limita únicamente a esta agregación.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'generic' unit for NumPy timedelta is deprecated.*",
            category=DeprecationWarning,
        )
        kwargs = {"origin": "start_day", "offset": pd.Timedelta(minutes=30)} if rule == "60min" else {}
        result = source.resample(rule, **kwargs).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    return result.dropna(subset=["Open", "High", "Low", "Close"])


def add_context_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for span in (9, 21, 50, 200):
        result[f"EMA{span}"] = result["Close"].ewm(
            span=span, adjust=False, min_periods=span
        ).mean()

    ema12 = result["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = result["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_signal"] = result["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    result["MACD_histogram"] = result["MACD"] - result["MACD_signal"]

    high_9 = result["High"].rolling(9, min_periods=9).max()
    low_9 = result["Low"].rolling(9, min_periods=9).min()
    high_26 = result["High"].rolling(26, min_periods=26).max()
    low_26 = result["Low"].rolling(26, min_periods=26).min()
    high_52 = result["High"].rolling(52, min_periods=52).max()
    low_52 = result["Low"].rolling(52, min_periods=52).min()
    result["Ichimoku_Tenkan"] = (high_9 + low_9) / 2
    result["Ichimoku_Kijun"] = (high_26 + low_26) / 2
    result["Ichimoku_Senkou_A"] = (
        (result["Ichimoku_Tenkan"] + result["Ichimoku_Kijun"]) / 2
    ).shift(26)
    result["Ichimoku_Senkou_B"] = ((high_52 + low_52) / 2).shift(26)
    return result


def classify_macro_trend(frame: pd.DataFrame) -> MacroTrend:
    usable = frame.dropna(subset=["EMA21", "EMA50", "MACD", "MACD_signal"])
    if usable.empty:
        return MacroTrend.NEUTRAL
    latest = usable.iloc[-1]
    close = float(latest["Close"])
    ema21 = float(latest["EMA21"])
    ema50 = float(latest["EMA50"])
    macd_bullish = float(latest["MACD"]) > float(latest["MACD_signal"])
    if close > ema21 > ema50 and macd_bullish:
        return MacroTrend.STRONG_BULLISH
    if close < ema21 < ema50 and not macd_bullish:
        return MacroTrend.STRONG_BEARISH
    if close > ema21 and macd_bullish:
        return MacroTrend.BULLISH
    if close < ema21 and not macd_bullish:
        return MacroTrend.BEARISH
    return MacroTrend.NEUTRAL


def calculate_liquidity_zones(
    daily: pd.DataFrame,
    sessions: int = 252,
    bins: int = 20,
    top_n: int = 3,
) -> tuple[LiquidityZone, ...]:
    """Aproxima nodos de alto volumen; no representa el libro real de órdenes."""

    sample = daily.tail(sessions)
    if sample.empty or float(sample["High"].max()) <= float(sample["Low"].min()):
        return ()
    typical_price = (sample["High"] + sample["Low"] + sample["Close"]) / 3
    buckets = pd.cut(typical_price, bins=bins, include_lowest=True)
    volume_by_bucket = sample["Volume"].groupby(buckets, observed=True).sum()
    total_volume = float(volume_by_bucket.sum())
    if total_volume <= 0:
        return ()
    zones: list[LiquidityZone] = []
    for interval, volume in volume_by_bucket.nlargest(top_n).items():
        lower, upper = float(interval.left), float(interval.right)
        zones.append(
            LiquidityZone(
                lower=lower,
                upper=upper,
                center=(lower + upper) / 2,
                volume_share_pct=float(volume) / total_volume * 100,
            )
        )
    return tuple(sorted(zones, key=lambda zone: zone.center, reverse=True))


def strict_confluence_gate(
    probability_up: float,
    signal: str,
    weekly_trend: MacroTrend,
    monthly_trend: MacroTrend,
    adx: float,
    overextended_unconfirmed: bool,
    preexisting_veto_reason: str = "",
    conditional_block_reason: str = "",
    activation_trigger: str = "",
    activation_trigger_met: bool | None = None,
) -> StrictConfluenceDecision:
    """Aplica veto de régimen y limita a 39/100 el score operativo."""

    is_buy = signal == "BUY"
    is_sell = signal == "SELL"
    operational = is_buy or is_sell
    reasons: list[str] = []
    if is_buy and weekly_trend is MacroTrend.STRONG_BEARISH:
        reasons.append("La tendencia semanal firme contradice la compra.")
    if is_sell and weekly_trend is MacroTrend.STRONG_BULLISH:
        reasons.append("La tendencia semanal firme contradice la venta.")
    if is_buy and monthly_trend is MacroTrend.STRONG_BEARISH:
        reasons.append("La tendencia mensual firme contradice la compra.")
    # Un SHORT contra un mensual fuertemente alcista se gestiona como táctico
    # mediante volumen y exposición reducida en el motor; no se veta aquí.
    if operational and adx < 20:
        reasons.append("ADX menor a 20: mercado lateral sin ventaja operativa.")
    if operational and overextended_unconfirmed:
        reasons.append("Precio sobreextendido respecto a EMA21 diaria y sin volumen.")
    if operational and preexisting_veto_reason:
        reasons.append(preexisting_veto_reason)
    if conditional_block_reason:
        reasons.append(conditional_block_reason)

    probability_up = round(max(0.0, min(100.0, probability_up)), 1)
    probability_down = round(100.0 - probability_up, 1)
    risk_veto = bool(reasons)
    if conditional_block_reason:
        operation_probability = 0.0
        alert = "ESPERAR CORRECCIÓN / ENTRADA LONG BLOQUEADA"
        scenario = (
            "PLAN CONDICIONAL: no ejecutar hasta que el precio y los osciladores "
            "validen el retroceso. Esperar que %K cruce a la baja de 80; para "
            "buscar un rebote LONG, exigir además una nueva señal desde sobreventa (<20)."
        )
    elif risk_veto:
        # El veto de régimen tiene precedencia sobre cualquier gatillo pendiente.
        directional_score = probability_up if is_buy else probability_down
        operation_probability = min(directional_score, 39.0)
        alert = "⚠️ RIESGO ALTO / RÉGIMEN SIN VENTAJA: EVITAR OPERACIÓN"
        scenario = alert
    elif activation_trigger and activation_trigger_met is False:
        operation_probability = 0.0
        alert = ""
        scenario = (
            "PLAN CONDICIONAL - DETONANTE PENDIENTE: "
            + (activation_trigger or "se requiere confirmación cuantitativa explícita.")
        )
    elif not operational:
        operation_probability = 0.0
        alert = ""
        scenario = "SIN OPERACIÓN: no existe un gatillo confirmado en 5 minutos."
    else:
        operation_probability = probability_up if is_buy else probability_down
        alert = ""
        direction = "COMPRA" if is_buy else "VENTA"
        if operation_probability >= 65:
            scenario = f"ESCENARIO {direction} VÁLIDO: score {operation_probability:.1f}/100; ejecutar solo con riesgo definido."
        else:
            scenario = f"ESPERAR: señal de {direction.lower()} con score {operation_probability:.1f}/100."
    return StrictConfluenceDecision(
        probability_up=probability_up,
        probability_down=probability_down,
        operation_probability=round(operation_probability, 1),
        risk_veto=risk_veto,
        alert=alert,
        reasons=tuple(reasons),
        scenario=scenario,
    )


def _latest_context_bias(frame: pd.DataFrame) -> float:
    """Resume EMA y MACD del último cierre en un sesgo entre -1 y +1."""

    if frame.empty:
        return 0.0
    latest = frame.iloc[-1]
    close = float(latest.get("Close", float("nan")))
    if not math.isfinite(close) or close == 0:
        return 0.0

    votes: list[float] = []
    for column, weight in (("EMA9", 1.0), ("EMA21", 1.2), ("EMA50", 1.4), ("EMA200", 1.6)):
        value = float(latest.get(column, float("nan")))
        if math.isfinite(value):
            votes.append(weight if close > value else -weight)
    macd = float(latest.get("MACD", float("nan")))
    macd_signal = float(latest.get("MACD_signal", float("nan")))
    if math.isfinite(macd) and math.isfinite(macd_signal):
        votes.append(1.3 if macd > macd_signal else -1.3)
    if not votes:
        return 0.0
    return max(-1.0, min(1.0, sum(votes) / sum(abs(item) for item in votes)))


def _projection_distribution(bias: float, risk_veto: bool, short_term: bool) -> tuple[float, float, float]:
    """Convierte sesgo en subida/rango/bajada conservando una suma exacta de 100."""

    bias = max(-1.0, min(1.0, bias))
    if risk_veto and short_term:
        bias *= 0.35
    range_probability = 42.0 - 16.0 * abs(bias)
    if risk_veto and short_term:
        range_probability += 12.0
    range_probability = max(22.0, min(62.0, range_probability))
    directional = 100.0 - range_probability
    probability_up = directional * (0.5 + 0.5 * bias)
    probability_up = round(max(4.0, min(directional - 4.0, probability_up)), 1)
    probability_range = round(range_probability, 1)
    probability_down = round(100.0 - probability_up - probability_range, 1)
    return probability_up, probability_range, probability_down


def _atr(frame: pd.DataFrame, fallback_price: float, period: int = 14) -> float:
    """ATR de Wilder con respaldo porcentual para historiales incompletos."""

    if frame.empty:
        return max(fallback_price * 0.015, 0.01)
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    series = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().dropna()
    value = float(series.iloc[-1]) if not series.empty else float(true_range.tail(period).mean())
    return value if math.isfinite(value) and value > 0 else max(fallback_price * 0.015, 0.01)


def _price_projection(
    frame: pd.DataFrame,
    price: float,
    horizon_bars: float,
    support_window: int,
) -> tuple[float, float, float, float, float, float, float]:
    """Combina ATR, Bollinger y estructura local en objetivos monetarios."""

    sample = frame.dropna(subset=["High", "Low", "Close"])
    if sample.empty:
        sample = pd.DataFrame({"High": [price], "Low": [price], "Close": [price]})
    close_sample = sample["Close"].tail(20)
    middle = float(close_sample.mean())
    deviation = float(close_sample.std(ddof=0)) if len(close_sample) > 1 else 0.0
    atr_value = _atr(sample, price)
    atr_move = max(atr_value * math.sqrt(max(horizon_bars, 1.0)), price * 0.0025)
    bollinger_upper = middle + 2 * deviation
    bollinger_lower = max(0.01, middle - 2 * deviation)
    structure = sample.tail(max(2, support_window))
    local_support = max(0.01, float(structure["Low"].min()))
    local_resistance = max(price, float(structure["High"].max()))

    upward_candidates = [price + atr_move]
    upward_candidates.extend(value for value in (bollinger_upper, local_resistance) if value > price)
    downward_candidates = [max(0.01, price - atr_move)]
    downward_candidates.extend(value for value in (bollinger_lower, local_support) if 0 < value < price)
    bullish_target = max(price, float(median(upward_candidates)))
    bearish_target = max(0.01, min(price, float(median(downward_candidates))))
    bullish_target = min(bullish_target, price + atr_move * 1.6)
    bearish_target = max(bearish_target, max(0.01, price - atr_move * 1.6))

    bollinger_half_width = max(deviation * 2, atr_value * 0.5)
    horizon_width_factor = min(1.05, 0.45 + 0.15 * math.sqrt(math.sqrt(max(horizon_bars, 1.0))))
    range_move = min(
        atr_move * 0.55,
        max(bollinger_half_width * horizon_width_factor, atr_value * 0.35),
    )
    range_low = max(0.01, price - range_move)
    range_high = price + range_move
    if local_support < price:
        range_low = max(range_low, local_support)
    if local_resistance > price:
        range_high = min(range_high, local_resistance)
    range_low = min(range_low, price)
    range_high = max(range_high, price)
    return tuple(round(value, 2) for value in (
        bullish_target,
        range_low,
        range_high,
        bearish_target,
        atr_value,
        local_support,
        local_resistance,
    ))  # type: ignore[return-value]


def calculate_horizon_projections(
    probability_up: float,
    risk_veto: bool,
    last_price: float,
    intraday: pd.DataFrame,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
) -> tuple[HorizonProjection, ...]:
    """Proyecta horizontes con tres motores desacoplados.

    El oscilador operativo de 5 minutos participa exclusivamente en 1h/6h. El
    motor swing usa precio/ATR/medias diarios y el táctico usa solo estructura
    semanal/mensual; el filtro fundamental se incorpora después mediante el
    corte versionado, sin contaminar los gatillos intradía.
    """

    engine_bias = max(-1.0, min(1.0, (float(probability_up) - 50.0) / 35.0))
    intraday_bias = _latest_context_bias(intraday)
    hourly_bias = _latest_context_bias(hourly)
    daily_bias = _latest_context_bias(daily)
    weekly_bias = _latest_context_bias(weekly)
    monthly_bias = _latest_context_bias(monthly)
    definitions = (
        ("1 Hora", 0.70 * engine_bias + 0.30 * intraday_bias, True, intraday, 12.0, 36, "Intradía / microestructura"),
        ("6 Horas", 0.50 * engine_bias + 0.30 * intraday_bias + 0.20 * hourly_bias, True, hourly, 6.0, 20, "Intradía / microestructura"),
        ("1 Día", 0.35 * hourly_bias + 0.65 * daily_bias, True, daily, 1.0, 20, "Swing / acción del precio"),
        ("1 Semana", 0.45 * daily_bias + 0.55 * weekly_bias, False, daily, 5.0, 30, "Swing / acción del precio"),
        ("1 Mes", 0.35 * weekly_bias + 0.65 * monthly_bias, False, daily, 22.0, 66, "Táctico / fundamental"),
        ("6 Meses", 0.20 * weekly_bias + 0.80 * monthly_bias, False, daily, 126.0, 252, "Táctico / fundamental"),
    )
    projections: list[HorizonProjection] = []
    for label, bias, short_term, frame, horizon_bars, support_window, engine_name in definitions:
        uses_intraday_score = engine_name.startswith("Intradía")
        if uses_intraday_score and engine_bias < 0 < bias:
            bias = engine_bias * 0.35
        elif uses_intraday_score and engine_bias > 0 > bias:
            bias = engine_bias * 0.35
        up, range_probability, down = _projection_distribution(bias, risk_veto, short_term)
        bias_label = "Alcista" if up > max(range_probability, down) else "Bajista" if down > max(up, range_probability) else "Rango"
        targets = _price_projection(frame, last_price, horizon_bars, support_window)
        projections.append(
            HorizonProjection(
                label, up, range_probability, down, bias_label, *targets,
                engine_name=engine_name,
            )
        )
    return tuple(projections)


def calculate_execution_levels(
    last_price: float,
    projections: tuple[HorizonProjection, ...],
    additional_supports: tuple[float, ...] = (),
    additional_resistances: tuple[float, ...] = (),
    probability_up: float = 50.0,
    signal: str = "NEUTRAL",
    intraday_atr: float | None = None,
    confirmed_pattern_target: float | None = None,
    confirmed_pattern_label: str = "",
    atr_stop_multiple: float = 2.25,
) -> ExecutionLevels:
    """Construye niveles coherentes con el sesgo largo o bajista vigente."""

    if len(projections) < 2:
        raise ValueError("Se requieren al menos los horizontes de 1 y 6 horas.")
    short = projections[0]
    atr_5m = (
        float(intraday_atr)
        if intraday_atr is not None
        and math.isfinite(float(intraday_atr))
        and float(intraday_atr) > 0
        else float(short.atr_value)
    )
    atr_5m = round(atr_5m, 4)
    stop_atr_multiple = max(2.0, min(2.5, float(atr_stop_multiple)))
    pattern_target_applied = False
    structural_stop_applied = False
    bearish_plan = signal == "SELL" or (signal != "BUY" and probability_up < 50.0)
    if bearish_plan:
        resistances = [short.local_resistance, short.range_high]
        resistances.extend(float(value) for value in additional_resistances)
        valid_levels = [value for value in resistances if math.isfinite(value) and value >= last_price]
        reference_support = min(valid_levels, default=last_price + short.atr_value)
        entry_high = max(last_price, reference_support)
        entry_low = max(0.01, min(entry_high, reference_support - short.atr_value * 0.35))
        # Invalida un corto tras 2.0-2.5 ATR o por encima de una resistencia
        # estructural próxima, tomando siempre el nivel más conservador.
        entry_high = round(entry_high, 2)
        atr_stop = entry_high + stop_atr_multiple * atr_5m
        nearby_resistances = [
            value
            for value in valid_levels
            if entry_high <= value <= entry_high + 3.0 * atr_5m
        ]
        structural_stop = (
            max(nearby_resistances) + 0.25 * atr_5m
            if nearby_resistances
            else atr_stop
        )
        stop_loss = max(atr_stop, structural_stop)
        structural_stop_applied = structural_stop > atr_stop
        take_profit_1 = min(last_price, short.bearish_target)
        minimum_tp1 = max(
            0.01,
            entry_high - 1.5 * (stop_loss - entry_high),
        )
        take_profit_1 = min(take_profit_1, minimum_tp1)
        take_profit_2 = min(take_profit_1, projections[1].bearish_target)
        tp1_reward_risk = (
            (entry_high - take_profit_1) / max(stop_loss - entry_high, 0.01)
        )
        direction = "SHORT"
    else:
        supports = [short.local_support, short.range_low]
        supports.extend(float(value) for value in additional_supports)
        valid_levels = [value for value in supports if math.isfinite(value) and 0 < value <= last_price]
        reference_support = max(valid_levels, default=max(0.01, last_price - short.atr_value))
        entry_low = max(0.01, reference_support)
        entry_high = max(entry_low, min(last_price, reference_support + short.atr_value * 0.35))
        # Para LONG se adopta el límite inferior como entrada conservadora. El
        # stop tolera 2.0-2.5 ATR y, si existe soporte técnico cercano inferior,
        # queda por debajo de esa estructura en vez de dentro del ruido normal.
        entry_low = round(entry_low, 2)
        atr_stop = entry_low - stop_atr_multiple * atr_5m
        nearby_supports = [
            value
            for value in valid_levels
            if entry_low - 3.0 * atr_5m <= value <= entry_low
        ]
        structural_stop = (
            min(nearby_supports) - 0.25 * atr_5m
            if nearby_supports
            else atr_stop
        )
        stop_loss = max(0.01, min(atr_stop, structural_stop))
        structural_stop_applied = structural_stop < atr_stop
        take_profit_1 = max(last_price, short.bullish_target)
        minimum_tp1 = entry_high + 1.5 * (entry_high - stop_loss)
        if (
            confirmed_pattern_target is not None
            and math.isfinite(float(confirmed_pattern_target))
            and float(confirmed_pattern_target) > last_price
        ):
            take_profit_1 = max(float(confirmed_pattern_target), minimum_tp1)
            pattern_target_applied = float(confirmed_pattern_target) >= minimum_tp1
        else:
            take_profit_1 = max(take_profit_1, minimum_tp1)
        take_profit_2 = max(take_profit_1, projections[1].bullish_target)
        tp1_reward_risk = (
            (take_profit_1 - entry_high) / max(entry_high - stop_loss, 0.01)
        )
        direction = "LONG"
    return ExecutionLevels(
        entry_low=round(entry_low, 2),
        entry_high=round(entry_high, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(take_profit_1, 2),
        take_profit_2=round(take_profit_2, 2),
        reference_support=round(reference_support, 2),
        direction=direction,
        atr_5m=atr_5m,
        stop_atr_multiple=stop_atr_multiple,
        pattern_target_applied=pattern_target_applied,
        pattern_target_label=confirmed_pattern_label if pattern_target_applied else "",
        structural_stop_applied=structural_stop_applied,
        take_profit_1_reward_risk=round(tp1_reward_risk, 3),
        minimum_reward_risk=1.5,
    )


def calculate_15_day_projection(
    last_price: float,
    daily: pd.DataFrame,
    probability_up: float,
    risk_veto: bool,
    as_of: date | datetime | pd.Timestamp,
    sessions: int = 15,
) -> tuple[DailyProjectionPoint, ...]:
    """Genera un escenario bootstrap reproducible y una banda Monte Carlo.

    Los choques proceden de retornos diarios históricos centrados. El drift
    incorpora momentum y confluencia, y una reacción suave evita ignorar los
    soportes y resistencias observados. No son velas futuras predichas.
    """

    if sessions <= 0:
        raise ValueError("El numero de sesiones proyectadas debe ser positivo.")
    if not math.isfinite(last_price) or last_price <= 0:
        raise ValueError("El ultimo precio debe ser positivo y finito.")

    usable = daily.dropna(subset=["High", "Low", "Close"])
    if usable.empty:
        raise ValueError("Se requieren velas diarias para proyectar la trayectoria.")
    atr_value = _atr(usable, last_price)
    atr_pct = atr_value / last_price

    recent_close = usable["Close"].tail(126)
    log_returns = np.log(recent_close / recent_close.shift(1)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    recent_drift = float(log_returns.tail(20).median()) if not log_returns.empty else 0.0
    confluence_bias = max(-1.0, min(1.0, (float(probability_up) - 50.0) / 35.0))
    context_bias = _latest_context_bias(usable)
    raw_daily_drift = (
        0.45 * recent_drift
        + 0.35 * confluence_bias * atr_pct * 0.22
        + 0.20 * context_bias * atr_pct * 0.18
    )
    daily_drift_cap = max(atr_pct * 0.30, 0.0005)
    daily_drift = max(-daily_drift_cap, min(daily_drift_cap, raw_daily_drift))
    if risk_veto:
        daily_drift *= 0.40

    residuals = log_returns - float(log_returns.median()) if not log_returns.empty else pd.Series([0.0])
    residual_values = residuals.clip(-2.5 * atr_pct, 2.5 * atr_pct).to_numpy(dtype=float)
    realized_sigma = float(log_returns.tail(60).std(ddof=0)) if len(log_returns) > 1 else 0.0
    shock_scale = max(
        atr_pct * 0.35,
        min(atr_pct * 1.20, realized_sigma or atr_pct * 0.50),
    )
    residual_sigma = float(np.std(residual_values))
    if residual_sigma > 1e-9:
        residual_values = residual_values / residual_sigma * shock_scale

    structure = usable.tail(60)
    support_levels = sorted(
        {
            float(structure["Low"].tail(window).min())
            for window in (5, 10, 20, 40, 60)
            if len(structure) >= window
        }
    )
    resistance_levels = sorted(
        {
            float(structure["High"].tail(window).max())
            for window in (5, 10, 20, 40, 60)
            if len(structure) >= window
        }
    )

    def react_to_structure(previous_price: float, proposed_price: float) -> float:
        nearby_supports = [level for level in support_levels if level <= previous_price]
        nearby_resistances = [level for level in resistance_levels if level >= previous_price]
        support = max(nearby_supports, default=0.01)
        resistance = min(nearby_resistances, default=float("inf"))
        if proposed_price < support and confluence_bias > -0.70:
            proposed_price += (support - proposed_price) * 0.45
        if proposed_price > resistance and confluence_bias < 0.70:
            proposed_price -= (proposed_price - resistance) * 0.45
        return max(0.01, proposed_price)

    seed_payload = "|".join(f"{value:.4f}" for value in recent_close.tail(60))
    seed_payload += f"|{pd.Timestamp(as_of).date()}|{probability_up:.2f}|{risk_veto}"
    seed = int.from_bytes(
        hashlib.sha256(seed_payload.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)

    def bootstrap_shocks(count: int) -> np.ndarray:
        if len(residual_values) < 3:
            return np.resize(residual_values, count)
        output: list[float] = []
        while len(output) < count:
            start = int(rng.integers(0, len(residual_values)))
            output.extend(
                float(residual_values[(start + offset) % len(residual_values)])
                for offset in range(3)
            )
        return np.asarray(output[:count], dtype=float)

    simulation_count = 320
    simulated = np.empty((simulation_count, sessions), dtype=float)
    for path_index in range(simulation_count):
        path_price = last_price
        for day_index, shock in enumerate(bootstrap_shocks(sessions)):
            path_price = react_to_structure(
                path_price,
                path_price * math.exp(daily_drift + shock),
            )
            simulated[path_index, day_index] = path_price

    central_prices: list[float] = []
    central_price = last_price
    for shock in bootstrap_shocks(sessions):
        central_price = react_to_structure(
            central_price,
            central_price * math.exp(daily_drift + shock * 0.72),
        )
        central_prices.append(central_price)

    previous_simulated = np.column_stack(
        (
            np.full(simulation_count, last_price, dtype=float),
            simulated[:, :-1],
        )
    )
    simulated_daily_moves = np.abs(simulated / previous_simulated - 1.0)
    local_move_quantiles = np.quantile(simulated_daily_moves, 0.70, axis=0)
    historical_range_pct = (
        (usable["High"] - usable["Low"])
        / usable["Close"].shift(1).replace(0.0, float("nan"))
    ).replace([np.inf, -np.inf], np.nan).dropna()
    typical_range_pct = (
        float(historical_range_pct.tail(60).quantile(0.65))
        if not historical_range_pct.empty
        else atr_pct * 0.80
    )
    historical_body_pct = log_returns.abs()
    body_cap_pct = min(
        atr_pct * 0.85,
        max(
            atr_pct * 0.35,
            float(historical_body_pct.tail(60).quantile(0.75))
            if not historical_body_pct.empty
            else atr_pct * 0.50,
        ),
    )
    anchor = pd.Timestamp(as_of).tz_localize(None).normalize()
    future_dates = pd.bdate_range(start=anchor + pd.offsets.BDay(1), periods=sessions)
    points: list[DailyProjectionPoint] = []
    projected_open = last_price
    for day_number, session_date in enumerate(future_dates, start=1):
        raw_close = central_prices[day_number - 1]
        raw_body_return = raw_close / projected_open - 1.0
        bounded_body_return = max(
            -body_cap_pct,
            min(body_cap_pct, raw_body_return),
        )
        expected_close = max(0.01, projected_open * (1.0 + bounded_body_return))
        body_size = abs(expected_close - projected_open)
        target_range_pct = max(
            atr_pct * 0.65,
            typical_range_pct,
            float(local_move_quantiles[day_number - 1]) * 1.15,
            body_size / projected_open / 0.72,
        )
        target_range_pct = min(atr_pct * 1.35, target_range_pct)
        target_range = projected_open * target_range_pct
        wick_budget = max(atr_value * 0.12, target_range - body_size)
        # Una vela alcista suele dejar algo más de mecha inferior y viceversa.
        lower_share = 0.55 if expected_close >= projected_open else 0.45
        lower_wick = wick_budget * lower_share
        upper_wick = wick_budget - lower_wick
        floor = max(0.01, min(projected_open, expected_close) - lower_wick)
        ceiling = max(projected_open, expected_close) + upper_wick
        points.append(
            DailyProjectionPoint(
                day_number=day_number,
                session_date=session_date.date(),
                expected_close=round(expected_close, 2),
                daily_floor=round(min(floor, expected_close), 2),
                daily_ceiling=round(max(ceiling, expected_close), 2),
                atr_value=round(atr_value, 4),
            )
        )
        projected_open = expected_close
    return tuple(points)
