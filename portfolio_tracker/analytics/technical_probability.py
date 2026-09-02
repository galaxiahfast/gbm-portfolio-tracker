"""Motor técnico explicable de probabilidad intradía, Fases 1, 2 y 3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import math

import pandas as pd

from portfolio_tracker.analytics.chart_patterns import (
    ChartPattern,
    ChartPatternType,
    PatternDirection,
    evaluate_pattern_influence,
    scan_multi_timeframe_patterns,
)
from portfolio_tracker.analytics.multi_timeframe import (
    ExecutionLevels,
    HorizonProjection,
    DailyProjectionPoint,
    LiquidityZone,
    MacroTrend,
    add_context_indicators,
    calculate_horizon_projections,
    calculate_execution_levels,
    calculate_15_day_projection,
    calculate_liquidity_zones,
    classify_macro_trend,
    resample_ohlcv,
    strict_confluence_gate,
)


class TechnicalSignal(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WATCH_BUY = "WATCH_BUY"
    WATCH_SELL = "WATCH_SELL"
    NEUTRAL = "NEUTRAL"


class DailyTrend(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MomentumState(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class ObvState(StrEnum):
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    CONFIRMING_UP = "CONFIRMING_UP"
    CONFIRMING_DOWN = "CONFIRMING_DOWN"
    NEUTRAL = "NEUTRAL"


class CloudPosition(StrEnum):
    ABOVE = "ABOVE"
    INSIDE = "INSIDE"
    BELOW = "BELOW"


class CandlePattern(StrEnum):
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    BULLISH_CONTINUATION = "BULLISH_CONTINUATION"
    BEARISH_CONTINUATION = "BEARISH_CONTINUATION"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class PivotLevels:
    pivot: float
    s1: float
    s2: float
    r1: float
    r2: float
    source_date: str


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    name: str
    impact_points: float
    detail: str


@dataclass(frozen=True, slots=True)
class FibonacciLevels:
    high: float
    low: float
    level_382: float
    level_500: float
    level_618: float
    nearest_level: float
    nearest_ratio: str
    distance_pct: float
    near_zone: bool
    role: str
    source_start: str
    source_end: str


@dataclass(frozen=True, slots=True)
class ProbabilityAnalysis:
    symbol: str
    as_of: datetime
    last_price: float
    probability_up: float
    probability_down: float
    signal: TechnicalSignal
    daily_trend: DailyTrend
    volume_confirmed: bool
    volume_ratio: float
    stochastic_k: float
    stochastic_d: float
    atr_5m: float
    stoch_overbought_extreme: bool
    stoch_oversold_extreme: bool
    rebound_watch_active: bool
    support_interaction: bool
    nearest_support: float
    long_entry_blocked: bool
    execution_plan_conditional: bool
    execution_plan_label: str
    activation_trigger: str
    activation_trigger_met: bool
    tactical_short: bool
    exposure_factor: float
    neckline_heat_warning: str
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    ema9: float
    ema21: float
    ema50: float
    ema200: float
    macd_5m: float
    macd_signal_5m: float
    macd_histogram_5m: float
    macd_daily: float
    macd_signal_daily: float
    macd_histogram_daily: float
    macd_state_5m: MomentumState
    macd_state_daily: MomentumState
    vwap: float
    price_vs_vwap_pct: float
    above_vwap: bool
    adx: float
    range_market: bool
    obv: float
    obv_state: ObvState
    obv_price_change_pct: float
    fibonacci: FibonacciLevels
    ichimoku_5m: CloudPosition
    ichimoku_daily: CloudPosition
    tenkan_5m: float
    kijun_5m: float
    cloud_upper_5m: float
    cloud_lower_5m: float
    tenkan_daily: float
    kijun_daily: float
    cloud_upper_daily: float
    cloud_lower_daily: float
    candle_pattern: CandlePattern
    candle_detail: str
    weekly_trend: MacroTrend
    monthly_trend: MacroTrend
    weekly_support: float
    weekly_resistance: float
    annual_fibonacci: FibonacciLevels
    liquidity_zones: tuple[LiquidityZone, ...]
    operation_probability: float
    risk_veto: bool
    risk_alert: str
    risk_reasons: tuple[str, ...]
    scenario: str
    overextended_unconfirmed: bool
    signal_rejected: bool
    score_breakdown: tuple[ScoreComponent, ...]
    pivots: PivotLevels
    suggested_level: float
    verdict: str
    observations: tuple[str, ...]
    warnings: tuple[str, ...]
    intraday_indicators: pd.DataFrame
    hourly_indicators: pd.DataFrame
    daily_indicators: pd.DataFrame
    weekly_indicators: pd.DataFrame
    monthly_indicators: pd.DataFrame
    chart_patterns: tuple[ChartPattern, ...]
    chart_pattern_impact: float
    chart_pattern_veto: bool
    horizon_projections: tuple[HorizonProjection, ...]
    execution_levels: ExecutionLevels
    daily_projection: tuple[DailyProjectionPoint, ...]
    fundamental_score: float = 0.0
    fundamental_label: str = "Sin actualizar"
    fundamental_reasons: tuple[str, ...] = ()
    fundamental_risk_veto: bool = False
    fundamental_snapshot_sha256: str = ""
    fundamental_as_of: str = ""
    raw_probability_up: float = 50.0
    probability_status: str = "Score heurístico preliminar"
    calibration_samples: int = 0
    calibration_brier_score: float | None = None
    event_risk_level: str = "BAJO"
    event_risk_window_until: str = ""
    fundamental_news_audit: tuple[str, ...] = ()
    market_regime: str = "SIN CLASIFICAR"
    position_size_policy: str = "CONDICIONAL"

    @property
    def has_empirical_probability(self) -> bool:
        """Solo permite nomenclatura probabilística con evidencia OOS amplia."""

        return bool(
            self.probability_status == "Probabilidad empíricamente calibrada"
            and self.calibration_samples >= 500
            and self.calibration_brier_score is not None
            and math.isfinite(self.calibration_brier_score)
        )

    @property
    def bullish_display_label(self) -> str:
        return (
            "Probabilidad empírica de subida"
            if self.has_empirical_probability
            else "Score heurístico alcista"
        )

    @property
    def bearish_display_label(self) -> str:
        return (
            "Probabilidad empírica de bajada"
            if self.has_empirical_probability
            else "Score heurístico bajista"
        )

    @property
    def calibration_disclosure(self) -> str:
        if self.has_empirical_probability:
            return (
                f"Calibración empírica OOS · n={self.calibration_samples} · "
                f"Brier {self.calibration_brier_score:.3f}"
            )
        return "Pendiente de calibración empírica Brier"


def validate_probability_analysis(analysis: ProbabilityAnalysis) -> None:
    """Aplica invariantes matemáticos antes de exponer o persistir un análisis."""

    scalar_values = {
        "precio": analysis.last_price,
        "probabilidad de subida": analysis.probability_up,
        "probabilidad de bajada": analysis.probability_down,
        "ATR 5m": analysis.atr_5m,
        "Bollinger inferior": analysis.bollinger_lower,
        "Bollinger media": analysis.bollinger_middle,
        "Bollinger superior": analysis.bollinger_upper,
        "Estocástico %K": analysis.stochastic_k,
        "Estocástico %D": analysis.stochastic_d,
        "VWAP": analysis.vwap,
    }
    invalid_scalars = [name for name, value in scalar_values.items() if not math.isfinite(float(value))]
    if invalid_scalars:
        raise ValueError("El análisis contiene valores no finitos: " + ", ".join(invalid_scalars) + ".")
    if analysis.last_price <= 0 or analysis.atr_5m <= 0 or analysis.vwap <= 0:
        raise ValueError("Precio, VWAP y ATR deben ser estrictamente positivos.")
    if not 0 <= analysis.probability_up <= 100 or not 0 <= analysis.probability_down <= 100:
        raise ValueError("Las probabilidades direccionales deben permanecer entre 0 y 100.")
    if not math.isclose(analysis.probability_up + analysis.probability_down, 100.0, abs_tol=0.11):
        raise ValueError("Las probabilidades de subida y bajada no suman 100%.")
    if not 0 <= analysis.operation_probability <= 100:
        raise ValueError("La probabilidad operativa quedó fuera de 0–100%.")
    if not 0 <= analysis.stochastic_k <= 100 or not 0 <= analysis.stochastic_d <= 100:
        raise ValueError("El Estocástico RSI quedó fuera de su dominio 0–100.")
    if not analysis.bollinger_lower <= analysis.bollinger_middle <= analysis.bollinger_upper:
        raise ValueError("Las Bandas de Bollinger no conservan el orden inferior ≤ media ≤ superior.")
    if not math.isclose(
        analysis.macd_5m - analysis.macd_signal_5m,
        analysis.macd_histogram_5m,
        abs_tol=1e-8,
    ) or not math.isclose(
        analysis.macd_daily - analysis.macd_signal_daily,
        analysis.macd_histogram_daily,
        abs_tol=1e-8,
    ):
        raise ValueError("El histograma MACD no coincide con MACD menos su señal.")

    required_frames = {
        "intradía": analysis.intraday_indicators,
        "diario": analysis.daily_indicators,
    }
    for label, frame in required_frames.items():
        if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"El marco {label} está vacío, desordenado o contiene fechas duplicadas.")
        for column in ("Open", "High", "Low", "Close"):
            if column not in frame or not pd.to_numeric(frame[column], errors="coerce").map(math.isfinite).all():
                raise ValueError(f"El marco {label} contiene datos inválidos en {column}.")
        if (
            (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).any()
            or (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).any()
        ):
            raise ValueError(f"El marco {label} contiene velas OHLC incoherentes.")

    if len(analysis.horizon_projections) != 6:
        raise ValueError("El mapa debe contener seis horizontes temporales.")
    for horizon in analysis.horizon_projections:
        total = horizon.probability_up + horizon.probability_range + horizon.probability_down
        if not math.isclose(total, 100.0, abs_tol=0.2):
            raise ValueError(f"Las probabilidades de {horizon.label} no suman 100%.")
        if min(horizon.bullish_target, horizon.range_low, horizon.range_high, horizon.bearish_target) <= 0:
            raise ValueError(f"El horizonte {horizon.label} contiene precios no positivos.")
        if horizon.range_low > horizon.range_high or horizon.atr_value < 0:
            raise ValueError(f"El rango o ATR de {horizon.label} es inválido.")

    if len(analysis.daily_projection) != 15:
        raise ValueError("La proyección debe contener exactamente 15 sesiones.")
    previous_date = None
    for expected_day, point in enumerate(analysis.daily_projection, start=1):
        if point.day_number != expected_day or (previous_date and point.session_date <= previous_date):
            raise ValueError("La proyección diaria está desordenada o contiene sesiones duplicadas.")
        if point.daily_floor <= 0 or not point.daily_floor <= point.expected_close <= point.daily_ceiling:
            raise ValueError(f"La vela proyectada del día {expected_day} es matemáticamente inválida.")
        previous_date = point.session_date

    levels = analysis.execution_levels
    if levels.direction == "LONG":
        if not levels.stop_loss < levels.entry_low <= levels.entry_high < levels.take_profit_1 <= levels.take_profit_2:
            raise ValueError("Los niveles LONG no respetan stop < entrada < objetivos.")
    elif levels.direction == "SHORT":
        if not levels.take_profit_2 <= levels.take_profit_1 < levels.entry_low <= levels.entry_high < levels.stop_loss:
            raise ValueError("Los niveles SHORT no respetan objetivos < entrada < invalidación.")
    else:
        raise ValueError("La dirección del plan de ejecución no es LONG ni SHORT.")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.where(average_loss != 0, 100.0)


def _add_macd(result: pd.DataFrame) -> None:
    ema12 = result["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = result["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_signal"] = result["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["MACD_histogram"] = result["MACD"] - result["MACD_signal"]


def _add_adx(result: pd.DataFrame, period: int = 14) -> None:
    previous_close = result["Close"].shift(1)
    true_range = pd.concat(
        [
            result["High"] - result["Low"],
            (result["High"] - previous_close).abs(),
            (result["Low"] - previous_close).abs(),
        ], axis=1,
    ).max(axis=1)
    upward = result["High"].diff()
    downward = -result["Low"].diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, float("nan"))
    directional_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / directional_sum
    result["ADX14"] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    result["ATR14"] = atr
    result["Plus_DI14"] = plus_di
    result["Minus_DI14"] = minus_di


def _add_session_vwap(result: pd.DataFrame) -> None:
    typical_price = (result["High"] + result["Low"] + result["Close"]) / 3
    weighted_price = typical_price * result["Volume"]
    session_key = pd.Index(result.index).date
    cumulative_volume = result["Volume"].groupby(session_key).cumsum()
    cumulative_value = weighted_price.groupby(session_key).cumsum()
    result["VWAP"] = cumulative_value / cumulative_volume.replace(0, float("nan"))


def _add_obv(result: pd.DataFrame) -> None:
    change = result["Close"].diff()
    signed_volume = pd.Series(0.0, index=result.index)
    signed_volume.loc[change > 0] = result.loc[change > 0, "Volume"]
    signed_volume.loc[change < 0] = -result.loc[change < 0, "Volume"]
    result["OBV"] = signed_volume.cumsum()


def _add_ichimoku(result: pd.DataFrame) -> None:
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


def add_intraday_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["RSI14"] = _rsi(result["Close"], 14)
    rsi_low = result["RSI14"].rolling(14, min_periods=14).min()
    rsi_high = result["RSI14"].rolling(14, min_periods=14).max()
    denominator = (rsi_high - rsi_low).replace(0, float("nan"))
    result["StochRSI_raw"] = (result["RSI14"] - rsi_low) / denominator * 100
    result["StochRSI_K"] = result["StochRSI_raw"].rolling(3, min_periods=3).mean()
    result["StochRSI_D"] = result["StochRSI_K"].rolling(3, min_periods=3).mean()
    result["BB_middle"] = result["Close"].rolling(20, min_periods=20).mean()
    deviation = result["Close"].rolling(20, min_periods=20).std(ddof=0)
    result["BB_upper"] = result["BB_middle"] + 2 * deviation
    result["BB_lower"] = result["BB_middle"] - 2 * deviation
    result["Volume_MA20"] = result["Volume"].shift(1).rolling(20, min_periods=20).mean()
    _add_macd(result)
    _add_session_vwap(result)
    _add_adx(result)
    _add_obv(result)
    _add_ichimoku(result)
    return result


def add_daily_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return add_context_indicators(frame)


def _momentum_state(line: float, signal: float) -> MomentumState:
    if line > signal:
        return MomentumState.BULLISH
    if line < signal:
        return MomentumState.BEARISH
    return MomentumState.NEUTRAL


def _obv_divergence(frame: pd.DataFrame, lookback: int = 12) -> tuple[ObvState, float]:
    reference = frame.iloc[-min(lookback + 1, len(frame))]
    latest = frame.iloc[-1]
    reference_price = float(reference["Close"])
    price_change = (float(latest["Close"]) / reference_price - 1) * 100 if reference_price else 0.0
    obv_change = float(latest["OBV"] - reference["OBV"])
    if price_change > 0.25 and obv_change < 0:
        return ObvState.DISTRIBUTION, price_change
    if price_change < -0.25 and obv_change > 0:
        return ObvState.ACCUMULATION, price_change
    if price_change > 0.25 and obv_change > 0:
        return ObvState.CONFIRMING_UP, price_change
    if price_change < -0.25 and obv_change < 0:
        return ObvState.CONFIRMING_DOWN, price_change
    return ObvState.NEUTRAL, price_change


def _cloud_assessment(
    row: pd.Series,
    base_weight: float,
) -> tuple[CloudPosition, float, str, float, float]:
    price = float(row["Close"])
    tenkan = float(row["Ichimoku_Tenkan"])
    kijun = float(row["Ichimoku_Kijun"])
    cloud_upper = max(float(row["Ichimoku_Senkou_A"]), float(row["Ichimoku_Senkou_B"]))
    cloud_lower = min(float(row["Ichimoku_Senkou_A"]), float(row["Ichimoku_Senkou_B"]))
    if price > cloud_upper:
        position = CloudPosition.ABOVE
        aligned = tenkan > kijun
        impact = base_weight + (1.0 if aligned else -1.0)
        detail = "Precio sobre la nube; Tenkan sobre Kijun." if aligned else "Precio sobre la nube, pero Tenkan no supera Kijun."
    elif price < cloud_lower:
        position = CloudPosition.BELOW
        aligned = tenkan < kijun
        impact = -base_weight - (1.0 if aligned else -1.0)
        detail = "Precio bajo la nube; Tenkan bajo Kijun." if aligned else "Precio bajo la nube, pero Tenkan no está bajo Kijun."
    else:
        position = CloudPosition.INSIDE
        impact = 0.0
        detail = "Precio dentro de la nube; contexto sin dirección limpia."
    return position, impact, detail, cloud_upper, cloud_lower


def _fibonacci_levels(
    daily: pd.DataFrame,
    session_date,
    price: float,
    tolerance_pct: float = 0.35,
    sessions: int = 22,
) -> FibonacciLevels:
    eligible = daily[pd.Index(daily.index).date < session_date].tail(sessions)
    if len(eligible) < 15:
        raise ValueError("No hay suficientes sesiones para Fibonacci mensual.")
    high = float(eligible["High"].max())
    low = float(eligible["Low"].min())
    price_range = high - low
    if price_range <= 0:
        raise ValueError("El rango mensual no permite calcular Fibonacci.")
    levels = {
        "0.382": high - price_range * 0.382,
        "0.500": high - price_range * 0.500,
        "0.618": high - price_range * 0.618,
    }
    nearest_ratio, nearest_level = min(
        levels.items(), key=lambda item: abs(price - item[1])
    )
    distance_pct = abs(price - nearest_level) / price * 100 if price else 0.0
    return FibonacciLevels(
        high=high,
        low=low,
        level_382=levels["0.382"],
        level_500=levels["0.500"],
        level_618=levels["0.618"],
        nearest_level=nearest_level,
        nearest_ratio=nearest_ratio,
        distance_pct=distance_pct,
        near_zone=distance_pct <= tolerance_pct,
        role="SOPORTE" if price >= nearest_level else "RESISTENCIA",
        source_start=pd.Timestamp(eligible.index[0]).date().isoformat(),
        source_end=pd.Timestamp(eligible.index[-1]).date().isoformat(),
    )


def _fibonacci_score(
    signal: TechnicalSignal,
    levels: FibonacciLevels,
) -> tuple[float, str]:
    if not levels.near_zone:
        return 0.0, f"Nivel más cercano {levels.nearest_ratio} a {levels.distance_pct:.2f}%; fuera de tolerancia."
    bullish = signal in (TechnicalSignal.BUY, TechnicalSignal.WATCH_BUY)
    bearish = signal in (TechnicalSignal.SELL, TechnicalSignal.WATCH_SELL)
    if bullish and levels.role == "SOPORTE":
        return 4.0, f"Señal alcista coincide con soporte Fibonacci {levels.nearest_ratio}."
    if bearish and levels.role == "RESISTENCIA":
        return -4.0, f"Señal bajista coincide con resistencia Fibonacci {levels.nearest_ratio}."
    if bullish and levels.role == "RESISTENCIA":
        return -2.0, "La señal alcista choca con resistencia Fibonacci cercana."
    if bearish and levels.role == "SOPORTE":
        return 2.0, "La señal bajista choca con soporte Fibonacci cercano."
    return 0.0, f"Precio en zona Fibonacci {levels.nearest_ratio}, sin señal direccional que bonificar."


def _detect_candlestick(frame: pd.DataFrame) -> tuple[CandlePattern, float, str]:
    previous, latest = frame.iloc[-2], frame.iloc[-1]
    open_price, close = float(latest["Open"]), float(latest["Close"])
    high, low = float(latest["High"]), float(latest["Low"])
    previous_open, previous_close = float(previous["Open"]), float(previous["Close"])
    candle_range = max(high - low, 1e-12)
    body = abs(close - open_price)
    effective_body = max(body, candle_range * 0.05)
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low

    bullish_engulfing = (
        previous_close < previous_open
        and close > open_price
        and open_price <= previous_close
        and close >= previous_open
    )
    bearish_engulfing = (
        previous_close > previous_open
        and close < open_price
        and open_price >= previous_close
        and close <= previous_open
    )
    if bullish_engulfing:
        return CandlePattern.BULLISH_ENGULFING, 4.0, "Envolvente alcista: el cuerpo actual cubre la vela bajista previa."
    if bearish_engulfing:
        return CandlePattern.BEARISH_ENGULFING, -4.0, "Envolvente bajista: el cuerpo actual cubre la vela alcista previa."
    if lower_wick >= 2 * effective_body and upper_wick <= effective_body and body / candle_range <= 0.45:
        return CandlePattern.HAMMER, 3.0, "Martillo: rechazo de precios inferiores en la última vela."
    if upper_wick >= 2 * effective_body and lower_wick <= effective_body and body / candle_range <= 0.45:
        return CandlePattern.SHOOTING_STAR, -3.0, "Estrella fugaz: rechazo de precios superiores en la última vela."
    if body / candle_range >= 0.75 and close > open_price:
        return CandlePattern.BULLISH_CONTINUATION, 2.0, "Vela alcista de cuerpo amplio; continuación básica."
    if body / candle_range >= 0.75 and close < open_price:
        return CandlePattern.BEARISH_CONTINUATION, -2.0, "Vela bajista de cuerpo amplio; continuación básica."
    return CandlePattern.NONE, 0.0, "Sin patrón de vela concluyente en la última vela cerrada."


def _pivot_levels(daily: pd.DataFrame, session_date) -> PivotLevels:
    eligible = daily[pd.Index(daily.index).date < session_date]
    source = eligible.iloc[-1] if not eligible.empty else daily.iloc[-2]
    source_timestamp = eligible.index[-1] if not eligible.empty else daily.index[-2]
    high, low, close = float(source["High"]), float(source["Low"]), float(source["Close"])
    pivot = (high + low + close) / 3
    return PivotLevels(pivot, 2 * pivot - high, pivot - (high - low), 2 * pivot - low, pivot + (high - low), pd.Timestamp(source_timestamp).date().isoformat())


def preliminary_probability(signal: TechnicalSignal, trend: DailyTrend, volume_confirmed: bool) -> tuple[float, str, tuple[str, ...]]:
    """Compatibilidad con el puntaje de Fase 1 y sus pruebas históricas."""
    warnings: list[str] = []
    if signal is TechnicalSignal.BUY:
        if trend is DailyTrend.BULLISH:
            probability_up, verdict = (70.0 if volume_confirmed else 65.0), "Confluencia alcista; vigilar entrada y riesgo."
        elif trend is DailyTrend.BEARISH:
            probability_up, verdict = 35.0, "Señal contra tendencia; no vale la pena entrar en Fase 1."
            warnings.append("La señal de compra contradice la tendencia diaria.")
        else:
            probability_up, verdict = (58.0 if volume_confirmed else 55.0), "Señal alcista sin confirmación clara del mapa diario."
    elif signal is TechnicalSignal.SELL:
        if trend is DailyTrend.BEARISH:
            probability_up, verdict = (30.0 if volume_confirmed else 35.0), "Confluencia bajista; priorizar protección de capital."
        elif trend is DailyTrend.BULLISH:
            probability_up, verdict = 65.0, "Señal bajista contra tendencia; no vale la pena entrar en Fase 1."
            warnings.append("La señal de venta contradice la tendencia diaria.")
        else:
            probability_up, verdict = (42.0 if volume_confirmed else 45.0), "Señal bajista sin confirmación clara del mapa diario."
    elif signal is TechnicalSignal.WATCH_BUY:
        probability_up, verdict = (56.0 if trend is DailyTrend.BULLISH else 52.0), "Sobreventa detectada; falta cruce de confirmación."
    elif signal is TechnicalSignal.WATCH_SELL:
        probability_up, verdict = (44.0 if trend is DailyTrend.BEARISH else 48.0), "Sobrecompra detectada; falta cruce de confirmación."
    else:
        probability_up = {DailyTrend.BULLISH: 54.0, DailyTrend.BEARISH: 46.0, DailyTrend.NEUTRAL: 50.0}[trend]
        verdict = "Sin señal intradía completa; esperar confirmación."
    if signal is not TechnicalSignal.NEUTRAL and not volume_confirmed:
        warnings.append("El volumen actual no supera su media previa de 20 velas.")
    return probability_up, verdict, tuple(warnings)


def _phase2_probability(
    signal: TechnicalSignal,
    trend: DailyTrend,
    volume_confirmed: bool,
    macd_5m: MomentumState,
    macd_daily: MomentumState,
    above_vwap: bool,
    adx: float,
    obv_state: ObvState,
    fibonacci_impact: float = 0.0,
    fibonacci_detail: str = "Sin evaluación Fibonacci.",
    ichimoku_5m_impact: float = 0.0,
    ichimoku_5m_detail: str = "Sin evaluación Ichimoku intradía.",
    ichimoku_daily_impact: float = 0.0,
    ichimoku_daily_detail: str = "Sin evaluación Ichimoku diaria.",
    candle_impact: float = 0.0,
    candle_detail: str = "Sin patrón de vela concluyente.",
    short_term_return_pct: float = 0.0,
    price_vs_vwap_pct: float = 0.0,
    macd_histogram_delta: float = 0.0,
    chart_pattern_impact: float = 0.0,
    chart_pattern_detail: str = "Sin patrón chartista confirmado.",
    stoch_overbought_extreme: bool = False,
) -> tuple[float, str, tuple[str, ...], tuple[ScoreComponent, ...], bool]:
    """Suma ponderada de Fases 1–3 previa al veto central de Fase 4."""
    components: list[ScoreComponent] = []
    warnings: list[str] = []

    def add(name: str, impact: float, detail: str) -> None:
        components.append(ScoreComponent(name, impact, detail))

    signal_impact = {TechnicalSignal.BUY: 12.0, TechnicalSignal.SELL: -12.0, TechnicalSignal.WATCH_BUY: 6.0, TechnicalSignal.WATCH_SELL: -6.0, TechnicalSignal.NEUTRAL: 0.0}[signal]
    add("Estocástico RSI + Bollinger", signal_impact, f"Señal {signal.value}.")
    trend_impact = {DailyTrend.BULLISH: 5.0, DailyTrend.BEARISH: -5.0, DailyTrend.NEUTRAL: 0.0}[trend]
    add("EMA diaria 9/21", trend_impact, f"Contexto {trend.value}.")
    macd_5m_impact = {MomentumState.BULLISH: 8.0, MomentumState.BEARISH: -8.0, MomentumState.NEUTRAL: 0.0}[macd_5m]
    add("MACD 5 min", macd_5m_impact, f"Impulso {macd_5m.value}.")
    histogram_impact = 0.0 if macd_histogram_delta == 0 else (2.5 if macd_histogram_delta > 0 else -2.5)
    add(
        "Aceleración MACD 5 min",
        histogram_impact,
        "El histograma acelera al alza." if histogram_impact > 0 else "El histograma acelera a la baja." if histogram_impact < 0 else "Histograma sin aceleración.",
    )
    macd_daily_impact = {MomentumState.BULLISH: 5.0, MomentumState.BEARISH: -5.0, MomentumState.NEUTRAL: 0.0}[macd_daily]
    add("MACD diario", macd_daily_impact, f"Impulso {macd_daily.value}.")
    vwap_impact = max(-6.0, min(6.0, price_vs_vwap_pct * 5.0))
    if abs(vwap_impact) < 1.0:
        vwap_impact = 1.0 if above_vwap else -1.0
    add(
        "VWAP de sesión",
        vwap_impact,
        f"Precio {price_vs_vwap_pct:+.2f}% respecto al VWAP.",
    )
    price_action_impact = max(-8.0, min(8.0, short_term_return_pct * 4.0))
    add(
        "Retorno reciente 5 min",
        price_action_impact,
        f"Cambio acumulado de {short_term_return_pct:+.2f}% en las últimas 6 velas cerradas.",
    )
    directional_signal = signal is not TechnicalSignal.NEUTRAL
    volume_impact = 0.0
    if volume_confirmed and directional_signal:
        volume_impact = 3.0 if signal in (TechnicalSignal.BUY, TechnicalSignal.WATCH_BUY) else -3.0
    add("Volumen relativo", volume_impact, "Confirma la dirección de la señal." if volume_impact else "Sin aporte direccional.")
    obv_impact = {ObvState.ACCUMULATION: 5.0, ObvState.DISTRIBUTION: -5.0, ObvState.CONFIRMING_UP: 3.0, ObvState.CONFIRMING_DOWN: -3.0, ObvState.NEUTRAL: 0.0}[obv_state]
    add("OBV", obv_impact, f"Lectura {obv_state.value}.")
    add("Fibonacci mensual", fibonacci_impact, fibonacci_detail)
    add("Ichimoku 5 min", ichimoku_5m_impact, ichimoku_5m_detail)
    add("Ichimoku diario", ichimoku_daily_impact, ichimoku_daily_detail)
    add("Vela japonesa 5 min", candle_impact, candle_detail)
    add("Patrones chartistas", chart_pattern_impact, chart_pattern_detail)

    probability_up = 50.0 + sum(item.impact_points for item in components)
    if adx < 20:
        reduced = 50.0 + (probability_up - 50.0) * 0.60
        add("ADX / mercado lateral", reduced - probability_up, f"ADX {adx:.1f}: reduce 40% el sesgo direccional.")
        probability_up = reduced
        warnings.append("Rango / Mercado Lateral: ADX menor a 20; fiabilidad reducida.")
    elif adx >= 25 and probability_up != 50:
        strength_impact = 2.0 if probability_up > 50 else -2.0
        add("ADX / tendencia fuerte", strength_impact, f"ADX {adx:.1f} confirma fuerza.")
        probability_up += strength_impact
    else:
        add("ADX", 0.0, f"ADX {adx:.1f}: fuerza intermedia.")

    # Se aplica después del ajuste ADX para que un mercado lateral no diluya
    # la penalización estricta requerida de 18 puntos porcentuales.
    if stoch_overbought_extreme:
        probability_up -= 18.0
        add(
            "Bloqueo Estocástico RSI extremo",
            -18.0,
            "%K o %D supera 80: resta 18 pp al sesgo alcista y bloquea un LONG inmediato.",
        )
        warnings.append(
            "Sobrecompra extrema 5 min: entrada LONG bloqueada hasta que %K cruce a la baja de 80; "
            "un rebote exige después una señal nueva desde sobreventa (<20)."
        )

    # Dirección y autorización de entrada son conceptos distintos. La ausencia
    # de gatillo mantiene operation_probability en cero en el veto central,
    # pero no recorta ni infla el sesgo técnico calculado.
    if signal is TechnicalSignal.NEUTRAL:
        add("Ausencia de disparador", 0.0, "La lectura direccional se conserva, pero no habilita una operación.")
    elif signal in (TechnicalSignal.WATCH_BUY, TechnicalSignal.WATCH_SELL):
        add("Señal aún no confirmada", 0.0, "La vigilancia conserva el sesgo, pero no habilita una operación.")

    daily_macd_conflict = (
        signal is TechnicalSignal.BUY and macd_daily is MomentumState.BEARISH
    ) or (
        signal is TechnicalSignal.SELL and macd_daily is MomentumState.BULLISH
    )
    if daily_macd_conflict:
        conflict_impact = -8.0 if signal is TechnicalSignal.BUY else 8.0
        probability_up += conflict_impact
        add(
            "Penalización MACD diario",
            conflict_impact,
            "El impulso diario contradice la señal intradía confirmada.",
        )
        warnings.append("MACD diario contrario: la señal pierde 8 puntos de fiabilidad.")

    rejected = (signal is TechnicalSignal.BUY and macd_5m is MomentumState.BEARISH) or (signal is TechnicalSignal.SELL and macd_5m is MomentumState.BULLISH)
    if rejected:
        before_rejection = probability_up
        probability_up = min(probability_up, 42.0) if signal is TechnicalSignal.BUY else max(probability_up, 58.0)
        add("Veto MACD intradía", probability_up - before_rejection, "El MACD contradice la señal confirmada; entrada rechazada.")
        warnings.append("Señal rechazada: el MACD de 5 minutos contradice al Estocástico RSI.")

    # Regla simétrica de actualidad: MACD y precio reciente alineados dominan
    # de inmediato el sesgo corto, aunque el contexto diario sea contrario.
    short_bearish = macd_5m is MomentumState.BEARISH and short_term_return_pct < 0
    short_bullish = macd_5m is MomentumState.BULLISH and short_term_return_pct > 0
    if short_bearish and probability_up >= 50.0:
        cap = 45.0 if (not above_vwap or candle_impact < 0) else 48.0
        add(
            "Dominio técnico corto plazo",
            cap - probability_up,
            "MACD bajista y retroceso reciente: el contexto lento no puede mantener un sesgo alcista.",
        )
        probability_up = cap
    elif short_bullish and probability_up <= 50.0:
        floor = 55.0 if (above_vwap or candle_impact > 0) else 52.0
        add(
            "Dominio técnico corto plazo",
            floor - probability_up,
            "MACD alcista y avance reciente: el contexto lento no puede mantener un sesgo bajista.",
        )
        probability_up = floor

    bounded = min(85.0, max(15.0, probability_up))
    if bounded != probability_up:
        add("Límite prudencial", bounded - probability_up, "Rango heurístico limitado a 15–85%.")
    probability_up = round(bounded, 1)
    if signal is TechnicalSignal.NEUTRAL:
        direction = "alcista" if probability_up > 52 else "bajista" if probability_up < 48 else "neutral"
        verdict = f"Sesgo {direction} informativo; sin gatillo operativo confirmado."
    elif rejected:
        verdict = "Señal invalidada por MACD; no vale la pena entrar sin nueva confirmación."
    elif adx < 20:
        verdict = "Mercado lateral: esperar ruptura y confirmación antes de considerar entrada."
    elif probability_up >= 65:
        verdict = "Confluencia alcista previa al veto macro."
    elif probability_up <= 35:
        verdict = "Confluencia bajista previa al veto macro."
    else:
        verdict = "Confluencia insuficiente; esperar una señal más limpia."
    return probability_up, verdict, tuple(warnings), tuple(components), rejected


def _technical_level(signal: TechnicalSignal, price: float, lower_band: float, upper_band: float, vwap: float, pivots: PivotLevels, fibonacci: FibonacciLevels) -> float:
    if signal in (TechnicalSignal.BUY, TechnicalSignal.WATCH_BUY):
        supports = [level for level in (lower_band, vwap, pivots.s1, pivots.s2, fibonacci.nearest_level) if level <= price]
        return max(supports) if supports else lower_band
    if signal in (TechnicalSignal.SELL, TechnicalSignal.WATCH_SELL):
        resistances = [level for level in (upper_band, vwap, pivots.r1, pivots.r2, fibonacci.nearest_level) if level >= price]
        return min(resistances) if resistances else upper_band
    return pivots.pivot


def _monthly_execution_policy(
    planned_direction: str,
    monthly_trend: MacroTrend,
) -> tuple[bool, float, float]:
    """Devuelve (short táctico, exposición relativa, volumen mínimo).

    La política es transversal a cualquier emisora: solo depende del sentido
    de la operación y del régimen mensual calculado.
    """

    tactical_short = (
        planned_direction == "SHORT"
        and monthly_trend is MacroTrend.STRONG_BULLISH
    )
    return (
        tactical_short,
        0.50 if tactical_short else 1.0,
        1.20 if tactical_short else 1.0,
    )


def analyze_probability(
    symbol: str,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    atr_stop_multiple: float = 2.25,
) -> ProbabilityAnalysis:
    if len(intraday) < 40:
        raise ValueError("Se requieren al menos 40 velas intradía utilizables.")
    if len(daily) < 35:
        raise ValueError("Se requieren al menos 35 velas diarias utilizables.")
    intraday_indicators = add_intraday_indicators(intraday).dropna(subset=["StochRSI_K", "StochRSI_D", "BB_upper", "BB_middle", "BB_lower", "Volume_MA20", "MACD", "MACD_signal", "MACD_histogram", "VWAP", "ADX14", "OBV", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B"])
    daily_indicators = add_daily_indicators(daily).dropna(subset=["EMA9", "EMA21", "MACD", "MACD_signal", "MACD_histogram", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B"])
    completed_intraday = intraday_indicators[intraday_indicators["Volume"] > 0]
    if len(completed_intraday) < 13 or daily_indicators.empty:
        raise ValueError("No hay suficientes velas para completar los indicadores.")
    intraday_indicators = intraday_indicators.loc[: completed_intraday.index[-1]].copy()
    previous, latest, daily_latest = intraday_indicators.iloc[-2], intraday_indicators.iloc[-1], daily_indicators.iloc[-1]
    k_value, d_value = float(latest["StochRSI_K"]), float(latest["StochRSI_D"])
    previous_k, previous_d = float(previous["StochRSI_K"]), float(previous["StochRSI_D"])
    price, lower_band, upper_band = float(latest["Close"]), float(latest["BB_lower"]), float(latest["BB_upper"])
    crossed_up, crossed_down = previous_k <= previous_d and k_value > d_value, previous_k >= previous_d and k_value < d_value
    stoch_overbought_extreme = k_value > 80 or d_value > 80
    stoch_oversold_extreme = k_value < 20 or d_value < 20
    oversold, overbought = min(previous_k, k_value) < 20 and price <= lower_band, max(previous_k, k_value) > 80 and price >= upper_band
    if oversold and crossed_up:
        signal = TechnicalSignal.BUY
    elif overbought and crossed_down:
        signal = TechnicalSignal.SELL
    elif oversold:
        signal = TechnicalSignal.WATCH_BUY
    elif overbought:
        signal = TechnicalSignal.WATCH_SELL
    else:
        signal = TechnicalSignal.NEUTRAL
    ema9, ema21, daily_close = float(daily_latest["EMA9"]), float(daily_latest["EMA21"]), float(daily_latest["Close"])
    ema50, ema200 = float(daily_latest["EMA50"]), float(daily_latest["EMA200"])
    trend = DailyTrend.BULLISH if ema9 > ema21 and daily_close >= ema9 else DailyTrend.BEARISH if ema9 < ema21 and daily_close <= ema9 else DailyTrend.NEUTRAL
    volume_average = float(latest["Volume_MA20"])
    volume_ratio = float(latest["Volume"]) / volume_average if volume_average > 0 else 0.0
    volume_confirmed = volume_ratio > 1.0
    macd_state_5m = _momentum_state(float(latest["MACD"]), float(latest["MACD_signal"]))
    macd_state_daily = _momentum_state(float(daily_latest["MACD"]), float(daily_latest["MACD_signal"]))
    vwap = float(latest["VWAP"])
    above_vwap = price >= vwap
    price_vs_vwap_pct = (price / vwap - 1) * 100 if vwap else 0.0
    short_reference = float(intraday_indicators["Close"].iloc[-7])
    short_term_return_pct = (price / short_reference - 1) * 100 if short_reference else 0.0
    macd_histogram_delta = float(latest["MACD_histogram"] - previous["MACD_histogram"])
    adx = float(latest["ADX14"])
    obv_state, obv_price_change_pct = _obv_divergence(intraday_indicators)
    latest_timestamp = pd.Timestamp(intraday_indicators.index[-1])
    completed_daily_source = daily[pd.Index(daily.index).date < latest_timestamp.date()]
    hourly_indicators = add_context_indicators(
        resample_ohlcv(intraday.loc[: latest_timestamp], "60min")
    )
    weekly_indicators = add_context_indicators(
        resample_ohlcv(completed_daily_source, "W-FRI")
    )
    monthly_indicators = add_context_indicators(
        resample_ohlcv(completed_daily_source, "ME")
    )
    weekly_trend = classify_macro_trend(weekly_indicators)
    monthly_trend = classify_macro_trend(monthly_indicators)
    weekly_structure = weekly_indicators.tail(20)
    weekly_support = float(weekly_structure["Low"].min())
    weekly_resistance = float(weekly_structure["High"].max())
    fibonacci = _fibonacci_levels(daily, latest_timestamp.date(), price)
    annual_fibonacci = _fibonacci_levels(
        daily,
        latest_timestamp.date(),
        price,
        tolerance_pct=0.50,
        sessions=252,
    )
    liquidity_zones = calculate_liquidity_zones(completed_daily_source)
    ichimoku_5m, ichimoku_5m_impact, ichimoku_5m_detail, cloud_upper_5m, cloud_lower_5m = _cloud_assessment(latest, 4.0)
    ichimoku_daily, ichimoku_daily_impact, ichimoku_daily_detail, cloud_upper_daily, cloud_lower_daily = _cloud_assessment(daily_latest, 5.0)
    candle_pattern, candle_impact, candle_detail = _detect_candlestick(intraday_indicators)
    chart_patterns = scan_multi_timeframe_patterns(
        intraday.loc[:latest_timestamp],
        completed_daily_source,
    )
    pivots = _pivot_levels(daily, latest_timestamp.date())
    atr_5m = float(latest["ATR14"])
    support_candidates = [pivots.s1, pivots.s2, weekly_support, lower_band]
    resistance_candidates = [pivots.r1, pivots.r2, weekly_resistance, upper_band]
    if vwap <= price:
        support_candidates.append(vwap)
    if fibonacci.role == "SOPORTE":
        support_candidates.append(fibonacci.nearest_level)
    elif fibonacci.role == "RESISTENCIA":
        resistance_candidates.append(fibonacci.nearest_level)
    valid_supports = sorted(
        level for level in support_candidates if 0 < level <= price + atr_5m
    )
    nearest_support = min(
        valid_supports,
        key=lambda level: abs(price - level),
        default=lower_band,
    )
    support_tolerance = max(0.50 * atr_5m, price * 0.002)
    support_interaction = (
        abs(price - nearest_support) <= support_tolerance
        or float(latest["Low"]) <= nearest_support <= float(latest["High"])
    )
    bullish_reversal_candle = candle_pattern in (
        CandlePattern.HAMMER,
        CandlePattern.BULLISH_ENGULFING,
    )
    bullish_reversal_pattern = any(
        pattern.timeframe == "5m"
        and pattern.valid
        and pattern.pattern_type
        in (ChartPatternType.DOUBLE_BOTTOM, ChartPatternType.TRIPLE_BOTTOM)
        for pattern in chart_patterns
    )
    rebound_watch_active = stoch_oversold_extreme and (
        support_interaction or bullish_reversal_candle or bullish_reversal_pattern
    )
    if rebound_watch_active:
        signal = TechnicalSignal.BUY if crossed_up else TechnicalSignal.WATCH_BUY
    elif stoch_oversold_extreme and signal is not TechnicalSignal.SELL:
        signal = TechnicalSignal.NEUTRAL

    fibonacci_impact, fibonacci_detail = _fibonacci_score(signal, fibonacci)
    pattern_influence = evaluate_pattern_influence(
        chart_patterns,
        signal=signal.value,
    )
    probability_up, verdict, scoring_warnings, breakdown, rejected = _phase2_probability(
        signal,
        trend,
        volume_confirmed,
        macd_state_5m,
        macd_state_daily,
        above_vwap,
        adx,
        obv_state,
        fibonacci_impact=fibonacci_impact,
        fibonacci_detail=fibonacci_detail,
        ichimoku_5m_impact=ichimoku_5m_impact,
        ichimoku_5m_detail=ichimoku_5m_detail,
        ichimoku_daily_impact=ichimoku_daily_impact,
        ichimoku_daily_detail=ichimoku_daily_detail,
        candle_impact=candle_impact,
        candle_detail=candle_detail,
        short_term_return_pct=short_term_return_pct,
        price_vs_vwap_pct=price_vs_vwap_pct,
        macd_histogram_delta=macd_histogram_delta,
        chart_pattern_impact=pattern_influence.impact_points,
        chart_pattern_detail=pattern_influence.detail,
        stoch_overbought_extreme=stoch_overbought_extreme,
    )
    daily_ema_distance_pct = abs(price / ema21 - 1) * 100 if ema21 else 0.0
    overextended_unconfirmed = daily_ema_distance_pct >= 8.0 and not volume_confirmed
    veto_reasons = []
    if rejected:
        veto_reasons.append("MACD de 5 minutos contradice el gatillo operativo.")
    if pattern_influence.veto:
        veto_reasons.append(pattern_influence.veto_reason)
    conditional_block_reason = (
        "Estocástico RSI 5 min en sobrecompra extrema (>80): LONG inmediato bloqueado."
        if stoch_overbought_extreme
        else ""
    )
    planned_direction = (
        "LONG"
        if signal in (TechnicalSignal.BUY, TechnicalSignal.WATCH_BUY)
        or rebound_watch_active
        else "SHORT"
        if signal in (TechnicalSignal.SELL, TechnicalSignal.WATCH_SELL)
        or probability_up < 50.0
        else "LONG"
    )
    tactical_short, exposure_factor, required_volume_ratio = (
        _monthly_execution_policy(planned_direction, monthly_trend)
    )
    ema_regime_aligned = (
        ema9 > ema21
        if planned_direction == "LONG"
        else ema9 < ema21
    )
    macd_regime_aligned = (
        macd_state_5m is MomentumState.BULLISH
        if planned_direction == "LONG"
        else macd_state_5m is MomentumState.BEARISH
    )
    trend_regime_confirmed = adx > 25 and ema_regime_aligned and macd_regime_aligned
    if adx < 20:
        exposure_factor = min(exposure_factor, 0.25)
    elif not trend_regime_confirmed:
        exposure_factor = min(exposure_factor, 0.50)
    valid_resistances = sorted(
        level for level in resistance_candidates if level >= price - atr_5m
    )
    nearest_resistance = min(
        valid_resistances,
        key=lambda level: abs(price - level),
        default=upper_band,
    )
    if planned_direction == "LONG":
        activation_trigger = (
            f"Activar LONG solo si %K cruza sobre %D desde <20, el cierre 5m mantiene "
            f"el soporte ${nearest_support:.2f}, MACD 5m deja de ser bajista y volumen "
            f"> SMA20 (ratio > {required_volume_ratio:.2f}x), con ADX >25 y EMA9 > EMA21."
        )
        activation_trigger_met = (
            signal is TechnicalSignal.BUY
            and rebound_watch_active
            and macd_state_5m is not MomentumState.BEARISH
            and volume_ratio > required_volume_ratio
            and trend_regime_confirmed
        )
    else:
        activation_trigger = (
            f"Activar SHORT solo si un cierre 5m rompe el soporte ${nearest_support:.2f}, "
            f"MACD 5m permanece bajista y volumen > SMA20 "
            f"(ratio > {required_volume_ratio:.2f}x), con ADX >25 y EMA9 < EMA21."
        )
        activation_trigger_met = (
            signal is TechnicalSignal.SELL
            and price < nearest_support
            and macd_state_5m is MomentumState.BEARISH
            and volume_ratio > required_volume_ratio
            and trend_regime_confirmed
        )
    decision = strict_confluence_gate(
        probability_up,
        signal.value,
        weekly_trend,
        monthly_trend,
        adx,
        overextended_unconfirmed,
        preexisting_veto_reason=" ".join(veto_reasons),
        conditional_block_reason=conditional_block_reason,
        activation_trigger=activation_trigger,
        activation_trigger_met=activation_trigger_met,
    )
    breakdown_list = list(breakdown)
    if decision.probability_up != probability_up:
        breakdown_list.append(
            ScoreComponent(
                "Veto central Fase 4",
                decision.probability_up - probability_up,
                " ".join(decision.reasons),
            )
        )
    breakdown = tuple(breakdown_list)
    probability_up = decision.probability_up
    verdict = decision.scenario
    horizon_projections = calculate_horizon_projections(
        probability_up=probability_up,
        risk_veto=decision.risk_veto,
        last_price=price,
        intraday=intraday_indicators,
        hourly=hourly_indicators,
        daily=daily_indicators,
        weekly=weekly_indicators,
        monthly=monthly_indicators,
    )
    confirmed_double_bottoms = [
        pattern
        for pattern in chart_patterns
        if pattern.timeframe == "5m"
        and pattern.pattern_type is ChartPatternType.DOUBLE_BOTTOM
        and pattern.valid
        and pattern.target_price > price
    ]
    active_double_bottom = max(
        confirmed_double_bottoms,
        key=lambda pattern: (pattern.confidence, pattern.detected_at),
        default=None,
    )
    execution_levels = calculate_execution_levels(
        last_price=price,
        projections=horizon_projections,
        additional_supports=tuple(support_candidates),
        additional_resistances=tuple(resistance_candidates),
        probability_up=probability_up,
        signal="BUY" if planned_direction == "LONG" else "SELL",
        intraday_atr=atr_5m,
        atr_stop_multiple=atr_stop_multiple,
        confirmed_pattern_target=(
            active_double_bottom.target_price if active_double_bottom else None
        ),
        confirmed_pattern_label=(
            f"Doble suelo 5m · cuello ${active_double_bottom.neckline:.2f}"
            if active_double_bottom
            else ""
        ),
    )
    as_of = latest_timestamp.to_pydatetime()
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    daily_projection = calculate_15_day_projection(
        last_price=price,
        daily=daily_indicators,
        probability_up=probability_up,
        risk_veto=decision.risk_veto,
        as_of=as_of,
    )
    suggested_level = _technical_level(signal, price, lower_band, upper_band, vwap, pivots, fibonacci)
    hot_patterns = [
        pattern
        for pattern in chart_patterns
        if pattern.timeframe == "5m" and pattern.valid
        and (
            abs(price - pattern.neckline) <= max(0.75 * atr_5m, price * 0.001)
            or (
                pattern.direction is PatternDirection.BULLISH
                and price >= pattern.neckline
            )
            or (
                pattern.direction is PatternDirection.BEARISH
                and price <= pattern.neckline
            )
        )
    ]
    neckline_heat_warning = ""
    if stoch_overbought_extreme and hot_patterns:
        hottest = min(hot_patterns, key=lambda pattern: abs(price - pattern.neckline))
        neckline_heat_warning = (
            f"ADVERTENCIA: precio en/tras la línea de cuello ${hottest.neckline:.2f} "
            f"del patrón {hottest.label} con Estocástico RSI quemado "
            f"(%K {k_value:.1f} / %D {d_value:.1f}); no perseguir la ruptura."
        )
    execution_plan_conditional = (
        stoch_overbought_extreme
        or decision.risk_veto
        or signal not in (TechnicalSignal.BUY, TechnicalSignal.SELL)
        or decision.operation_probability < 65
        or not activation_trigger_met
    )
    execution_plan_label = (
        f"PLAN CONDICIONAL - {activation_trigger}"
        if execution_plan_conditional
        else "PLAN DE EJECUCIÓN VALIDADO"
    )
    observations = (
        f"Stoch RSI %K {k_value:.1f} / %D {d_value:.1f}; señal {signal.value}.",
        f"Vigilancia de rebote: {'ACTIVA' if rebound_watch_active else 'INACTIVA'}; soporte más cercano ${nearest_support:.2f}.",
        f"Detonante {'CUMPLIDO' if activation_trigger_met else 'PENDIENTE'}: {activation_trigger}",
        f"Exposición relativa {exposure_factor:.2f}x; SHORT táctico mensual: {'SÍ' if tactical_short else 'NO'}.",
        f"Régimen central: {'TENDENCIA CONFIRMADA' if trend_regime_confirmed else 'LATERAL' if adx < 20 else 'TRANSICIÓN'}; ADX {adx:.1f}, EMA y MACD {'alineados' if ema_regime_aligned and macd_regime_aligned else 'sin alineación completa'}.",
        f"MACD 5 min {float(latest['MACD']):.3f} vs señal {float(latest['MACD_signal']):.3f} ({macd_state_5m.value}).",
        f"MACD diario {float(daily_latest['MACD']):.3f} vs señal {float(daily_latest['MACD_signal']):.3f} ({macd_state_daily.value}).",
        f"VWAP {vwap:.2f}; precio {price_vs_vwap_pct:+.2f}% respecto a la sesión.",
        f"Retorno de las últimas 6 velas de 5 min: {short_term_return_pct:+.2f}%.",
        f"ADX 14 en 5 min: {adx:.1f} ({'RANGO / LATERAL' if adx < 20 else 'TENDENCIA'}).",
        f"OBV {obv_state.value}; cambio de precio de {obv_price_change_pct:+.2f}% en 12 velas.",
        f"Volumen {volume_ratio:.2f}× su media previa de 20 velas.",
        f"EMA9 {ema9:.2f} frente a EMA21 {ema21:.2f}.",
        f"Fibonacci {fibonacci.nearest_ratio} en {fibonacci.nearest_level:.2f}; distancia {fibonacci.distance_pct:.2f}% ({fibonacci.role}).",
        f"Ichimoku 5 min {ichimoku_5m.value}; Tenkan {float(latest['Ichimoku_Tenkan']):.2f} / Kijun {float(latest['Ichimoku_Kijun']):.2f}.",
        f"Ichimoku diario {ichimoku_daily.value}; Tenkan {float(daily_latest['Ichimoku_Tenkan']):.2f} / Kijun {float(daily_latest['Ichimoku_Kijun']):.2f}.",
        f"Vela 5 min {candle_pattern.value}: {candle_detail}",
        (
            f"Patrones chartistas: {pattern_influence.detail}"
            if chart_patterns
            else "Patrones chartistas: no se detectaron estructuras recientes utilizables."
        ),
        f"Tendencia semanal {weekly_trend.value}; tendencia mensual {monthly_trend.value}.",
        f"Estructura semanal: soporte {weekly_support:.2f} / resistencia {weekly_resistance:.2f}.",
        f"Distancia a EMA21 diaria {daily_ema_distance_pct:.2f}%; sobreextensión sin confirmar: {'SÍ' if overextended_unconfirmed else 'NO'}.",
    )
    warnings = list(scoring_warnings)
    warnings.extend(decision.reasons)
    if neckline_heat_warning:
        warnings.append(neckline_heat_warning)
    if datetime.now(timezone.utc) - as_of.astimezone(timezone.utc) > timedelta(hours=24):
        warnings.append("La última vela corresponde a una sesión anterior.")
    warnings.append("Puntaje heurístico no calibrado; no es recomendación ni garantía de resultado.")
    analysis = ProbabilityAnalysis(
        symbol=symbol.upper(), as_of=as_of, last_price=price, probability_up=probability_up, probability_down=decision.probability_down, signal=signal, daily_trend=trend,
        volume_confirmed=volume_confirmed, volume_ratio=volume_ratio, stochastic_k=k_value, stochastic_d=d_value, atr_5m=atr_5m, stoch_overbought_extreme=stoch_overbought_extreme, stoch_oversold_extreme=stoch_oversold_extreme, rebound_watch_active=rebound_watch_active, support_interaction=support_interaction, nearest_support=nearest_support, long_entry_blocked=stoch_overbought_extreme, execution_plan_conditional=execution_plan_conditional, execution_plan_label=execution_plan_label, activation_trigger=activation_trigger, activation_trigger_met=activation_trigger_met, tactical_short=tactical_short, exposure_factor=exposure_factor, neckline_heat_warning=neckline_heat_warning, bollinger_upper=upper_band, bollinger_middle=float(latest["BB_middle"]), bollinger_lower=lower_band,
        ema9=ema9, ema21=ema21, ema50=ema50, ema200=ema200, macd_5m=float(latest["MACD"]), macd_signal_5m=float(latest["MACD_signal"]), macd_histogram_5m=float(latest["MACD_histogram"]),
        macd_daily=float(daily_latest["MACD"]), macd_signal_daily=float(daily_latest["MACD_signal"]), macd_histogram_daily=float(daily_latest["MACD_histogram"]), macd_state_5m=macd_state_5m, macd_state_daily=macd_state_daily,
        vwap=vwap, price_vs_vwap_pct=price_vs_vwap_pct, above_vwap=above_vwap, adx=adx, range_market=adx < 20, obv=float(latest["OBV"]), obv_state=obv_state, obv_price_change_pct=obv_price_change_pct,
        fibonacci=fibonacci, ichimoku_5m=ichimoku_5m, ichimoku_daily=ichimoku_daily, tenkan_5m=float(latest["Ichimoku_Tenkan"]), kijun_5m=float(latest["Ichimoku_Kijun"]), cloud_upper_5m=cloud_upper_5m, cloud_lower_5m=cloud_lower_5m,
        tenkan_daily=float(daily_latest["Ichimoku_Tenkan"]), kijun_daily=float(daily_latest["Ichimoku_Kijun"]), cloud_upper_daily=cloud_upper_daily, cloud_lower_daily=cloud_lower_daily, candle_pattern=candle_pattern, candle_detail=candle_detail,
        weekly_trend=weekly_trend, monthly_trend=monthly_trend, weekly_support=weekly_support, weekly_resistance=weekly_resistance, annual_fibonacci=annual_fibonacci, liquidity_zones=liquidity_zones, operation_probability=decision.operation_probability, risk_veto=decision.risk_veto, risk_alert=decision.alert, risk_reasons=decision.reasons, scenario=decision.scenario, overextended_unconfirmed=overextended_unconfirmed,
        signal_rejected=rejected or decision.risk_veto, score_breakdown=breakdown, pivots=pivots, suggested_level=suggested_level, verdict=verdict, observations=observations, warnings=tuple(warnings), intraday_indicators=intraday_indicators, hourly_indicators=hourly_indicators, daily_indicators=daily_indicators, weekly_indicators=weekly_indicators, monthly_indicators=monthly_indicators, chart_patterns=chart_patterns, chart_pattern_impact=pattern_influence.impact_points, chart_pattern_veto=pattern_influence.veto, horizon_projections=horizon_projections, execution_levels=execution_levels, daily_projection=daily_projection, raw_probability_up=probability_up, market_regime=("TENDENCIA CONFIRMADA" if trend_regime_confirmed else "RANGO / VETO" if adx < 20 else "TRANSICIÓN"), position_size_policy=("NORMAL" if trend_regime_confirmed else "REDUCIDA 25%" if adx < 20 else "REDUCIDA 50% / ESPERAR"),
    )
    validate_probability_analysis(analysis)
    return analysis
