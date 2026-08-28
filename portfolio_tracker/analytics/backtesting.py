"""Backtesting fuera de muestra para los setups estructurales Fase 4.

El módulo no descarga datos ni accede a SQLite. Recibe OHLCV, calcula señales
causales, calibra probabilidades solo con el tramo de entrenamiento y congela
esa calibración antes de evaluar validación.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Mapping

import pandas as pd

from portfolio_tracker.analytics.multi_timeframe import (
    add_context_indicators,
    resample_ohlcv,
)
from portfolio_tracker.analytics.technical_probability import add_intraday_indicators


ENGINE_VERSION = "oos-phase4-v1"
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    training_fraction: float = 0.70
    stop_atr_multiple: float = 2.25
    reward_risk: float = 2.0
    holding_sessions: int = 10
    risk_per_trade_pct: float = 1.0
    max_position_exposure_pct: float = 50.0
    commission_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 5.0
    minimum_probability: float = 0.55
    minimum_reward_risk: float = 1.50
    minimum_adx: float = 20.0
    minimum_validation_trades: int = 5
    maximum_drawdown_pct: float = 15.0
    minimum_profit_factor: float = 1.10

    def __post_init__(self) -> None:
        if not 0.55 <= self.training_fraction <= 0.85:
            raise ValueError("El tramo de entrenamiento debe estar entre 55% y 85%.")
        if not 2.0 <= self.stop_atr_multiple <= 2.5:
            raise ValueError("El stop debe estar entre 2.0 y 2.5 ATR.")
        if self.reward_risk < self.minimum_reward_risk:
            raise ValueError("El objetivo no alcanza el riesgo/beneficio mínimo.")
        if self.holding_sessions < 1:
            raise ValueError("La duración máxima debe ser positiva.")
        if self.risk_per_trade_pct <= 0 or self.max_position_exposure_pct <= 0:
            raise ValueError("El riesgo y la exposición deben ser positivos.")
        if not 0.50 <= self.minimum_probability <= 0.80:
            raise ValueError("La probabilidad mínima debe estar entre 50% y 80%.")


@dataclass(frozen=True, slots=True)
class SetupCandidate:
    signal_position: int
    signal_date: str
    side: str
    score: float
    strength_bucket: int
    atr: float
    volume_ratio: float
    adx: float
    monthly_regime: str
    exposure_factor: float
    trigger: str


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    symbol: str
    split: str
    signal_date: str
    entry_date: str
    exit_date: str
    side: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    quantity: float
    probability: float
    score: float
    reward_risk: float
    gross_pnl_usd: float
    costs_usd: float
    net_pnl_usd: float
    net_r_multiple: float
    outcome: str
    exit_reason: str


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    setups: int
    trades: int
    rejected: int
    wins: int
    losses: int
    win_rate: float
    win_rate_lower_bound: float
    profit_factor: float | None
    maximum_drawdown_pct: float
    net_return_pct: float
    gross_profit_usd: float
    gross_loss_usd: float
    costs_usd: float
    brier_score: float | None


@dataclass(frozen=True, slots=True)
class SymbolBacktestResult:
    symbol: str
    data_start: str
    data_end: str
    split_date: str
    dataset_sha256: str
    training: PerformanceMetrics
    validation: PerformanceMetrics
    decision: str
    decision_reasons: tuple[str, ...]
    rejected_reasons: tuple[tuple[str, int], ...]
    training_trades: tuple[SimulatedTrade, ...]
    validation_trades: tuple[SimulatedTrade, ...]
    validation_equity_curve: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class BacktestBatchResult:
    engine_version: str
    generated_at: str
    starting_capital_usd: float
    config: BacktestConfig
    results: tuple[SymbolBacktestResult, ...]
    aggregate: PerformanceMetrics
    aggregate_decision: str
    dataset_sha256: str


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("El histórico OHLCV está vacío.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Faltan columnas OHLCV: " + ", ".join(missing) + ".")
    result = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["Open", "High", "Low", "Close"])
    result["Volume"] = result["Volume"].fillna(0.0)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if len(result) < 300:
        raise ValueError("Se requieren al menos 300 sesiones para separar entrenamiento y validación.")
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    return result


def dataset_sha256(symbol: str, frame: pd.DataFrame) -> str:
    normalized = _normalize_frame(frame)
    digest = hashlib.sha256(symbol.upper().encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(normalized, index=True).values.tobytes())
    return digest.hexdigest()


def _macro_bias(frame: pd.DataFrame, rule: str, target_index: pd.DatetimeIndex) -> pd.Series:
    context = add_context_indicators(resample_ohlcv(frame, rule))
    usable = context.dropna(subset=["EMA21", "EMA50", "MACD", "MACD_signal"])
    bias = pd.Series(0.0, index=context.index)
    if not usable.empty:
        bullish = (
            (usable["Close"] > usable["EMA21"])
            & (usable["EMA21"] > usable["EMA50"])
            & (usable["MACD"] > usable["MACD_signal"])
        )
        bearish = (
            (usable["Close"] < usable["EMA21"])
            & (usable["EMA21"] < usable["EMA50"])
            & (usable["MACD"] < usable["MACD_signal"])
        )
        bias.loc[usable.index[bullish]] = 2.0
        bias.loc[usable.index[bearish]] = -2.0
        moderate_bull = (usable["Close"] > usable["EMA21"]) & ~bullish
        moderate_bear = (usable["Close"] < usable["EMA21"]) & ~bearish
        bias.loc[usable.index[moderate_bull]] = 1.0
        bias.loc[usable.index[moderate_bear]] = -1.0
    # Las etiquetas semanales/mensuales solo se vuelven visibles al cerrar su periodo.
    return bias.reindex(target_index, method="ffill").fillna(0.0)


def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_frame(frame)
    result = add_intraday_indicators(source)
    for span in (9, 21, 50, 200):
        result[f"EMA{span}"] = result["Close"].ewm(
            span=span, adjust=False, min_periods=span
        ).mean()
    result["WeeklyBias"] = _macro_bias(source, "W-FRI", result.index)
    result["MonthlyBias"] = _macro_bias(source, "ME", result.index)
    result["PriorHigh20"] = result["High"].shift(1).rolling(20, min_periods=20).max()
    result["PriorLow20"] = result["Low"].shift(1).rolling(20, min_periods=20).min()
    return result.dropna(
        subset=[
            "StochRSI_K", "StochRSI_D", "MACD", "MACD_signal",
            "MACD_histogram", "ATR14", "ADX14", "Volume_MA20",
            "EMA21", "PriorHigh20", "PriorLow20",
        ]
    )


def _generate_candidates(
    indicators: pd.DataFrame, config: BacktestConfig
) -> tuple[list[SetupCandidate], Counter[str]]:
    candidates: list[SetupCandidate] = []
    rejected: Counter[str] = Counter()
    for position in range(1, len(indicators) - config.holding_sessions - 1):
        previous = indicators.iloc[position - 1]
        row = indicators.iloc[position]
        volume_average = float(row["Volume_MA20"])
        volume_ratio = float(row["Volume"]) / volume_average if volume_average > 0 else 0.0
        k, d = float(row["StochRSI_K"]), float(row["StochRSI_D"])
        previous_k, previous_d = float(previous["StochRSI_K"]), float(previous["StochRSI_D"])
        bullish_cross = previous_k <= previous_d and k > d and min(previous_k, k) < 25
        bearish_cross = previous_k >= previous_d and k < d and max(previous_k, k) > 75
        bullish_breakout = float(row["Close"]) > float(row["PriorHigh20"]) and volume_ratio > 1.20
        bearish_breakout = float(row["Close"]) < float(row["PriorLow20"]) and volume_ratio > 1.20
        long_trigger = bullish_cross or bullish_breakout
        short_trigger = bearish_cross or bearish_breakout
        if long_trigger == short_trigger:
            continue
        side = "LONG" if long_trigger else "SHORT"
        if volume_ratio <= 1.0:
            rejected["Volumen no supera su media de 20 sesiones"] += 1
            continue
        adx = float(row["ADX14"])
        if adx < config.minimum_adx:
            rejected["ADX menor a 20: mercado lateral"] += 1
            continue

        monthly_bias = float(row["MonthlyBias"])
        exposure_factor = 1.0
        if side == "SHORT" and monthly_bias >= 2.0:
            if volume_ratio <= 1.20:
                rejected["SHORT contra régimen mensual alcista sin volumen 1.20x"] += 1
                continue
            exposure_factor = 0.50
        elif side == "LONG" and monthly_bias <= -2.0:
            if volume_ratio <= 1.20:
                rejected["LONG contra régimen mensual bajista sin volumen 1.20x"] += 1
                continue
            exposure_factor = 0.50

        direction = 1.0 if side == "LONG" else -1.0
        score = 3.0 * direction
        score += direction if float(row["MACD"]) > float(row["MACD_signal"]) else -direction
        score += direction if float(row["Close"]) > float(row["EMA21"]) else -direction
        score += 0.75 * float(row["WeeklyBias"])
        score += 1.00 * monthly_bias
        score += 0.5 * direction if volume_ratio > 1.20 else 0.0
        # Penalización simétrica: una señal nunca recibe sesgo alcista por defecto.
        directional_score = score * direction
        if directional_score < 2.0:
            rejected["Confluencia direccional insuficiente"] += 1
            continue
        bucket = max(2, min(8, int(math.floor(directional_score))))
        trigger = (
            "Cruce alcista de StochRSI desde sobreventa con volumen confirmado"
            if bullish_cross
            else "Ruptura alcista de máximo 20 sesiones con volumen >1.20x"
            if bullish_breakout
            else "Cruce bajista de StochRSI desde sobrecompra con volumen confirmado"
            if bearish_cross
            else "Ruptura bajista de mínimo 20 sesiones con volumen >1.20x"
        )
        monthly_regime = (
            "STRONG_BULLISH" if monthly_bias >= 2 else
            "STRONG_BEARISH" if monthly_bias <= -2 else
            "BULLISH" if monthly_bias > 0 else
            "BEARISH" if monthly_bias < 0 else "NEUTRAL"
        )
        candidates.append(
            SetupCandidate(
                signal_position=position,
                signal_date=pd.Timestamp(indicators.index[position]).isoformat(),
                side=side,
                score=round(directional_score, 4),
                strength_bucket=bucket,
                atr=float(row["ATR14"]),
                volume_ratio=volume_ratio,
                adx=adx,
                monthly_regime=monthly_regime,
                exposure_factor=exposure_factor,
                trigger=trigger,
            )
        )
    return candidates, rejected


def _simulate_trade(
    symbol: str,
    split: str,
    indicators: pd.DataFrame,
    candidate: SetupCandidate,
    config: BacktestConfig,
    equity: float,
    probability: float,
) -> SimulatedTrade:
    signal_position = candidate.signal_position
    entry_position = signal_position + 1
    entry_row = indicators.iloc[entry_position]
    entry = float(entry_row["Open"])
    atr = max(candidate.atr, entry * 0.001)
    signal_row = indicators.iloc[signal_position]
    if candidate.side == "LONG":
        atr_stop = entry - config.stop_atr_multiple * atr
        support = float(signal_row["PriorLow20"])
        structural_stop = support - 0.10 * atr
        stop = min(atr_stop, structural_stop) if 0 < entry - support <= 2.5 * atr else atr_stop
        risk_per_share = entry - stop
        target = entry + config.reward_risk * risk_per_share
    else:
        atr_stop = entry + config.stop_atr_multiple * atr
        resistance = float(signal_row["PriorHigh20"])
        structural_stop = resistance + 0.10 * atr
        stop = max(atr_stop, structural_stop) if 0 < resistance - entry <= 2.5 * atr else atr_stop
        risk_per_share = stop - entry
        target = entry - config.reward_risk * risk_per_share
    risk_budget = equity * config.risk_per_trade_pct / 100 * candidate.exposure_factor
    quantity_by_risk = risk_budget / max(risk_per_share, 0.01)
    quantity_by_exposure = (
        equity * config.max_position_exposure_pct / 100 * candidate.exposure_factor
    ) / max(entry, 0.01)
    quantity = max(0.0, min(quantity_by_risk, quantity_by_exposure))

    exit_price = float(indicators.iloc[entry_position]["Close"])
    exit_position = entry_position
    exit_reason = "Vencimiento temporal"
    last_position = min(
        len(indicators) - 1, entry_position + config.holding_sessions - 1
    )
    for position in range(entry_position, last_position + 1):
        row = indicators.iloc[position]
        low, high = float(row["Low"]), float(row["High"])
        # Si ambos niveles aparecen en la misma vela diaria, se asume el stop:
        # criterio conservador que evita inventar el orden intradía favorable.
        if candidate.side == "LONG" and low <= stop:
            exit_price, exit_position, exit_reason = stop, position, "Stop ATR/estructural"
            break
        if candidate.side == "SHORT" and high >= stop:
            exit_price, exit_position, exit_reason = stop, position, "Stop ATR/estructural"
            break
        if candidate.side == "LONG" and high >= target:
            exit_price, exit_position, exit_reason = target, position, "Take profit"
            break
        if candidate.side == "SHORT" and low <= target:
            exit_price, exit_position, exit_reason = target, position, "Take profit"
            break
        exit_price = float(row["Close"])
        exit_position = position

    direction = 1.0 if candidate.side == "LONG" else -1.0
    gross_pnl = quantity * (exit_price - entry) * direction
    round_trip_bps = config.commission_bps_per_side + config.slippage_bps_per_side
    costs = quantity * (entry + exit_price) * round_trip_bps / 10_000
    net_pnl = gross_pnl - costs
    net_r = net_pnl / risk_budget if risk_budget > 0 else 0.0
    return SimulatedTrade(
        symbol=symbol,
        split=split,
        signal_date=candidate.signal_date,
        entry_date=pd.Timestamp(indicators.index[entry_position]).isoformat(),
        exit_date=pd.Timestamp(indicators.index[exit_position]).isoformat(),
        side=candidate.side,
        entry_price=round(entry, 6),
        stop_price=round(stop, 6),
        target_price=round(target, 6),
        exit_price=round(exit_price, 6),
        quantity=round(quantity, 8),
        probability=round(probability, 6),
        score=candidate.score,
        reward_risk=round(abs(target - entry) / max(abs(entry - stop), 0.01), 4),
        gross_pnl_usd=round(gross_pnl, 6),
        costs_usd=round(costs, 6),
        net_pnl_usd=round(net_pnl, 6),
        net_r_multiple=round(net_r, 6),
        outcome="WIN" if net_pnl > 0 else "LOSS",
        exit_reason=exit_reason,
    )


def _wilson_lower_bound(wins: int, trades: int, z: float = 1.644854) -> float:
    if trades <= 0:
        return 0.0
    proportion = wins / trades
    denominator = 1 + z * z / trades
    centre = proportion + z * z / (2 * trades)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * trades)) / trades
    )
    return max(0.0, (centre - margin) / denominator)


def calculate_metrics(
    trades: list[SimulatedTrade] | tuple[SimulatedTrade, ...],
    *,
    setups: int,
    rejected: int,
    starting_capital: float,
) -> tuple[PerformanceMetrics, tuple[tuple[str, float], ...]]:
    ordered = sorted(trades, key=lambda trade: (trade.exit_date, trade.symbol))
    equity = starting_capital
    peak = equity
    maximum_drawdown = 0.0
    curve: list[tuple[str, float]] = []
    for trade in ordered:
        equity += trade.net_pnl_usd
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0.0
        maximum_drawdown = max(maximum_drawdown, drawdown)
        curve.append((trade.exit_date, round(equity, 6)))
    wins = sum(trade.net_pnl_usd > 0 for trade in ordered)
    losses = len(ordered) - wins
    gross_profit = sum(max(0.0, trade.net_pnl_usd) for trade in ordered)
    gross_loss = sum(min(0.0, trade.net_pnl_usd) for trade in ordered)
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else None
    brier = (
        sum(
            (trade.probability - (1.0 if trade.net_pnl_usd > 0 else 0.0)) ** 2
            for trade in ordered
        ) / len(ordered)
        if ordered
        else None
    )
    metrics = PerformanceMetrics(
        setups=setups,
        trades=len(ordered),
        rejected=rejected,
        wins=wins,
        losses=losses,
        win_rate=wins / len(ordered) if ordered else 0.0,
        win_rate_lower_bound=_wilson_lower_bound(wins, len(ordered)),
        profit_factor=profit_factor,
        maximum_drawdown_pct=maximum_drawdown,
        net_return_pct=(equity / starting_capital - 1) * 100 if starting_capital > 0 else 0.0,
        gross_profit_usd=gross_profit,
        gross_loss_usd=gross_loss,
        costs_usd=sum(trade.costs_usd for trade in ordered),
        brier_score=brier,
    )
    return metrics, tuple(curve)


def _calibration_table(trades: list[SimulatedTrade]) -> tuple[dict[tuple[str, int], float], float]:
    if not trades:
        return {}, 0.50
    overall = (sum(trade.net_pnl_usd > 0 for trade in trades) + 1) / (len(trades) + 2)
    grouped: dict[tuple[str, int], list[SimulatedTrade]] = {}
    for trade in trades:
        grouped.setdefault((trade.side, max(2, min(8, int(trade.score)))), []).append(trade)
    calibrated: dict[tuple[str, int], float] = {}
    prior_weight = 5.0
    for key, values in grouped.items():
        wins = sum(trade.net_pnl_usd > 0 for trade in values)
        calibrated[key] = (wins + prior_weight * overall) / (len(values) + prior_weight)
    return calibrated, overall


def evaluate_capital_preservation(
    metrics: PerformanceMetrics, config: BacktestConfig
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if metrics.trades < config.minimum_validation_trades:
        reasons.append(
            f"Solo hay {metrics.trades} operaciones OOS; se requieren {config.minimum_validation_trades}."
        )
    break_even_probability = 1 / (1 + config.reward_risk)
    if metrics.win_rate_lower_bound <= break_even_probability:
        reasons.append(
            "El límite inferior de acierto no supera la probabilidad de equilibrio."
        )
    if metrics.profit_factor is not None and metrics.profit_factor < config.minimum_profit_factor:
        reasons.append(
            f"Profit factor {metrics.profit_factor:.2f} menor al mínimo {config.minimum_profit_factor:.2f}."
        )
    if metrics.net_return_pct <= 0:
        reasons.append("El rendimiento neto fuera de muestra no es positivo.")
    if metrics.maximum_drawdown_pct > config.maximum_drawdown_pct:
        reasons.append(
            f"Drawdown {metrics.maximum_drawdown_pct:.2f}% excede el límite {config.maximum_drawdown_pct:.2f}%."
        )
    return ("RECHAZADO", tuple(reasons)) if reasons else (
        "APROBADO ESTADÍSTICAMENTE",
        ("El tramo OOS supera los límites mínimos de preservación de capital.",),
    )


def run_symbol_backtest(
    symbol: str,
    frame: pd.DataFrame,
    config: BacktestConfig,
    *,
    starting_capital_usd: float,
) -> SymbolBacktestResult:
    indicators = _prepare_indicators(frame)
    candidates, hard_rejections = _generate_candidates(indicators, config)
    split_position = max(1, min(len(indicators) - 2, int(len(indicators) * config.training_fraction)))
    split_date = pd.Timestamp(indicators.index[split_position]).isoformat()
    # Ninguna operación de entrenamiento puede consumir precios del tramo OOS.
    training_limit = split_position - config.holding_sessions
    training_candidates = [
        candidate for candidate in candidates
        if candidate.signal_position < training_limit
    ]
    validation_candidates = [candidate for candidate in candidates if candidate.signal_position >= split_position]

    training_trades: list[SimulatedTrade] = []
    training_equity = starting_capital_usd
    last_training_exit = ""
    for candidate in training_candidates:
        if last_training_exit and candidate.signal_date <= last_training_exit:
            hard_rejections["Señal solapada con una posición activa"] += 1
            continue
        trade = _simulate_trade(
            symbol.upper(), "TRAIN", indicators, candidate, config,
            training_equity, 0.50,
        )
        training_trades.append(trade)
        training_equity += trade.net_pnl_usd
        last_training_exit = trade.exit_date

    calibration, fallback_probability = _calibration_table(training_trades)
    validation_trades: list[SimulatedTrade] = []
    validation_equity = starting_capital_usd
    last_validation_exit = ""
    statistical_rejections = 0
    for candidate in validation_candidates:
        probability = calibration.get(
            (candidate.side, candidate.strength_bucket), fallback_probability
        )
        expected_r = probability * config.reward_risk - (1 - probability)
        if probability < config.minimum_probability:
            hard_rejections["Probabilidad calibrada menor al mínimo"] += 1
            statistical_rejections += 1
            continue
        if config.reward_risk < config.minimum_reward_risk or expected_r <= 0:
            hard_rejections["Valor esperado o riesgo/beneficio insuficiente"] += 1
            statistical_rejections += 1
            continue
        if last_validation_exit and candidate.signal_date <= last_validation_exit:
            hard_rejections["Señal solapada con una posición activa"] += 1
            statistical_rejections += 1
            continue
        trade = _simulate_trade(
            symbol.upper(), "VALIDATION", indicators, candidate, config,
            validation_equity, probability,
        )
        validation_trades.append(trade)
        validation_equity += trade.net_pnl_usd
        last_validation_exit = trade.exit_date

    training_metrics, _ = calculate_metrics(
        training_trades,
        setups=len(training_candidates),
        rejected=max(0, len(training_candidates) - len(training_trades)),
        starting_capital=starting_capital_usd,
    )
    validation_metrics, validation_curve = calculate_metrics(
        validation_trades,
        setups=len(validation_candidates),
        rejected=statistical_rejections,
        starting_capital=starting_capital_usd,
    )
    decision, reasons = evaluate_capital_preservation(validation_metrics, config)
    normalized = _normalize_frame(frame)
    return SymbolBacktestResult(
        symbol=symbol.upper(),
        data_start=pd.Timestamp(normalized.index[0]).isoformat(),
        data_end=pd.Timestamp(normalized.index[-1]).isoformat(),
        split_date=split_date,
        dataset_sha256=dataset_sha256(symbol, normalized),
        training=training_metrics,
        validation=validation_metrics,
        decision=decision,
        decision_reasons=reasons,
        rejected_reasons=tuple(sorted(hard_rejections.items())),
        training_trades=tuple(training_trades),
        validation_trades=tuple(validation_trades),
        validation_equity_curve=validation_curve,
    )


def run_backtest_batch(
    frames: Mapping[str, pd.DataFrame],
    config: BacktestConfig,
    *,
    starting_capital_usd: float,
) -> BacktestBatchResult:
    if not frames:
        raise ValueError("Selecciona al menos una emisora para ejecutar el backtest.")
    if starting_capital_usd <= 0:
        raise ValueError("El capital inicial del backtest debe ser mayor que cero.")
    results = tuple(
        run_symbol_backtest(
            symbol, frame, config, starting_capital_usd=starting_capital_usd
        )
        for symbol, frame in sorted(frames.items())
    )
    validation_trades = [
        trade for result in results for trade in result.validation_trades
    ]
    aggregate, _ = calculate_metrics(
        validation_trades,
        setups=sum(result.validation.setups for result in results),
        rejected=sum(result.validation.rejected for result in results),
        starting_capital=starting_capital_usd * len(results),
    )
    aggregate_decision, _ = evaluate_capital_preservation(aggregate, config)
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.symbol.encode("utf-8"))
        digest.update(result.dataset_sha256.encode("ascii"))
    return BacktestBatchResult(
        engine_version=ENGINE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        starting_capital_usd=starting_capital_usd,
        config=config,
        results=results,
        aggregate=aggregate,
        aggregate_decision=aggregate_decision,
        dataset_sha256=digest.hexdigest(),
    )


def batch_to_payload(batch: BacktestBatchResult) -> str:
    """Serialización canónica usada tanto por SQLite como por la auditoría."""

    return json.dumps(
        asdict(batch), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def payload_sha256(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
