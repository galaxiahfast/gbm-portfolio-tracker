"""Independent regime → setup → trigger contracts, without accounting writes."""
from dataclasses import dataclass
from enum import StrEnum
import math

import pandas as pd

INTRADAY_READY = [
    'StochRSI_K', 'StochRSI_D', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume_MA20',
    'MACD', 'MACD_signal', 'MACD_histogram', 'VWAP', 'ADX14', 'OBV',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B',
]


def causal_revision():
    """Reject promotion after the actual causal rules change, even without a version bump."""
    from pathlib import Path
    import hashlib
    digest = hashlib.sha256()
    for name in ('causal_core.py','decision_engines.py','technical_probability.py',
                 'closed_bars.py','multi_timeframe.py','replay.py','backtesting.py','technical_validity.py'):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


class Permission(StrEnum):
    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    BOTH_REDUCED = "BOTH_REDUCED"
    NO_TRADE = "NO_TRADE"


def adx_hysteresis(value, previous=False):
    if not math.isfinite(float(value)):
        return False
    return value > 20 if previous else value > 25


def trend(frame):
    if len(frame) < 21:
        return None
    close = frame.Close
    fast, slow = close.ewm(span=9, adjust=False).mean().iloc[-1], close.ewm(span=21, adjust=False).mean().iloc[-1]
    return 1 if close.iloc[-1] > fast > slow else -1 if close.iloc[-1] < fast < slow else 0


@dataclass(frozen=True)
class Regime:
    permission: Permission
    trending: bool
    detail: str


class RegimeEngine:
    def evaluate(self, weekly, daily, four_hour, previous_trending=False):
        votes = [trend(weekly), trend(daily), trend(four_hour)]
        if any(v is None for v in votes) or "ADX14" not in daily or daily.empty:
            return Regime(Permission.NO_TRADE, False, "Histórico macro insuficiente: semanal, diario y 4h obligatorios.")
        active = adx_hysteresis(float(daily.ADX14.iloc[-1]), previous_trending)
        # Higher frames must agree; microstructure has no vote.
        permission = Permission.BOTH_REDUCED
        if active and all(v == 1 for v in votes):
            permission = Permission.LONG_ONLY
        elif active and all(v == -1 for v in votes):
            permission = Permission.SHORT_ONLY
        return Regime(permission, active, f"Votos S/D/4h={votes}; ADX diario={daily.ADX14.iloc[-1]:.1f}; histéresis >25 / <=20.")


@dataclass(frozen=True)
class Setup:
    long_allowed: bool
    short_allowed: bool
    support: float
    resistance: float
    detail: str


class SetupEngine:
    def evaluate(self, four_hour, hourly):
        votes = [trend(four_hour), trend(hourly)]
        if any(v is None for v in votes):
            return Setup(False, False, 0, 0, "Faltan al menos 21 velas cerradas 1h/4h.")
        support = min(float(four_hour.Low.tail(20).min()), float(hourly.Low.tail(20).min()))
        resistance = max(float(four_hour.High.tail(20).max()), float(hourly.High.tail(20).max()))
        return Setup(all(v >= 0 for v in votes), all(v <= 0 for v in votes), support, resistance, f"Estructura 4h/1h={votes}; soporte {support:.2f}, resistencia {resistance:.2f}.")


@dataclass(frozen=True)
class Trigger:
    direction: str
    activated: bool
    detail: str


class TriggerEngine:
    def evaluate(self, bars, regime, setup):
        if len(bars) < 22:
            return Trigger("NONE", False, "Esperar histórico cerrado 5m suficiente.")
        long_momentum, short_momentum = trigger_momentum(bars)
        long = regime.permission == Permission.LONG_ONLY and setup.long_allowed and long_momentum
        short = regime.permission == Permission.SHORT_ONLY and setup.short_allowed and short_momentum
        direction = "LONG" if long else "SHORT" if short else "NONE"
        detail = ("Activar LONG: régimen LONG_ONLY, setup 4h/1h, cruce K>D cerrado sin sobrecompra, "
                  "MACD>=señal, cierre>máximo previo y volumen>media previa20. "
                  "Activar SHORT: SHORT_ONLY, setup 4h/1h, cruce K<D cerrado, MACD<=señal, "
                  "cierre<mínimo previo y volumen>media previa20. En BOTH_REDUCED se espera; no entrada inmediata.")
        return Trigger(direction, long or short, detail)


def trigger_momentum(bars):
        """Necessary trigger conditions; shared cheap replay prefilter, not authorization."""
        if len(bars) < 22:
            return False, False
        previous, latest = bars.iloc[-2], bars.iloc[-1]
        volume = latest.Volume > bars.Volume.iloc[-21:-1].mean() > 0
        up = previous.StochRSI_K <= previous.StochRSI_D and latest.StochRSI_K > latest.StochRSI_D
        down = previous.StochRSI_K >= previous.StochRSI_D and latest.StochRSI_K < latest.StochRSI_D
        long = (volume
                and up and max(latest.StochRSI_K, latest.StochRSI_D) <= 80
                and latest.MACD >= latest.MACD_signal and latest.Close >= previous.High)
        short = (volume
                 and down and latest.MACD <= latest.MACD_signal and latest.Close <= previous.Low)
        return bool(long), bool(short)

@dataclass(frozen=True)
class CausalDecision:
    regime: Regime
    setup: Setup
    trigger: Trigger


def evaluate_causal_core(*, weekly, daily, four_hour, hourly, intraday, previous_trending=False):
    """One gate for live and replay. Inputs must contain ONLY closed bars."""
    regime = RegimeEngine().evaluate(weekly, daily, four_hour, previous_trending)
    setup = SetupEngine().evaluate(four_hour, hourly)
    trigger = TriggerEngine().evaluate(intraday, regime, setup)
    return CausalDecision(regime, setup, trigger)


def directional_confluence(side, *, macd_delta, price_delta, weekly_bias, monthly_bias, volume_ratio):
    """Each signed market vote is projected onto the operation exactly once."""
    if side not in ('LONG', 'SHORT'):
        raise ValueError('Dirección no reconocida.')
    values = (macd_delta, price_delta, weekly_bias, monthly_bias, volume_ratio)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Confluencia requiere valores finitos.')
    direction = 1 if side == 'LONG' else -1
    sign = lambda value: (value > 0) - (value < 0)
    return (3.0 + direction * (sign(macd_delta) + sign(price_delta)
            + 0.75 * weekly_bias + monthly_bias) + (0.5 if volume_ratio > 1.2 else 0.0))
