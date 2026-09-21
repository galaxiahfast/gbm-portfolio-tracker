"""Pure expected-value math for an informational trading recommendation."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    symbol: str
    direction: str
    raw_probability: float
    adjusted_probability: float
    brier_touch: float | None
    brier_close: float | None
    reward_per_share: float
    risk_per_share: float
    reward_risk: float
    expected_value_per_share: float
    calibration_available: bool


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} no es numérico.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} no es finito.")
    return result


def calculate_expectation(
    symbol,
    entry_price,
    stop_loss,
    take_profit,
    *,
    bullish_score=50.0,
    bearish_score=None,
    brier_touch=None,
    brier_close=None,
    calibrated_probability=None,
    direction="LONG",
):
    """Calculate EV only from an already calibrated directional probability.

    Zone-touch Brier parameters remain accepted for API compatibility but are
    deliberately not used: they describe a different statistical target.
    """
    symbol = str(symbol).strip().upper()
    direction = str(direction).strip().upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction debe ser LONG o SHORT.")
    entry = _number(entry_price, "entry_price")
    stop = _number(stop_loss, "stop_loss")
    target = _number(take_profit, "take_profit")
    score = _number(
        bullish_score if direction == "LONG" else
        (100.0 - float(bullish_score) if bearish_score is None else bearish_score),
        "directional_score",
    )
    if not 0 <= score <= 100 or min(entry, stop, target) <= 0:
        raise ValueError("Precios y score fuera de rango.")
    if direction == "LONG":
        risk, reward = entry - stop, target - entry
    else:
        risk, reward = stop - entry, entry - target
    if risk <= 0 or reward <= 0:
        raise ValueError("Stop y objetivo no corresponden a la dirección.")

    if calibrated_probability is None:
        raise ValueError("Se requiere una probabilidad direccional empíricamente calibrada.")
    probability = _number(calibrated_probability, "calibrated_probability")
    if not 0 < probability < 1:
        raise ValueError("La probabilidad calibrada debe estar en (0,1).")
    expected = probability * reward - (1.0 - probability) * risk
    return ExpectationResult(
        symbol, direction, score / 100.0, probability, None, None,
        reward, risk, round(reward / risk, 12), expected, True,
    )
