"""Conservative, cost-aware ranking of calibrated first-passage opportunities.

TIMEOUT is valued at the stop exit as a stress assumption, not as a claim
about its realized price. Gap losses can exceed this estimate. The module is
pure and never submits an order or reads the portfolio ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TradingCostPolicy:
    commission_bps_per_side: float = 25.0
    slippage_bps_per_side: float = 5.0

    def __post_init__(self):
        if any(
            not math.isfinite(float(value)) or float(value) < 0
            for value in (self.commission_bps_per_side, self.slippage_bps_per_side)
        ):
            raise ValueError("Los costes por lado deben ser finitos y no negativos.")

    @property
    def rate_per_side(self) -> float:
        return (self.commission_bps_per_side + self.slippage_bps_per_side) / 10_000.0


@dataclass(frozen=True, slots=True)
class NetOpportunity:
    horizon: str
    entry: float
    stop: float
    take_profit: float
    tp_first: float
    sl_first: float
    timeout: float
    net_tp_per_share: float
    net_sl_per_share: float
    net_ev_per_share: float
    net_ev_total: float
    net_reward_risk: float
    shares: int
    monetary_risk: float
    risk_budget: float
    cost_rate_per_side: float
    eligible: bool
    reason: str


def _finite_positive(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} inválido.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} debe ser positivo y finito.")
    return number


def evaluate_long_opportunity(
    horizon: str,
    *,
    entry: float,
    stop: float,
    take_profit: float,
    tp_first: float,
    sl_first: float,
    timeout: float,
    capital: float,
    cash: float,
    current_market_value: float = 0.0,
    risk_fraction: float = 0.02,
    maximum_concentration: float = 0.30,
    minimum_net_reward_risk: float = 1.5,
    costs: TradingCostPolicy | None = None,
) -> NetOpportunity:
    """Evaluate a LONG with first-hit probabilities and two-sided costs.

    Position size is capped by *net* stop loss, entry cash including costs,
    and portfolio concentration. A timeout is stress-priced at the stop; it
    must not be silently treated as a profitable or cost-free outcome.
    """
    entry = _finite_positive(entry, "Entrada")
    stop = _finite_positive(stop, "Stop")
    take_profit = _finite_positive(take_profit, "Take profit")
    capital = _finite_positive(capital, "Capital")
    if not stop < entry < take_profit:
        raise ValueError("Las barreras LONG deben cumplir stop < entrada < TP.")
    probabilities = (float(tp_first), float(sl_first), float(timeout))
    if (any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8)):
        raise ValueError("TP/SL/timeout deben ser probabilidades finitas que sumen uno.")
    if (not math.isfinite(float(cash)) or not math.isfinite(float(current_market_value))
            or cash < 0 or current_market_value < 0
            or not 0 < float(risk_fraction) <= 0.02
            or not 0 < float(maximum_concentration) <= 1
            or not math.isfinite(float(minimum_net_reward_risk))
            or minimum_net_reward_risk < 1):
        raise ValueError("Parámetros de capital o riesgo inválidos.")
    costs = costs or TradingCostPolicy()
    rate = costs.rate_per_side
    entry_cost = entry * rate
    net_tp = (take_profit - entry) - entry_cost - take_profit * rate
    net_sl = (stop - entry) - entry_cost - stop * rate
    if net_sl >= 0:
        raise ValueError("El stop debe representar una pérdida neta.")
    # Conservative timeout stress: exit at the stop, including both sides.
    net_ev = probabilities[0] * net_tp + (probabilities[1] + probabilities[2]) * net_sl
    net_rr = net_tp / -net_sl
    budget = capital * float(risk_fraction)
    headroom = max(0.0, float(maximum_concentration) * capital - current_market_value)
    shares = max(0, min(
        math.floor(budget / -net_sl),
        math.floor(cash / (entry + entry_cost)),
        math.floor(headroom / entry),
    ))
    reasons = []
    if net_tp <= 0 or net_rr + 1e-12 < minimum_net_reward_risk:
        reasons.append("R:R neto insuficiente")
    if net_ev <= 0:
        reasons.append("expectativa neta no positiva")
    if shares == 0:
        reasons.append("capital, efectivo o concentración insuficiente")
    return NetOpportunity(
        horizon=str(horizon), entry=entry, stop=stop, take_profit=take_profit,
        tp_first=probabilities[0], sl_first=probabilities[1], timeout=probabilities[2],
        net_tp_per_share=net_tp, net_sl_per_share=net_sl,
        net_ev_per_share=net_ev, net_ev_total=net_ev * shares,
        net_reward_risk=net_rr, shares=shares,
        monetary_risk=shares * -net_sl, risk_budget=budget,
        cost_rate_per_side=rate, eligible=not reasons,
        reason="Apta" if not reasons else "; ".join(reasons),
    )


def select_highest_net_expectation(opportunities) -> NetOpportunity | None:
    """Rank actionable plans by net portfolio EV, never by heuristic score."""
    eligible = [item for item in opportunities if item.eligible]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item.net_ev_total, item.net_ev_per_share,
                          -item.monetary_risk, item.horizon),
    )
