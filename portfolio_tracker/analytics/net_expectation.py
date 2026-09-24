"""Cost-aware first-passage ranking with development-only observed gap losses.

Fill probability, spread and slippage are explicit assumptions, not measured
fills. The module is pure and never submits an order or reads the ledger.
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
class FillModel:
    no_fill_probability: float = 0.05
    spread_bps: float = 4.0  # full quoted spread; pay half at entry and exit

    def __post_init__(self):
        if (not math.isfinite(float(self.no_fill_probability))
                or not 0 <= self.no_fill_probability < 1
                or not math.isfinite(float(self.spread_bps))
                or self.spread_bps < 0):
            raise ValueError("Supuestos de fill/spread inválidos.")


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
    worst_sl_per_share: float
    theoretical_net_ev_per_share: float
    theoretical_net_ev_total: float
    observed_net_ev_per_share: float
    observed_net_ev_total: float
    fill_probability: float
    observed_sl_samples: int
    spread_bps: float
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
    observed_sl_loss_multiples: tuple[float, ...] | None = None,
    fills: FillModel | None = None,
) -> NetOpportunity:
    """Evaluate a LONG with empirical stop exits and explicit fill assumptions.

    Missing observed SL exits makes the opportunity non-actionable. Position
    size uses the worst observed stop fill; TIMEOUT is stressed at that same
    adverse exit. This is conservative but not a guarantee against a new gap.
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
    fills = fills or FillModel()
    rate = costs.rate_per_side
    entry_cost = entry * rate
    theoretical_tp = (take_profit - entry) - entry_cost - take_profit * rate
    theoretical_sl = (stop - entry) - entry_cost - stop * rate
    if theoretical_sl >= 0:
        raise ValueError("El stop debe representar una pérdida neta.")
    theoretical_ev = (
        probabilities[0] * theoretical_tp
        + (probabilities[1] + probabilities[2]) * theoretical_sl
    )
    observed = tuple(float(value) for value in (observed_sl_loss_multiples or ()))
    if any(not math.isfinite(value) or value < 1.0 - 1e-9 for value in observed):
        raise ValueError("Las pérdidas observadas SL deben ser múltiplos finitos >= 1 del stop.")
    half_spread = fills.spread_bps / 20_000.0
    paid_entry = entry * (1 + half_spread)
    paid_tp = take_profit * (1 - half_spread)
    net_tp = paid_tp - paid_entry - rate * (paid_entry + paid_tp)
    realized_stops = tuple(
        entry - (entry - stop) * multiple for multiple in observed
    )
    if any(exit_price <= 0 for exit_price in realized_stops):
        raise ValueError("Pérdida observada excede el precio disponible.")
    net_sl_distribution = tuple(
        exit_price * (1 - half_spread) - paid_entry
        - rate * (paid_entry + exit_price * (1 - half_spread))
        for exit_price in realized_stops
    )
    # No empirical SL exits -> display the theoretical column, but never
    # fabricate an 'observed' EV or authorize a trade.
    net_sl = sum(net_sl_distribution) / len(net_sl_distribution) if observed else theoretical_sl
    worst_sl = min(net_sl_distribution) if observed else theoretical_sl
    conditional_ev = (
        probabilities[0] * net_tp
        + probabilities[1] * net_sl
        + probabilities[2] * worst_sl
    ) if observed else 0.0
    net_ev = (1 - fills.no_fill_probability) * conditional_ev
    net_rr = net_tp / -net_sl
    budget = capital * float(risk_fraction)
    headroom = max(0.0, float(maximum_concentration) * capital - current_market_value)
    shares = max(0, min(
        math.floor(budget / -worst_sl),
        math.floor(cash / (paid_entry * (1 + rate))),
        math.floor(headroom / entry),
    ))
    reasons = []
    if not observed:
        reasons.append("sin distribución observada de pérdidas SL_FIRST")
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
        worst_sl_per_share=worst_sl,
        theoretical_net_ev_per_share=theoretical_ev,
        theoretical_net_ev_total=theoretical_ev * shares,
        observed_net_ev_per_share=net_ev,
        observed_net_ev_total=net_ev * shares,
        fill_probability=1 - fills.no_fill_probability,
        observed_sl_samples=len(observed), spread_bps=fills.spread_bps,
        net_ev_per_share=net_ev, net_ev_total=net_ev * shares,
        net_reward_risk=net_rr, shares=shares,
        monetary_risk=shares * -worst_sl, risk_budget=budget,
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
