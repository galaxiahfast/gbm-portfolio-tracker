"""Read-only execution adviser with separated statistical contracts."""
from __future__ import annotations

from dataclasses import dataclass

from portfolio_tracker.analytics.directional_probability import (
    MIN_EFFECTIVE_SAMPLES,
    MIN_HOLDOUT_SAMPLES,
    assess_directional_projection,
    dominant_class,
)
from portfolio_tracker.analytics.execution_decision import (
    ActivationCheck,
    build_long_activation_checklist,
)
from portfolio_tracker.analytics.expected_value import calculate_expectation
from portfolio_tracker.analytics.horizon_selector import select_best_horizon
from portfolio_tracker.services.position_sizing import (
    calculate_position_size,
    portfolio_risk_context,
)


@dataclass(frozen=True, slots=True)
class SystemDecision:
    action: str
    horizon: str
    direction: str
    preliminary_horizon: str
    preliminary_bias: str
    calibration_status: str
    entry_low: float | None
    entry_high: float | None
    stop_loss: float | None
    take_profit: float | None
    position_size: int
    current_shares: float
    average_price: float | None
    total_capital: float
    cash_available: float
    concentration: float
    monetary_risk: float
    risk_budget: float
    reward_risk: float | None
    expected_value_per_share: float | None
    expected_value_total: float | None
    adjusted_win_probability: float | None
    directional_brier: float | None
    directional_baseline_brier: float | None
    brier_touch: float | None
    brier_close: float | None
    validated_sessions: int
    trigger_met: bool
    risk_veto: bool
    activation_checks: tuple[ActivationCheck, ...]
    reasons: tuple[str, ...]
    explanation: str


def _long_plan(analysis, projection, zone_snapshot, minimum_rr):
    price = float(analysis.last_price)
    atr = max(float(projection.atr_value), float(analysis.atr_5m))
    zone = zone_snapshot.buys[0] if zone_snapshot and zone_snapshot.buys else None
    entry_low = float(zone.low) if zone and zone.low else min(price, float(projection.local_support))
    entry_high = float(zone.high) if zone and zone.high else entry_low
    entry_high = max(entry_low, entry_high)
    stop = max(0.01, entry_low - 2.25 * atr)
    risk = entry_high - stop
    target = max(float(projection.bullish_target), entry_high + minimum_rr * risk)
    return entry_low, entry_high, stop, target


def _short_plan(analysis, projection, zone_snapshot, minimum_rr):
    price = float(analysis.last_price)
    atr = max(float(projection.atr_value), float(analysis.atr_5m))
    zone = zone_snapshot.sales[0] if zone_snapshot and zone_snapshot.sales else None
    entry_low = max(price, float(zone.low)) if zone and zone.low else max(price, float(projection.local_resistance))
    entry_high = max(entry_low, float(zone.high)) if zone and zone.high else entry_low
    stop = entry_high + 2.25 * atr
    risk = stop - entry_low
    target = min(float(projection.bearish_target), entry_low - minimum_rr * risk)
    return entry_low, entry_high, stop, max(0.01, target)


def _rr(direction, entry, stop, target):
    risk = entry - stop if direction == "LONG" else stop - entry
    reward = target - entry if direction == "LONG" else entry - target
    return reward / risk if risk > 0 and reward > 0 else None


def _preliminary_projection(projections):
    """Descriptive context only; never authorizes an operation."""
    def rank(item):
        directional = max(float(item.probability_up), float(item.probability_down))
        return directional - float(item.probability_range)
    return max(projections, key=rank)


def generate_decision(
    symbol,
    *,
    analysis,
    repository,
    zone_snapshot=None,
    risk_fraction=0.02,
    minimum_reward_risk=1.5,
    minimum_validated_samples=MIN_EFFECTIVE_SAMPLES,
    minimum_holdout_samples=MIN_HOLDOUT_SAMPLES,
):
    """Generate one recommendation; no order, ledger or account state is written."""
    symbol = str(symbol).strip().upper()
    context = portfolio_risk_context(repository, symbol, analysis.last_price)
    activation_checks = build_long_activation_checklist(analysis, zone_snapshot)
    selection = select_best_horizon(
        analysis.horizon_projections,
        min_validated_samples=minimum_validated_samples,
        min_holdout_samples=minimum_holdout_samples,
        direction="AUTO",
    )
    preliminary = _preliminary_projection(analysis.horizon_projections)
    preliminary_class = dominant_class(preliminary)
    preliminary_bias = {
        "UP": "Alcista", "DOWN": "Bajista", "RANGE": "Lateral", "UNKNOWN": "Indefinido",
    }[preliminary_class]
    reasons = []
    if selection is None:
        projection = preliminary
        evidence = assess_directional_projection(
            projection,
            minimum_effective_samples=minimum_validated_samples,
            minimum_holdout_samples=minimum_holdout_samples,
        )
        direction = "LONG" if preliminary_class == "UP" else "SHORT" if preliminary_class == "DOWN" else "NEUTRAL"
        horizon = "Sin horizonte validado"
        reasons.append("No existe un horizonte direccional calibrado que mejore su baseline OOS.")
        reasons.append(evidence.reason)
    else:
        projection = selection.projection
        evidence = assess_directional_projection(
            projection,
            minimum_effective_samples=minimum_validated_samples,
            minimum_holdout_samples=minimum_holdout_samples,
        )
        direction, horizon = selection.direction, selection.label

    plan_direction = direction if direction in {"LONG", "SHORT"} else (
        "LONG" if float(projection.probability_up) >= float(projection.probability_down) else "SHORT"
    )
    entry_low, entry_high, stop, target = (
        _long_plan(analysis, projection, zone_snapshot, minimum_reward_risk)
        if plan_direction == "LONG" else
        _short_plan(analysis, projection, zone_snapshot, minimum_reward_risk)
    )
    plan_entry = entry_high if plan_direction == "LONG" else entry_low
    reward_risk = _rr(plan_direction, plan_entry, stop, target)
    expectation = None
    if selection is not None and evidence.eligible:
        calibrated_probability = (
            float(projection.probability_up) / 100.0
            if direction == "LONG" else float(projection.probability_down) / 100.0
        )
        expectation = calculate_expectation(
            symbol, plan_entry, stop, target,
            bullish_score=projection.probability_up,
            bearish_score=projection.probability_down,
            calibrated_probability=calibrated_probability,
            direction=direction,
        )

    sizing = (
        calculate_position_size(
            context, entry_high, stop, risk_fraction=risk_fraction,
            maximum_concentration=0.30,
        ) if plan_direction == "LONG" else None
    )
    current_price = float(analysis.last_price)
    has_position = context.current_shares > 0
    active_long_plan = (
        analysis.execution_levels
        if getattr(analysis.execution_levels, "direction", "") == "LONG"
        else analysis.buy_levels
    )
    risk_veto = bool(analysis.risk_veto or analysis.fundamental_risk_veto)

    if risk_veto:
        action = "ESPERAR"
        reasons.append("Veto de riesgo técnico, fundamental, noticioso o de evento activo.")
    elif (has_position and active_long_plan is not None
          and current_price <= float(active_long_plan.stop_loss)):
        action = "VENDER"
        reasons.append("El precio perforó el stop del plan persistente de la posición.")
    elif (has_position and active_long_plan is not None
          and current_price >= float(active_long_plan.take_profit_1)):
        action = "VENDER"
        reasons.append("El precio alcanzó el take profit del plan persistente de la posición.")
    elif has_position and direction == "SHORT" and selection is not None:
        action = "VENDER"
        reasons.append(f"El horizonte validado {horizon} cambió a sesgo bajista.")
    elif has_position:
        action = "MANTENER"
        reasons.append("La posición sigue entre stop y objetivo sin invalidación confirmada.")
        if context.concentration > 0.30:
            reasons.append("No aumentar: la posición ya supera 30% del portafolio.")
    else:
        gates = {
            "probabilidad direccional calibrada": selection is not None and evidence.eligible,
            "régimen LONG permitido": analysis.macro_permission in {"LONG_ONLY", "BOTH_REDUCED"} and direction == "LONG",
            "gatillo confirmado": bool(analysis.activation_trigger_met),
            "R:R mínimo": reward_risk is not None and reward_risk + 1e-9 >= minimum_reward_risk,
            "expectativa positiva": expectation is not None and expectation.expected_value_per_share > 0,
            "tamaño permitido": sizing is not None and sizing.shares > 0,
            "concentración máxima": context.concentration <= 0.30,
        }
        failed = [name for name, passed in gates.items() if not passed]
        if not failed:
            action = "COMPRAR"
            reasons.append("Todas las condiciones cuantitativas y de capital están cumplidas.")
        else:
            action = "ESPERAR"
            reasons.append("Falta: " + ", ".join(failed) + ".")

    if has_position and active_long_plan is not None:
        entry_low = float(active_long_plan.entry_low)
        entry_high = float(active_long_plan.entry_high)
        stop = float(active_long_plan.stop_loss)
        target = float(active_long_plan.take_profit_1)
        reward_risk = _rr("LONG", entry_high, stop, target)

    size = int(sizing.shares) if sizing is not None and action == "COMPRAR" else 0
    monetary_risk = sizing.monetary_risk if sizing is not None and action == "COMPRAR" else 0.0
    ev_per_share = expectation.expected_value_per_share if expectation is not None else None
    total_ev = ev_per_share * size if ev_per_share is not None and size else None
    action_context = (
        f"Horizonte validado: {horizon}." if selection is not None else
        f"Sesgo preliminar más fuerte: {preliminary_bias.lower()} · {preliminary.label}."
    )
    rr_text = "N/D" if reward_risk is None else f"{reward_risk:.2f}"
    ev_text = "N/D (muestra insuficiente)" if ev_per_share is None else f"${ev_per_share:+.2f} por acción"
    explanation = (
        f"Acción actual: {action}. {action_context} R:R {rr_text}; EV {ev_text}. "
        + " ".join(reasons)
    )
    return SystemDecision(
        action=action, horizon=horizon, direction=direction,
        preliminary_horizon=preliminary.label, preliminary_bias=preliminary_bias,
        calibration_status=evidence.status,
        entry_low=entry_low, entry_high=entry_high, stop_loss=stop,
        take_profit=target, position_size=size,
        current_shares=context.current_shares, average_price=context.average_price,
        total_capital=context.total_capital, cash_available=context.cash_available,
        concentration=context.concentration, monetary_risk=monetary_risk,
        risk_budget=context.total_capital * min(float(risk_fraction), 0.02),
        reward_risk=reward_risk,
        expected_value_per_share=ev_per_share,
        expected_value_total=total_ev,
        adjusted_win_probability=(expectation.adjusted_probability if expectation else None),
        directional_brier=evidence.brier_score,
        directional_baseline_brier=evidence.baseline_brier_score,
        brier_touch=None, brier_close=None,
        validated_sessions=evidence.holdout_samples,
        trigger_met=bool(analysis.activation_trigger_met), risk_veto=risk_veto,
        activation_checks=activation_checks,
        reasons=tuple(reasons), explanation=explanation,
    )
