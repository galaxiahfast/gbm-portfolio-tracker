"""Read-only execution adviser with separated statistical contracts."""
from __future__ import annotations

from dataclasses import dataclass

from portfolio_tracker.analytics.directional_probability import (
    MIN_EFFECTIVE_SAMPLES,
    MIN_HOLDOUT_SAMPLES,
    assess_directional_projection,
    dominant_class,
)
from portfolio_tracker.analytics.horizon_models import feature_vector
from portfolio_tracker.analytics.net_expectation import (
    TradingCostPolicy,
    evaluate_long_opportunity,
    select_highest_net_expectation,
)
from portfolio_tracker.analytics.operational_calibration import predict_calibrated_scores
from portfolio_tracker.analytics.execution_decision import (
    ActivationCheck,
    build_long_activation_checklist,
)
from portfolio_tracker.services.model_execution_record import build_replay_snapshot, prediction_snapshot
from portfolio_tracker.services.operational_model_registry import latest_approved_operational_models
from portfolio_tracker.services.position_sizing import portfolio_risk_context


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
    operational_brier: float | None = None
    operational_baseline_brier: float | None = None
    sl_first_probability: float | None = None
    timeout_probability: float | None = None
    cost_rate_per_side: float | None = None


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


def _operational_opportunities(analysis, context, models, risk_fraction, minimum_rr, costs):
    """Score the frozen live LONG plan against each approved first-hit horizon."""
    plan = getattr(analysis, "execution_levels", None)
    if not models or plan is None or getattr(plan, "direction", "") != "LONG":
        return ()
    price = float(analysis.last_price)
    stop, target = float(plan.stop_loss), float(plan.take_profit_1)
    if not stop < price < target:
        return ()
    try:
        snapshot = build_replay_snapshot(
            analysis, observed_at=analysis.source_bar_closed_at,
            protocol="READ_ONLY_OPERATIONAL_SELECTION_V1",
        )
    except (TypeError, ValueError, OSError):
        return ()
    opportunities = []
    for projection in analysis.horizon_projections:
        model = models.get(projection.label)
        if model is None:
            continue
        horizon_record = {
            "prediction": prediction_snapshot(projection),
            "operational_contract": {
                "entry_price": price, "stop_loss": stop, "take_profit": target,
            },
        }
        try:
            prediction = predict_calibrated_scores(
                model, feature_vector({"feature_snapshot": snapshot}, horizon_record)
            )
            probabilities = prediction["scores"]
            opportunity = evaluate_long_opportunity(
                projection.label, entry=price, stop=stop, take_profit=target,
                tp_first=probabilities["TP_FIRST"],
                sl_first=probabilities["SL_FIRST"],
                timeout=probabilities["TIMEOUT"],
                capital=context.total_capital, cash=context.cash_available,
                current_market_value=context.current_market_value,
                risk_fraction=risk_fraction,
                minimum_net_reward_risk=minimum_rr, costs=costs,
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            # A missing, malformed or stale model is never promoted to a trade.
            continue
        opportunities.append(opportunity)
    return tuple(opportunities)


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
    operational_models=None,
    cost_policy: TradingCostPolicy | None = None,
):
    """Recommend from net first-hit EV; never execute or write an order."""
    symbol = str(symbol).strip().upper()
    context = portfolio_risk_context(repository, symbol, analysis.last_price)
    activation_checks = build_long_activation_checklist(analysis, zone_snapshot)
    preliminary = _preliminary_projection(analysis.horizon_projections)
    preliminary_class = dominant_class(preliminary)
    preliminary_bias = {
        "UP": "Alcista", "DOWN": "Bajista", "RANGE": "Lateral", "UNKNOWN": "Indefinido",
    }[preliminary_class]
    evidence = assess_directional_projection(
        preliminary,
        minimum_effective_samples=minimum_validated_samples,
        minimum_holdout_samples=minimum_holdout_samples,
    )
    current_price = float(analysis.last_price)
    has_position = context.current_shares > 0
    risk_veto = bool(analysis.risk_veto or analysis.fundamental_risk_veto)
    models = {}
    opportunities = ()
    if not has_position and not risk_veto:
        models = (
            operational_models if operational_models is not None else
            latest_approved_operational_models(symbol, analysis.source_bar_closed_at)
        )
        opportunities = _operational_opportunities(
            analysis, context, models, risk_fraction, minimum_reward_risk,
            cost_policy or TradingCostPolicy(),
        )
    selection = select_highest_net_expectation(opportunities)
    model_record = models.get(selection.horizon) if selection is not None else None
    direction = "LONG" if has_position or selection is not None else "NEUTRAL"
    horizon = selection.horizon if selection is not None else "Sin horizonte validado"
    reasons = []
    if selection is None and not has_position:
        if not models:
            reasons.append("No hay un modelo TP/SL/timeout calibrado y aprobado para este corte.")
        elif not opportunities:
            reasons.append("No existe un plan LONG compatible con las barreras del modelo operativo.")
        else:
            reasons.append("Ningún horizonte ofrece expectativa neta positiva, R:R neto y tamaño admisible.")
            reasons.extend(f"{item.horizon}: {item.reason}" for item in opportunities)

    projection = next(
        (item for item in analysis.horizon_projections if item.label == horizon), preliminary,
    )
    if selection is not None:
        entry_low = entry_high = selection.entry
        stop, target = selection.stop, selection.take_profit
        reward_risk = selection.net_reward_risk
    else:
        plan_direction = "LONG" if has_position or preliminary_class != "DOWN" else "SHORT"
        entry_low, entry_high, stop, target = (
            _long_plan(analysis, projection, zone_snapshot, minimum_reward_risk)
            if plan_direction == "LONG" else
            _short_plan(analysis, projection, zone_snapshot, minimum_reward_risk)
        )
        reward_risk = _rr(
            plan_direction, entry_high if plan_direction == "LONG" else entry_low,
            stop, target,
        )
    active_long_plan = (
        analysis.execution_levels
        if getattr(analysis.execution_levels, "direction", "") == "LONG"
        else analysis.buy_levels
    )
    if (has_position and active_long_plan is not None
          and current_price <= float(active_long_plan.stop_loss)):
        action = "VENDER"
        reasons.append("El precio perforó el stop del plan persistente de la posición.")
    elif (has_position and active_long_plan is not None
          and current_price >= float(active_long_plan.take_profit_1)):
        action = "VENDER"
        reasons.append("El precio alcanzó el take profit del plan persistente de la posición.")
    elif has_position:
        action = "MANTENER"
        reasons.append("La posición sigue entre stop y objetivo sin invalidación confirmada.")
        if context.concentration > 0.30:
            reasons.append("No aumentar: la posición ya supera 30% del portafolio.")
        if risk_veto:
            reasons.append("Veto activo: no abrir ni ampliar exposición; revisar la posición.")
    elif risk_veto:
        action = "ESPERAR"
        reasons.append("Veto de riesgo técnico, fundamental, noticioso o de evento activo.")
    else:
        gates = {
            "modelo TP/SL/timeout aprobado y EV neta positiva": selection is not None,
            "régimen LONG permitido": analysis.macro_permission in {"LONG_ONLY", "BOTH_REDUCED"},
            "gatillo confirmado": bool(analysis.activation_trigger_met),
            "plan no condicional": not bool(getattr(analysis, "execution_plan_conditional", True)),
            "señal no rechazada": not bool(getattr(analysis, "signal_rejected", False)),
            "R:R neto mínimo": selection is not None and reward_risk + 1e-9 >= minimum_reward_risk,
            "tamaño permitido": selection is not None and selection.shares > 0,
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

    size = selection.shares if selection is not None and action == "COMPRAR" else 0
    monetary_risk = selection.monetary_risk if size else 0.0
    ev_per_share = selection.net_ev_per_share if selection is not None else None
    total_ev = selection.net_ev_total if size else None
    action_context = (
        f"Horizonte elegido por EV neta: {horizon}." if selection is not None else
        f"Sesgo preliminar más fuerte: {preliminary_bias.lower()} · {preliminary.label}."
    )
    rr_text = "N/D" if reward_risk is None else f"{reward_risk:.2f}"
    ev_text = "N/D (muestra insuficiente)" if ev_per_share is None else f"${ev_per_share:+.2f} por acción"
    explanation = (
        f"Acción actual: {action}. {action_context} R:R {'neto ' if selection else ''}{rr_text}; "
        f"EV neta {ev_text}. "
        + " ".join(reasons)
    )
    metrics = (model_record or {}).get("final_holdout", {}).get("metrics") or {}
    return SystemDecision(
        action=action, horizon=horizon, direction=direction,
        preliminary_horizon=preliminary.label, preliminary_bias=preliminary_bias,
        entry_low=entry_low, entry_high=entry_high, stop_loss=stop,
        take_profit=target, position_size=size,
        current_shares=context.current_shares, average_price=context.average_price,
        total_capital=context.total_capital, cash_available=context.cash_available,
        concentration=context.concentration, monetary_risk=monetary_risk,
        risk_budget=context.total_capital * min(float(risk_fraction), 0.02),
        reward_risk=reward_risk,
        expected_value_per_share=ev_per_share,
        expected_value_total=total_ev,
        adjusted_win_probability=(selection.tp_first if selection else None),
        directional_brier=None,
        directional_baseline_brier=None,
        brier_touch=None, brier_close=None,
        validated_sessions=int(metrics.get("samples") or 0),
        trigger_met=bool(analysis.activation_trigger_met), risk_veto=risk_veto,
        activation_checks=activation_checks,
        reasons=tuple(reasons), explanation=explanation,
        operational_brier=metrics.get("brier"),
        operational_baseline_brier=metrics.get("baseline_brier"),
        sl_first_probability=(selection.sl_first if selection else None),
        timeout_probability=(selection.timeout if selection else None),
        cost_rate_per_side=(selection.cost_rate_per_side if selection else None),
        calibration_status=(
            "Histórico OOS calibrado; pendiente de validación forward"
            if selection is not None else evidence.status
        ),
    )
