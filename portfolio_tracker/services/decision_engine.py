"""Read-only execution adviser with separated statistical contracts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

from portfolio_tracker.analytics.directional_probability import (
    MIN_EFFECTIVE_SAMPLES,
    MIN_HOLDOUT_SAMPLES,
    assess_directional_projection,
    dominant_class,
)
from portfolio_tracker.analytics.horizon_models import (
    FeatureContractMismatch, PROFESSIONAL_MINIMUM_SAMPLES,
    feature_vector, model_feature_contract,
)
from portfolio_tracker.analytics.net_expectation import (
    FillModel,
    TradingCostPolicy,
    evaluate_long_opportunity,
    select_highest_net_expectation,
)
from portfolio_tracker.analytics.temporal_contract import ANCHOR_VERSION, is_actionable_emission
from portfolio_tracker.analytics.operational_calibration import predict_calibrated_scores
from portfolio_tracker.analytics.multi_timeframe import ExecutionLevels
from portfolio_tracker.analytics.execution_decision import (
    ActivationCheck,
    build_long_activation_checklist,
)
from portfolio_tracker.services.model_execution_record import (
    build_replay_snapshot, prediction_snapshot, technical_horizon,
)
from portfolio_tracker.services.operational_model_registry import (
    _approved_result,
    latest_approved_operational_models,
)
from portfolio_tracker.services.position_sizing import portfolio_risk_context
from portfolio_tracker.services.operational_state import read_state


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
    anchor_version: str = ANCHOR_VERSION
    actionable_at_emission: bool = False
    waiting_cause: str = ""
    theoretical_expected_value_per_share: float | None = None
    theoretical_expected_value_total: float | None = None
    observed_expected_value_per_share: float | None = None
    observed_expected_value_total: float | None = None
    fill_probability: float | None = None
    observed_sl_samples: int = 0
    spread_bps: float | None = None
    exposure_factor_applied: float = 1.0
    recommendation_mode: str = "CONSERVADOR"


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


def _operational_opportunities(analysis, context, models, risk_fraction, minimum_rr, costs, fills):
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
        if model is None or not _approved_result(model):
            continue
        horizon_record = {
            "prediction": prediction_snapshot(projection),
            "model_prediction": prediction_snapshot(technical_horizon(analysis, projection.label)),
            "operational_contract": {
                "entry_price": price, "stop_loss": stop, "take_profit": target,
            },
        }
        try:
            cut = {"feature_snapshot": snapshot}
            manifest = model_feature_contract(cut, horizon_record)
            prediction = predict_calibrated_scores(
                model, feature_vector(cut, horizon_record),
                feature_version=manifest["version"],
                available_features=manifest["available_features"],
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
                observed_sl_loss_multiples=tuple(
                    (model.get("execution_evidence") or {}).get("gross_loss_multiples") or ()
                ),
                fills=fills,
            )
        except FeatureContractMismatch:
            raise
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
    fill_policy: FillModel | None = None,
    inference_at=None,
    validation_counts=None,
):
    """Recommend from net first-hit EV; never execute or write an order."""
    symbol = str(symbol).strip().upper()
    with repository.database.connect() as connection:
        persistent_state, _ = read_state(connection, symbol)
    exit_pending = bool(persistent_state and persistent_state.get("status") == "EXIT_PENDING")
    emitted_at = inference_at if inference_at is not None else datetime.now(timezone.utc)
    actionable_at_emission = is_actionable_emission(
        emitted_at, analysis.source_bar_closed_at,
    )
    context = portfolio_risk_context(repository, symbol, analysis.last_price)
    # Forward-resolved, executable entries are a second independent gate.
    # A historical artifact alone cannot promote sparse live evidence to BUY.
    validation_counts = (
        validation_counts if validation_counts is not None
        else repository.operational_validation_counts(symbol)
    )
    activation_checks = build_long_activation_checklist(analysis, zone_snapshot)
    if exit_pending:
        activation_checks = (*activation_checks, ActivationCheck(
            "persistent_exit", "Salida persistente resuelta", False,
            "EXIT_PENDING: confirmar el fill real antes de evaluar una nueva entrada.",
        ))
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
    feature_contract_error = ""
    if not has_position and not risk_veto and not exit_pending and actionable_at_emission:
        models = (
            operational_models if operational_models is not None else
            latest_approved_operational_models(symbol, analysis.source_bar_closed_at)
        )
        # Treat stale/hypothetical-only records as absent evidence, not as a
        # pending price trigger. The registry gate also covers injected models.
        models = {
            label: row for label, row in models.items()
            if _approved_result(row)
            and int(validation_counts.get(label, {}).get("eligible", 0)) >= PROFESSIONAL_MINIMUM_SAMPLES
        }
        try:
            opportunities = _operational_opportunities(
                analysis, context, models, risk_fraction, minimum_reward_risk,
                cost_policy or TradingCostPolicy(), fill_policy or FillModel(),
            )
        except FeatureContractMismatch as exc:
            feature_contract_error = str(exc)
            opportunities = ()
    selection = select_highest_net_expectation(opportunities)
    model_record = models.get(selection.horizon) if selection is not None else None
    direction = "LONG" if has_position or selection is not None else "NEUTRAL"
    horizon = selection.horizon if selection is not None else "Sin horizonte validado"
    reasons = []
    if selection is None and not has_position and not exit_pending:
        if feature_contract_error:
            reasons.append(f"Error explícito de paridad train/live: {feature_contract_error}")
        elif not models:
            available = max(
                (int(item.get("eligible", 0)) for item in validation_counts.values()),
                default=0,
            )
            reasons.append(
                f"Falta de evidencia: n={available}/{PROFESSIONAL_MINIMUM_SAMPLES}, "
                "modelo no aprobado; sin evidencia de entradas ejecutables suficiente "
                "para validar un modelo TP/SL/timeout. Los toques de barreras hipotéticas no autorizan compras."
            )
        elif not opportunities:
            reasons.append("No existe un plan LONG compatible con las barreras del modelo operativo.")
        else:
            reasons.append("Ningún horizonte ofrece expectativa neta positiva, R:R neto y tamaño admisible.")
            reasons.extend(f"{item.horizon}: {item.reason}" for item in opportunities)

    if selection is not None:
        entry_low = entry_high = selection.entry
        stop, target = selection.stop, selection.take_profit
        reward_risk = selection.net_reward_risk
    else:
        entry_low = entry_high = stop = target = reward_risk = None
    # The regime's exposure permission is a hard cap, not merely a label in
    # the report. Apply it before checking whether an entry can be emitted.
    exposure_factor = 1.0
    if analysis.macro_permission == "BOTH_REDUCED":
        declared = float(getattr(analysis, "exposure_factor", 0.25))
        exposure_factor = min(0.25, declared) if math.isfinite(declared) and declared > 0 else 0.0
    proposed_size = math.floor(selection.shares * exposure_factor) if selection is not None else 0
    active_long_plan = (
        analysis.execution_levels
        if getattr(analysis.execution_levels, "direction", "") == "LONG"
        else analysis.buy_levels
    )
    if exit_pending:
        # Persistent state > latest price signal: an intrabar stop/TP touch or
        # higher-timeframe invalidation cannot be undone by a recovered close.
        # Only a real fill reconciled in the ledger releases EXIT_PENDING.
        frozen_levels = persistent_state.get("levels")
        active_long_plan = None
        if frozen_levels:
            try:
                active_long_plan = ExecutionLevels(**frozen_levels)
            except (TypeError, ValueError):
                active_long_plan = None
        management = str(persistent_state.get("management") or "Salida pendiente de verificación.")
        if has_position:
            action = "CONFIRMAR_SALIDA"
            reasons.append(management)
            reasons.append("Verifica fill, cantidad y posible gap en GBM+; no se registra una venta automática.")
        else:
            action = "ESPERAR"
            reasons.append("La salida figura pendiente, pero el libro ya no muestra acciones; reconciliar el estado antes de otra entrada.")
    elif (has_position and active_long_plan is not None
          and current_price <= float(active_long_plan.stop_loss)):
        action = "VENDER"
        reasons.append("El precio perforó el stop del plan persistente de la posición.")
    elif (has_position and active_long_plan is not None
          and current_price >= float(active_long_plan.take_profit_1)):
        action = "VENDER"
        reasons.append("El precio alcanzó el take profit del plan persistente de la posición.")
    elif has_position:
        action = "MANTENER" if actionable_at_emission else "NO_ACCIONABLE"
        reasons.append("La posición sigue entre stop y objetivo sin invalidación confirmada.")
        if not actionable_at_emission:
            reasons.append(f"Fuera del corte temporal {ANCHOR_VERSION}; no se emite nueva señal de mantenimiento o entrada.")
        if context.concentration > 0.30:
            reasons.append("No aumentar: la posición ya supera 30% del portafolio.")
        if risk_veto:
            reasons.append("Veto activo: no abrir ni ampliar exposición; revisar la posición.")
    elif not actionable_at_emission:
        # Only the preregistered 11:00 NY cut can authorize a NEW entry.
        # This does not suppress EXIT_PENDING or immediate stop/TP management.
        action = "NO_ACCIONABLE"
        reasons.append(
            f"Fuera del corte temporal {ANCHOR_VERSION}: la señal no autoriza una entrada."
        )
        if risk_veto:
            reasons.append("Veto de riesgo técnico, fundamental, noticioso o de evento activo.")
    elif risk_veto:
        action = "ESPERAR"
        reasons.append("Veto de riesgo técnico, fundamental, noticioso o de evento activo.")
    else:
        gates = {
            "modelo TP/SL/timeout aprobado y EV neta positiva": selection is not None,
            "300 entradas ejecutables validadas en el horizonte": (
                selection is not None and int(validation_counts.get(selection.horizon, {}).get("eligible", 0))
                >= PROFESSIONAL_MINIMUM_SAMPLES
            ),
            "régimen LONG permitido": analysis.macro_permission in {"LONG_ONLY", "BOTH_REDUCED"},
            "gatillo confirmado": bool(analysis.activation_trigger_met),
            "plan no condicional": not bool(getattr(analysis, "execution_plan_conditional", True)),
            "señal no rechazada": not bool(getattr(analysis, "signal_rejected", False)),
            "R:R neto mínimo": selection is not None and reward_risk + 1e-9 >= minimum_reward_risk,
            "tamaño permitido tras reducir exposición": proposed_size > 0,
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
        reward_risk = _rr(active_long_plan.direction, entry_high, stop, target)

    size = proposed_size if selection is not None and action == "COMPRAR" else 0
    monetary_risk = size * -selection.worst_sl_per_share if size else 0.0
    ev_per_share = selection.net_ev_per_share if selection is not None else None
    total_ev = ev_per_share * size if size else None
    action_context = (
        f"Horizonte elegido por EV neta: {horizon}." if selection is not None else
        f"Sesgo preliminar más fuerte: {preliminary_bias.lower()} · {preliminary.label}."
    )
    rr_text = "N/D" if reward_risk is None else f"{reward_risk:.2f}"
    ev_text = "N/D (muestra insuficiente)" if ev_per_share is None else f"${ev_per_share:+.2f} por acción"
    explanation = (
        f"Acción actual: {action}. {action_context} R:R {'neto ' if selection else ''}{rr_text}; "
        f"EV neta observada/realista {ev_text}. "
        + " ".join(reasons)
    )
    metrics = (model_record or {}).get("final_holdout", {}).get("metrics") or {}
    waiting_cause = ""
    if action == "ESPERAR":
        waiting_cause = (
            "VETO_RIESGO" if risk_veto or exit_pending else
            "CONTRATO_FEATURES" if feature_contract_error else
            "FALTA_EVIDENCIA" if not has_position and not models else
            "GATILLO_PENDIENTE"
        )
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
        theoretical_expected_value_per_share=(selection.theoretical_net_ev_per_share if selection else None),
        theoretical_expected_value_total=(selection.theoretical_net_ev_per_share * size if size else None),
        observed_expected_value_per_share=ev_per_share,
        observed_expected_value_total=total_ev,
        fill_probability=(selection.fill_probability if selection else None),
        observed_sl_samples=(selection.observed_sl_samples if selection else 0),
        spread_bps=(selection.spread_bps if selection else None),
        exposure_factor_applied=exposure_factor,
        adjusted_win_probability=(selection.tp_first if selection else None),
        directional_brier=None,
        directional_baseline_brier=None,
        brier_touch=None, brier_close=None,
        validated_sessions=int(metrics.get("samples") or 0),
        trigger_met=bool(analysis.activation_trigger_met) and not exit_pending and actionable_at_emission,
        risk_veto=risk_veto or exit_pending,
        activation_checks=activation_checks,
        reasons=tuple(reasons), explanation=explanation,
        operational_brier=metrics.get("brier"),
        operational_baseline_brier=metrics.get("baseline_brier"),
        sl_first_probability=(selection.sl_first if selection else None),
        timeout_probability=(selection.timeout if selection else None),
        cost_rate_per_side=(selection.cost_rate_per_side if selection else None),
        anchor_version=ANCHOR_VERSION,
        actionable_at_emission=actionable_at_emission,
        waiting_cause=waiting_cause,
        calibration_status=(
            "Histórico OOS calibrado; pendiente de validación forward"
            if selection is not None else
            "EVIDENCIA INSUFICIENTE · sin evidencia de entradas ejecutables"
            if waiting_cause == "FALTA_EVIDENCIA" else evidence.status
        ),
        recommendation_mode="VALIDADO" if selection is not None else "CONSERVADOR",
    )
