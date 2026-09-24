"""Compatibility adapter for the shared causal core; no duplicated rules."""
from dataclasses import replace
from .causal_core import (
    Permission, Regime, Setup, Trigger, RegimeEngine, SetupEngine, TriggerEngine,
    adx_hysteresis, trend, evaluate_causal_core,
)


def apply_hierarchy(analysis, previous_trending=False):
    """Final authorization layer. Calibration can never override this gate."""
    decision = evaluate_causal_core(
        weekly=analysis.weekly_indicators, daily=analysis.daily_indicators,
        four_hour=analysis.four_hour_indicators, hourly=analysis.hourly_indicators,
        intraday=analysis.intraday_indicators, previous_trending=previous_trending,
    )
    regime, setup, trigger = decision.regime, decision.setup, decision.trigger
    same_direction = trigger.direction == analysis.execution_levels.direction
    allowed = trigger.activated and same_direction and not analysis.risk_veto and not analysis.signal_rejected
    directional_score = analysis.probability_up if trigger.direction == "LONG" else analysis.probability_down
    reason = f"{regime.detail} {setup.detail}"
    if regime.permission == Permission.NO_TRADE:
        market_regime = "NO_TRADE"
    elif regime.trending:
        market_regime = "TREND"
    else:
        daily_adx = float(analysis.daily_indicators.ADX14.iloc[-1])
        market_regime = "RANGE" if daily_adx < 20 else "TRANSITION"
    return replace(analysis, macro_permission=regime.permission.value, macro_trending=regime.trending,
                   structural_support=setup.support, structural_resistance=setup.resistance,
                   activation_trigger=trigger.detail, activation_trigger_met=allowed,
                   signal=type(analysis.signal)("BUY" if trigger.direction == "LONG" else "SELL") if allowed else analysis.signal,
                   operation_probability=directional_score if allowed else 0.0,
                   execution_plan_conditional=not allowed or directional_score < 65,
                   execution_plan_label="GATILLO CERRADO · respetar umbral de riesgo" if allowed else "PLAN CONDICIONAL · jerarquía mayor / gatillo pendiente",
                   market_regime=market_regime, hierarchy_detail=reason,
                   exposure_factor=min(analysis.exposure_factor, 0.25) if regime.permission == Permission.BOTH_REDUCED else analysis.exposure_factor)
