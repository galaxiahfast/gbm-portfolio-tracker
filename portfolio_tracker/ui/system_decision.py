"""Compact Streamlit presentation for the read-only system recommendation."""
from __future__ import annotations

import streamlit as st


def render_validation_banner(status, slot=None):
    """Keep the statistical caveat visible above the predictor tabs."""
    target = slot if slot is not None else st
    if status["preliminary"]:
        target.warning(str(status["banner"]), icon=":material/shield:")
    else:
        target.info(str(status["banner"]), icon=":material/verified:")


def render_system_decision(decision):
    with st.container(border=True, key="system_decision"):
        st.markdown("### DECISIÓN DEL SISTEMA")
        if getattr(decision, "recommendation_mode", "CONSERVADOR") == "CONSERVADOR":
            st.caption("PRELIMINAR · Modo CONSERVADOR: no hay compra autorizada en este corte. "
                       "Se exigen 300 entradas ejecutables verificadas y un modelo aprobado por horizonte.")
        if decision.action == "ESPERAR":
            reason = {
                "VETO_RIESGO": "VETO DE RIESGO",
                "FALTA_EVIDENCIA": "FALTA DE EVIDENCIA",
                "CONTRATO_FEATURES": "ERROR DE PARIDAD DE FEATURES",
                "GATILLO_PENDIENTE": "GATILLO PENDIENTE",
            }.get(getattr(decision, "waiting_cause", ""),
                  "VETO DE RIESGO" if decision.risk_veto else "GATILLO PENDIENTE")
            headline = f"ESPERAR AHORA · {reason}"
        else:
            headline = decision.action.replace("_", " ")
        message = f"**{headline}**  \n{decision.explanation}"
        if decision.action == "COMPRAR":
            st.success(message, icon=":material/trending_up:")
        elif decision.action in {"VENDER", "CONFIRMAR_SALIDA"}:
            st.error(message, icon=":material/trending_down:")
        elif decision.action == "MANTENER":
            st.info(message, icon=":material/pause_circle:")
        else:
            st.warning(message, icon=":material/schedule:")
        first, second, third, fourth = st.columns(4, gap="small")
        first.metric(
            "Entrada",
            "N/D" if decision.entry_low is None else
            f"${decision.entry_low:,.2f}–${decision.entry_high:,.2f}",
        )
        second.metric("Stop loss", "N/D" if decision.stop_loss is None else f"${decision.stop_loss:,.2f}")
        third.metric("Take profit", "N/D" if decision.take_profit is None else f"${decision.take_profit:,.2f}")
        fourth.metric("R:R", "N/D" if decision.reward_risk is None else f"{decision.reward_risk:.2f}")
        capital, position, risk = st.columns(3, gap="small")
        capital.metric("Capital / efectivo", f"${decision.total_capital:,.2f} / ${decision.cash_available:,.2f}")
        position.metric(
            "Posición actual / nueva",
            f"{decision.current_shares:g} / {decision.position_size:d} acciones",
            help="La segunda cifra solo es distinta de cero cuando COMPRAR está autorizado.",
        )
        risk.metric(
            "Riesgo máximo estimado / EV realista",
            f"${decision.monetary_risk:,.2f} / "
            + ("N/D" if decision.expected_value_total is None else f"${decision.expected_value_total:+,.2f}"),
            help=f"Riesgo dimensionado con la peor salida SL observada. Presupuesto: ${decision.risk_budget:,.2f}.",
        )
        observed_column, theoretical_column = st.columns(2, gap="small")
        observed_column.metric(
            "EV neta observada/realista · por acción",
            "N/D" if decision.observed_expected_value_per_share is None
            else f"${decision.observed_expected_value_per_share:+,.2f}",
            help="Usa salidas SL observadas en desarrollo, spread, deslizamiento y posibilidad de no ejecución.",
        )
        theoretical_column.metric(
            "EV neta teórica · por acción",
            "N/D" if decision.theoretical_expected_value_per_share is None
            else f"${decision.theoretical_expected_value_per_share:+,.2f}",
            help="Supone ejecución y salida exacta en el stop; solo sirve como referencia.",
        )
        if decision.adjusted_win_probability is None:
            st.caption(
                "TP primero / SL primero / timeout: N/D · sin modelo operativo "
                "calibrado y aprobado. Los scores direccionales no autorizan la compra."
            )
        else:
            brier = (
                "N/D" if decision.operational_brier is None
                else f"{decision.operational_brier:.4f}"
            )
            baseline = (
                "N/D" if decision.operational_baseline_brier is None
                else f"{decision.operational_baseline_brier:.4f}"
            )
            st.caption(
                f"TP primero {decision.adjusted_win_probability:.1%} · "
                f"SL primero {decision.sl_first_probability:.1%} · "
                f"timeout {decision.timeout_probability:.1%} · "
                f"Brier operativo OOS {brier} · baseline {baseline} · "
                f"{decision.validated_sessions} muestras holdout. "
                f"Costes estimados por lado {decision.cost_rate_per_side:.2%}; "
                f"{decision.observed_sl_samples} salidas SL observadas en desarrollo; "
                f"fill supuesto {decision.fill_probability:.0%}; "
                f"spread supuesto {decision.spread_bps:.1f} pb. "
                "Timeout estresado a la peor salida SL observada. "
                f"{decision.calibration_status}. "
                "Un gap futuro puede superar incluso la peor pérdida histórica."
            )
        st.markdown("**Condiciones de activación LONG**")
        st.table([
            {
                "Condición": item.label,
                "Estado": "Sí" if item.passed else "No",
                "Detalle": item.detail,
            }
            for item in decision.activation_checks
        ])
        st.caption(
            f"Acción ahora: {decision.action} · Sesgo descriptivo: "
            f"{decision.preliminary_bias} ({decision.preliminary_horizon}). "
            "Recomendación informativa; ejecución manual."
        )
