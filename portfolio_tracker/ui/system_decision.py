"""Compact Streamlit presentation for the read-only system recommendation."""
from __future__ import annotations

import streamlit as st


def render_system_decision(decision):
    with st.container(border=True, key="system_decision"):
        st.markdown("### DECISIÓN DEL SISTEMA")
        if decision.action == "ESPERAR":
            reason = "VETO DE RIESGO" if decision.risk_veto else "GATILLO PENDIENTE"
            headline = f"ESPERAR AHORA · {reason}"
        else:
            headline = decision.action
        message = f"**{headline}**  \n{decision.explanation}"
        if decision.action == "COMPRAR":
            st.success(message, icon=":material/trending_up:")
        elif decision.action == "VENDER":
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
            "Riesgo / expectativa",
            f"${decision.monetary_risk:,.2f} / "
            + ("N/D" if decision.expected_value_total is None else f"${decision.expected_value_total:+,.2f}"),
            help=f"Presupuesto máximo de riesgo: ${decision.risk_budget:,.2f}.",
        )
        probability = (
            "N/D · evidencia insuficiente"
            if decision.adjusted_win_probability is None
            else f"{decision.adjusted_win_probability:.1%}"
        )
        st.caption(
            f"Probabilidad direccional ajustada: {probability} · Brier direccional OOS: "
            f"{'N/D' if decision.directional_brier is None else f'{decision.directional_brier:.4f}'} · "
            f"Baseline: {'N/D' if decision.directional_baseline_brier is None else f'{decision.directional_baseline_brier:.4f}'} · "
            f"{decision.validated_sessions} muestras holdout. {decision.calibration_status}. "
            "El Brier de zonas no interviene en esta decisión."
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
