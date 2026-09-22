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
            "Riesgo / expectativa neta",
            f"${decision.monetary_risk:,.2f} / "
            + ("N/D" if decision.expected_value_total is None else f"${decision.expected_value_total:+,.2f}"),
            help=f"Presupuesto máximo de riesgo: ${decision.risk_budget:,.2f}.",
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
                "timeout valorado conservadoramente al stop. "
                "Un gap puede exceder la pérdida estimada."
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
