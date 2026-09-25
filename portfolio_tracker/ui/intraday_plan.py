"""Minimal Streamlit presentation of the signed, sequential session plan."""
from __future__ import annotations

import streamlit as st

from portfolio_tracker.services.intraday_plan import IntradayPlan


def render_intraday_plan(plan: IntradayPlan) -> None:
    with st.container(border=True, key="quant_intraday_sequence"):
        st.markdown("**Plan secuencial de la sesión · PRELIMINAR**")
        if plan.entry is None:
            st.caption(plan.detail)
            return
        observed = plan.observed_at.tz_convert("America/New_York").strftime("%H:%M")
        st.caption(
            f"Niveles del contrato 1 día congelados al corte firmado {observed} NY; "
            "la secuencia visible observa solo hasta el cierre de esta sesión. "
            "Los cambios de 5 min no mueven entrada, stop ni objetivo."
        )
        cols = st.columns(4, gap="small")
        for column, title, value in zip(
            cols,
            ("Entrada condicional", "Objetivo posterior", "Stop posterior", "Estado"),
            (f"${plan.entry:,.2f}", f"${plan.target:,.2f}",
             f"${plan.stop:,.2f}", plan.status.replace("_", " ")),
        ):
            with column:
                st.caption(title)
                st.markdown(f"**{value}**")
        st.caption(plan.detail)
        if not plan.eligible_at_cut:
            st.caption("El corte no autorizó una entrada ejecutable. Un toque posterior no cambia ese permiso.")
        st.caption(
            "Probabilidad de entrada → objetivo antes de stop: N/D (no calibrada OOS). "
            "Los porcentajes de toque de las zonas son marginales y no indican el orden, "
            "el precio mínimo/máximo futuro ni una operación ejecutada. "
            "La recomendación automática y el control de riesgo siguen siendo independientes. "
            "Otra oportunidad requiere un nuevo corte prospectivo; no se reconstruye usando el máximo o mínimo ya visto."
        )
