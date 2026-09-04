"""Native compact context panel; no custom styling or navigation changes."""
import streamlit as st

from ..services.cross_asset_presentation import cross_asset_rows


def render_cross_asset(analysis):
    context = getattr(analysis, "cross_asset_context", {})
    if not context:
        return
    with st.container(border=True):
        st.markdown(f"**Correlación · Relación con {context.get('peer', 'activo par')}**")
        if context.get("status") == "unavailable":
            st.caption(context["detail"])
            return
        st.table([{"Lectura": key, "Valor": value} for key, value in cross_asset_rows(context)])
        for alert in context.get("alerts", []):
            st.caption(alert)
        st.caption(context["detail"])
        st.caption("Correlación no implica causalidad ni autoriza entradas. "
                   "El ajuste no modifica probabilidades de alcance, stops ni objetivos.")
