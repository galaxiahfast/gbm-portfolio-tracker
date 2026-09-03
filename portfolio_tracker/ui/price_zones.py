"""Presentation of the shared read-only zone snapshot."""
from typing import TYPE_CHECKING
import streamlit as st
from portfolio_tracker.services.price_zones import (
    DisplayZone, build_zone_lists, distance_to_zone, price_location,
    _positive, build_zone_snapshot,
)
if TYPE_CHECKING:
    from portfolio_tracker.analytics.technical_probability import ProbabilityAnalysis


def _render_list(zones, price, estimates):
    for zone, estimate in zip(zones, estimates):
        level = "Sin nivel disponible" if zone.low is None else (
            f"${zone.low:,.2f}" if zone.low == zone.high else f"${zone.low:,.2f} – ${zone.high:,.2f}")
        st.markdown(f"**{zone.label}**  \n{level}")
        distance = distance_to_zone(price, zone)
        proximity = "Distancia N/D" if distance is None else (
            "En zona" if distance[0] == 0 else f"Distancia {distance[0]:+,.2f} USD ({distance[1]:+.2f}%)")
        if estimate.model.startswith('conditional-'):
            touch = 'N/D' if estimate.probability is None else f'{estimate.probability:.0f}%'
            close = 'N/D' if estimate.close_probability is None else f'{estimate.close_probability:.0f}%'
            direction = 'debajo' if estimate.close_direction == 'BELOW' else 'encima'
            st.markdown(f'**Probabilidad estimada de toque hoy: {touch}**')
            st.caption(f'Probabilidad de cierre {direction}: {close}')
            if estimate.probability is not None:
                st.caption(f'IC 95% toque {estimate.lower:.0f}–{estimate.upper:.0f}% · cierre {estimate.close_lower:.0f}–{estimate.close_upper:.0f}%')
            effective = '' if estimate.effective_samples is None else f' ({estimate.effective_samples:.1f} efectivas)'
            st.caption(f'{estimate.samples} sesiones{effective} · {estimate.status}')
        elif estimate.probability is None:
            st.caption(f"Alcance hoy: N/D · {estimate.status}")
        else:
            st.caption(f"Probabilidad estimada de alcance hoy: {estimate.probability:.0f}%")
            st.caption(f"IC 95%: {estimate.lower:.0f}–{estimate.upper:.0f}% · {estimate.samples} sesiones · {estimate.status}")
        st.caption(proximity)
        st.caption(zone.source)


def render_price_zones(analysis: "ProbabilityAnalysis", zone_snapshot=None) -> None:
    """Shared presentation with read-only reach estimates; no execution writes."""
    snapshot = zone_snapshot if zone_snapshot is not None else build_zone_snapshot(analysis)
    buys, sales, estimates = snapshot.buys, snapshot.sales, snapshot.estimates
    with st.container(key="quant_zone_lists"):
        price = _positive(analysis.last_price)
        with st.container(key="quant_three_panels"):
            current, left, right = st.columns(3, gap="small", vertical_alignment="top", border=True, wrap=False)
            with current:
                st.markdown("**Precio actual**")
                st.markdown(f"{analysis.symbol}  \n" + (f"${price:,.2f} USD" if price else "No disponible"))
                st.caption(f"Corte: {analysis.as_of:%d/%m/%Y %H:%M %Z}")
                st.caption("Último cierre de 5m disponible. Se actualiza con el motor; no es una cotización tick a tick.")
                if estimates and estimates[0].model.startswith('conditional-'):
                    st.caption(estimates[0].detail)
                    st.caption('Cierre = cierre final de hoy más allá de la zona. No equivale a una señal de compra o venta.')
            with left:
                st.markdown("**Zona de compra / Entrada ideal**")
                _render_list(buys, price, estimates[:3])
            with right:
                st.markdown("**Zona de venta / Objetivos y resistencia**")
                _render_list(sales, price, estimates[3:])
        # Keep display calculations intact, but never enqueue the removed UI.
        # Rendering then clearing a placeholder can briefly expose it on reruns.
        location = price_location(price, buys + sales)
        if location is not None:
            low, high, ratio, status = location
            distances = [(z, distance_to_zone(price, z)) for z in buys + sales if z.low is not None]
            zone, distance = min(distances, key=lambda item: abs(item[1][0]))
