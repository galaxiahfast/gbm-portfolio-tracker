"""Presentation of the shared read-only zone snapshot."""
import math
from typing import TYPE_CHECKING
import streamlit as st
from portfolio_tracker.services.price_zones import (
    DisplayZone, build_zone_lists, distance_to_zone, price_location,
    _positive, build_zone_snapshot, market_session_status,
    projected_extended_levels,
)
if TYPE_CHECKING:
    from portfolio_tracker.analytics.technical_probability import ProbabilityAnalysis


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _formatted(value, pattern, fallback="N/D"):
    number = _finite(value)
    if number is None:
        return fallback
    try:
        return format(number, pattern)
    except (TypeError, ValueError):
        return fallback


def _confidence_text(estimate, *, include_close):
    try:
        bounds = [
            _finite(getattr(estimate, "lower", None)),
            _finite(getattr(estimate, "upper", None)),
        ]
        if include_close:
            bounds.extend([
                _finite(getattr(estimate, "close_lower", None)),
                _finite(getattr(estimate, "close_upper", None)),
            ])
        available = bool(getattr(estimate, "confidence_available", False)) and all(
            value is not None for value in bounds
        )
        if not available:
            return "IC 95%: N/D (muestra insuficiente)"
        if include_close:
            return (
                f"IC 95% toque {_formatted(bounds[0], '.0f')}–{_formatted(bounds[1], '.0f')}% · "
                f"cierre {_formatted(bounds[2], '.0f')}–{_formatted(bounds[3], '.0f')}%"
            )
        return f"IC 95%: {_formatted(bounds[0], '.0f')}–{_formatted(bounds[1], '.0f')}%"
    except (AttributeError, TypeError, ValueError):
        return "IC 95%: N/D (muestra insuficiente)"


def _render_list(zones, price, estimates):
    for zone, estimate in zip(zones, estimates):
        low, high = _positive(zone.low), _positive(zone.high)
        level = "Sin nivel disponible" if low is None or high is None else (
            f"${_formatted(low, ',.2f')}" if low == high
            else f"${_formatted(low, ',.2f')} – ${_formatted(high, ',.2f')}")
        st.markdown(f"**{zone.label}**  \n{level}")
        distance = distance_to_zone(price, zone)
        proximity = "Distancia N/D" if distance is None else (
            "En zona" if distance[0] == 0 else (
                f"Distancia {_formatted(distance[0], '+,.2f')} USD "
                f"({_formatted(distance[1], '+.2f')}%)"
            ))
        if estimate is None:
            st.caption("Alcance hoy: N/D · Estimación no disponible")
            st.caption(proximity)
            st.caption(str(zone.source or "Sin procedencia disponible"))
            continue
        model = str(getattr(estimate, "model", "") or "")
        if model.startswith(('conditional-', 'dynamic-')):
            touch_value = _finite(getattr(estimate, "probability", None))
            close_value = _finite(getattr(estimate, "close_probability", None))
            touch = 'N/D' if touch_value is None else f'{_formatted(touch_value, ".0f")}%'
            close = 'N/D' if close_value is None else f'{_formatted(close_value, ".0f")}%'
            direction = 'debajo' if getattr(estimate, 'close_direction', '') == 'BELOW' else 'encima'
            st.markdown(f'**Probabilidad estimada de toque hoy: {touch}**')
            st.caption(f'Probabilidad de cierre {direction}: {close}')
            st.caption(_confidence_text(estimate, include_close=True))
            effective_value = _finite(getattr(estimate, "effective_samples", None))
            effective = '' if effective_value is None else f' ({_formatted(effective_value, ".1f")} efectivas)'
            samples = getattr(estimate, "samples", 0)
            status = str(getattr(estimate, "status", "Estimación no disponible") or "Estimación no disponible")
            st.caption(f'{samples} sesiones{effective} · {status}')
        elif _finite(getattr(estimate, "probability", None)) is None:
            st.caption(f"Alcance hoy: N/D · {getattr(estimate, 'status', 'Estimación no disponible')}")
        else:
            probability = _finite(getattr(estimate, "probability", None))
            st.caption(f"Probabilidad estimada de alcance hoy: {_formatted(probability, '.0f')}%")
            interval = _confidence_text(estimate, include_close=False)
            st.caption(
                f"{interval} · {getattr(estimate, 'samples', 0)} sesiones · "
                f"{getattr(estimate, 'status', 'Estimación no disponible')}"
            )
        st.caption(proximity)
        st.caption(str(zone.source or "Sin procedencia disponible"))


def render_price_zones(analysis: "ProbabilityAnalysis", zone_snapshot=None) -> None:
    """Shared presentation with read-only reach estimates; no execution writes."""
    snapshot = zone_snapshot if zone_snapshot is not None else build_zone_snapshot(analysis)
    buys, sales, estimates = snapshot.buys, snapshot.sales, snapshot.estimates
    with st.container(key="quant_zone_lists"):
        price = _positive(analysis.last_price)
        session = market_session_status()
        if not session.is_open:
            st.info(session.message, icon=":material/schedule:")
        with st.container(key="quant_three_panels"):
            current, left, right = st.columns(3, gap="small", vertical_alignment="top", border=True, wrap=False)
            with current:
                st.markdown("**Precio actual**")
                st.markdown(f"{analysis.symbol}  \n" + (f"${price:,.2f} USD" if price else "No disponible"))
                try:
                    cut = analysis.as_of.strftime("%d/%m/%Y %H:%M %Z")
                except (AttributeError, TypeError, ValueError):
                    cut = "N/D"
                st.caption(f"Corte: {cut}")
                st.caption("Último cierre de 5m disponible. Se actualiza con el motor; no es una cotización tick a tick.")
                first_model = str(getattr(estimates[0], "model", "") or "") if estimates else ""
                if first_model.startswith(('conditional-', 'dynamic-')):
                    st.caption(str(getattr(estimates[0], "detail", "") or "Sin detalle estadístico disponible"))
                    st.caption('Cierre = cierre final de hoy más allá de la zona. No equivale a una señal de compra o venta.')
            with left:
                st.markdown("**Zona de compra / Entrada ideal**")
                _render_list(buys, price, estimates[:3])
            with right:
                st.markdown("**Zona de venta / Objetivos y resistencia**")
                _render_list(sales, price, estimates[3:])
        extended = tuple(getattr(snapshot, "extended_levels", ()) or ())
        if not extended:
            extended = projected_extended_levels(analysis, snapshot)
        if extended:
            heading = (
                "Soportes extendidos (Proyectados)"
                if extended[0].direction == "BELOW"
                else "Niveles extendidos (Proyectados)"
            )
            with st.container(border=True, key="quant_extended_levels"):
                st.markdown(f"**{heading}**")
                st.caption(
                    "Referencias visuales calculadas en tiempo real. No pertenecen al plan original, "
                    "no constituyen una señal y no se guardan en el registro forward."
                )
                columns = st.columns(len(extended), gap="small")
                for column, level in zip(columns, extended):
                    with column:
                        st.markdown(f"**{level.label}**")
                        st.markdown(f"${level.price:,.2f}")
                        st.caption(level.source)
        # Keep display calculations intact, but never enqueue the removed UI.
        # Rendering then clearing a placeholder can briefly expose it on reruns.
        location = price_location(price, buys + sales)
        if location is not None:
            low, high, ratio, status = location
            distances = [(z, distance_to_zone(price, z)) for z in buys + sales if z.low is not None]
            zone, distance = min(distances, key=lambda item: abs(item[1][0]))
