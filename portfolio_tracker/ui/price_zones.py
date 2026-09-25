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


def _render_list(zones, price, estimates, *, compact=False):
    for index, zone in enumerate(zones):
        estimate = estimates[index] if index < len(estimates) else None
        low, high = _positive(zone.low), _positive(zone.high)
        level = "Sin nivel disponible" if low is None or high is None else (
            f"${_formatted(low, ',.2f')}" if low == high
            else f"${_formatted(low, ',.2f')} – ${_formatted(high, ',.2f')}")
        st.markdown(f"**{level}**" if compact else f"**{zone.label}**  \n{level}")
        if compact:
            touch = _finite(getattr(estimate, "probability", None))
            close = _finite(getattr(estimate, "close_probability", None))
            direction = "debajo" if getattr(estimate, "close_direction", "") == "BELOW" else "encima"
            st.markdown(f"**PRELIMINAR · Probabilidad estimada de toque hoy: {'N/D' if touch is None else f'{touch:.0f}%'}**")
            st.caption(f"PRELIMINAR · Probabilidad de cierre {direction}: {'N/D' if close is None else f'{close:.0f}%'}")
            continue
        distance = distance_to_zone(price, zone)
        proximity = "Distancia N/D" if distance is None else (
            "En zona" if distance[0] == 0 else (
                f"Distancia {_formatted(distance[0], '+,.2f')} USD "
                f"({_formatted(distance[1], '+.2f')}%)"
            ))
        if estimate is None:
            st.caption("PRELIMINAR · Alcance hoy: N/D · Estimación no disponible")
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
            st.markdown(f'**PRELIMINAR · Probabilidad estimada de toque hoy: {touch}**')
            st.caption(f'PRELIMINAR · Probabilidad de cierre {direction}: {close}')
            st.caption(_confidence_text(estimate, include_close=True))
            effective_value = _finite(getattr(estimate, "effective_samples", None))
            effective = '' if effective_value is None else f' ({_formatted(effective_value, ".1f")} efectivas)'
            samples = getattr(estimate, "samples", 0)
            status = str(getattr(estimate, "status", "Estimación no disponible") or "Estimación no disponible")
            st.caption(f'{samples} sesiones{effective} · {status}')
        elif _finite(getattr(estimate, "probability", None)) is None:
            st.caption(f"PRELIMINAR · Alcance hoy: N/D · {getattr(estimate, 'status', 'Estimación no disponible')}")
        else:
            probability = _finite(getattr(estimate, "probability", None))
            st.caption(f"PRELIMINAR · Probabilidad estimada de alcance hoy: {_formatted(probability, '.0f')}%")
            interval = _confidence_text(estimate, include_close=False)
            st.caption(
                f"{interval} · {getattr(estimate, 'samples', 0)} sesiones · "
                f"{getattr(estimate, 'status', 'Estimación no disponible')}"
            )
        st.caption(proximity)
        st.caption(str(zone.source or "Sin procedencia disponible"))


def _operational_action(analysis):
    """Translate existing engine state into one unambiguous UI action."""
    if analysis.position_state == "EXIT_PENDING":
        return "VENDER / PROTEGER CAPITAL", "danger", "La posición abierta tiene una salida pendiente."
    if analysis.position_state != "FLAT":
        return "GESTIONAR POSICIÓN ABIERTA", "warning", (
            "No abrir otra entrada por ruido de 5 minutos; gestionar stop, objetivo e invalidación estructural."
        )
    if analysis.risk_veto or analysis.signal_rejected:
        reason = analysis.risk_reasons[0] if analysis.risk_reasons else analysis.verdict
        return "NO COMPRAR · ESPERAR", "danger", str(reason)
    if (
        analysis.signal.value == "BUY"
        and analysis.activation_trigger_met
        and analysis.operation_probability >= 65
        and not analysis.long_entry_blocked
    ):
        return "COMPRA AUTORIZADA POR EL MOTOR", "success", (
            f"Gatillo cerrado y score operativo {analysis.operation_probability:.1f}/100."
        )
    if (
        analysis.signal.value == "SELL"
        and analysis.activation_trigger_met
        and analysis.operation_probability >= 65
    ):
        return "VENDER / NO ABRIR LONG", "danger", (
            f"Gatillo bajista cerrado y score operativo {analysis.operation_probability:.1f}/100."
        )
    return "ESPERAR · SIN ENTRADA AUTORIZADA", "warning", (
        "El precio puede moverse, pero el gatillo completo del motor todavía no autoriza una operación."
    )


def _operational_levels(analysis, snapshot):
    """Risk-aware visual plan based on current adaptive zones; never persisted."""
    price = _positive(analysis.last_price)
    atr = _positive(analysis.atr_5m)
    if price is None or atr is None or not snapshot.buys or not snapshot.sales:
        return None
    entry_zone = snapshot.buys[0]
    entry_low, entry_high = _positive(entry_zone.low), _positive(entry_zone.high)
    if entry_low is None or entry_high is None:
        return None
    stop_candidates = [entry_low - 2.25 * atr]
    if len(snapshot.buys) > 1 and _positive(snapshot.buys[1].low) is not None:
        stop_candidates.append(float(snapshot.buys[1].low) - 0.25 * atr)
    stop = max(0.01, min(stop_candidates))
    risk = max(entry_high - stop, 0.01)
    minimum_target = entry_high + 1.5 * risk
    technical_targets = sorted({
        float(zone.low) for zone in snapshot.sales
        if _positive(zone.low) is not None and float(zone.low) >= minimum_target
    })
    target = technical_targets[0] if technical_targets else minimum_target
    reward_risk = (target - entry_high) / risk
    return entry_low, entry_high, stop, target, reward_risk


def render_operational_signal(analysis: "ProbabilityAnalysis", zone_snapshot) -> None:
    """Put the buy/sell/wait decision before secondary touch probabilities."""
    if zone_snapshot is None:
        zone_snapshot = build_zone_snapshot(analysis)
    action, tone, reason = _operational_action(analysis)
    horizon = next(
        (item for item in analysis.horizon_projections if item.label == "6 Horas"),
        analysis.horizon_projections[0] if analysis.horizon_projections else None,
    )
    with st.container(border=True, key="quant_operational_signal"):
        if tone == "success":
            st.success(f"**SEÑAL ACTUAL: {action}**  \n{reason}", icon=":material/check_circle:")
        elif tone == "danger":
            st.error(f"**SEÑAL ACTUAL: {action}**  \n{reason}", icon=":material/block:")
        else:
            st.warning(f"**SEÑAL ACTUAL: {action}**  \n{reason}", icon=":material/schedule:")
        if horizon is not None:
            suffix = "%" if analysis.has_empirical_probability else "/100"
            columns = st.columns(3, gap="small")
            columns[0].metric(
                "Subida · próximas 6 horas",
                f"{horizon.probability_up:.1f}{suffix}",
                help=analysis.calibration_disclosure,
            )
            columns[1].metric(
                "Rango · próximas 6 horas",
                f"{horizon.probability_range:.1f}{suffix}",
                help=analysis.calibration_disclosure,
            )
            columns[2].metric(
                "Bajada · próximas 6 horas",
                f"{horizon.probability_down:.1f}{suffix}",
                help=analysis.calibration_disclosure,
            )
        levels = _operational_levels(analysis, zone_snapshot)
        if levels is not None:
            entry_low, entry_high, stop, target, reward_risk = levels
            st.markdown(
                f"**Plan LONG condicionado:** comprar únicamente después de que se cumpla el gatillo, "
                f"en `${entry_low:,.2f}–{entry_high:,.2f}` · stop `${stop:,.2f}` · "
                f"objetivo técnico mínimo `${target:,.2f}` · R:R `{reward_risk:.2f}`."
            )
        st.markdown(f"**Gatillo vigente:** {analysis.activation_trigger}")
        st.caption(
            f"Estado: {'CUMPLIDO' if analysis.activation_trigger_met else 'PENDIENTE'} · "
            f"Régimen: {analysis.macro_permission} · Exposición relativa: {analysis.exposure_factor:.2f}x. "
            "Las lecturas son preliminares mientras no exista calibración OOS suficiente."
        )


def render_price_zones(analysis: "ProbabilityAnalysis", zone_snapshot=None, *, reference_only=False) -> None:
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
                if reference_only:
                    st.markdown(f"${price:,.2f} USD" if price else "No disponible")
                else:
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
                st.markdown("**Bajada**" if reference_only else "**Zona de compra / Entrada ideal**")
                _render_list(buys, price, estimates[:3], compact=reference_only)
            with right:
                st.markdown("**Subida**" if reference_only else "**Zona de venta / Objetivos y resistencia**")
                _render_list(sales, price, estimates[3:], compact=reference_only)
        extended = tuple(getattr(snapshot, "extended_levels", ()) or ())
        if not extended:
            extended = projected_extended_levels(analysis, snapshot)
        sale_prices = {
            round(value, 2)
            for zone in sales
            for value in (_positive(zone.low), _positive(zone.high))
            if value is not None
        }
        extended = tuple(level for level in extended if round(level.price, 2) not in sale_prices)
        if extended and not reference_only:
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
