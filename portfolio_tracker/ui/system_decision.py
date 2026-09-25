"""Compact Streamlit presentation for the read-only system recommendation."""
from __future__ import annotations

import math

import streamlit as st


def _positive_or_none(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def render_validation_banner(status, slot=None):
    """Keep the statistical caveat visible above the predictor tabs."""
    target = slot if slot is not None else st
    if status["preliminary"]:
        target.warning(str(status["banner"]), icon=":material/shield:")
    else:
        target.info(str(status["banner"]), icon=":material/verified:")


def _visible_levels(decision, intraday_plan=None):
    """Choose one reference: the real holding, or one prospective entry.

    Presentation never converts a hypothetical zone touch into a ledger fill.
    The decision engine and its risk gates remain the source of authorization.
    """
    holding = float(decision.current_shares) > 0
    if holding:
        entry = decision.average_price
        stop = decision.stop_loss
        target = decision.take_profit
        entry_label = "Compra registrada · promedio"
        stop_label = "Stop de la posición"
        target_label = "Salida objetivo"
        rr_label = "R:R desde compra"
        note = "Posición registrada: no se muestra otra entrada. Verifica la ejecución real de cualquier salida en GBM+."
    elif decision.action == "COMPRAR":
        entry = decision.entry_high if decision.entry_high is not None else decision.entry_low
        stop = decision.stop_loss
        target = decision.take_profit
        entry_label = "Única entrada autorizada"
        stop_label = "Stop del plan"
        target_label = "Objetivo del plan"
        rr_label = "R:R del plan"
        note = "Entrada sujeta a la decisión vigente y a ejecución manual en GBM+."
    elif (intraday_plan is not None and intraday_plan.entry is not None
          and intraday_plan.status in {"ESPERAR_ENTRADA", "VERIFICAR_GATILLO"}):
        entry = intraday_plan.entry
        stop = intraday_plan.stop
        target = intraday_plan.target
        entry_label = "Única entrada condicional"
        stop_label = "Stop si se ejecuta"
        target_label = "Objetivo si se ejecuta"
        rr_label = "R:R del plan"
        note = (
            f"Corte firmado: {intraday_plan.status.replace('_', ' ').lower()}. "
            "Es una referencia prospectiva, no un fill ni autorización de compra."
        )
    else:
        entry = stop = target = None
        entry_label = "Entrada disponible"
        stop_label = "Stop"
        target_label = "Objetivo"
        rr_label = "R:R"
        note = (
            "La oportunidad del corte ya terminó; esperar un nuevo corte prospectivo."
            if intraday_plan is not None and intraday_plan.status in {"OBJETIVO_OBSERVADO", "STOP_OBSERVADO"}
            else "Sin una entrada autorizada ni un corte vigente evaluable; no se inventa un precio de compra."
        )
    entry, stop, target = map(_positive_or_none, (entry, stop, target))
    valid = all(value is not None for value in (entry, stop, target))
    ratio = (target - entry) / (entry - stop) if valid and stop < entry < target else None
    return (entry_label, entry, stop_label, stop, target_label, target,
            rr_label, ratio, note, holding)


def render_system_decision(decision, *, intraday_plan=None, current_price=None):
    """Show actual holding levels, or one concise flat-portfolio state."""
    if float(decision.current_shares) <= 0:
        action = getattr(decision, "action", None)
        if action == "COMPRAR":
            entry = _positive_or_none(getattr(decision, "entry_high", None))
            if entry is not None:
                st.caption(
                    f"COMPRAR · entrada autorizada ${entry:,.2f} · ejecución manual; "
                    "confirmar precio y disponibilidad en GBM+."
                )
        elif action in {"ESPERAR", "NO_ACCIONABLE"}:
            cause = str(getattr(decision, "waiting_cause", "") or "").strip()
            if not cause:
                reasons = getattr(decision, "reasons", ()) or ()
                cause = str(reasons[0]).strip() if reasons else "sin entrada validada"
            # Scores and marginal touches do not authorize a purchase.
            st.caption(f"ESPERAR · {cause}")
        return

    stop = _positive_or_none(decision.stop_loss)
    target = _positive_or_none(decision.take_profit)
    average = _positive_or_none(decision.average_price)
    last_close = _positive_or_none(current_price)
    exit_pending = decision.action in {"VENDER", "CONFIRMAR_SALIDA"}
    target_passed = target is not None and last_close is not None and target <= last_close
    with st.container(border=True, key="system_decision"):
        stop_col, target_col, average_col, shares_col = st.columns(4, gap="small")
        stop_label = (
            "Stop loss · confirmar salida"
            if exit_pending
            else "Stop loss móvil"
        )
        stop_col.metric(stop_label, "N/D" if stop is None else f"${stop:,.2f}")
        if exit_pending or target_passed:
            target_col.metric(
                "Salida pendiente · último cierre" if exit_pending else "Objetivo superado · último cierre",
                "N/D" if last_close is None else f"${last_close:,.2f}",
                help="Referencia del último cierre de 5 minutos; no es un precio de venta ejecutado ni garantizado.",
            )
        else:
            target_col.metric("Precio objetivo de salida", "N/D" if target is None else f"${target:,.2f}")
        average_col.metric("Precio promedio de compra", "N/D" if average is None else f"${average:,.2f}")
        shares_col.metric("Acciones en cartera", f"{decision.current_shares:g}")
