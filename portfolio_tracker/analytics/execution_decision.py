"""Pure execution checklist; it consumes analysis but never submits orders."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ActivationCheck:
    key: str
    label: str
    passed: bool
    detail: str


def _inside(price, zone) -> bool:
    try:
        low, high = float(zone.low), float(zone.high)
        return all(math.isfinite(value) for value in (price, low, high)) and low <= price <= high
    except (AttributeError, TypeError, ValueError):
        return False


def build_long_activation_checklist(analysis, zone_snapshot=None) -> tuple[ActivationCheck, ...]:
    buys = tuple(getattr(zone_snapshot, "buys", ()) or ())
    price = float(analysis.last_price)
    signal = getattr(getattr(analysis, "signal", None), "value", "UNKNOWN")
    macd_state = getattr(getattr(analysis, "macd_state_5m", None), "value", "UNKNOWN")
    volume_ratio = float(getattr(analysis, "volume_ratio", 0.0) or 0.0)
    macro = str(getattr(analysis, "macro_permission", "NO_TRADE"))
    veto = bool(getattr(analysis, "risk_veto", False) or getattr(analysis, "fundamental_risk_veto", False))
    checks = (
        ActivationCheck("price_zone", "Precio dentro de zona", bool(buys and _inside(price, buys[0])),
                        "El precio debe entrar en la primera zona vigente."),
        ActivationCheck("closed_rebound", "Vela de rebote cerrada",
                        signal == "BUY" and bool(getattr(analysis, "rebound_watch_active", False)),
                        f"Señal cerrada: {signal}."),
        ActivationCheck("volume", "Volumen superior a media", bool(getattr(analysis, "volume_confirmed", False)),
                        f"Ratio de volumen: {volume_ratio:.2f}x."),
        ActivationCheck("macd", "MACD 5m confirmado", macd_state == "ALCISTA",
                        f"Estado MACD: {macd_state}."),
        ActivationCheck("macro", "Régimen autoriza LONG", macro in {"LONG_ONLY", "BOTH_REDUCED"},
                        f"Permiso superior: {macro}."),
        ActivationCheck("veto", "Veto de riesgo inactivo", not veto,
                        "Sin veto técnico, fundamental, noticioso o de evento." if not veto else "Existe un veto activo."),
        ActivationCheck("trigger", "Gatillo completo", bool(getattr(analysis, "activation_trigger_met", False)),
                        "Resultado causal del TriggerEngine sobre velas cerradas."),
    )
    return checks
