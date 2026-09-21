"""Contrato estadístico para escenarios direccionales por horizonte.

Este módulo no conoce zonas de precio. Un Brier de toque/cierre intradía no
puede promover ni penalizar una distribución UP/RANGE/DOWN.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


CALIBRATED_STATUS = "Probabilidad empíricamente calibrada"
MIN_EFFECTIVE_SAMPLES = 500
MIN_HOLDOUT_SAMPLES = 100


@dataclass(frozen=True, slots=True)
class DirectionalEvidence:
    eligible: bool
    status: str
    dominant_class: str
    effective_samples: int
    holdout_samples: int
    brier_score: float | None
    baseline_brier_score: float | None
    reason: str


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dominant_class(projection) -> str:
    """Return UP, RANGE or DOWN using the complete three-class contract."""
    values = {
        "UP": _finite(getattr(projection, "probability_up", None)),
        "RANGE": _finite(getattr(projection, "probability_range", None)),
        "DOWN": _finite(getattr(projection, "probability_down", None)),
    }
    if any(value is None or not 0 <= value <= 100 for value in values.values()):
        return "UNKNOWN"
    return max(values, key=values.get)


def assess_directional_projection(
    projection,
    *,
    minimum_effective_samples: int = MIN_EFFECTIVE_SAMPLES,
    minimum_holdout_samples: int = MIN_HOLDOUT_SAMPLES,
) -> DirectionalEvidence:
    """Approve only a calibrated model that beats its holdout baseline."""
    status = str(getattr(projection, "probability_status", "Score heurístico preliminar"))
    effective = int(getattr(projection, "calibration_samples", 0) or 0)
    holdout = int(getattr(projection, "calibration_holdout_samples", 0) or 0)
    brier = _finite(getattr(projection, "brier_score", None))
    baseline = _finite(getattr(projection, "baseline_brier_score", None))
    dominant = dominant_class(projection)
    reasons = []
    if status != CALIBRATED_STATUS:
        reasons.append("el calibrador todavía clasifica el resultado como preliminar")
    if effective < int(minimum_effective_samples):
        reasons.append(f"{effective}/{minimum_effective_samples} muestras efectivas")
    if holdout < int(minimum_holdout_samples):
        reasons.append(f"{holdout}/{minimum_holdout_samples} muestras holdout")
    if brier is None or baseline is None:
        reasons.append("Brier OOS o baseline no disponible")
    elif brier >= baseline:
        reasons.append(f"Brier {brier:.4f} no mejora baseline {baseline:.4f}")
    if dominant in {"UNKNOWN", "RANGE"}:
        reasons.append("el escenario dominante no es direccional")
    eligible = not reasons
    return DirectionalEvidence(
        eligible=eligible,
        status=status,
        dominant_class=dominant,
        effective_samples=effective,
        holdout_samples=holdout,
        brier_score=brier,
        baseline_brier_score=baseline,
        reason=("Evidencia direccional OOS aprobada." if eligible else "; ".join(reasons) + "."),
    )


def fifteen_day_up_score(projections) -> float:
    """Blend final 1-week/1-month distributions for the 15-session scenario."""
    by_label = {item.label: item for item in projections}
    week = by_label.get("1 Semana")
    month = by_label.get("1 Mes")
    if week is None or month is None:
        raise ValueError("La proyección de 15 días requiere horizontes de 1 semana y 1 mes.")
    return 0.60 * float(week.probability_up) + 0.40 * float(month.probability_up)
