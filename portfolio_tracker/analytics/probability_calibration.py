"""Calibracion empirica causal para probabilidades direccionales.

No depende de Streamlit ni de SQLite. Recibe predicciones ya resueltas y aplica
una curva isotónica (PAV) únicamente cuando existe una muestra mínima. Antes de
ese umbral conserva el score original y lo etiqueta como preliminar.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


# Una curva puede estimarse con menos observaciones, pero no se presenta al
# usuario como probabilidad hasta reunir una base amplia fuera de muestra.
MIN_CALIBRATION_SAMPLES = 500


@dataclass(frozen=True, slots=True)
class ReliabilityPoint:
    predicted_mean: float
    observed_frequency: float
    samples: int


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    raw_probability: float
    calibrated_probability: float
    status: str
    sample_size: int
    brier_score: float | None
    reliability_curve: tuple[ReliabilityPoint, ...]

    @property
    def empirically_calibrated(self) -> bool:
        return self.status == "Probabilidad empíricamente calibrada"


def _clean_samples(
    samples: Iterable[tuple[float, int | bool]],
) -> list[tuple[float, int]]:
    cleaned: list[tuple[float, int]] = []
    for probability, outcome in samples:
        try:
            value = float(probability)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            continue
        cleaned.append((value, int(bool(outcome))))
    return cleaned


def reliability_curve(
    samples: Iterable[tuple[float, int | bool]],
    *,
    bins: int = 10,
) -> tuple[ReliabilityPoint, ...]:
    """Agrupa evidencia sin ocultar bins vacíos ni inventar observaciones."""

    if bins < 2:
        raise ValueError("La curva de calibración requiere al menos dos bins.")
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, outcome in _clean_samples(samples):
        index = min(bins - 1, int(probability * bins))
        grouped[index].append((probability, outcome))
    return tuple(
        ReliabilityPoint(
            predicted_mean=sum(item[0] for item in values) / len(values),
            observed_frequency=sum(item[1] for item in values) / len(values),
            samples=len(values),
        )
        for values in grouped
        if values
    )


def _isotonic_blocks(
    points: tuple[ReliabilityPoint, ...],
) -> tuple[ReliabilityPoint, ...]:
    """Pool Adjacent Violators ponderado por número de observaciones."""

    blocks = [
        [point.predicted_mean, point.observed_frequency, point.samples]
        for point in points
    ]
    position = 0
    while position < len(blocks) - 1:
        if blocks[position][1] <= blocks[position + 1][1]:
            position += 1
            continue
        left, right = blocks[position], blocks[position + 1]
        samples = int(left[2] + right[2])
        merged = [
            (left[0] * left[2] + right[0] * right[2]) / samples,
            (left[1] * left[2] + right[1] * right[2]) / samples,
            samples,
        ]
        blocks[position : position + 2] = [merged]
        position = max(0, position - 1)
    return tuple(
        ReliabilityPoint(float(predicted), float(observed), int(samples))
        for predicted, observed, samples in blocks
    )


def calibrate_probability(
    raw_probability: float,
    samples: Iterable[tuple[float, int | bool]],
    *,
    minimum_samples: int = MIN_CALIBRATION_SAMPLES,
) -> CalibrationResult:
    """Convierte un score en frecuencia observada solo con evidencia suficiente."""

    raw = max(0.0, min(1.0, float(raw_probability)))
    cleaned = _clean_samples(samples)
    curve = reliability_curve(cleaned)
    brier = (
        sum((probability - outcome) ** 2 for probability, outcome in cleaned)
        / len(cleaned)
        if cleaned
        else None
    )
    if len(cleaned) < minimum_samples or len(curve) < 2:
        return CalibrationResult(
            raw_probability=raw,
            calibrated_probability=raw,
            status="Score heurístico preliminar",
            sample_size=len(cleaned),
            brier_score=brier,
            reliability_curve=curve,
        )
    isotonic = _isotonic_blocks(curve)
    nearest = min(isotonic, key=lambda point: abs(point.predicted_mean - raw))
    # Suavizado conservador evita convertir una frecuencia finita en certeza.
    empirical = (nearest.observed_frequency * nearest.samples + 0.5 * 8) / (
        nearest.samples + 8
    )
    calibrated = max(0.02, min(0.98, empirical))
    return CalibrationResult(
        raw_probability=raw,
        calibrated_probability=calibrated,
        status="Probabilidad empíricamente calibrada",
        sample_size=len(cleaned),
        brier_score=brier,
        reliability_curve=isotonic,
    )
