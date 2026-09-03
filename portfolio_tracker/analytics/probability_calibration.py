"""Calibracion empirica causal para probabilidades direccionales.

Sin Streamlit ni SQLite: división cronológica 60/20/20, ajuste isotónico solo
en calibración y Brier/reliability solo en holdout. La API multiclase purga
resultados tardíos y horizontes solapados. Sin evidencia suficiente conserva
el score preliminar; un ajuste estadístico no garantiza utilidad operativa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    raw_brier_score: float | None = None
    training_samples: int = 0
    calibration_samples: int = 0
    holdout_samples: int = 0
    isotonic_curve: tuple[ReliabilityPoint, ...] = ()

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
        if outcome not in (0, 1):
            continue
        cleaned.append((value, int(outcome)))
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


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """Forecast already emitted and resolved; class order UP/RANGE/DOWN."""
    probabilities: tuple[float, ...]
    outcome: int
    observed_at: datetime
    available_at: datetime
    resolved_at: datetime


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    training: tuple
    calibration: tuple
    holdout: tuple
    nominal_sizes: tuple[int, int, int]
    excluded: int = 0


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Las muestras requieren timestamps con zona horaria.")
    return value.astimezone(timezone.utc)


def chronological_split(samples) -> ChronologicalSplit:
    """60/20/20 by emission, then purge future/overlapping labels, never shuffle.

    Untimed pairs are a compatibility API: caller MUST supply chronological
    order. Production uses CalibrationSample, with explicit label availability.
    Training is reserved for the upstream model/baselines, never isotonic fit.
    """
    rows = list(samples)
    timed = bool(rows) and isinstance(rows[0], CalibrationSample)
    if any(isinstance(row, CalibrationSample) != timed for row in rows):
        raise ValueError("No mezclar muestras con y sin timestamps.")
    if timed:
        for row in rows:
            if not _utc(row.observed_at) < _utc(row.available_at) <= _utc(row.resolved_at):
                raise ValueError("Orden temporal inválido en la muestra.")
        rows.sort(key=lambda row: _utc(row.observed_at))
    first, second = len(rows)*60//100, len(rows)*80//100
    blocks = [rows[:first], rows[first:second], rows[second:]]
    nominal = tuple(map(len, blocks))
    if timed:
        for i, block in enumerate(blocks):
            boundary = (rows[first] if i == 0 and first < len(rows)
                        else rows[second] if i == 1 and second < len(rows) else None)
            boundary_time = _utc(boundary.observed_at) if boundary else None
            selected, end = [], None
            for row in block:
                observed = _utc(row.observed_at)
                # available_at is target maturity, resolved_at is when its
                # label actually entered our system. Both must be in the past.
                known = _utc(row.resolved_at)
                if boundary_time is not None and known >= boundary_time:
                    continue
                if end is not None and observed <= end:
                    continue
                selected.append(row)
                end = _utc(row.available_at)
            blocks[i] = selected
    return ChronologicalSplit(*(tuple(b) for b in blocks), nominal,
                              len(rows)-sum(map(len, blocks)))


def _fit_isotonic(samples) -> tuple[ReliabilityPoint, ...]:
    """PAV over exact scores (ties pooled), fit ONLY on calibration."""
    grouped = {}
    for probability, outcome in samples:
        total, count = grouped.get(probability, (0, 0))
        grouped[probability] = (total+outcome, count+1)
    # Smooth BEFORE PAV: smoothing pooled blocks afterwards can destroy
    # monotonicity when their sample counts differ.
    points = tuple(ReliabilityPoint(p, (total+4.0)/(count+8.0), count)
                   for p, (total,count) in sorted(grouped.items()))
    return _isotonic_blocks(points)


def _predict_isotonic(raw, curve):
    if not curve:
        raise ValueError("Curva isotónica vacía.")
    point = min(curve, key=lambda p: abs(p.predicted_mean-raw))
    return point.observed_frequency


def calibrate_probability(
    raw_probability: float,
    samples: Iterable[tuple[float, int | bool]],
    *,
    minimum_samples: int = MIN_CALIBRATION_SAMPLES,
) -> CalibrationResult:
    """Binary compatibility API. Brier/reliability use ONLY the last 20%."""
    raw = float(raw_probability)
    if not math.isfinite(raw) or not 0 <= raw <= 1:
        raise ValueError("Probabilidad fuera de [0,1].")
    cleaned = _clean_samples(samples)
    split = chronological_split(cleaned)
    curve = _fit_isotonic(split.calibration)
    eligible = (len(cleaned) >= minimum_samples and bool(split.holdout)
                and len({p for p,y in split.calibration}) >= 2)
    predict = (lambda p: _predict_isotonic(p, curve)) if eligible else (lambda p: p)
    evaluated = [(predict(p), y) for p,y in split.holdout]
    brier = sum((p-y)**2 for p,y in evaluated)/len(evaluated) if evaluated else None
    raw_brier = (sum((p-y)**2 for p,y in split.holdout)/len(split.holdout)
                 if split.holdout else None)
    return CalibrationResult(
        raw_probability=raw, calibrated_probability=predict(raw),
        status="Probabilidad empíricamente calibrada" if eligible else "Score heurístico preliminar",
        sample_size=len(cleaned), brier_score=brier,
        reliability_curve=reliability_curve(evaluated), raw_brier_score=raw_brier,
        training_samples=len(split.training), calibration_samples=len(split.calibration),
        holdout_samples=len(split.holdout), isotonic_curve=curve,
    )


@dataclass(frozen=True, slots=True)
class MulticlassCalibrationResult:
    probabilities: tuple[float, float, float]
    status: str
    sample_size: int
    brier_score: float | None
    raw_brier_score: float | None
    baseline_brier_score: float | None
    split: ChronologicalSplit
    isotonic_curves: tuple[tuple[ReliabilityPoint, ...], ...]
    reliability_curves: tuple[tuple[ReliabilityPoint, ...], ...]
    reason: str

    @property
    def empirically_calibrated(self):
        return self.status == "Probabilidad empíricamente calibrada"


def validate_distribution(values):
    vector = tuple(float(x) for x in values)
    if len(vector) != 3 or not all(math.isfinite(x) and 0 <= x <= 1 for x in vector):
        raise ValueError("Se requiere distribución subida/rango/bajada válida.")
    if not math.isclose(sum(vector), 1.0, abs_tol=1e-9):
        raise ValueError("Los tres escenarios deben sumar 1; no se recortan en UI.")
    return vector


def _joint_prediction(vector, curves):
    # OVR isotonic + simplex coupling is ONE fixed model. Evaluate the coupled
    # vector on holdout, never report the Brier of uncoupled binary estimators.
    scores = [_predict_isotonic(p, curve) for p,curve in zip(vector, curves)]
    total = sum(scores)
    return tuple(p/total for p in scores)


def _multiclass_brier(predictions, outcomes):
    if not predictions:
        return None
    # Unscaled multiclass Brier: sum over classes, range [0,2].
    return sum(sum((p-int(k==y))**2 for k,p in enumerate(vector))
               for vector,y in zip(predictions,outcomes))/len(predictions)


def calibrate_scenarios(raw_probabilities, samples, *, minimum_samples=500):
    """Timestamped OOS 60/20/20 evaluation for one immutable model contract."""
    raw = validate_distribution(raw_probabilities)
    rows = list(samples)
    for row in rows:
        validate_distribution(row.probabilities)
        if row.outcome not in (0,1,2):
            raise ValueError("Clase de cierre inválida.")
    split = chronological_split(rows)
    cal, holdout = split.calibration, split.holdout
    curves = tuple(_fit_isotonic([(r.probabilities[k], int(r.outcome==k)) for r in cal])
                   for k in range(3))
    effective = sum(len(b) for b in (split.training, cal, holdout))
    eligible = (effective >= minimum_samples and len(cal) >= 100 and len(holdout) >= 100
                and {r.outcome for r in cal} == {0,1,2}
                and all(len({r.probabilities[k] for r in cal}) >= 2 for k in range(3)))
    predict = (lambda p: _joint_prediction(p, curves)) if eligible else (lambda p: p)
    evaluated = [predict(r.probabilities) for r in holdout]
    outcomes = [r.outcome for r in holdout]
    # Training-only class frequencies: descriptive control, never isotonic input.
    prior = tuple(sum(r.outcome==k for r in split.training)/len(split.training)
                  for k in range(3)) if split.training else None
    reason = ("Isotónica ajustada solo en calibración; Brier multiclase [0,2] solo en holdout."
              if eligible else "Evidencia OOS insuficiente: se conserva la distribución heurística.")
    return MulticlassCalibrationResult(
        probabilities=predict(raw),
        status="Probabilidad empíricamente calibrada" if eligible else "Score heurístico preliminar",
        sample_size=effective, brier_score=_multiclass_brier(evaluated,outcomes),
        raw_brier_score=_multiclass_brier([r.probabilities for r in holdout],outcomes),
        baseline_brier_score=_multiclass_brier([prior]*len(holdout),outcomes) if prior else None,
        split=split, isotonic_curves=curves,
        reliability_curves=tuple(reliability_curve([(p[k],int(y==k)) for p,y in zip(evaluated,outcomes)])
                                 for k in range(3)), reason=reason,
    )
