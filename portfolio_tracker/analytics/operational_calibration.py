"""Temperature calibration for skillful first-passage horizon models.

The caller supplies chronologically separated out-of-fold development scores.
This module never reads the final holdout, SQLite, or live observations.
"""
from __future__ import annotations

import math

import numpy as np

from .horizon_models import (
    FEATURE_NAMES,
    TARGET_CLASSES,
    _brier,
    _log_loss,
    _sha,
    _softmax,
)
from .metric_uncertainty import score_intervals


DEFAULT_TEMPERATURE_GRID = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def _distribution_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != len(TARGET_CLASSES)
            or len(matrix) == 0 or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or np.any(matrix > 1)
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-8)):
        raise ValueError("Se requieren vectores TP/SL/timeout finitos que sumen uno.")
    return matrix


def temperature_scale(probabilities, temperature: float) -> np.ndarray:
    """Normalize scaled log scores without changing their class ordering."""

    matrix = _distribution_matrix(probabilities)
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("La temperatura debe ser positiva y finita.")
    logits = np.log(np.clip(matrix, 1e-12, 1.0)) / value
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _labels(values, expected: int) -> np.ndarray:
    labels = np.asarray(values)
    if (labels.ndim != 1 or len(labels) != expected
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= len(TARGET_CLASSES))):
        raise ValueError("Etiquetas operativas inválidas para calibración.")
    return labels.astype(int)


def _metrics(probabilities, labels):
    intervals = score_intervals(probabilities, labels)
    return {
        "brier": _brier(probabilities, labels),
        "log_loss": _log_loss(probabilities, labels),
        "confidence_intervals": {
            name: intervals[name] for name in ("brier", "log_loss")
        },
    }


def select_and_validate_temperature(
    fit_scores,
    fit_labels,
    validation_scores,
    validation_labels,
    validation_baselines,
    *,
    grid=DEFAULT_TEMPERATURE_GRID,
    minimum_fit: int = 60,
    minimum_validation: int = 30,
    minimum_per_class: int = 5,
) -> dict:
    """Choose temperature on OOF fit data; gate on later untouched OOF data."""

    candidates = tuple(sorted(set(float(value) for value in grid)))
    if not candidates or 1.0 not in candidates or any(
        not math.isfinite(value) or value <= 0 for value in candidates
    ):
        raise ValueError("La rejilla debe ser positiva e incluir temperatura 1.0.")
    fit = _distribution_matrix(fit_scores)
    validation = _distribution_matrix(validation_scores)
    baselines = _distribution_matrix(validation_baselines)
    if len(validation) != len(baselines):
        raise ValueError("El baseline debe corresponder a las mismas observaciones.")
    y_fit = _labels(fit_labels, len(fit))
    y_validation = _labels(validation_labels, len(validation))
    fit_counts = {
        name: int(sum(y_fit == index)) for index, name in enumerate(TARGET_CLASSES)
    }
    validation_counts = {
        name: int(sum(y_validation == index)) for index, name in enumerate(TARGET_CLASSES)
    }
    if (len(fit) < minimum_fit or len(validation) < minimum_validation
            or min(fit_counts.values()) < minimum_per_class
            or min(validation_counts.values()) < minimum_per_class):
        return {
            "status": "INSUFFICIENT_OOF_CLASS_SUPPORT" if (
                min(fit_counts.values()) < minimum_per_class
                or min(validation_counts.values()) < minimum_per_class
            ) else "INSUFFICIENT_OOF_CALIBRATION",
            "approved": False,
            "reason": (
                f"Cada clase TP_FIRST/SL_FIRST/TIMEOUT necesita al menos "
                f"{minimum_per_class} casos tanto en ajuste como en validación."
            ),
            "temperature": None,
            "fit_samples": len(fit),
            "validation_samples": len(validation),
            "fit_class_counts": fit_counts,
            "validation_class_counts": validation_counts,
            "candidates": [],
            "validation_metrics": None,
        }
    results = []
    for temperature in candidates:
        scores = temperature_scale(fit, temperature)
        results.append({"temperature": temperature, **_metrics(scores, y_fit)})
    selected = min(results, key=lambda row: (row["log_loss"], row["brier"], row["temperature"]))
    raw = _metrics(validation, y_validation)
    calibrated = _metrics(temperature_scale(validation, selected["temperature"]), y_validation)
    baseline = _metrics(baselines, y_validation)
    raw_beats_baseline = (
        raw["brier"] < baseline["brier"] - 1e-12
        and raw["log_loss"] < baseline["log_loss"] - 1e-12
    )
    improves_raw = (
        calibrated["brier"] < raw["brier"] - 1e-12
        and calibrated["log_loss"] < raw["log_loss"] - 1e-12
    )
    beats_baseline = (
        calibrated["brier"] < baseline["brier"] - 1e-12
        and calibrated["log_loss"] < baseline["log_loss"] - 1e-12
    )
    approved = bool(
        raw_beats_baseline and selected["temperature"] != 1.0
        and improves_raw and beats_baseline
    )
    return {
        "status": "CALIBRATED_OOF_VALIDATED" if approved else "NO_CALIBRATION_GAIN",
        "approved": approved,
        "method": "MULTICLASS_TEMPERATURE_SCALING_V1",
        "raw_model_beats_baseline": bool(raw_beats_baseline),
        "temperature": selected["temperature"],
        "fit_samples": len(fit),
        "validation_samples": len(validation),
        "fit_class_counts": fit_counts,
        "validation_class_counts": validation_counts,
        "selection_source": "EARLIER_OUTER_OOF_ONLY",
        "validation_source": "LATER_OUTER_OOF_ONLY",
        "candidates": results,
        "validation_metrics": {
            "raw": raw,
            "calibrated": calibrated,
            "baseline": baseline,
        },
    }


def predict_calibrated_scores(
    model_record, features, *, feature_version=None, available_features=None,
) -> dict:
    """Infer only from a signed, approved and calibrated horizon result."""

    if model_record.get("result_sha256") != _sha({
        key: value for key, value in model_record.items() if key != "result_sha256"
    }):
        raise ValueError("Firma del modelo calibrado inválida.")
    if (not model_record.get("promotable")
            or model_record.get("status") != "APPROVED_SEALED_HOLDOUT_CALIBRATED"
            or model_record.get("score_semantics") != "HISTORICAL_OOS_CALIBRATED_PRELIMINARY"
            or model_record.get("feature_names") != list(FEATURE_NAMES)
            or model_record.get("feature_schema_sha256") != _sha(list(FEATURE_NAMES))):
        raise ValueError("Modelo calibrado no aprobado o esquema incompatible.")
    from .horizon_models import FeatureContractMismatch, MODEL_FEATURE_VERSION
    if (feature_version != MODEL_FEATURE_VERSION
            or model_record.get("feature_version") != MODEL_FEATURE_VERSION
            or list(available_features or ()) != model_record.get("available_features")):
        raise FeatureContractMismatch("Features train/live incompatibles: versión o disponibilidad distinta.")
    model = model_record.get("model") or {}
    calibration = model.get("calibration") or {}
    if calibration.get("method") != "MULTICLASS_TEMPERATURE_SCALING_V1":
        raise ValueError("Calibrador del modelo no reconocido.")
    values = np.asarray(features, dtype=float)
    if values.shape != (len(FEATURE_NAMES),):
        raise ValueError("Vector de features incompatible.")
    actual_available = [name for name, value in zip(FEATURE_NAMES, values) if math.isfinite(value)]
    if actual_available != list(available_features or ()):
        raise FeatureContractMismatch("Features train/live incompatibles: disponibilidad declarada no coincide con el vector.")
    medians = np.asarray(model["imputation_medians"], dtype=float)
    means = np.asarray(model["standardization_means"], dtype=float)
    scales = np.asarray(model["standardization_scales"], dtype=float)
    coefficients = np.asarray(model["intercept_and_coefficients"], dtype=float)
    if (medians.shape != values.shape or means.shape != values.shape
            or scales.shape != values.shape
            or coefficients.shape != (len(values) + 1, len(TARGET_CLASSES))
            or not np.isfinite(medians).all() or not np.isfinite(means).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not np.isfinite(coefficients).all()):
        raise ValueError("Coeficientes o preprocesamiento inválidos.")
    filled = np.where(np.isfinite(values), values, medians)
    standardized = (filled - means) / scales
    raw = _softmax(np.concatenate(([1.0], standardized)).reshape(1, -1) @ coefficients)
    adjusted = temperature_scale(raw, calibration["temperature"])[0]
    return {
        "semantics": "HISTORICAL_OOS_CALIBRATED_PRELIMINARY",
        "horizon": model_record["horizon"],
        "scores": {
            name: float(adjusted[index]) for index, name in enumerate(TARGET_CLASSES)
        },
        "live_forward_validated": False,
    }
