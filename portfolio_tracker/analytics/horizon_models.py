"""Regularized operational classifiers trained independently per horizon.

The training target is the causal first event ``TP_FIRST / SL_FIRST /
TIMEOUT`` frozen by :mod:`historical_replay`.  Each horizon owns a different
model and chronological split.  Feature preprocessing and L2 selection never
inspect the final holdout, and labels which were not yet known at a split
boundary are purged.

The resulting softmax values are deliberately named *uncalibrated scores*.
Calibration is a later, separate operation and replay evidence never becomes
live/OOS evidence merely because a model was fitted to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .historical_replay import validate_historical_replay
from ..services.directional_collection import HORIZON_MINUTES
from ..services.model_observations import canonical


HORIZON_MODEL_CONTRACT = "REGULARIZED_HORIZON_FIRST_PASSAGE_V1"
TARGET_CLASSES = ("TP_FIRST", "SL_FIRST", "TIMEOUT")
PROFESSIONAL_MINIMUM_SAMPLES = 300


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Immutable protocol controls; defaults intentionally reject small data."""

    minimum_samples: int = PROFESSIONAL_MINIMUM_SAMPLES
    minimum_class_samples: int = 20
    minimum_holdout_samples: int = 60
    l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    maximum_iterations: int = 1_200
    learning_rate: float = 0.08

    def validate(self) -> None:
        if self.minimum_samples < 30:
            raise ValueError("minimum_samples no puede ser menor que 30.")
        if self.minimum_class_samples < 1 or self.minimum_holdout_samples < 1:
            raise ValueError("Los mínimos por clase y holdout deben ser positivos.")
        if not self.l2_grid or any(not math.isfinite(v) or v <= 0 for v in self.l2_grid):
            raise ValueError("l2_grid debe contener penalizaciones positivas finitas.")
        if self.maximum_iterations < 100 or not 0 < self.learning_rate <= 1:
            raise ValueError("Configuración de optimización inválida.")


@dataclass(frozen=True, slots=True)
class HorizonSample:
    observed_at: str
    label_available_at: str
    features: tuple[float, ...]
    outcome: str
    cut_id: str = ""


NUMERIC_FEATURES = (
    "last_price", "atr_to_price", "price_vs_vwap_pct", "adx",
    "stochastic_k", "stochastic_d", "volume_ratio", "operation_probability",
    "heuristic_up", "heuristic_down", "bb_position", "ema9_distance_atr",
    "ema21_distance_atr", "ema50_distance_atr", "ema200_distance_atr",
    "macd_5m_atr", "macd_histogram_5m_atr", "macd_daily_atr",
    "support_distance_atr", "resistance_distance_atr", "weekly_span_atr",
    "fundamental_score", "chart_pattern_impact", "exposure_factor",
    "cross_correlation", "cross_ratio_deviation_pct", "cross_applied_impact",
    "horizon_up", "horizon_range", "horizon_down", "horizon_atr_to_price",
    "bullish_distance_atr", "bearish_distance_atr", "expected_range_atr",
    "operational_reward_atr", "operational_risk_atr",
)

CATEGORICAL_FEATURES = {
    "market_regime": ("TREND", "RANGE", "TRANSITION", "NO_TRADE", "UNKNOWN"),
    "macro_permission": ("LONG_ONLY", "SHORT_ONLY", "BOTH_REDUCED", "NO_TRADE", "UNKNOWN"),
    "signal": ("BUY", "SELL", "WATCH_BUY", "WATCH_SELL", "NEUTRAL", "UNKNOWN"),
    "weekly_trend": ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH", "UNKNOWN"),
    "daily_trend": ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH", "UNKNOWN"),
    "monthly_trend": ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH", "UNKNOWN"),
}

BOOLEAN_FEATURES = (
    "above_vwap", "volume_confirmed", "range_market", "macro_trending",
    "risk_veto", "fundamental_risk_veto", "activation_trigger_met",
    "chart_pattern_veto", "tactical_short",
)

FEATURE_NAMES = (
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
    *(f"{name}={category}" for name, categories in CATEGORICAL_FEATURES.items()
      for category in categories),
)


def _sha(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _number(value, default=np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_div(numerator, denominator) -> float:
    left, right = _number(numerator), _number(denominator)
    if not math.isfinite(left) or not math.isfinite(right) or abs(right) < 1e-12:
        return np.nan
    return left / right


def _normalized_distance(value, price, atr) -> float:
    return _safe_div(_number(value) - _number(price), atr)


def _category(value) -> str:
    text = str(value or "UNKNOWN").strip().upper().replace(" ", "_")
    return text or "UNKNOWN"


def feature_vector(cut: Mapping, horizon: Mapping) -> tuple[float, ...]:
    """Extract only information frozen at emission; never inspect outcomes."""

    snapshot = cut.get("feature_snapshot") or {}
    features = snapshot.get("features") or {}
    prediction = horizon.get("prediction") or {}
    contract = horizon.get("operational_contract") or {}
    cross = features.get("cross_asset_context") or {}
    ratio = cross.get("ratio") or {}
    price = _number(features.get("last_price"))
    atr = _number(features.get("atr_5m"))
    bb_low = _number(features.get("bollinger_lower"))
    bb_high = _number(features.get("bollinger_upper"))
    bb_position = _safe_div(price - bb_low, bb_high - bb_low)
    support = _number(features.get("nearest_support"))
    resistance = _number(features.get("structural_resistance"))
    if not math.isfinite(resistance) or resistance <= 0:
        resistance = _number(features.get("weekly_resistance"))
    horizon_atr = _number(prediction.get("atr_value"))
    if not math.isfinite(horizon_atr) or horizon_atr <= 0:
        horizon_atr = atr
    entry = _number(contract.get("entry_price"))
    stop = _number(contract.get("stop_loss"))
    take = _number(contract.get("take_profit"))

    numeric = {
        "last_price": price,
        "atr_to_price": _safe_div(atr, price),
        "price_vs_vwap_pct": _number(features.get("price_vs_vwap_pct")),
        "adx": _number(features.get("adx")),
        "stochastic_k": _number(features.get("stochastic_k")),
        "stochastic_d": _number(features.get("stochastic_d")),
        "volume_ratio": _number(features.get("volume_ratio")),
        "operation_probability": _number(features.get("operation_probability")),
        "heuristic_up": _number(features.get("probability_up")),
        "heuristic_down": _number(features.get("probability_down")),
        "bb_position": bb_position,
        "ema9_distance_atr": _normalized_distance(features.get("ema9"), price, atr),
        "ema21_distance_atr": _normalized_distance(features.get("ema21"), price, atr),
        "ema50_distance_atr": _normalized_distance(features.get("ema50"), price, atr),
        "ema200_distance_atr": _normalized_distance(features.get("ema200"), price, atr),
        "macd_5m_atr": _safe_div(features.get("macd_5m"), atr),
        "macd_histogram_5m_atr": _safe_div(features.get("macd_histogram_5m"), atr),
        "macd_daily_atr": _safe_div(features.get("macd_daily"), atr),
        "support_distance_atr": _normalized_distance(support, price, atr),
        "resistance_distance_atr": _normalized_distance(resistance, price, atr),
        "weekly_span_atr": _safe_div(
            _number(features.get("weekly_resistance")) - _number(features.get("weekly_support")), atr
        ),
        "fundamental_score": _number(features.get("fundamental_score")),
        "chart_pattern_impact": _number(features.get("chart_pattern_impact")),
        "exposure_factor": _number(features.get("exposure_factor")),
        "cross_correlation": _number(cross.get("correlation")),
        "cross_ratio_deviation_pct": _number(ratio.get("deviation_pct")),
        "cross_applied_impact": _number(cross.get("applied_impact")),
        "horizon_up": _number(prediction.get("probability_up")),
        "horizon_range": _number(prediction.get("probability_range")),
        "horizon_down": _number(prediction.get("probability_down")),
        "horizon_atr_to_price": _safe_div(horizon_atr, price),
        "bullish_distance_atr": _normalized_distance(prediction.get("bullish_target"), price, horizon_atr),
        "bearish_distance_atr": _normalized_distance(prediction.get("bearish_target"), price, horizon_atr),
        "expected_range_atr": _safe_div(
            _number(prediction.get("range_high")) - _number(prediction.get("range_low")), horizon_atr
        ),
        "operational_reward_atr": _safe_div(take - entry, horizon_atr),
        "operational_risk_atr": _safe_div(entry - stop, horizon_atr),
    }
    values = [numeric[name] for name in NUMERIC_FEATURES]
    values.extend(1.0 if bool(features.get(name)) else 0.0 for name in BOOLEAN_FEATURES)
    for name, categories in CATEGORICAL_FEATURES.items():
        current = _category(features.get(name))
        if current not in categories:
            current = "UNKNOWN"
        values.extend(1.0 if current == category else 0.0 for category in categories)
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("Contrato de features inconsistente.")
    return tuple(float(value) for value in values)


def horizon_samples(replay: Mapping, horizon_label: str) -> tuple[HorizonSample, ...]:
    """Build resolved, point-in-time samples for exactly one horizon."""

    validate_historical_replay(dict(replay))
    if horizon_label not in HORIZON_MINUTES:
        raise ValueError(f"Horizonte desconocido: {horizon_label}.")
    rows = []
    for cut in replay.get("observations", ()):
        matches = [row for row in cut.get("horizons", ()) if row.get("label") == horizon_label]
        if len(matches) != 1:
            raise ValueError("Cada corte debe tener exactamente un contrato por horizonte.")
        horizon = matches[0]
        result = horizon.get("operational_result") or {}
        outcome = result.get("outcome")
        available = result.get("exit_at")
        if result.get("status") != "RESOLVED" or outcome not in TARGET_CLASSES or not available:
            continue
        observed = pd.Timestamp(cut.get("observed_at"))
        known = pd.Timestamp(available)
        if pd.isna(observed) or pd.isna(known) or observed.tzinfo is None or known.tzinfo is None:
            raise ValueError("Timestamps del replay deben ser timezone-aware.")
        if known < observed:
            raise ValueError("Una etiqueta no puede conocerse antes de la predicción.")
        rows.append(HorizonSample(
            observed_at=observed.tz_convert("UTC").isoformat(),
            label_available_at=known.tz_convert("UTC").isoformat(),
            features=feature_vector(cut, horizon),
            outcome=str(outcome),
            cut_id=str(cut.get("cut_id") or ""),
        ))
    rows.sort(key=lambda item: item.observed_at)
    if len({row.observed_at for row in rows}) != len(rows):
        raise ValueError("Cortes duplicados en un horizonte.")
    return tuple(rows)


def _class_counts(samples: Sequence[HorizonSample]) -> dict[str, int]:
    return {name: sum(row.outcome == name for row in samples) for name in TARGET_CLASSES}


def chronological_purged_split(samples: Sequence[HorizonSample]):
    """60/20/20 chronological split with label-availability purging."""

    ordered = tuple(sorted(samples, key=lambda item: item.observed_at))
    size = len(ordered)
    train_end, calibration_end = int(size * 0.60), int(size * 0.80)
    if train_end < 1 or calibration_end <= train_end or calibration_end >= size:
        return (), (), (), {"training": 0, "calibration": 0}
    calibration_start = pd.Timestamp(ordered[train_end].observed_at)
    holdout_start = pd.Timestamp(ordered[calibration_end].observed_at)
    raw_training = ordered[:train_end]
    raw_calibration = ordered[train_end:calibration_end]
    holdout = ordered[calibration_end:]
    training = tuple(row for row in raw_training if pd.Timestamp(row.label_available_at) <= calibration_start)
    calibration = tuple(row for row in raw_calibration if pd.Timestamp(row.label_available_at) <= holdout_start)
    return training, calibration, holdout, {
        "training": len(raw_training) - len(training),
        "calibration": len(raw_calibration) - len(calibration),
    }


def _fit_preprocessor(samples: Sequence[HorizonSample]):
    matrix = np.asarray([row.features for row in samples], dtype=float)
    finite = np.isfinite(matrix)
    medians = np.zeros(matrix.shape[1], dtype=float)
    for column in range(matrix.shape[1]):
        observed = matrix[finite[:, column], column]
        medians[column] = float(np.median(observed)) if len(observed) else 0.0
    filled = np.where(finite, matrix, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    return medians, means, scales


def _transform(samples, medians, means, scales):
    matrix = np.asarray([row.features for row in samples], dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    return (matrix - means) / scales


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -700, 0))
    return exponential / exponential.sum(axis=1, keepdims=True)


def _fit_softmax(x, y, l2, iterations, learning_rate):
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros((design.shape[1], len(TARGET_CLASSES)), dtype=float)
    target = np.eye(len(TARGET_CLASSES), dtype=float)[y]
    previous = float("inf")
    for step in range(iterations):
        probabilities = _softmax(design @ weights)
        gradient = design.T @ (probabilities - target) / len(design)
        gradient[1:] += float(l2) * weights[1:]
        rate = learning_rate / math.sqrt(1.0 + step / 100.0)
        weights -= rate * gradient
        if step % 25 == 0:
            loss = _log_loss(probabilities, y) + 0.5 * float(l2) * float(np.sum(weights[1:] ** 2))
            if abs(previous - loss) < 1e-9:
                break
            previous = loss
    return weights


def _labels(samples):
    lookup = {name: index for index, name in enumerate(TARGET_CLASSES)}
    return np.asarray([lookup[row.outcome] for row in samples], dtype=int)


def _log_loss(probabilities, labels):
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def _brier(probabilities, labels):
    expected = np.eye(len(TARGET_CLASSES), dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - expected) ** 2, axis=1)))


def _accuracy(probabilities, labels):
    return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def _clean_numbers(value):
    if isinstance(value, np.ndarray):
        return _clean_numbers(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_clean_numbers(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _clean_numbers(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 12) if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _rejection(horizon_label, samples, config, status, reason, split=None, purged=None):
    payload = {
        "horizon": horizon_label,
        "horizon_minutes": HORIZON_MINUTES[horizon_label],
        "target": "TP_FIRST_SL_FIRST_TIMEOUT_V1",
        "classes": list(TARGET_CLASSES),
        "score_semantics": "NO_SCORE_MODEL_NOT_TRAINED",
        "status": status,
        "promotable": False,
        "reason": reason,
        "resolved_samples": len(samples),
        "class_counts": _class_counts(samples),
        "split_samples": split or {"training": 0, "calibration": 0, "holdout": 0},
        "purged_labels": purged or {"training": 0, "calibration": 0},
        "minimum_samples_required": config.minimum_samples,
        "professional_minimum_samples": PROFESSIONAL_MINIMUM_SAMPLES,
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "model": None,
        "metrics": None,
    }
    payload["model_sha256"] = _sha(payload)
    return payload


def train_horizon_samples(
    horizon_label: str,
    samples: Sequence[HorizonSample],
    config: TrainingConfig | None = None,
):
    """Fit one L2 multinomial model without accessing any other horizon."""

    config = config or TrainingConfig()
    config.validate()
    if horizon_label not in HORIZON_MINUTES:
        raise ValueError(f"Horizonte desconocido: {horizon_label}.")
    samples = tuple(sorted(samples, key=lambda item: item.observed_at))
    if any(len(row.features) != len(FEATURE_NAMES) for row in samples):
        raise ValueError("Longitud de features incompatible con el contrato.")
    if any(row.outcome not in TARGET_CLASSES for row in samples):
        raise ValueError("Clase operacional desconocida.")
    if len(samples) < config.minimum_samples:
        return _rejection(
            horizon_label, samples, config, "INSUFFICIENT_DATA",
            f"{len(samples)}/{config.minimum_samples} observaciones resueltas; no se entrena.",
        )

    training, calibration, holdout, purged = chronological_purged_split(samples)
    split = {"training": len(training), "calibration": len(calibration), "holdout": len(holdout)}
    if min(split.values()) <= 0 or len(holdout) < config.minimum_holdout_samples:
        return _rejection(
            horizon_label, samples, config, "INSUFFICIENT_PURGED_SPLIT",
            "La división cronológica purgada no conserva evidencia suficiente.", split, purged,
        )
    train_counts = _class_counts(training)
    if min(train_counts.values()) < config.minimum_class_samples:
        return _rejection(
            horizon_label, samples, config, "INSUFFICIENT_CLASS_SUPPORT",
            f"Training por clase insuficiente: {train_counts}.", split, purged,
        )

    medians, means, scales = _fit_preprocessor(training)
    x_train = _transform(training, medians, means, scales)
    x_calibration = _transform(calibration, medians, means, scales)
    x_holdout = _transform(holdout, medians, means, scales)
    y_train, y_calibration, y_holdout = _labels(training), _labels(calibration), _labels(holdout)
    candidates = []
    for l2 in sorted(set(float(value) for value in config.l2_grid)):
        weights = _fit_softmax(
            x_train, y_train, l2, config.maximum_iterations, config.learning_rate
        )
        probabilities = _softmax(np.column_stack([np.ones(len(x_calibration)), x_calibration]) @ weights)
        candidates.append({
            "l2": l2,
            "calibration_log_loss": _log_loss(probabilities, y_calibration),
            "calibration_brier": _brier(probabilities, y_calibration),
        })
    selected = min(candidates, key=lambda item: (item["calibration_log_loss"], item["l2"]))

    final_samples = tuple(training) + tuple(calibration)
    x_final = _transform(final_samples, medians, means, scales)
    y_final = _labels(final_samples)
    weights = _fit_softmax(
        x_final, y_final, selected["l2"], config.maximum_iterations, config.learning_rate
    )
    probabilities = _softmax(np.column_stack([np.ones(len(x_holdout)), x_holdout]) @ weights)
    priors = np.asarray(
        [sum(y_final == index) / len(y_final) for index in range(len(TARGET_CLASSES))], dtype=float
    )
    baseline = np.repeat(priors.reshape(1, -1), len(holdout), axis=0)
    holdout_brier = _brier(probabilities, y_holdout)
    baseline_brier = _brier(baseline, y_holdout)
    holdout_log_loss = _log_loss(probabilities, y_holdout)
    baseline_log_loss = _log_loss(baseline, y_holdout)
    professional_sample = len(samples) >= PROFESSIONAL_MINIMUM_SAMPLES
    has_skill = holdout_brier < baseline_brier and holdout_log_loss < baseline_log_loss
    promotable = bool(professional_sample and has_skill)
    status = (
        "TRAINED_OOS_UNCALIBRATED" if promotable else
        "RESEARCH_ONLY_BELOW_PROFESSIONAL_SAMPLE" if not professional_sample else
        "REJECTED_NO_OOS_SKILL"
    )
    reason = (
        "Modelo regularizado supera ambos controles OOS; scores aún no calibrados."
        if promotable else
        "Entrenamiento permitido para investigación, pero no alcanza 300 observaciones."
        if not professional_sample else
        "El modelo no supera Brier y log-loss del prior de entrenamiento en holdout."
    )
    payload = {
        "horizon": horizon_label,
        "horizon_minutes": HORIZON_MINUTES[horizon_label],
        "target": "TP_FIRST_SL_FIRST_TIMEOUT_V1",
        "classes": list(TARGET_CLASSES),
        "score_semantics": "REGULARIZED_SCORE_UNCALIBRATED",
        "status": status,
        "promotable": promotable,
        "reason": reason,
        "resolved_samples": len(samples),
        "class_counts": _class_counts(samples),
        "split_samples": split,
        "split_policy": "CHRONOLOGICAL_60_20_20_PURGED_BY_LABEL_AVAILABILITY",
        "purged_labels": purged,
        "minimum_samples_required": config.minimum_samples,
        "professional_minimum_samples": PROFESSIONAL_MINIMUM_SAMPLES,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "selection": {
            "source": "CALIBRATION_ONLY",
            "candidates": candidates,
            "selected_l2": selected["l2"],
        },
        "model": {
            "family": "MULTINOMIAL_LOGISTIC_L2",
            "intercept_and_coefficients": weights,
            "imputation_medians": medians,
            "standardization_means": means,
            "standardization_scales": scales,
            "training_class_priors": priors,
        },
        "metrics": {
            "source": "HOLDOUT_ONLY",
            "multiclass_brier": holdout_brier,
            "baseline_brier": baseline_brier,
            "log_loss": holdout_log_loss,
            "baseline_log_loss": baseline_log_loss,
            "accuracy": _accuracy(probabilities, y_holdout),
            "holdout_class_counts": _class_counts(holdout),
        },
    }
    payload = _clean_numbers(payload)
    payload["model_sha256"] = _sha(payload)
    return payload


def train_replay_horizon_models(replay: Mapping, config: TrainingConfig | None = None):
    """Train six isolated models from one verified replay artifact."""

    validate_historical_replay(dict(replay))
    config = config or TrainingConfig()
    config.validate()
    models = [
        train_horizon_samples(label, horizon_samples(replay, label), config)
        for label in HORIZON_MINUTES
    ]
    deterministic = {
        "contract": HORIZON_MODEL_CONTRACT,
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "symbol": str(replay.get("symbol") or "").upper(),
        "source_replay_id": replay.get("replay_id"),
        "source_replay_sha256": replay.get("artifact_sha256"),
        "target": "TP_FIRST_SL_FIRST_TIMEOUT_V1",
        "horizon_isolation": "ONE_MODEL_PER_SYMBOL_AND_HORIZON",
        "live_oos_policy": "HISTORICAL_TRAINING_NEVER_COUNTS_AS_LIVE_OOS",
        "configuration": _clean_numbers({
            "minimum_samples": config.minimum_samples,
            "minimum_class_samples": config.minimum_class_samples,
            "minimum_holdout_samples": config.minimum_holdout_samples,
            "l2_grid": config.l2_grid,
            "maximum_iterations": config.maximum_iterations,
            "learning_rate": config.learning_rate,
        }),
        "models": models,
    }
    payload = {
        **deterministic,
        "training_run_id": _sha(deterministic),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["artifact_sha256"] = _sha(payload)
    validate_horizon_model_artifact(payload)
    return payload


def validate_horizon_model_artifact(payload: Mapping) -> bool:
    if not isinstance(payload, Mapping) or payload.get("contract") != HORIZON_MODEL_CONTRACT:
        raise ValueError("Contrato de modelos por horizonte desconocido.")
    models = payload.get("models")
    if not isinstance(models, list) or [row.get("horizon") for row in models] != list(HORIZON_MINUTES):
        raise ValueError("El artefacto debe contener seis modelos aislados y ordenados.")
    for row in models:
        unsigned = {key: value for key, value in row.items() if key != "model_sha256"}
        if row.get("model_sha256") != _sha(unsigned):
            raise ValueError(f"Firma de modelo inválida: {row.get('horizon')}.")
        if row.get("target") != "TP_FIRST_SL_FIRST_TIMEOUT_V1":
            raise ValueError("Objetivo operacional inválido.")
        if row.get("model") is not None and row.get("score_semantics") != "REGULARIZED_SCORE_UNCALIBRATED":
            raise ValueError("Un modelo entrenado debe declarar scores no calibrados.")
    unsigned_artifact = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload.get("artifact_sha256") != _sha(unsigned_artifact):
        raise ValueError("Firma integral de modelos por horizonte inválida.")
    deterministic = {
        key: payload.get(key) for key in (
            "contract", "trainer_sha256", "numpy_version", "symbol",
            "source_replay_id", "source_replay_sha256", "target",
            "horizon_isolation", "live_oos_policy", "configuration", "models",
        )
    }
    if payload.get("training_run_id") != _sha(deterministic):
        raise ValueError("Identidad de entrenamiento inválida.")
    return True


def predict_uncalibrated_scores(model_record: Mapping, features: Sequence[float]):
    """Return named softmax scores; rejected/untrained models fail closed."""

    if not model_record.get("promotable") or model_record.get("model") is None:
        raise ValueError("El modelo no está aprobado para inferencia operativa.")
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("Vector de inferencia incompatible.")
    model = model_record["model"]
    medians = np.asarray(model["imputation_medians"], dtype=float)
    values = np.asarray(features, dtype=float)
    values = np.where(np.isfinite(values), values, medians)
    transformed = (values - np.asarray(model["standardization_means"], dtype=float)) / np.asarray(
        model["standardization_scales"], dtype=float
    )
    design = np.concatenate([[1.0], transformed]).reshape(1, -1)
    probabilities = _softmax(design @ np.asarray(model["intercept_and_coefficients"], dtype=float))[0]
    return {
        "semantics": "REGULARIZED_SCORE_UNCALIBRATED",
        "scores": {name: round(float(value) * 100.0, 6) for name, value in zip(TARGET_CLASSES, probabilities)},
        "sum": round(float(probabilities.sum()) * 100.0, 6),
    }


def write_horizon_model_artifact(payload: Mapping, path) -> Path:
    validate_horizon_model_artifact(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(canonical(payload), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
