"""Nested, embargoed walk-forward validation with a sealed final holdout.

Every symbol/horizon is evaluated independently.  Hyperparameters are chosen
inside inner expanding folds; outer folds estimate development performance.
The final holdout is committed before any fitting and is opened only after the
walk-forward protocol has been frozen and has passed its development gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .closed_bars import NY, _calendar
from .historical_replay import validate_historical_replay
from .operational_target import TARGET_VERSION
from .operational_calibration import (
    DEFAULT_TEMPERATURE_GRID,
    select_and_validate_temperature,
    temperature_scale,
)
from .holdout_registry import (
    DEFAULT_HOLDOUT_REGISTRY, reserve_holdout_opening, verify_holdout_opening,
)
from .metric_uncertainty import proportion_interval, score_intervals
from .horizon_models import (
    FEATURE_NAMES,
    FeatureContractMismatch,
    MODEL_FEATURE_VERSION,
    HORIZON_MODEL_CONTRACT,
    PROFESSIONAL_MINIMUM_SAMPLES,
    TARGET_CLASSES,
    HorizonSample,
    _accuracy,
    _brier,
    _class_counts,
    _clean_numbers,
    _fit_preprocessor,
    _fit_softmax,
    _labels,
    _log_loss,
    _sha,
    _softmax,
    _transform,
    horizon_samples,
    replay_population_report,
)
from ..services.directional_collection import HORIZON_MINUTES
from ..services.model_observations import SESSION_HORIZONS, canonical


NESTED_WALK_FORWARD_CONTRACT = "NESTED_EMBARGOED_WALK_FORWARD_V4"
PREVIOUS_NESTED_WALK_FORWARD_CONTRACT = "NESTED_EMBARGOED_WALK_FORWARD_V3"
OLDER_NESTED_WALK_FORWARD_CONTRACT = "NESTED_EMBARGOED_WALK_FORWARD_V2"
LEGACY_NESTED_WALK_FORWARD_CONTRACT = "NESTED_EMBARGOED_WALK_FORWARD_V1"
EMBARGO_SESSIONS = {
    "1 Hora": 1,
    "6 Horas": 1,
    "1 Día": 1,
    "1 Semana": SESSION_HORIZONS[10_080],
    "1 Mes": SESSION_HORIZONS[43_200],
    "6 Meses": SESSION_HORIZONS[259_200],
}


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    minimum_samples: int = PROFESSIONAL_MINIMUM_SAMPLES
    minimum_class_samples: int = 20
    final_holdout_fraction: float = 0.20
    minimum_final_holdout: int = 60
    outer_folds: int = 4
    inner_folds: int = 3
    minimum_outer_training: int = 100
    minimum_outer_test: int = 20
    minimum_inner_training: int = 50
    minimum_inner_validation: int = 12
    minimum_skillful_fold_ratio: float = 0.50
    minimum_calibration_fit: int = 60
    minimum_calibration_validation: int = 30
    minimum_calibration_per_class: int = 5
    temperature_grid: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    maximum_iterations: int = 1_000
    learning_rate: float = 0.08

    def validate(self) -> None:
        if self.minimum_samples < PROFESSIONAL_MINIMUM_SAMPLES:
            raise ValueError(
                f"minimum_samples no puede ser menor que {PROFESSIONAL_MINIMUM_SAMPLES}."
            )
        if self.minimum_class_samples < 1:
            raise ValueError("minimum_class_samples debe ser positivo.")
        if not 0.10 <= self.final_holdout_fraction <= 0.30:
            raise ValueError("final_holdout_fraction debe estar entre 0.10 y 0.30.")
        if self.minimum_final_holdout < 20:
            raise ValueError("El holdout final debe contener al menos 20 observaciones.")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Walk-forward anidado requiere al menos dos folds por nivel.")
        if min(
            self.minimum_outer_training,
            self.minimum_outer_test,
            self.minimum_inner_training,
            self.minimum_inner_validation,
        ) < 1:
            raise ValueError("Los mínimos de folds deben ser positivos.")
        if not 0.0 < self.minimum_skillful_fold_ratio <= 1.0:
            raise ValueError("minimum_skillful_fold_ratio debe estar en (0,1].")
        if min(
            self.minimum_calibration_fit,
            self.minimum_calibration_validation,
            self.minimum_calibration_per_class,
        ) < 1:
            raise ValueError("Los mínimos de calibración deben ser positivos.")
        if not self.temperature_grid or 1.0 not in self.temperature_grid or any(
            not math.isfinite(value) or value <= 0 for value in self.temperature_grid
        ):
            raise ValueError("temperature_grid debe ser positiva e incluir 1.0.")
        if not self.l2_grid or any(not math.isfinite(value) or value <= 0 for value in self.l2_grid):
            raise ValueError("l2_grid debe contener valores positivos finitos.")
        if self.maximum_iterations < 100 or not 0 < self.learning_rate <= 1:
            raise ValueError("Configuración de optimización inválida.")


def minimum_samples_for_horizon(horizon_label: str, config: WalkForwardConfig) -> int:
    """Conservative one-cut-per-session floor after all three embargo layers.

    The first outer training block must survive its own embargo and leave an
    inner training block after another embargo. At least ``outer_folds`` test
    blocks and chronological OOF calibration fit/validation must remain. We
    then add the final holdout embargo and solve n - holdout(n) - embargo >=
    required development. Actual XNYS cuts are still purged and checked below.
    """
    embargo = EMBARGO_SESSIONS[horizon_label]
    inner_training_capacity = max(
        config.minimum_inner_training + config.inner_folds * config.minimum_inner_validation,
        2 * (config.minimum_inner_training + embargo),
    )
    outer_initial = embargo + max(config.minimum_outer_training, inner_training_capacity)
    oof_required = max(
        math.ceil((config.minimum_calibration_fit + embargo) / 0.60),
        math.ceil(config.minimum_calibration_validation / 0.40),
    )
    development_required = 2 * max(
        outer_initial,
        config.outer_folds * config.minimum_outer_test,
        oof_required,
    )
    n = max(config.minimum_samples, config.minimum_final_holdout + 1)
    while n - max(config.minimum_final_holdout, math.ceil(n * config.final_holdout_fraction)) - embargo < development_required:
        n += 1
    return n


def _development_stop_evidence(samples: Sequence[HorizonSample]) -> dict:
    """Observed stop losses from development only, never the final holdout."""
    stop_rows = [row for row in samples if row.outcome == "SL_FIRST" and row.entry_side == "LONG"]
    multiples = [float(row.sl_loss_multiple) for row in stop_rows
                 if row.sl_loss_multiple is not None and math.isfinite(row.sl_loss_multiple)
                 and row.sl_loss_multiple > 0]
    return {
        "source": "ELIGIBLE_LONG_DEVELOPMENT_ONLY",
        "target": TARGET_VERSION,
        "observed_sl_samples": len(multiples),
        "gap_samples": sum("gap-open" in (row.sl_exit_source or "") for row in stop_rows),
        "gross_loss_multiples": sorted(multiples),
        "semantics": "OBSERVED_EXIT_VS_FROZEN_STOP_DISTANCE_BEFORE_FILL_COSTS",
    }


def _sample_manifest(sample: HorizonSample) -> dict:
    return {
        "cut_id": sample.cut_id,
        "observed_at": sample.observed_at,
        "label_available_at": sample.label_available_at,
        "features_sha256": hashlib.sha256(
            canonical(_clean_numbers(sample.features)).encode("utf-8")
        ).hexdigest(),
        "outcome": sample.outcome,
        "eligible_at_emission": sample.eligible_at_emission,
        "target_version": sample.target_version,
        "net_pnl_per_share": sample.net_pnl_per_share,
        "feature_version": sample.feature_version,
        "available_features": list(sample.available_features),
        "sl_loss_multiple": sample.sl_loss_multiple,
        "sl_exit_source": sample.sl_exit_source,
        "entry_side": sample.entry_side,
    }


def _session_day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Walk-forward requiere timestamps timezone-aware.")
    return pd.Timestamp(stamp.tz_convert(NY).date())


def embargo_and_purge_before(
    samples: Sequence[HorizonSample],
    boundary_at: str,
    embargo_sessions: int,
):
    """Keep rows known before boundary and outside its XNYS embargo."""

    if isinstance(embargo_sessions, bool) or int(embargo_sessions) < 0:
        raise ValueError("embargo_sessions debe ser un entero no negativo.")
    boundary = pd.Timestamp(boundary_at)
    boundary_day = _session_day(boundary)
    years = [_session_day(row.observed_at).year for row in samples] or [boundary_day.year]
    schedule = _calendar(min(years) - 1, boundary_day.year + 1).schedule
    if boundary_day not in schedule.index:
        raise ValueError("El límite del fold no pertenece a una sesión XNYS.")
    boundary_position = int(schedule.index.get_loc(boundary_day))
    last_allowed_position = boundary_position - int(embargo_sessions) - 1
    last_allowed_day = schedule.index[last_allowed_position] if last_allowed_position >= 0 else None
    kept = []
    embargoed = 0
    label_unknown = 0
    for row in samples:
        observed = pd.Timestamp(row.observed_at)
        known = pd.Timestamp(row.label_available_at)
        if observed >= boundary:
            continue
        day = _session_day(observed)
        if last_allowed_day is None or day > last_allowed_day:
            embargoed += 1
            continue
        if known >= boundary:
            label_unknown += 1
            continue
        kept.append(row)
    return tuple(kept), {
        "embargoed": embargoed,
        "label_unknown_at_boundary": label_unknown,
    }


def _expanding_windows(size: int, folds: int, minimum_training: int, minimum_test: int):
    initial = max(int(minimum_training), size // 2)
    remaining = size - initial
    if remaining < folds * minimum_test:
        return ()
    base, extra = divmod(remaining, folds)
    windows = []
    start = initial
    for index in range(folds):
        width = base + (1 if index < extra else 0)
        windows.append((start, start + width))
        start += width
    return tuple(windows)


def _candidate_metrics(
    samples: Sequence[HorizonSample],
    *,
    embargo_sessions: int,
    config: WalkForwardConfig,
):
    windows = _expanding_windows(
        len(samples), config.inner_folds,
        config.minimum_inner_training, config.minimum_inner_validation,
    )
    if len(windows) != config.inner_folds:
        return None, (), "Folds internos insuficientes después del embargo."
    aggregates = {float(value): [] for value in sorted(set(config.l2_grid))}
    fold_manifest = []
    for fold_index, (start, end) in enumerate(windows, start=1):
        validation = tuple(samples[start:end])
        training, exclusions = embargo_and_purge_before(
            samples[:start], validation[0].observed_at, embargo_sessions
        )
        counts = _class_counts(training)
        if len(training) < config.minimum_inner_training or min(counts.values()) < config.minimum_class_samples:
            return None, (), f"Soporte de clase insuficiente en fold interno {fold_index}: {counts}."
        medians, means, scales = _fit_preprocessor(training)
        x_train = _transform(training, medians, means, scales)
        x_validation = _transform(validation, medians, means, scales)
        y_train, y_validation = _labels(training), _labels(validation)
        scores = []
        for l2 in aggregates:
            weights = _fit_softmax(
                x_train, y_train, l2, config.maximum_iterations, config.learning_rate
            )
            probabilities = _softmax(
                np.column_stack([np.ones(len(x_validation)), x_validation]) @ weights
            )
            intervals = score_intervals(probabilities, y_validation)
            row = {
                "l2": l2,
                "log_loss": _log_loss(probabilities, y_validation),
                "brier": _brier(probabilities, y_validation),
                "confidence_intervals": {
                    name: intervals[name] for name in ("brier", "log_loss")
                },
                "samples": len(validation),
            }
            aggregates[l2].append(row)
            scores.append(row)
        fold_manifest.append({
            "fold": fold_index,
            "training_samples": len(training),
            "validation_samples": len(validation),
            "training_class_counts": counts,
            "validation_start": validation[0].observed_at,
            "validation_end": validation[-1].observed_at,
            "excluded": exclusions,
            "candidates": scores,
        })
    ranking = []
    for l2, values in aggregates.items():
        total = sum(row["samples"] for row in values)
        ranking.append({
            "l2": l2,
            "mean_log_loss": sum(row["log_loss"] * row["samples"] for row in values) / total,
            "mean_brier": sum(row["brier"] * row["samples"] for row in values) / total,
            "validation_samples": total,
        })
    selected = min(ranking, key=lambda row: (row["mean_log_loss"], row["mean_brier"], row["l2"]))
    return float(selected["l2"]), tuple(fold_manifest), ""


def _evaluate_fold(training, test, selected_l2, config, *, return_predictions=False):
    medians, means, scales = _fit_preprocessor(training)
    x_training = _transform(training, medians, means, scales)
    x_test = _transform(test, medians, means, scales)
    y_training, y_test = _labels(training), _labels(test)
    weights = _fit_softmax(
        x_training, y_training, selected_l2,
        config.maximum_iterations, config.learning_rate,
    )
    probabilities = _softmax(np.column_stack([np.ones(len(x_test)), x_test]) @ weights)
    priors = np.asarray(
        [sum(y_training == index) / len(y_training) for index in range(len(TARGET_CLASSES))],
        dtype=float,
    )
    baseline = np.repeat(priors.reshape(1, -1), len(test), axis=0)
    prediction_intervals = score_intervals(probabilities, y_test)
    baseline_intervals = score_intervals(baseline, y_test)
    metrics = {
        "samples": len(test),
        "brier": _brier(probabilities, y_test),
        "baseline_brier": _brier(baseline, y_test),
        "log_loss": _log_loss(probabilities, y_test),
        "baseline_log_loss": _log_loss(baseline, y_test),
        "accuracy": _accuracy(probabilities, y_test),
        "confidence_intervals": {
            **prediction_intervals,
            "baseline_brier": baseline_intervals["brier"],
            "baseline_log_loss": baseline_intervals["log_loss"],
        },
    }
    if return_predictions:
        observations = tuple(
            (sample, probabilities[index].copy(), priors.copy())
            for index, sample in enumerate(test)
        )
        return metrics, observations
    return metrics


def _aggregate_outer(folds, observations):
    total = sum(row["metrics"]["samples"] for row in folds)
    names = ("brier", "baseline_brier", "log_loss", "baseline_log_loss", "accuracy")
    metrics = {
        name: sum(row["metrics"][name] * row["metrics"]["samples"] for row in folds) / total
        for name in names
    }
    skillful = sum(
        row["metrics"]["brier"] < row["metrics"]["baseline_brier"]
        and row["metrics"]["log_loss"] < row["metrics"]["baseline_log_loss"]
        for row in folds
    )
    y = _labels([item[0] for item in observations])
    predictions = np.asarray([item[1] for item in observations])
    baselines = np.asarray([item[2] for item in observations])
    prediction_intervals = score_intervals(predictions, y)
    baseline_intervals = score_intervals(baselines, y)
    return {
        **metrics,
        "confidence_intervals": {
            **prediction_intervals,
            "baseline_brier": baseline_intervals["brier"],
            "baseline_log_loss": baseline_intervals["log_loss"],
            "skillful_fold_ratio": proportion_interval(skillful, len(folds)),
        },
        "samples": total,
        "skillful_folds": skillful,
        "folds": len(folds),
        "skillful_fold_ratio": skillful / len(folds),
    }


def _calibrate_development_oof(observations, embargo_sessions, config):
    """Fit on early outer OOF scores and verify on later outer OOF scores."""

    ordered = tuple(sorted(observations, key=lambda item: item[0].observed_at))
    first_validation = int(len(ordered) * 0.60)
    if first_validation < 1 or first_validation >= len(ordered):
        return {"status": "INSUFFICIENT_OOF_CALIBRATION", "approved": False}
    raw_fit = ordered[:first_validation]
    validation = ordered[first_validation:]
    eligible, excluded = embargo_and_purge_before(
        [item[0] for item in raw_fit],
        validation[0][0].observed_at,
        embargo_sessions,
    )
    retained = {(item.observed_at, item.cut_id) for item in eligible}
    fit = tuple(
        item for item in raw_fit
        if (item[0].observed_at, item[0].cut_id) in retained
    )
    evidence = [
        {
            "sample": _sample_manifest(sample),
            "raw_scores": list(probabilities),
            "baseline": list(priors),
        }
        for sample, probabilities, priors in ordered
    ]
    metadata = {
        "oof_evidence_sha256": _sha(_clean_numbers(evidence)),
        "oof_samples": len(ordered),
        "fit_start": fit[0][0].observed_at if fit else None,
        "fit_end": fit[-1][0].observed_at if fit else None,
        "validation_start": validation[0][0].observed_at,
        "validation_end": validation[-1][0].observed_at,
        "excluded_before_validation": excluded,
    }
    if not fit:
        return {
            **metadata,
            "status": "INSUFFICIENT_OOF_CALIBRATION",
            "approved": False,
            "fit_samples": 0,
            "validation_samples": len(validation),
        }
    labels = {name: index for index, name in enumerate(TARGET_CLASSES)}
    result = select_and_validate_temperature(
        [item[1] for item in fit],
        [labels[item[0].outcome] for item in fit],
        [item[1] for item in validation],
        [labels[item[0].outcome] for item in validation],
        [item[2] for item in validation],
        grid=config.temperature_grid,
        minimum_fit=config.minimum_calibration_fit,
        minimum_validation=config.minimum_calibration_validation,
        minimum_per_class=config.minimum_calibration_per_class,
    )
    return _clean_numbers({**result, **metadata})


def _rejected(
    horizon_label,
    samples,
    config,
    embargo_sessions,
    holdout,
    holdout_commitment,
    status,
    reason,
    **details,
):
    payload = {
        "horizon": horizon_label,
        "horizon_minutes": HORIZON_MINUTES[horizon_label],
        "base_model_contract": HORIZON_MODEL_CONTRACT,
        "target": TARGET_VERSION,
        "classes": list(TARGET_CLASSES),
        "status": status,
        "promotable": False,
        "reason": reason,
        "resolved_samples": len(samples),
        # Class counts of a sealed holdout would expose its outcome mix.
        "class_counts": None,
        "embargo_sessions": embargo_sessions,
        "final_holdout": {
            "status": "SEALED_UNOPENED",
            "samples": len(holdout),
            "commitment_sha256": holdout_commitment,
            "metrics": None,
        },
        "walk_forward": details.get("walk_forward"),
        "calibration": details.get("calibration"),
        "protocol_frozen_sha256": details.get("protocol_frozen_sha256"),
        "development_exclusions": details.get("development_exclusions"),
        "development_class_counts": details.get("development_class_counts"),
        "model": None,
        "score_semantics": "NO_SCORE_MODEL_NOT_APPROVED",
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "minimum_samples_required": minimum_samples_for_horizon(horizon_label, config),
    }
    payload = _clean_numbers(payload)
    payload["result_sha256"] = _sha(payload)
    return payload


def nested_walk_forward_horizon(
    horizon_label: str,
    samples: Sequence[HorizonSample],
    config: WalkForwardConfig | None = None,
    *,
    symbol: str = "UNSPECIFIED",
    dataset_sha256: str | None = None,
    registry_path: str | Path = DEFAULT_HOLDOUT_REGISTRY,
):
    """Validate one horizon and open its final holdout only after protocol freeze."""

    config = config or WalkForwardConfig()
    config.validate()
    if horizon_label not in HORIZON_MINUTES:
        raise ValueError(f"Horizonte desconocido: {horizon_label}.")
    embargo_sessions = EMBARGO_SESSIONS[horizon_label]
    # Eligibility is frozen at emission. Never validate a hypothetical barrier
    # as though an entry could actually have been placed.
    samples = tuple(sorted(
        (row for row in samples if row.eligible_at_emission and row.target_version == TARGET_VERSION),
        key=lambda row: row.observed_at,
    ))
    if any(len(row.features) != len(FEATURE_NAMES) for row in samples):
        raise ValueError("Longitud de features incompatible con el contrato.")
    if samples and (any(row.feature_version != MODEL_FEATURE_VERSION for row in samples)
                    or any(row.available_features != tuple(FEATURE_NAMES) for row in samples)
                    or any(row.available_features != tuple(
                        name for name, value in zip(FEATURE_NAMES, row.features)
                        if math.isfinite(value)
                    ) for row in samples)):
        raise FeatureContractMismatch("Features train/replay incompatibles: versión o cobertura histórica incompleta.")
    if any(row.outcome not in TARGET_CLASSES for row in samples):
        raise ValueError("Clase operacional desconocida.")
    holdout_size = max(
        config.minimum_final_holdout,
        int(math.ceil(len(samples) * config.final_holdout_fraction)),
    )
    if holdout_size >= len(samples):
        holdout = samples
        development_raw = ()
    else:
        holdout = samples[-holdout_size:]
        development_raw = samples[:-holdout_size]
    holdout_commitment = _sha([_sample_manifest(row) for row in holdout])
    required_samples = minimum_samples_for_horizon(horizon_label, config)
    if len(samples) < required_samples:
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "INSUFFICIENT_DATA_FINAL_HOLDOUT_UNOPENED",
            f"{len(samples)}/{required_samples} observaciones elegibles; "
            f"embargo de {embargo_sessions} sesiones y desarrollo anidado protegidos.",
        )
    if len(holdout) < config.minimum_final_holdout or not development_raw:
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "INSUFFICIENT_FINAL_HOLDOUT",
            "No existe un holdout final del tamaño preregistrado.",
        )

    development, final_exclusions = embargo_and_purge_before(
        development_raw, holdout[0].observed_at, embargo_sessions
    )
    windows = _expanding_windows(
        len(development), config.outer_folds,
        config.minimum_outer_training, config.minimum_outer_test,
    )
    if len(windows) != config.outer_folds:
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "INSUFFICIENT_EMBARGOED_DEVELOPMENT",
            "El desarrollo restante no alcanza para los folds externos.",
            development_exclusions=final_exclusions,
        )

    outer_results = []
    outer_oof = []
    for fold_index, (start, end) in enumerate(windows, start=1):
        test = tuple(development[start:end])
        training, exclusions = embargo_and_purge_before(
            development[:start], test[0].observed_at, embargo_sessions
        )
        counts = _class_counts(training)
        if len(training) < config.minimum_outer_training or min(counts.values()) < config.minimum_class_samples:
            return _rejected(
                horizon_label, samples, config, embargo_sessions, holdout,
                holdout_commitment, "INSUFFICIENT_OUTER_FOLD_SUPPORT",
                f"Fold externo {fold_index} sin soporte suficiente: {counts}.",
                development_exclusions=final_exclusions,
            )
        selected_l2, inner_folds, error = _candidate_metrics(
            training, embargo_sessions=embargo_sessions, config=config
        )
        if selected_l2 is None:
            return _rejected(
                horizon_label, samples, config, embargo_sessions, holdout,
                holdout_commitment, "INSUFFICIENT_INNER_FOLD_SUPPORT",
                f"Fold externo {fold_index}: {error}",
                development_exclusions=final_exclusions,
            )
        fold_metrics, fold_oof = _evaluate_fold(
            training, test, selected_l2, config, return_predictions=True
        )
        outer_oof.extend(fold_oof)
        outer_results.append({
            "fold": fold_index,
            "training_samples": len(training),
            "test_samples": len(test),
            "training_class_counts": counts,
            "test_start": test[0].observed_at,
            "test_end": test[-1].observed_at,
            "excluded": exclusions,
            "selected_l2": selected_l2,
            "inner_folds": inner_folds,
            "metrics": fold_metrics,
        })
    aggregate = _aggregate_outer(outer_results, outer_oof)
    walk_forward = {
        "policy": "EXPANDING_NESTED_WALK_FORWARD_XNYS_EMBARGO_V1",
        "outer_folds": outer_results,
        "aggregate": aggregate,
    }
    development_skill = (
        aggregate["brier"] < aggregate["baseline_brier"]
        and aggregate["log_loss"] < aggregate["baseline_log_loss"]
        and aggregate["skillful_fold_ratio"] >= config.minimum_skillful_fold_ratio
    )
    if not development_skill:
        protocol = {
            "horizon": horizon_label,
            "embargo_sessions": embargo_sessions,
            "walk_forward": walk_forward,
            "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
            "decision": "REJECTED_BEFORE_FINAL_HOLDOUT",
        }
        protocol_sha = _sha(_clean_numbers(protocol))
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "REJECTED_WALK_FORWARD_FINAL_HOLDOUT_UNOPENED",
            "El modelo no supera el baseline de desarrollo; holdout final no abierto.",
            walk_forward=walk_forward,
            protocol_frozen_sha256=protocol_sha,
            development_exclusions=final_exclusions,
        )

    final_l2, final_inner_folds, error = _candidate_metrics(
        development, embargo_sessions=embargo_sessions, config=config
    )
    if final_l2 is None:
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "INSUFFICIENT_FINAL_INNER_SUPPORT",
            error, walk_forward=walk_forward, development_exclusions=final_exclusions,
        )
    calibration = _calibrate_development_oof(outer_oof, embargo_sessions, config)
    if not calibration.get("approved"):
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "REJECTED_CALIBRATION_FINAL_HOLDOUT_UNOPENED",
            "El ajuste multiclase no mejora al score original y al baseline en OOF posterior.",
            walk_forward=walk_forward,
            calibration=calibration,
            development_exclusions=final_exclusions,
        )
    stop_evidence = _development_stop_evidence(development)
    if stop_evidence["observed_sl_samples"] < config.minimum_class_samples:
        return _rejected(
            horizon_label, samples, config, embargo_sessions, holdout,
            holdout_commitment, "INSUFFICIENT_OBSERVED_STOP_EVIDENCE",
            "No hay pérdidas SL_FIRST observadas suficientes para estimar gaps.",
            walk_forward=walk_forward, calibration=calibration,
            development_exclusions=final_exclusions,
        )
    frozen_protocol = _clean_numbers({
        "contract": NESTED_WALK_FORWARD_CONTRACT,
        "horizon": horizon_label,
        "horizon_minutes": HORIZON_MINUTES[horizon_label],
        "embargo_sessions": embargo_sessions,
        "target": TARGET_VERSION,
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "selected_l2": final_l2,
        "walk_forward": walk_forward,
        "final_inner_folds": final_inner_folds,
        "calibration": calibration,
        "execution_evidence": stop_evidence,
        "holdout_commitment_sha256": holdout_commitment,
    })
    protocol_sha = _sha(frozen_protocol)

    # Persistent reservation precedes all holdout-label access. A duplicate
    # dataset or overlapping cut raises, even across processes and restarts.
    dataset_fingerprint = dataset_sha256 or _sha([
        _sample_manifest(row) for row in samples
    ])
    opening = reserve_holdout_opening(
        registry_path=registry_path,
        dataset_sha256=dataset_fingerprint,
        symbol=symbol.upper(),
        horizon=horizon_label,
        protocol_sha256=protocol_sha,
        commitment_sha256=holdout_commitment,
        members=tuple((row.cut_id, row.observed_at) for row in holdout),
    )
    # HOLDOUT IS OPENED ONLY BELOW THIS LINE, after durable registration.
    medians, means, scales = _fit_preprocessor(development)
    x_development = _transform(development, medians, means, scales)
    x_holdout = _transform(holdout, medians, means, scales)
    y_development, y_holdout = _labels(development), _labels(holdout)
    weights = _fit_softmax(
        x_development, y_development, final_l2,
        config.maximum_iterations, config.learning_rate,
    )
    raw_probabilities = _softmax(np.column_stack([np.ones(len(x_holdout)), x_holdout]) @ weights)
    probabilities = temperature_scale(raw_probabilities, calibration["temperature"])
    priors = np.asarray(
        [sum(y_development == index) / len(y_development) for index in range(len(TARGET_CLASSES))],
        dtype=float,
    )
    baseline = np.repeat(priors.reshape(1, -1), len(holdout), axis=0)
    calibrated_intervals = score_intervals(probabilities, y_holdout)
    raw_intervals = score_intervals(raw_probabilities, y_holdout)
    baseline_intervals = score_intervals(baseline, y_holdout)
    final_metrics = {
        "source": "SEALED_FINAL_HOLDOUT_ONLY",
        "samples": len(holdout),
        "class_counts": _class_counts(holdout),
        "brier": _brier(probabilities, y_holdout),
        "baseline_brier": _brier(baseline, y_holdout),
        "log_loss": _log_loss(probabilities, y_holdout),
        "baseline_log_loss": _log_loss(baseline, y_holdout),
        "raw_brier": _brier(raw_probabilities, y_holdout),
        "raw_log_loss": _log_loss(raw_probabilities, y_holdout),
        "accuracy": _accuracy(probabilities, y_holdout),
        "confidence_intervals": {
            **calibrated_intervals,
            "baseline_brier": baseline_intervals["brier"],
            "baseline_log_loss": baseline_intervals["log_loss"],
            "raw_brier": raw_intervals["brier"],
            "raw_log_loss": raw_intervals["log_loss"],
        },
        "opened_against_protocol_sha256": protocol_sha,
    }
    final_class_support = min(final_metrics["class_counts"].values()) >= config.minimum_calibration_per_class
    final_skill = (
        final_class_support
        and final_metrics["brier"] < final_metrics["baseline_brier"]
        and final_metrics["log_loss"] < final_metrics["baseline_log_loss"]
        and final_metrics["brier"] < final_metrics["raw_brier"]
        and final_metrics["log_loss"] < final_metrics["raw_log_loss"]
    )
    status = (
        "REJECTED_FINAL_HOLDOUT_CLASS_SUPPORT" if not final_class_support else
        "APPROVED_SEALED_HOLDOUT_CALIBRATED" if final_skill else
        "REJECTED_CALIBRATED_FINAL_HOLDOUT"
    )
    reason = (
        "Holdout sin representación mínima de TP_FIRST, SL_FIRST y TIMEOUT."
        if not final_class_support else
        "Walk-forward, calibración OOF y holdout final superan los controles preregistrados."
        if final_skill else
        "La calibración no mejoró al score original y al baseline en holdout final."
    )
    payload = {
        "horizon": horizon_label,
        "horizon_minutes": HORIZON_MINUTES[horizon_label],
        "base_model_contract": HORIZON_MODEL_CONTRACT,
        "target": TARGET_VERSION,
        "classes": list(TARGET_CLASSES),
        "status": status,
        "promotable": bool(final_skill),
        "reason": reason,
        "resolved_samples": len(samples),
        "class_counts": _class_counts(samples),
        "embargo_sessions": embargo_sessions,
        "development_samples": len(development),
        "development_exclusions": final_exclusions,
        "walk_forward": walk_forward,
        "calibration": calibration,
        "execution_evidence": stop_evidence,
        "frozen_protocol": frozen_protocol,
        "protocol_frozen_sha256": protocol_sha,
        "final_holdout": {
            "status": "OPENED_ONCE_AFTER_PROTOCOL_FREEZE",
            "samples": len(holdout),
            "commitment_sha256": holdout_commitment,
            "opening": opening,
            "metrics": final_metrics,
        },
        "model": {
            "family": "MULTINOMIAL_LOGISTIC_L2",
            "trained_on": "DEVELOPMENT_ONLY_FINAL_HOLDOUT_EXCLUDED",
            "selected_l2": final_l2,
            "calibration": {
                "method": calibration["method"],
                "temperature": calibration["temperature"],
                "oof_evidence_sha256": calibration["oof_evidence_sha256"],
            },
            "intercept_and_coefficients": weights,
            "imputation_medians": medians,
            "standardization_means": means,
            "standardization_scales": scales,
            "training_class_priors": priors,
        } if final_skill else None,
        "score_semantics": (
            "HISTORICAL_OOS_CALIBRATED_PRELIMINARY" if final_skill else "NO_SCORE_MODEL_NOT_APPROVED"
        ),
        "feature_names": list(FEATURE_NAMES),
        "feature_version": MODEL_FEATURE_VERSION,
        "available_features": list(samples[0].available_features),
        "feature_schema_sha256": _sha(list(FEATURE_NAMES)),
        "minimum_samples_required": required_samples,
    }
    payload = _clean_numbers(payload)
    payload["result_sha256"] = _sha(payload)
    return payload


def run_nested_walk_forward_replay(
    replay: Mapping, config: WalkForwardConfig | None = None,
    *, registry_path: str | Path = DEFAULT_HOLDOUT_REGISTRY,
):
    validate_historical_replay(dict(replay))
    config = config or WalkForwardConfig()
    config.validate()
    results = []
    for label in HORIZON_MINUTES:
        population = replay_population_report(replay, label)
        result = nested_walk_forward_horizon(
            label, horizon_samples(replay, label), config,
            symbol=str(replay.get("symbol") or "UNSPECIFIED"),
            dataset_sha256=str(replay["dataset_sha256"]),
            registry_path=registry_path,
        )
        result["population"] = population
        required = minimum_samples_for_horizon(label, config)
        if population["trainable_current_target_n"] < required:
            result["status"] = "EVIDENCIA_INSUFICIENTE"
            result["reason"] = (
                "Sin evidencia de entradas ejecutables suficiente: "
                f"{population['trainable_current_target_n']}/{required} "
                "muestras elegibles del contrato vigente. Holdout final sellado."
            )
            result["promotable"] = False
        result["result_sha256"] = _sha({
            key: value for key, value in result.items() if key != "result_sha256"
        })
        results.append(result)
    deterministic = {
        "contract": NESTED_WALK_FORWARD_CONTRACT,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "symbol": str(replay.get("symbol") or "").upper(),
        "source_replay_id": replay.get("replay_id"),
        "source_replay_sha256": replay.get("artifact_sha256"),
        "horizon_isolation": "ONE_NESTED_PROTOCOL_PER_SYMBOL_AND_HORIZON",
        "holdout_policy": "SEALED_FINAL_HOLDOUT_OPEN_ONLY_AFTER_DEVELOPMENT_GATE",
        "configuration": _clean_numbers(config.__dict__ if hasattr(config, "__dict__") else {
            name: getattr(config, name) for name in config.__slots__
        }),
        "results": results,
    }
    payload = {
        **deterministic,
        "validation_run_id": _sha(deterministic),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["artifact_sha256"] = _sha(payload)
    validate_nested_walk_forward_artifact(payload)
    return payload


def validate_nested_walk_forward_artifact(payload: Mapping) -> bool:
    if (not isinstance(payload, Mapping)
            or payload.get("contract") not in (
                NESTED_WALK_FORWARD_CONTRACT,
                PREVIOUS_NESTED_WALK_FORWARD_CONTRACT,
                OLDER_NESTED_WALK_FORWARD_CONTRACT,
                LEGACY_NESTED_WALK_FORWARD_CONTRACT,
            )):
        raise ValueError("Contrato walk-forward desconocido.")
    results = payload.get("results")
    if not isinstance(results, list) or [row.get("horizon") for row in results] != list(HORIZON_MINUTES):
        raise ValueError("El artefacto debe contener seis horizontes aislados.")
    for row in results:
        unsigned = {key: value for key, value in row.items() if key != "result_sha256"}
        if row.get("result_sha256") != _sha(unsigned):
            raise ValueError(f"Firma walk-forward inválida: {row.get('horizon')}.")
        final = row.get("final_holdout") or {}
        if final.get("status") == "SEALED_UNOPENED" and final.get("metrics") is not None:
            raise ValueError("Un holdout sellado no puede contener métricas.")
        metrics = final.get("metrics")
        if metrics is not None and metrics.get("opened_against_protocol_sha256") != row.get("protocol_frozen_sha256"):
            raise ValueError("El holdout fue abierto contra otro protocolo.")
        if payload["contract"] in (
            NESTED_WALK_FORWARD_CONTRACT, PREVIOUS_NESTED_WALK_FORWARD_CONTRACT,
            OLDER_NESTED_WALK_FORWARD_CONTRACT,
        ):
            calibration = row.get("calibration")
            if row.get("promotable"):
                if (row.get("status") != "APPROVED_SEALED_HOLDOUT_CALIBRATED"
                        or not isinstance(calibration, Mapping)
                        or not calibration.get("approved")
                        or (row.get("model") or {}).get("calibration", {}).get("temperature")
                        != calibration.get("temperature")):
                    raise ValueError("Modelo promovido sin calibración aprobada.")
            if final.get("status") == "SEALED_UNOPENED" and row.get("class_counts") is not None:
                raise ValueError("Un holdout sellado no puede publicar frecuencias de clase.")
            if final.get("status") == "OPENED_ONCE_AFTER_PROTOCOL_FREEZE":
                frozen = row.get("frozen_protocol")
                if (not isinstance(frozen, Mapping)
                        or row.get("protocol_frozen_sha256") != _sha(frozen)
                        or frozen.get("calibration") != calibration
                        or frozen.get("holdout_commitment_sha256") != final.get("commitment_sha256")):
                    raise ValueError("Protocolo de calibración/holdout no coincide.")
                if payload["contract"] == NESTED_WALK_FORWARD_CONTRACT:
                    stop_evidence = row.get("execution_evidence") or {}
                    multiples = stop_evidence.get("gross_loss_multiples") or []
                    if (frozen.get("execution_evidence") != stop_evidence
                            or stop_evidence.get("source") != "ELIGIBLE_LONG_DEVELOPMENT_ONLY"
                            or stop_evidence.get("observed_sl_samples") != len(multiples)
                            or (row.get("promotable") and len(multiples) < int(
                                (payload.get("configuration") or {}).get("minimum_class_samples", 20)
                            ))):
                        raise ValueError("Distribución observada de stops ausente o no congelada.")
                    opening = final.get("opening") or {}
                    if (not opening.get("opening_id") or not opening.get("opened_at_utc")
                            or not opening.get("dataset_sha256")
                            or opening.get("member_count") != final.get("samples")):
                        raise ValueError("Holdout abierto sin registro duradero verificable.")
                    if not verify_holdout_opening(
                        opening, symbol=payload.get("symbol", "UNSPECIFIED"),
                        horizon=row["horizon"],
                        protocol_sha256=row["protocol_frozen_sha256"],
                        commitment_sha256=final["commitment_sha256"],
                    ):
                        raise ValueError("Apertura de holdout ausente del registro local.")
                    intervals = (metrics or {}).get("confidence_intervals") or {}
                    for name in (
                        "brier", "baseline_brier", "log_loss", "baseline_log_loss",
                        "raw_brier", "raw_log_loss", "accuracy",
                    ):
                        bounds = intervals.get(name) or {}
                        if (not isinstance(bounds.get("lower"), (int, float))
                                or not isinstance(bounds.get("upper"), (int, float))
                                or bounds["lower"] > metrics[name]
                                or bounds["upper"] < metrics[name]):
                            raise ValueError(f"Métrica {name} sin intervalo válido.")
                    if row.get("promotable") and min(
                        (metrics.get("class_counts") or {}).values(), default=0
                    ) < int((payload.get("configuration") or {}).get("minimum_calibration_per_class", 5)):
                        raise ValueError("Modelo promovido sin soporte de todas las clases finales.")
                    if row.get("promotable"):
                        threshold = int((payload.get("configuration") or {}).get("minimum_calibration_per_class", 5))
                        for part in ("fit_class_counts", "validation_class_counts"):
                            counts = (calibration or {}).get(part) or {}
                            if min((counts.get(name, 0) for name in TARGET_CLASSES), default=0) < threshold:
                                raise ValueError(f"Calibración aprobada sin soporte de clase: {part}.")
            if (calibration is not None and calibration.get("approved")
                    and not row.get("walk_forward", {}).get("aggregate")):
                raise ValueError("Calibración sin evidencia walk-forward.")
    unsigned_artifact = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload.get("artifact_sha256") != _sha(unsigned_artifact):
        raise ValueError("Firma integral walk-forward inválida.")
    deterministic = {
        key: payload.get(key) for key in (
            "contract", "validator_sha256", "numpy_version", "symbol",
            "source_replay_id", "source_replay_sha256", "horizon_isolation",
            "holdout_policy", "configuration", "results",
        )
    }
    if payload.get("validation_run_id") != _sha(deterministic):
        raise ValueError("Identidad walk-forward inválida.")
    return True


def write_nested_walk_forward_artifact(payload: Mapping, path) -> Path:
    validate_nested_walk_forward_artifact(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(canonical(payload), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
