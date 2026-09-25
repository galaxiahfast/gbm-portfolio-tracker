"""Read-only access to the latest sealed walk-forward model per symbol.

Only the newest valid V4 artifact is considered. Rejected or insufficient
retraining must not silently fall back to an older approved run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from portfolio_tracker.analytics.nested_walk_forward import (
    NESTED_WALK_FORWARD_CONTRACT,
    validate_nested_walk_forward_artifact,
)
from portfolio_tracker.analytics.horizon_models import (
    FEATURE_NAMES, HORIZON_MODEL_CONTRACT, MODEL_FEATURE_VERSION,
    PROFESSIONAL_MINIMUM_SAMPLES, _sha,
)
from portfolio_tracker.analytics.operational_target import TARGET_VERSION


DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "output" / "nested_walk_forward"


def _strict_improvement(candidate, reference) -> bool:
    try:
        left, right = float(candidate), float(reference)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left) and math.isfinite(right) and left < right - 1e-12


def _approved_result(row) -> bool:
    if (row.get("status") != "APPROVED_SEALED_HOLDOUT_CALIBRATED"
            or row.get("promotable") is not True
            or row.get("base_model_contract") != HORIZON_MODEL_CONTRACT
            or row.get("target") != TARGET_VERSION
            or row.get("feature_version") != MODEL_FEATURE_VERSION
            or row.get("feature_names") != list(FEATURE_NAMES)
            or row.get("feature_schema_sha256") != _sha(list(FEATURE_NAMES))
            or row.get("available_features") != list(FEATURE_NAMES)
            or row.get("score_semantics") != "HISTORICAL_OOS_CALIBRATED_PRELIMINARY"):
        return False
    population = (row.get("population") or {}).get("executable_entries") or {}
    try:
        eligible_n = int(population.get("n") or 0)
        required_n = max(
            PROFESSIONAL_MINIMUM_SAMPLES,
            int(row.get("minimum_samples_required") or PROFESSIONAL_MINIMUM_SAMPLES),
        )
        resolved_n = int(row.get("resolved_samples") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    # Old artifacts trained on hypothetical barriers cannot authorize trades.
    if eligible_n < required_n or resolved_n != eligible_n:
        return False
    final = row.get("final_holdout") or {}
    stop_evidence = row.get("execution_evidence") or {}
    multiples = stop_evidence.get("gross_loss_multiples") or ()
    if (stop_evidence.get("source") != "ELIGIBLE_LONG_DEVELOPMENT_ONLY"
            or stop_evidence.get("semantics") != "OHLC_SIMULATED_EXIT_VS_POSSIBLE_FILL_STOP_DISTANCE_BEFORE_COSTS"
            or stop_evidence.get("observed_sl_samples") != len(multiples)
            or len(multiples) < 20):
        return False
    try:
        if any(not math.isfinite(float(value)) or float(value) < 1 for value in multiples):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    calibration = row.get("calibration") or {}
    validation = calibration.get("validation_metrics") or {}
    metrics = final.get("metrics") or {}
    raw = validation.get("raw") or {}
    calibrated = validation.get("calibrated") or {}
    baseline = validation.get("baseline") or {}
    try:
        sample_count = int(metrics.get("samples") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    if (final.get("status") != "OPENED_ONCE_AFTER_PROTOCOL_FREEZE"
            or calibration.get("approved") is not True
            or sample_count < 60):
        return False
    return all(
        _strict_improvement(first, second)
        for first, second in (
            (raw.get("brier"), baseline.get("brier")),
            (raw.get("log_loss"), baseline.get("log_loss")),
            (calibrated.get("brier"), raw.get("brier")),
            (calibrated.get("log_loss"), raw.get("log_loss")),
            (calibrated.get("brier"), baseline.get("brier")),
            (calibrated.get("log_loss"), baseline.get("log_loss")),
            (metrics.get("brier"), metrics.get("raw_brier")),
            (metrics.get("log_loss"), metrics.get("raw_log_loss")),
            (metrics.get("brier"), metrics.get("baseline_brier")),
            (metrics.get("log_loss"), metrics.get("baseline_log_loss")),
        )
    )


def latest_approved_operational_models(symbol, observed_at, *, directory=None) -> dict:
    """Return signed approved horizon results known by the decision cut."""
    symbol = str(symbol).strip().upper()
    if not symbol or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for char in symbol):
        raise ValueError("Símbolo inválido.")
    observed = pd.Timestamp(observed_at)
    if pd.isna(observed) or observed.tzinfo is None:
        raise ValueError("El corte de decisión debe tener zona horaria.")
    folder = Path(directory) if directory is not None else DEFAULT_MODEL_DIRECTORY
    if not folder.is_dir():
        return {}
    candidates = []
    for path in folder.glob(f"{symbol}_*.json"):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            validate_nested_walk_forward_artifact(artifact)
            if (artifact.get("contract") != NESTED_WALK_FORWARD_CONTRACT
                    or artifact.get("symbol") != symbol):
                continue
            generated = pd.Timestamp(artifact["generated_at"])
            if generated.tzinfo is None or generated > observed:
                continue
            candidates.append((generated, str(artifact["validation_run_id"]), artifact))
        except (OSError, ValueError, TypeError, KeyError):
            # Corrupt artifacts are never used as model evidence.
            continue
    if not candidates:
        return {}
    latest = max(candidates, key=lambda row: (row[0], row[1]))[2]
    return {
        row["horizon"]: row for row in latest["results"]
        if _approved_result(row)
    }
