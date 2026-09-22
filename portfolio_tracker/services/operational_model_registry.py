"""Read-only access to the latest sealed walk-forward model per symbol.

Only the newest valid V2 artifact is considered. Rejected or insufficient
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
            or row.get("score_semantics") != "HISTORICAL_OOS_CALIBRATED_PRELIMINARY"):
        return False
    final = row.get("final_holdout") or {}
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
