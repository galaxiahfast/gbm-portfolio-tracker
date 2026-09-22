"""Regularized models remain causal, horizon-specific and fail closed."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics.horizon_models import (
    FEATURE_NAMES,
    HorizonSample,
    TrainingConfig,
    chronological_purged_split,
    predict_uncalibrated_scores,
    train_horizon_samples,
    train_replay_horizon_models,
    validate_horizon_model_artifact,
)
from portfolio_tracker.analytics.historical_replay import (
    HISTORICAL_REPLAY_CONTRACT,
    _sha as replay_sha,
)
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES


def _samples(size=330):
    rows = []
    for index in range(size):
        observed = pd.Timestamp("2020-01-02", tz="UTC") + pd.DateOffset(days=index)
        category = index % 3
        values = np.zeros(len(FEATURE_NAMES), dtype=float)
        # A stable, learnable relationship present across every chronological block.
        values[0] = (-2.5, 2.5, 0.0)[category]
        values[1] = (0.2, 0.3, 0.8)[category]
        values[2] = np.sin(index / 7.0) * 0.05
        rows.append(HorizonSample(
            observed_at=observed.isoformat(),
            label_available_at=(observed + pd.DateOffset(hours=1)).isoformat(),
            features=tuple(values),
            outcome=("TP_FIRST", "SL_FIRST", "TIMEOUT")[category],
            cut_id=f"cut-{index}",
        ))
    return tuple(rows)


def _config():
    return TrainingConfig(
        minimum_samples=300,
        minimum_class_samples=20,
        minimum_holdout_samples=60,
        maximum_iterations=500,
    )


def test_regularized_model_is_per_horizon_and_scores_are_explicitly_uncalibrated():
    rows = _samples()
    one_hour = train_horizon_samples("1 Hora", rows, _config())
    one_day = train_horizon_samples("1 Día", rows, _config())
    assert one_hour["horizon_minutes"] == 60
    assert one_day["horizon_minutes"] == 1_440
    assert one_hour["model_sha256"] != one_day["model_sha256"]
    assert one_hour["status"] == "TRAINED_OOS_UNCALIBRATED"
    assert one_hour["promotable"] is True
    assert one_hour["metrics"]["source"] == "HOLDOUT_ONLY"
    assert one_hour["metrics"]["multiclass_brier"] < one_hour["metrics"]["baseline_brier"]
    prediction = predict_uncalibrated_scores(one_hour, rows[-1].features)
    assert prediction["semantics"] == "REGULARIZED_SCORE_UNCALIBRATED"
    assert prediction["sum"] == pytest.approx(100.0)
    assert set(prediction["scores"]) == {"TP_FIRST", "SL_FIRST", "TIMEOUT"}


def test_l2_selection_does_not_look_at_holdout_labels():
    original = _samples()
    split_at = int(len(original) * 0.8)
    changed = original[:split_at] + tuple(
        replace(row, outcome="SL_FIRST" if row.outcome != "SL_FIRST" else "TP_FIRST")
        for row in original[split_at:]
    )
    before = train_horizon_samples("6 Horas", original, _config())
    after = train_horizon_samples("6 Horas", changed, _config())
    assert before["selection"] == after["selection"]
    assert before["model"] == after["model"]
    assert before["metrics"] != after["metrics"]


def test_chronological_split_purges_labels_unknown_at_next_boundary():
    rows = list(_samples(30))
    # Both belong to training chronologically, but their outcomes become known
    # only after calibration has begun.
    rows[0] = replace(rows[0], label_available_at=rows[20].observed_at)
    rows[1] = replace(rows[1], label_available_at=rows[29].observed_at)
    training, calibration, holdout, purged = chronological_purged_split(rows)
    assert (len(training), len(calibration), len(holdout)) == (16, 6, 6)
    assert purged == {"training": 2, "calibration": 0}
    assert max(pd.Timestamp(row.observed_at) for row in training) < min(
        pd.Timestamp(row.observed_at) for row in calibration
    )


def test_small_replay_cannot_create_or_promote_a_model():
    rejected = train_horizon_samples("1 Semana", _samples(60), _config())
    assert rejected["status"] == "INSUFFICIENT_DATA"
    assert rejected["promotable"] is False
    assert rejected["model"] is None
    assert rejected["score_semantics"] == "NO_SCORE_MODEL_NOT_TRAINED"
    with pytest.raises(ValueError, match="no está aprobado"):
        predict_uncalibrated_scores(rejected, rejected.get("features", ()))


def test_invalid_feature_shape_and_unknown_horizon_fail_closed():
    bad = (replace(_samples(1)[0], features=(1.0, 2.0)),) * 300
    with pytest.raises(ValueError, match="Longitud de features"):
        train_horizon_samples("1 Hora", bad, _config())
    with pytest.raises(ValueError, match="Horizonte desconocido"):
        train_horizon_samples("Todos", _samples(), _config())


def _minimal_signed_replay():
    observed = "2026-09-01T15:00:00+00:00"
    horizons = []
    for label, minutes in HORIZON_MINUTES.items():
        horizons.append({
            "label": label,
            "horizon_minutes": minutes,
            "prediction": {},
            "scenario_contract": {},
            "scenario_result": {"status": "RIGHT_CENSORED"},
            "operational_contract": {},
            "operational_result": {
                "status": "RESOLVED",
                "outcome": "TIMEOUT",
                "exit_at": "2026-09-01T16:00:00+00:00",
            },
        })
    cut = {
        "cut_id": "cut-1",
        "symbol": "SMCI",
        "observed_at": observed,
        "source_bar_closed_at": observed,
        "dataset_sha256": "dataset",
        "feature_snapshot": {"features": {}},
        "horizons": horizons,
        "limitations": ["HISTORICAL_REPLAY_NOT_LIVE_OOS"],
    }
    cut["cut_sha256"] = replay_sha(cut)
    deterministic = {
        "contract": HISTORICAL_REPLAY_CONTRACT,
        "symbol": "SMCI",
        "dataset_sha256": "dataset",
        "peer_dataset_sha256": None,
        "parameters": {"minimum_probability": 0.55, "stop_atr_multiple": 2.25, "risk_per_trade_pct": 1.0},
        "requested_start": None,
        "requested_end": None,
        "max_cuts": None,
        "candidate_cuts": 1,
        "observations": [cut],
        "rejected": {},
        "separation_policy": "REPLAY_NEVER_COUNTS_AS_LIVE_OOS",
    }
    replay = {
        **deterministic,
        "replay_id": replay_sha({
            "contract": HISTORICAL_REPLAY_CONTRACT,
            "symbol": "SMCI",
            "dataset_sha256": "dataset",
            "parameters": deterministic["parameters"],
            "requested_start": None,
            "requested_end": None,
            "max_cuts": None,
        }),
        "generated_at": "2026-09-02T00:00:00+00:00",
    }
    replay["content_sha256"] = replay_sha(deterministic)
    replay["artifact_sha256"] = replay_sha(replay)
    return replay


def test_six_model_artifact_is_signed_and_tamper_evident():
    artifact = train_replay_horizon_models(_minimal_signed_replay())
    assert validate_horizon_model_artifact(artifact)
    assert [row["horizon"] for row in artifact["models"]] == list(HORIZON_MINUTES)
    assert all(row["status"] == "INSUFFICIENT_DATA" for row in artifact["models"])
    artifact["models"][0]["resolved_samples"] = 999
    with pytest.raises(ValueError, match="Firma de modelo"):
        validate_horizon_model_artifact(artifact)
