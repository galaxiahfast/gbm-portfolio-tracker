"""Nested walk-forward must embargo overlaps and keep final evidence sealed."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics.closed_bars import _calendar
from portfolio_tracker.analytics.horizon_models import FEATURE_NAMES, HorizonSample, _sha
from portfolio_tracker.analytics.operational_calibration import predict_calibrated_scores
from portfolio_tracker.analytics.nested_walk_forward import (
    EMBARGO_SESSIONS,
    NESTED_WALK_FORWARD_CONTRACT,
    WalkForwardConfig,
    embargo_and_purge_before,
    nested_walk_forward_horizon,
    validate_nested_walk_forward_artifact,
)
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES


def _config():
    return WalkForwardConfig(
        minimum_samples=300,
        minimum_class_samples=10,
        minimum_final_holdout=60,
        outer_folds=3,
        inner_folds=2,
        minimum_outer_training=90,
        minimum_outer_test=20,
        minimum_inner_training=40,
        minimum_inner_validation=12,
        maximum_iterations=350,
    )


def _samples(size=420, *, informative=True):
    schedule = _calendar(2018, 2025).schedule.iloc[:size]
    rows = []
    for index, session in enumerate(schedule.itertuples()):
        category = index % 3
        values = np.zeros(len(FEATURE_NAMES), dtype=float)
        if informative:
            values[0] = (-3.0, 3.0, 0.0)[category]
            values[1] = (0.2, 0.4, 0.9)[category]
        values[2] = np.sin(index / 9.0) * 0.03 if informative else 0.0
        observed = pd.Timestamp(session.open) + pd.DateOffset(minutes=90)
        rows.append(HorizonSample(
            observed_at=observed.isoformat(),
            label_available_at=(observed + pd.DateOffset(hours=1)).isoformat(),
            features=tuple(values),
            outcome=("TP_FIRST", "SL_FIRST", "TIMEOUT")[category],
            cut_id=f"cut-{index}",
        ))
    return tuple(rows)


def test_nested_walk_forward_opens_sealed_holdout_only_after_development_skill():
    result = nested_walk_forward_horizon("1 Hora", _samples(), _config())
    assert result["status"] == "APPROVED_SEALED_HOLDOUT_CALIBRATED"
    assert result["promotable"] is True
    assert result["embargo_sessions"] == EMBARGO_SESSIONS["1 Hora"] == 1
    assert len(result["walk_forward"]["outer_folds"]) == 3
    assert result["walk_forward"]["aggregate"]["brier"] < result["walk_forward"]["aggregate"]["baseline_brier"]
    assert result["final_holdout"]["status"] == "OPENED_ONCE_AFTER_PROTOCOL_FREEZE"
    assert result["final_holdout"]["metrics"]["source"] == "SEALED_FINAL_HOLDOUT_ONLY"
    assert result["final_holdout"]["metrics"]["opened_against_protocol_sha256"] == result["protocol_frozen_sha256"]
    assert result["model"]["trained_on"] == "DEVELOPMENT_ONLY_FINAL_HOLDOUT_EXCLUDED"
    assert result["calibration"]["approved"] is True
    assert result["calibration"]["validation_metrics"]["calibrated"]["brier"] < result["calibration"]["validation_metrics"]["baseline"]["brier"]
    assert result["final_holdout"]["metrics"]["brier"] < result["final_holdout"]["metrics"]["raw_brier"]
    prediction = predict_calibrated_scores(result, _samples(1)[0].features)
    assert prediction["semantics"] == "HISTORICAL_OOS_CALIBRATED_PRELIMINARY"
    assert sum(prediction["scores"].values()) == pytest.approx(1.0)


def test_final_holdout_labels_cannot_change_selection_or_fitted_model():
    original = _samples()
    holdout_size = max(60, int(np.ceil(len(original) * 0.20)))
    changed = original[:-holdout_size] + tuple(
        replace(row, outcome="SL_FIRST" if row.outcome != "SL_FIRST" else "TP_FIRST")
        for row in original[-holdout_size:]
    )
    before = nested_walk_forward_horizon("1 Hora", original, _config())
    after = nested_walk_forward_horizon("1 Hora", changed, _config())
    assert before["walk_forward"] == after["walk_forward"]
    before_protocol = dict(before["frozen_protocol"])
    after_protocol = dict(after["frozen_protocol"])
    before_protocol.pop("holdout_commitment_sha256")
    after_protocol.pop("holdout_commitment_sha256")
    assert before_protocol == after_protocol
    assert before["final_holdout"]["commitment_sha256"] != after["final_holdout"]["commitment_sha256"]
    assert before["final_holdout"]["metrics"] != after["final_holdout"]["metrics"]


def test_failed_development_gate_never_opens_final_holdout(monkeypatch):
    import portfolio_tracker.analytics.nested_walk_forward as workflow

    def forbidden(*args, **kwargs):
        raise AssertionError("No se debe calibrar un modelo inferior al baseline.")

    monkeypatch.setattr(workflow, "_calibrate_development_oof", forbidden)
    result = nested_walk_forward_horizon("1 Hora", _samples(informative=False), _config())
    assert result["status"] == "REJECTED_WALK_FORWARD_FINAL_HOLDOUT_UNOPENED"
    assert result["promotable"] is False
    assert result["final_holdout"]["status"] == "SEALED_UNOPENED"
    assert result["final_holdout"]["metrics"] is None
    assert result["model"] is None
    assert result["calibration"] is None
    assert result["class_counts"] is None


def test_failed_calibration_gate_keeps_final_holdout_sealed(monkeypatch):
    import portfolio_tracker.analytics.nested_walk_forward as workflow

    monkeypatch.setattr(
        workflow,
        "_calibrate_development_oof",
        lambda *args: {"status": "NO_CALIBRATION_GAIN", "approved": False},
    )
    result = nested_walk_forward_horizon("1 Hora", _samples(), _config())
    assert result["walk_forward"]["aggregate"]["brier"] < result["walk_forward"]["aggregate"]["baseline_brier"]
    assert result["status"] == "REJECTED_CALIBRATION_FINAL_HOLDOUT_UNOPENED"
    assert result["final_holdout"]["status"] == "SEALED_UNOPENED"
    assert result["final_holdout"]["metrics"] is None
    assert result["promotable"] is False
    assert result["model"] is None


def test_embargo_uses_xnys_sessions_and_label_availability():
    rows = list(_samples(20))
    boundary = rows[15].observed_at
    # Row 10 is well outside a two-session embargo but its outcome arrives late.
    rows[10] = replace(rows[10], label_available_at=rows[16].observed_at)
    selected, excluded = embargo_and_purge_before(rows[:15], boundary, 2)
    assert rows[14] not in selected and rows[13] not in selected
    assert rows[10] not in selected
    assert excluded == {"embargoed": 2, "label_unknown_at_boundary": 1}
    assert selected[-1].cut_id == "cut-12"


def test_short_history_keeps_holdout_sealed_and_unopened():
    result = nested_walk_forward_horizon("6 Meses", _samples(80), _config())
    assert result["status"] == "INSUFFICIENT_DATA_FINAL_HOLDOUT_UNOPENED"
    assert result["embargo_sessions"] == 126
    assert result["final_holdout"]["metrics"] is None
    assert result["protocol_frozen_sha256"] is None


def test_walk_forward_artifact_detects_tampering():
    results = [
        nested_walk_forward_horizon(label, _samples(80), _config())
        for label in HORIZON_MINUTES
    ]
    deterministic = {
        "contract": NESTED_WALK_FORWARD_CONTRACT,
        "validator_sha256": "validator",
        "numpy_version": np.__version__,
        "symbol": "SMCI",
        "source_replay_id": "replay",
        "source_replay_sha256": "source",
        "horizon_isolation": "ONE_NESTED_PROTOCOL_PER_SYMBOL_AND_HORIZON",
        "holdout_policy": "SEALED_FINAL_HOLDOUT_OPEN_ONLY_AFTER_DEVELOPMENT_GATE",
        "configuration": {"test": True},
        "results": results,
    }
    artifact = {
        **deterministic,
        "validation_run_id": _sha(deterministic),
        "generated_at": "2026-09-22T00:00:00+00:00",
    }
    artifact["artifact_sha256"] = _sha(artifact)
    assert validate_nested_walk_forward_artifact(artifact)
    artifact["results"][0]["status"] = "APPROVED"
    with pytest.raises(ValueError, match="Firma walk-forward"):
        validate_nested_walk_forward_artifact(artifact)
