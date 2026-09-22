"""Only models with independent OOF skill receive temperature calibration."""
from __future__ import annotations

import numpy as np
import pytest

from portfolio_tracker.analytics.operational_calibration import (
    select_and_validate_temperature,
    temperature_scale,
)


def _evidence(size):
    labels = np.arange(size, dtype=int) % 3
    predictions = np.full((size, 3), 0.05, dtype=float)
    for index, label in enumerate(labels):
        predicted = label if index % 6 else (label + 1) % 3
        predictions[index, predicted] = 0.90
    baseline = np.full_like(predictions, 1 / 3)
    return predictions, labels, baseline


def test_temperature_fits_early_oof_and_improves_later_oof():
    fit, y_fit, _ = _evidence(150)
    validation, y_validation, baseline = _evidence(90)
    result = select_and_validate_temperature(
        fit, y_fit, validation, y_validation, baseline
    )
    assert result["approved"] is True
    assert result["temperature"] > 1.0
    assert result["selection_source"] == "EARLIER_OUTER_OOF_ONLY"
    assert result["validation_source"] == "LATER_OUTER_OOF_ONLY"
    assert result["validation_metrics"]["calibrated"]["brier"] < result["validation_metrics"]["raw"]["brier"]
    assert result["validation_metrics"]["calibrated"]["log_loss"] < result["validation_metrics"]["baseline"]["log_loss"]


def test_later_labels_do_not_select_temperature():
    fit, y_fit, _ = _evidence(150)
    validation, y_validation, baseline = _evidence(90)
    before = select_and_validate_temperature(fit, y_fit, validation, y_validation, baseline)
    after = select_and_validate_temperature(
        fit, y_fit, validation, (y_validation + 1) % 3, baseline
    )
    assert before["temperature"] == after["temperature"]
    assert before["candidates"] == after["candidates"]
    assert before["validation_metrics"] != after["validation_metrics"]
    assert after["raw_model_beats_baseline"] is False
    assert after["approved"] is False


def test_uninformative_or_small_sample_cannot_be_calibrated():
    fit, y_fit, baseline = _evidence(150)
    validation, y_validation, _ = _evidence(90)
    flat = np.full_like(fit, 1 / 3)
    flat_validation = np.full_like(validation, 1 / 3)
    uninformative = select_and_validate_temperature(
        flat, y_fit, flat_validation, y_validation, baseline[:90]
    )
    assert uninformative["approved"] is False
    assert uninformative["status"] == "NO_CALIBRATION_GAIN"
    insufficient = select_and_validate_temperature(
        fit[:30], y_fit[:30], validation, y_validation, np.full_like(validation, 1 / 3)
    )
    assert insufficient["approved"] is False
    assert insufficient["temperature"] is None


def test_temperature_preserves_simplex_and_rejects_invalid_distribution():
    scaled = temperature_scale([[0.8, 0.1, 0.1]], 2.0)
    assert scaled.shape == (1, 3)
    assert scaled.sum() == pytest.approx(1.0)
    assert scaled[0, 0] < 0.8
    with pytest.raises(ValueError, match="sumen uno"):
        temperature_scale([[0.9, 0.9, 0.2]], 1.5)
    with pytest.raises(ValueError, match="positiva"):
        temperature_scale([[0.8, 0.1, 0.1]], 0)
