import math

from portfolio_tracker.analytics.probability_calibration import (
    calibrate_probability,
    reliability_curve,
)


def test_small_sample_remains_explicitly_heuristic() -> None:
    samples = [(0.70, 1), (0.65, 0), (0.40, 0)]
    result = calibrate_probability(0.72, samples)

    assert result.status == "Score heurístico preliminar"
    assert result.calibrated_probability == 0.72
    assert result.sample_size == 3
    assert result.brier_score is not None


def test_large_sample_uses_monotonic_empirical_reliability_curve() -> None:
    # Chronological 300 train / 100 calibration / 100 independent holdout.
    samples = [(0.30, 0), (0.70, 1)] * 250
    result = calibrate_probability(0.70, samples)

    assert result.empirically_calibrated
    assert result.sample_size == 500
    expected_high = (50 + 4) / (50 + 8)  # Fit uses only 50 calibration positives.
    assert math.isclose(result.calibrated_probability, expected_high)
    assert (result.training_samples, result.calibration_samples, result.holdout_samples) == (300,100,100)
    assert all(
        left.observed_frequency <= right.observed_frequency
        for left, right in zip(result.isotonic_curve, result.isotonic_curve[1:])
    )
    assert math.isclose(
        result.brier_score or 0,
        (1-expected_high)**2,
    )
    assert math.isclose(result.raw_brier_score, 0.09)
    assert not math.isclose(result.brier_score, sum((p-y)**2 for p,y in samples)/500)


def test_reliability_curve_rejects_invalid_bin_count() -> None:
    try:
        reliability_curve([(0.5, 1)], bins=1)
    except ValueError as exc:
        assert "dos bins" in str(exc)
    else:
        raise AssertionError("Se aceptó una curva sin resolución suficiente")
