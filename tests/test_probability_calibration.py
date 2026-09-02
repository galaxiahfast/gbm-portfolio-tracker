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
    samples = (
        [(0.30, 1 if index < 100 else 0) for index in range(250)]
        + [(0.70, 1 if index < 200 else 0) for index in range(250)]
    )
    result = calibrate_probability(0.70, samples)

    assert result.empirically_calibrated
    assert result.sample_size == 500
    assert 0.70 < result.calibrated_probability < 0.85
    assert all(
        left.observed_frequency <= right.observed_frequency
        for left, right in zip(result.reliability_curve, result.reliability_curve[1:])
    )
    assert math.isclose(
        result.brier_score or 0,
        sum((probability - outcome) ** 2 for probability, outcome in samples) / 500,
    )


def test_reliability_curve_rejects_invalid_bin_count() -> None:
    try:
        reliability_curve([(0.5, 1)], bins=1)
    except ValueError as exc:
        assert "dos bins" in str(exc)
    else:
        raise AssertionError("Se aceptó una curva sin resolución suficiente")
