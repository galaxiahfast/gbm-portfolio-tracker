"""Regression guards for the isolated development-only trigger study."""
import numpy as np
import pytest

from portfolio_tracker.analytics.calibrated_trigger_research import (
    apply_platt, count_selection, fit_platt, frozen_quantile_threshold, net_ev,
)
from portfolio_tracker.analytics.horizon_models import TARGET_CLASSES


def test_platt_needs_all_three_classes_and_preserves_simplex():
    probabilities = np.asarray([[.65, .25, .10], [.30, .60, .10], [.25, .20, .55]] * 4)
    labels = list(TARGET_CLASSES) * 4
    fitted = fit_platt(probabilities, labels)
    result = apply_platt(fitted, probabilities)
    assert fitted["calibration_n"] == 12
    assert result.shape == probabilities.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))
    assert np.allclose(result.sum(axis=1), 1)
    with pytest.raises(ValueError, match="soporte mínimo"):
        fit_platt(probabilities, [TARGET_CLASSES[0]] * len(probabilities))


def test_quantile_is_frozen_from_calibration_not_test():
    calibration = np.arange(20, dtype=float) / 100
    threshold = frozen_quantile_threshold(calibration, 0.10)
    assert threshold == pytest.approx(np.quantile(calibration, 0.90))
    # A large test-period EV must not change the preregistered threshold.
    assert threshold == frozen_quantile_threshold(calibration, 0.10)
    with pytest.raises(ValueError, match="preregistrado"):
        frozen_quantile_threshold(calibration, 0.15)


def test_net_ev_uses_already_net_class_payoffs_once():
    p = np.asarray([[.4, .5, .1]])
    payoff = dict(zip(TARGET_CLASSES, (.02, -.03, -.004)))
    assert net_ev(p, payoff)[0] == pytest.approx(.4 * .02 - .5 * .03 - .1 * .004)


def test_selected_signal_is_not_equivalent_to_possible_fill():
    rows = [{"outcome": "TP_FIRST", "net_return": .02},
            {"outcome": None, "net_return": None},
            {"outcome": "SL_FIRST", "net_return": -.03}]
    got = count_selection(rows, [True, True, False])
    assert got["long_signals"] == 2
    assert got["possible_ohlc_fills"] == 1
    assert got["outcomes"]["TP_FIRST"] == 1
    assert got["net_return_sum_hypothetical"] == pytest.approx(.02)
