"""Causality and numeric stability checks for the research-only pilot."""
from __future__ import annotations

import numpy as np

from scripts import train_prospective_pilot as pilot


def _row(session, cut, label):
    return {"session": session, "cut_at_utc": cut, "label_at_utc": label,
            "symbol": "SMCI", "snapshot_id": session}


def test_fold_purges_unavailable_labels_and_embargoes_a_session():
    sessions = [f"2026-10-{day:02d}" for day in range(1, 6)]
    rows = [
        _row(sessions[0], "2026-10-01T15:00:00+00:00", "2026-10-02T20:00:00+00:00"),
        _row(sessions[1], "2026-10-02T15:00:00+00:00", "2026-10-05T20:00:00+00:00"),
        _row(sessions[2], "2026-10-03T15:00:00+00:00", "2026-10-03T20:00:00+00:00"),
        _row(sessions[3], "2026-10-04T15:00:00+00:00", "2026-10-04T20:00:00+00:00"),
    ]
    train, test, _ = pilot._split_fold(rows, sessions, 3, test_length=1, embargo=1)
    assert train == [rows[0]]  # day 2 label arrived after the test cut
    assert test == [rows[3]]


def test_missing_sentiment_uses_train_median_and_small_variance_is_bounded():
    tech = [1.0] * len(pilot.TECH_FEATURES)
    train = [
        {"tech": tech, "news": [0.2, 0.3, 0.0]},
        {"tech": tech, "news": [0.4, 0.5, 0.0]},
    ]
    test = [{"tech": [99.0] + tech[1:], "news": [None, 0.7, 1.0]}]
    x, t, medians, _, scales = pilot._preprocess(train, test, True)
    assert np.isclose(medians[len(tech)], 0.3)
    assert scales[0] == pilot.SCALE_FLOOR
    assert np.isfinite(x).all() and np.isfinite(t).all()
    assert np.abs(t).max() <= pilot.STANDARDIZED_CLIP


def test_group_weights_do_not_count_five_same_day_cuts_as_five_events():
    rows = ([{"symbol": "SMCI", "session": "2026-10-01"}] * 5 +
            [{"symbol": "NVDA", "session": "2026-10-01"}])
    weights = pilot._group_weights(rows)
    assert np.isclose(weights[:5].sum(), 1)
    assert np.isclose(weights[5], 1)
