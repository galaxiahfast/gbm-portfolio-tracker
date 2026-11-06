"""Research-only 19+1 L2 multinomial logit for a predeclared news window."""
from __future__ import annotations

from collections import Counter
import numpy as np

from .horizon_models import (
    TARGET_CLASSES, _fit_preprocessor, _fit_softmax, _labels, _softmax, _transform,
)
from .learned_trigger_research import FEATURE_NAMES


class _Sample:
    def __init__(self, row):
        self.features = row["features"]
        self.outcome = row["outcome"]


def fit_headline_model(rows, *, hours, l2=1.0):
    if hours not in (4, 24):
        raise ValueError("Unregistered FinBERT headline window")
    names = [*FEATURE_NAMES, f"finbert_headline_mean_{hours}h"]
    selected = [row for row in rows if row.get("outcome") in TARGET_CLASSES
                and row.get("net_return") is not None]
    counts = Counter(row["outcome"] for row in selected)
    if len(selected) < 30 or any(counts[category] < 3 for category in TARGET_CLASSES):
        raise ValueError("Insufficient headline-present filled training examples by class")
    matrix = np.asarray([row["features"] for row in selected], dtype=float)
    if matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise ValueError("FinBERT feature vector mismatch")
    samples = [_Sample(row) for row in selected]
    medians, means, scales = _fit_preprocessor(samples)
    x = _transform(samples, medians, means, scales)
    weights = _fit_softmax(x, _labels(samples), l2, 1200, 0.08)
    payoffs = {category: float(np.mean([row["net_return"] for row in selected
                                        if row["outcome"] == category]))
               for category in TARGET_CLASSES}
    return {"weights": weights.tolist(), "means": means.tolist(),
            "scales": scales.tolist(), "payoff_by_class_net_return": payoffs,
            "training_n": len(selected), "training_class_counts": dict(counts),
            "feature_names": names, "l2": float(l2)}


def predict_headline(model, rows, *, hours):
    if model["feature_names"] != [*FEATURE_NAMES, f"finbert_headline_mean_{hours}h"]:
        raise ValueError("FinBERT feature version mismatch")
    if not rows:
        return np.empty((0, 3))
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    if matrix.shape[1] != len(model["feature_names"]) or not np.isfinite(matrix).all():
        raise ValueError("Missing FinBERT feature at inference")
    x = (matrix - np.asarray(model["means"])) / np.asarray(model["scales"])
    return _softmax(np.column_stack([np.ones(len(rows)), x]) @ np.asarray(model["weights"]))
