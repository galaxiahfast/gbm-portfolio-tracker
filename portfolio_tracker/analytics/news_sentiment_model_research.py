"""Same offline L2 multinomial logit, with one appended PIT news feature.

No live code imports this module. It reuses the existing preprocessing and
optimizer; only the research feature vector length changes from 19 to 20.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from .horizon_models import (
    TARGET_CLASSES, _fit_preprocessor, _fit_softmax, _labels, _softmax,
    _transform,
)
from .learned_trigger_research import FEATURE_NAMES


NEWS_FEATURE_NAMES = (*FEATURE_NAMES, "news_sentiment_24h")


class _Sample:
    def __init__(self, row):
        self.features = row["features"]
        self.outcome = row["outcome"]


def fit_augmented_model(rows, l2=1.0):
    selected = [row for row in rows if row.get("outcome") in TARGET_CLASSES
                and row.get("net_return") is not None]
    counts = Counter(row["outcome"] for row in selected)
    if len(selected) < 30 or any(counts[category] < 3 for category in TARGET_CLASSES):
        raise ValueError("Insufficient filled training examples by class.")
    if any(len(row["features"]) != len(NEWS_FEATURE_NAMES) for row in selected):
        raise ValueError("News feature vector mismatch.")
    if not np.isfinite(np.asarray([row["features"] for row in selected], dtype=float)).all():
        raise ValueError("Non-finite news feature vector.")
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
            "feature_names": list(NEWS_FEATURE_NAMES), "l2": float(l2)}


def predict_augmented(model, rows):
    if not rows:
        return np.empty((0, 3))
    if model["feature_names"] != list(NEWS_FEATURE_NAMES):
        raise ValueError("News model feature version mismatch.")
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    if matrix.shape[1] != len(NEWS_FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Missing PIT news feature at inference.")
    x = (matrix - np.asarray(model["means"])) / np.asarray(model["scales"])
    return _softmax(np.column_stack([np.ones(len(rows)), x]) @ np.asarray(model["weights"]))
