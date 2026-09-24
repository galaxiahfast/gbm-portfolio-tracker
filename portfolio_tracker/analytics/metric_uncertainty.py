"""Deterministic time-block bootstrap intervals for chronological OOS scores."""
from __future__ import annotations

import math

import numpy as np


def proportion_interval(successes: int, total: int) -> dict:
    """95% Wilson interval for a fold-success proportion."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Proporción sin soporte válido.")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
        "confidence": 0.95,
        "method": "WILSON_SCORE_V1",
    }


def score_intervals(probabilities, labels, *, replicates: int = 400) -> dict:
    """95% moving-block intervals for multiclass Brier, log loss and accuracy.

    Rows remain paired with their outcomes. Adjacent observations are resampled
    in blocks instead of assuming independent intraday/trading-day signals.
    These describe sampling uncertainty, not a guarantee of future coverage.
    """
    scores = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if scores.ndim != 2 or len(scores) != len(y) or len(y) == 0:
        raise ValueError("Scores y etiquetas incompatibles para intervalos.")
    expected = np.eye(scores.shape[1], dtype=float)[y]
    losses = {
        "brier": np.sum((scores - expected) ** 2, axis=1),
        "log_loss": -np.log(np.clip(scores[np.arange(len(y)), y], 1e-12, 1.0)),
        "accuracy": (np.argmax(scores, axis=1) == y).astype(float),
    }
    n = len(y)
    block = min(n, max(1, int(math.sqrt(n))))
    blocks = int(math.ceil(n / block))
    rng = np.random.default_rng(20_260_923 + n)
    starts = rng.integers(0, n, size=(replicates, blocks))
    positions = (starts[:, :, None] + np.arange(block)) % n
    indices = positions.reshape(replicates, -1)[:, :n]
    result = {}
    for name, values in losses.items():
        distribution = values[indices].mean(axis=1)
        lower, upper = np.quantile(distribution, [0.025, 0.975])
        point = float(values.mean())
        result[name] = {
            "lower": float(min(lower, point)),
            "upper": float(max(upper, point)),
            "confidence": 0.95,
            "method": "MOVING_BLOCK_BOOTSTRAP_V1",
            "block_size": block,
            "replicates": replicates,
        }
    return result
