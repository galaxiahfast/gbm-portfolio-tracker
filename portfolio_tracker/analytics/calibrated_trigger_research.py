"""Offline calibration and net-EV selection for the frozen 1D LONG replay.

No live decision path imports this module. Class payoffs already include the
replay's assumed commission, spread and slippage; charging them twice would
artificially suppress every signal.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .horizon_models import TARGET_CLASSES


CALIBRATION_VERSION = "OVR_PLATT_RENORMALIZED_V1"


def fit_platt(probabilities, labels, *, minimum_per_class=3, ridge=0.1):
    """Fit three binary Platt curves on data disjoint from base-model training.

    Renormalizing one-vs-rest estimates retains the three-class probability
    contract. This is a small-sample research calibrator, not OOS approval.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(TARGET_CLASSES) or len(p) != len(labels):
        raise ValueError("Probabilidades/etiquetas de calibración incompatibles.")
    if not len(p) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilidades de calibración inválidas.")
    counts = Counter(labels)
    if any(counts[category] < minimum_per_class for category in TARGET_CLASSES):
        raise ValueError("Calibración sin soporte mínimo por clase.")
    if any(label not in TARGET_CLASSES for label in labels):
        raise ValueError("Clase de calibración desconocida.")
    coefficients = []
    for category_index, category in enumerate(TARGET_CLASSES):
        clipped = np.clip(p[:, category_index], 1e-6, 1 - 1e-6)
        feature = np.log(clipped / (1 - clipped))
        x = np.column_stack((np.ones(len(p)), feature))
        y = np.asarray([label == category for label in labels], dtype=float)
        beta = np.asarray([math.log(y.mean() / (1 - y.mean())), 0.0])
        penalty = np.diag([1e-8, ridge])
        for _ in range(80):
            z = np.clip(x @ beta, -35, 35)
            fitted = 1 / (1 + np.exp(-z))
            gradient = x.T @ (fitted - y) + penalty @ beta
            hessian = x.T @ ((fitted * (1 - fitted))[:, None] * x) + penalty
            step = np.linalg.solve(hessian, gradient)
            beta -= np.clip(step, -2, 2)
            if float(np.max(np.abs(step))) < 1e-7:
                break
        coefficients.append(beta.tolist())
    return {"version": CALIBRATION_VERSION, "coefficients": coefficients,
            "calibration_n": len(p), "class_counts": dict(counts)}


def apply_platt(calibrator, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(TARGET_CLASSES):
        raise ValueError("Matriz de predicciones inválida.")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilidades inválidas.")
    coefficients = np.asarray(calibrator["coefficients"], dtype=float)
    if calibrator["version"] != CALIBRATION_VERSION or coefficients.shape != (3, 2):
        raise ValueError("Versión de calibración incompatible.")
    safe = np.clip(p, 1e-6, 1 - 1e-6)
    logits = np.log(safe / (1 - safe))
    z = np.clip(coefficients[:, 0] + logits * coefficients[:, 1], -35, 35)
    binary = 1 / (1 + np.exp(-z))
    return binary / binary.sum(axis=1, keepdims=True)


def net_ev(probabilities, training_net_payoffs):
    """Per-share net return; cost is ALREADY in each training payoff."""
    payoff = np.asarray([training_net_payoffs[category] for category in TARGET_CLASSES], dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(payoff).all():
        raise ValueError("EV de clases inválido.")
    return p @ payoff


def frozen_quantile_threshold(calibration_ev, top_fraction):
    """Threshold from *past calibration only*, never from test outcomes/EVs."""
    if top_fraction not in (0.05, 0.10, 0.20):
        raise ValueError("Cuantil no preregistrado.")
    values = np.asarray(calibration_ev, dtype=float)
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("Sin muestras de calibración para umbral.")
    return float(np.quantile(values, 1 - top_fraction))


def count_selection(rows, selected):
    """A selected plan is a signal; outcome/fill remain ex-post OHLC only."""
    if len(rows) != len(selected):
        raise ValueError("Selección y planes desalineados.")
    chosen = [row for row, flag in zip(rows, selected) if flag]
    filled = [row for row in chosen if row.get("outcome") in TARGET_CLASSES]
    return {"long_signals": len(chosen), "possible_ohlc_fills": len(filled),
            "outcomes": {category: sum(row["outcome"] == category for row in filled)
                         for category in TARGET_CLASSES},
            "net_return_sum_hypothetical": float(sum(row["net_return"] for row in filled)),
            "net_return_mean_per_possible_fill":
                float(np.mean([row["net_return"] for row in filled])) if filled else None}
