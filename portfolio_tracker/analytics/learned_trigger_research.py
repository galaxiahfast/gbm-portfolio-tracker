"""Offline LONG trigger experiment; never imported by the live decision path.

The operational plan, triple barriers and OHLC fill evaluator come from the
existing replay. This module only constructs point-in-time predictors and fits
an L2 multinomial logit to TP_FIRST / SL_FIRST / TIMEOUT. No output here is a
calibrated trading probability or broker fill.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from .horizon_models import (
    TARGET_CLASSES, _brier, _fit_preprocessor, _fit_softmax, _labels,
    _log_loss, _softmax, _transform,
)


RESEARCH_VERSION = "LEARNED_LONG_TRIGGER_RESEARCH_V1"
FEATURE_VERSION = "CAUSAL_GROUPED_LONG_FEATURES_V1"
FEATURE_NAMES = (
    "zone_entry_distance_atr", "support_distance_atr",
    "resistance_distance_atr", "log_volume_ratio",
    "momentum_5m_composite", "stoch_rsi_spread",
    "adx_scaled", "daily_atr_consumed",
    "ema50_distance_atr", "trend_1h_ema_gap_pct",
    "trend_4h_ema_gap_pct", "trend_1d_ema_gap_pct",
    "regime_trend", "regime_range", "permission_long_only",
    "permission_both_reduced", "peer_corr_20_completed_days",
    "hour_ny_scaled", "symbol_nvda",
)
L2_GRID = (0.01, 0.1, 1.0, 10.0)


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature ausente: {name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Feature no finita: {name}")
    return number


def _tail_gap(snapshot, frame, price):
    record = snapshot["indicator_tail"][frame]
    values = record["values"]
    return (_finite(values["EMA9"], frame + ".EMA9")
            - _finite(values["EMA21"], frame + ".EMA21")) / price


def completed_peer_correlation(primary_daily, peer_daily, cut):
    """Twenty paired returns, ending at the prior *completed* XNYS session."""
    session = pd.Timestamp(cut).tz_convert("America/New_York").date()
    left = primary_daily.copy()
    right = peer_daily.copy()
    left.index = pd.Index([pd.Timestamp(v).date() for v in left.index])
    right.index = pd.Index([pd.Timestamp(v).date() for v in right.index])
    common = left.index.intersection(right.index)
    common = sorted(day for day in common if day < session)
    if len(common) < 21:
        raise ValueError("Faltan 21 cierres diarios previos para correlación causal.")
    a = left.loc[common[-21:], "Close"].astype(float).to_numpy()
    b = right.loc[common[-21:], "Close"].astype(float).to_numpy()
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Cierres de correlación inválidos.")
    returns_a, returns_b = np.diff(np.log(a)), np.diff(np.log(b))
    if min(np.std(returns_a), np.std(returns_b)) < 1e-12:
        raise ValueError("Correlación indefinida con retornos constantes.")
    return float(np.corrcoef(returns_a, returns_b)[0, 1])


def extract_grouped_features(cut, horizon, primary_dataset, peer_dataset):
    """Only use the closed emission snapshot and bars whose close <= cut."""
    observed = pd.Timestamp(cut["observed_at"])
    if observed.tzinfo is None:
        raise ValueError("Corte sin zona horaria.")
    snapshot = cut["feature_snapshot"]
    source = pd.Timestamp(snapshot["source_bar_closed_at"])
    if source > observed:
        raise ValueError("Snapshot mira al futuro.")
    features = snapshot["model_features"]
    contract = horizon["operational_contract"]
    if not contract or contract["side"] != "LONG":
        raise ValueError("Este experimento solo estudia planes LONG existentes.")
    price = _finite(features["last_price"], "last_price")
    atr = _finite(features["atr_5m"], "atr_5m")
    if min(price, atr) <= 0:
        raise ValueError("Precio o ATR no positivo.")
    daily_tail = snapshot["indicator_tail"]["daily_indicators"]
    if pd.Timestamp(daily_tail["at"]).date() >= observed.tz_convert("America/New_York").date():
        raise ValueError("ATR diario de una sesión aún abierta.")
    atr_daily = _finite(daily_tail["values"]["ATR14"], "daily.ATR14")
    if atr_daily <= 0:
        raise ValueError("ATR diario no positivo.")
    ny = observed.tz_convert("America/New_York")
    intraday = primary_dataset.intraday
    mask = ((intraday.index.tz_convert("America/New_York").date == ny.date())
            & (intraday.index + pd.Timedelta(minutes=5) <= observed))
    completed = intraday.loc[mask]
    if completed.empty or completed.index.max() + pd.Timedelta(minutes=5) > observed:
        raise ValueError("No hay velas intradía cerradas en el corte.")
    consumed = (float(completed.High.max()) - float(completed.Low.min())) / atr_daily
    macd = _finite(features["macd_histogram_5m"], "macd_histogram_5m")
    ema9 = _finite(features["ema9"], "ema9")
    ema21 = _finite(features["ema21"], "ema21")
    momentum = 0.5 * (math.tanh(macd / atr) + math.tanh((ema9 - ema21) / atr))
    volume = _finite(features["volume_ratio"], "volume_ratio")
    if volume < 0:
        raise ValueError("Ratio de volumen negativo.")
    regime = str(features["market_regime"])
    permission = str(features["macro_permission"])
    if regime not in {"TREND", "RANGE", "TRANSITION"}:
        raise ValueError("Régimen no comparable con replay.")
    if permission not in {"LONG_ONLY", "SHORT_ONLY", "BOTH_REDUCED", "NO_TRADE"}:
        raise ValueError("Permiso macro desconocido.")
    correlation = completed_peer_correlation(primary_dataset.daily, peer_dataset.daily, observed)
    result = (
        (_finite(contract["entry_price"], "entry") - price) / atr,
        (_finite(features["nearest_support"], "support") - price) / atr,
        (_finite(features["structural_resistance"], "resistance") - price) / atr,
        math.log1p(volume), momentum,
        (_finite(features["stochastic_k"], "stoch_k")
         - _finite(features["stochastic_d"], "stoch_d")) / 100.0,
        _finite(features["adx"], "adx") / 50.0, consumed,
        (price - _finite(features["ema50"], "ema50")) / atr,
        _tail_gap(snapshot, "hourly_indicators", price),
        _tail_gap(snapshot, "four_hour_indicators", price),
        _tail_gap(snapshot, "daily_indicators", price),
        float(regime == "TREND"), float(regime == "RANGE"),
        float(permission == "LONG_ONLY"), float(permission == "BOTH_REDUCED"),
        correlation, (ny.hour + ny.minute / 60.0 - 9.5) / 6.5,
        float(cut["symbol"] == "NVDA"),
    )
    if len(result) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in result):
        raise ValueError("Vector agrupado incompleto.")
    return tuple(float(v) for v in result)


def _training_rows(rows):
    return [r for r in rows if r.get("outcome") in TARGET_CLASSES
            and r.get("net_return") is not None]


def fit_model(rows, l2):
    """Reuse the existing project's L2 softmax and train-only preprocessing."""
    selected = _training_rows(rows)
    if len(selected) < 30 or any(Counter(r["outcome"] for r in selected)[c] < 3
                               for c in TARGET_CLASSES):
        raise ValueError("Evidencia de fill/clase insuficiente para logit multinomial.")
    medians, means, scales = _fit_preprocessor([_as_sample(r) for r in selected])
    x = _transform([_as_sample(r) for r in selected], medians, means, scales)
    y = _labels([_as_sample(r) for r in selected])
    weights = _fit_softmax(x, y, l2, 1200, 0.08)
    counts = Counter(r["outcome"] for r in selected)
    prior = np.asarray([(counts[c] + 1) / (len(selected) + 3) for c in TARGET_CLASSES])
    payoff = {c: float(np.mean([r["net_return"] for r in selected if r["outcome"] == c]))
              for c in TARGET_CLASSES}
    ema_prior = {}
    for flag in (False, True):
        group = [r for r in selected if bool(r["ema_rule"]) == flag]
        group_counts = Counter(r["outcome"] for r in group)
        ema_prior[str(flag)] = [(group_counts[c] + 1) / (len(group) + 3)
                                for c in TARGET_CLASSES]
    return {"l2": float(l2), "medians": medians.tolist(), "means": means.tolist(),
            "scales": scales.tolist(), "weights": weights.tolist(),
            "prior": prior.tolist(), "ema_prior": ema_prior,
            "payoff_by_class_net_return": payoff,
            "training_n": len(selected), "training_class_counts": dict(counts)}


class _Sample:
    def __init__(self, row):
        self.features = row["features"]
        self.outcome = row["outcome"]


def _as_sample(row):
    return _Sample(row)


def predict(model, rows):
    if not rows:
        return np.empty((0, 3))
    matrix = np.asarray([r["features"] for r in rows], dtype=float)
    if matrix.shape[1] != len(FEATURE_NAMES) or not np.isfinite(matrix).all():
        raise ValueError("Paridad de features del modelo de investigación inválida.")
    x = (matrix - np.asarray(model["means"])) / np.asarray(model["scales"])
    return _softmax(np.column_stack([np.ones(len(rows)), x]) @ np.asarray(model["weights"]))


def score_labels(rows, probabilities):
    scored = [(r, p) for r, p in zip(rows, probabilities) if r.get("outcome") in TARGET_CLASSES]
    if not scored:
        return {"n": 0, "brier": None, "log_loss": None}
    y = np.asarray([TARGET_CLASSES.index(r["outcome"]) for r, _ in scored])
    p = np.asarray([vector for _, vector in scored])
    return {"n": len(y), "brier": _brier(p, y), "log_loss": _log_loss(p, y)}


def embargoed_training(rows, before_session, *, sessions, embargo=1):
    """Purge future-known labels and skip preceding XNYS sessions."""
    boundary = sessions.index(before_session)
    allowed = set(sessions[:max(0, boundary - embargo)])
    start = pd.Timestamp(before_session + "T11:00:00", tz="America/New_York").tz_convert("UTC")
    return [r for r in rows if r["session_ny"] in allowed
            and r.get("outcome") in TARGET_CLASSES
            and pd.Timestamp(r["label_available_at"]) <= start]


def strategy_counts(rows, probabilities, model):
    """Research signals at cut; possible fills and net returns only ex post."""
    output = {}
    payoff = np.asarray([model["payoff_by_class_net_return"][c] for c in TARGET_CLASSES])
    for name in ("LEARNED", "CURRENT_AND", "EMA9_21"):
        signals = 0
        filled = 0
        net = []
        outcomes = Counter()
        for row, p in zip(rows, probabilities):
            if name == "LEARNED":
                take = bool(p[0] > p[1] and float(np.dot(p, payoff)) > 0)
            elif name == "CURRENT_AND":
                take = bool(row["current_and"])
            else:
                take = bool(row["ema_rule"])
            if not take:
                continue
            signals += 1
            if row.get("outcome") in TARGET_CLASSES:
                filled += 1
                net.append(float(row["net_return"]))
                outcomes[row["outcome"]] += 1
        output[name] = {"long_signals": signals, "possible_ohlc_fills": filled,
                        "outcomes": {c: outcomes[c] for c in TARGET_CLASSES},
                        "net_return_sum_hypothetical": sum(net) if net else None,
                        "net_return_mean_per_possible_fill": float(np.mean(net)) if net else None}
    return output
