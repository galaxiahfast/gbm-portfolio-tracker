"""Research-only first-stage LONG model from signed prospective snapshots.

Usage: prepare, then run.  This never imports the app, reads portfolio.db,
changes the live gate, or produces an actionable trading probability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.prospective_archive_common import (
    ARCHIVE, DATA_SOURCE, FEATURE_VERSION, hash_file, label_path, signed_payload,
)


VERSION = "PROSPECTIVE_LONG_1D_PILOT_V2"
OUT = ROOT / "output" / "prospective_research" / "long_1d_pilot_v2"
PROTOCOL = OUT / "preregistration.json"
TECH_FEATURES = (
    "log_volume_ratio", "ema_gap_atr", "macd_hist_atr",
    "stoch_spread", "adx_scaled", "support_gap_atr",
    "pattern_impact", "regime_trend", "symbol_nvda",
)
NEWS_FEATURES = ("sentiment_4h", "sentiment_24h", "sentiment_4h_missing")
RIDGE = 10.0  # fixed before looking at any fold outcome
SCALE_FLOOR = 0.1  # guards nearly constant train features from exploding in test
STANDARDIZED_CLIP = 5.0
FOLD_STARTS = (9, 12, 15, 18)  # session indexes; three test sessions each
EMBARGO_SESSIONS = 1


def _paths():
    return [p for p in sorted((ARCHIVE / "snapshots").glob("*/*.json"))
            if p.name.endswith("_1D.json")]


def prepare():
    """Freeze source identities and evaluation rules before reading labels."""
    if PROTOCOL.exists():
        return read_signed(PROTOCOL)
    items = []
    for path in _paths():
        row = signed_payload(path)
        if row.get("synthetic_example") or row.get("data_source", DATA_SOURCE) != DATA_SOURCE:
            raise ValueError("Snapshot de otra fuente o sintético.")
        if row.get("training_performed") is not False or row["horizon"] != "1D":
            raise ValueError("Snapshot de otra población.")
        items.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "file_sha256": hash_file(path), "snapshot_id": row["snapshot_id"],
                      "session_ny": row["session_ny"], "symbol": row["symbol"]})
    sessions = sorted({r["session_ny"] for r in items})
    if len(sessions) < 19:
        raise ValueError("Menos de 19 sesiones; no se ejecuta el piloto fijado.")
    protocol = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "SMCI_NVDA_1D_LONG_POSSIBLE_OHLC_FILL_ONLY",
        "source": DATA_SOURCE, "snapshot_manifest": items,
        "sessions": sessions, "fold_starts": list(FOLD_STARTS),
        "fold_test_sessions": 3, "embargo_sessions": EMBARGO_SESSIONS,
        "technical_features": list(TECH_FEATURES), "news_features": list(NEWS_FEATURES),
        "models": ["TRAIN_FREQUENCY", "RIDGE_TECH", "RIDGE_TECH_NEWS"],
        "ridge": RIDGE,
        "preprocessing": {"scale_floor": SCALE_FLOOR,
                          "standardized_clip": STANDARDIZED_CLIP},
        "target": "TP_FIRST vs SL_FIRST/TIMEOUT after possible OHLC fill",
        "missing_news": "train-only median plus explicit 4h missing flag; never zero-fill",
        "weights": "one total unit per symbol-session, shared among same-day cuts",
        "status": "EXPLORATORY_ALREADY_INSPECTED_COHORT_NO_VIRGIN_HOLDOUT",
        "five_year_policy": "No five-year intraday/PIT-news cohort exists; do not mix feeds or fabricate backdated news.",
        "promotion_allowed": False,
    }
    write_signed(PROTOCOL, protocol)
    return protocol


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Feature no finita: {name}")
    return number


def _features(snapshot):
    f = snapshot["features"]
    price, atr = _finite(f["last_price"], "last_price"), _finite(f["atr_5m"], "atr_5m")
    if min(price, atr) <= 0:
        raise ValueError("Precio/ATR inválido.")
    volume = _finite(f["volume_ratio"], "volume_ratio")
    if volume < 0:
        raise ValueError("Volumen relativo negativo.")
    tech = [
        math.log1p(volume),
        math.tanh((_finite(f["ema9"], "ema9") - _finite(f["ema21"], "ema21")) / atr),
        math.tanh(_finite(f["macd_histogram_5m"], "macd_histogram_5m") / atr),
        (_finite(f["stochastic_k"], "stochastic_k") - _finite(f["stochastic_d"], "stochastic_d")) / 100.0,
        _finite(f["adx"], "adx") / 50.0,
        math.tanh((price - _finite(f["nearest_support"], "nearest_support")) / atr),
        _finite(f["chart_pattern_impact"], "chart_pattern_impact") / 20.0,
        float(f["market_regime"] == "TREND"),
        float(snapshot["symbol"] == "NVDA"),
    ]
    scores = []
    for key in ("sentiment_4h", "sentiment_24h"):
        item = snapshot[key]
        if item["status"] == "AVAILABLE":
            scores.append(_finite(item["score"], key))
        elif item["status"] == "N/D":
            scores.append(None)
        else:
            raise ValueError("Estado de sentimiento desconocido.")
    return tech, scores + [float(scores[0] is None)]


def _snapshot_row(path, snapshot, source):
    """Validated trainable row, or None when the cut is not a labelled LONG fill."""
    if (snapshot["feature_version"] != FEATURE_VERSION or
            snapshot.get("policy", {}).get("market_feed") != "Alpaca IEX"):
        raise ValueError("Feature o fuente de mercado fuera del contrato.")
    cut = pd.Timestamp(snapshot["cut_at_utc"])
    if (pd.Timestamp(snapshot["source_bar_closed_at_utc"]) > cut or
            pd.Timestamp(snapshot["news_collected_at_utc"]) > cut):
        raise ValueError("Features o noticias posteriores al corte.")
    for headline in snapshot["headlines"]:
        published = pd.Timestamp(headline["published_at_utc"])
        collected = pd.Timestamp(headline["collected_at_utc"])
        if published > collected or collected > cut:
            raise ValueError("Titular futuro o no disponible al corte.")
    contract = snapshot.get("operational_contract")
    if not contract or contract["side"] != "LONG":
        return None
    label_file = label_path(path)
    if not label_file.exists():
        return None
    label = signed_payload(label_file)
    if (label["parent_snapshot_sha256"] != sha(snapshot) or
            label["snapshot_id"] != snapshot["snapshot_id"] or
            label.get("data_source", DATA_SOURCE) != source or
            label["outcome"] not in ("TP_FIRST", "SL_FIRST", "TIMEOUT")):
        raise ValueError("Etiqueta o procedencia no coincide con el snapshot.")
    if label["fill_possible"] is not True:
        return None
    if pd.Timestamp(label["label_at_utc"]) <= cut:
        raise ValueError("Etiqueta registrada antes del corte.")
    tech, news = _features(snapshot)
    return {
        "snapshot_id": snapshot["snapshot_id"],
        "session": snapshot["session_ny"], "symbol": snapshot["symbol"],
        "cut_at_utc": snapshot["cut_at_utc"],
        "label_at_utc": label["label_at_utc"],
        "outcome": label["outcome"], "y": int(label["outcome"] == "TP_FIRST"),
        "tech": tech, "news": news,
    }


def _rows(protocol):
    result = []
    finbert_identity = None
    for item in protocol["snapshot_manifest"]:
        path = ROOT / item["path"]
        if hash_file(path) != item["file_sha256"]:
            raise ValueError("Snapshot modificado desde el preregistro.")
        snapshot = signed_payload(path)
        if (snapshot["snapshot_id"] != item["snapshot_id"] or
                snapshot["session_ny"] != item["session_ny"] or
                snapshot["symbol"] != item["symbol"] or
                snapshot.get("data_source", DATA_SOURCE) != protocol["source"]):
            raise ValueError("Identidad/procedencia de snapshot cambió.")
        identity = (snapshot["model_version"], snapshot["finbert_manifest_sha256"])
        if finbert_identity is None:
            finbert_identity = identity
        elif finbert_identity != identity:
            raise ValueError("Se mezclaron versiones de FinBERT.")
        row = _snapshot_row(path, snapshot, protocol["source"])
        if row is not None:
            result.append(row)
    return result


def _preprocess(train, test, with_news):
    names = TECH_FEATURES + NEWS_FEATURES if with_news else TECH_FEATURES
    def matrix(rows):
        return np.asarray([r["tech"] + (r["news"] if with_news else []) for r in rows], dtype=object)
    x, t = matrix(train), matrix(test)
    if x.shape[1] != len(names) or t.shape[1] != len(names):
        raise ValueError("Vector de features distinto entre entrenamiento y evaluación.")
    medians = []
    for i in range(len(names)):
        available = [float(v) for v in x[:, i] if v is not None]
        if not available:
            raise ValueError("Feature sin cobertura en entrenamiento: " + names[i])
        medians.append(float(np.median(available)))
    for array in (x, t):
        for i, median in enumerate(medians):
            array[:, i] = [median if v is None else float(v) for v in array[:, i]]
    x, t = x.astype(float), t.astype(float)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.maximum(scales, SCALE_FLOOR)
    return (np.clip((x - means) / scales, -STANDARDIZED_CLIP, STANDARDIZED_CLIP),
            np.clip((t - means) / scales, -STANDARDIZED_CLIP, STANDARDIZED_CLIP),
            medians, means, scales)


def _group_weights(rows):
    counts = Counter((r["symbol"], r["session"]) for r in rows)
    return np.asarray([1.0 / counts[r["symbol"], r["session"]] for r in rows])


def _fit(x, y, weights):
    """Deterministic L2 logistic with weighted, normalized full-batch gradient."""
    design = np.column_stack([np.ones(len(x)), x])
    coeff = np.zeros(design.shape[1])
    total = float(weights.sum())
    for _ in range(2500):
        z = np.clip(design @ coeff, -35, 35)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (weights * (p - y)) / total
        gradient[1:] += RIDGE * coeff[1:] / total
        next_coeff = coeff - 0.15 * gradient
        if np.max(np.abs(next_coeff - coeff)) < 1e-8:
            coeff = next_coeff
            break
        coeff = next_coeff
    return coeff


def _predict(x, coeff):
    z = np.clip(np.column_stack([np.ones(len(x)), x]) @ coeff, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _score(rows, p):
    y = np.asarray([r["y"] for r in rows], dtype=float)
    w = _group_weights(rows)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8)
    return {
        "n_cuts": len(rows), "n_symbol_sessions": len(set((r["symbol"], r["session"]) for r in rows)),
        "brier": float(np.average((p-y)**2, weights=w)),
        "log_loss": float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)), weights=w)),
    }


def _paired_bootstrap(predictions, left_name, right_name, *, draws=2000,
                      method="paired day-cluster bootstrap; exploratory, already inspected dates"):
    """Day-clustered uncertainty for a paired Brier improvement (right vs left)."""
    left = predictions[left_name]
    right = predictions[right_name]
    if ([r["snapshot_id"] for r, _ in left] !=
            [r["snapshot_id"] for r, _ in right]):
        raise ValueError("Predicciones de modelos no alineadas.")
    days = sorted({r["session"] for r, _ in left})
    if not days:
        return None
    by_day = {}
    for day in days:
        indexes = [i for i, (r, _) in enumerate(left) if r["session"] == day]
        day_rows = [left[i][0] for i in indexes]
        a = _score(day_rows, [left[i][1] for i in indexes])
        b = _score(day_rows, [right[i][1] for i in indexes])
        by_day[day] = (a["brier"], b["brier"], a["n_symbol_sessions"])
    rng = np.random.default_rng(20261028)
    differences = []
    for _ in range(draws):
        chosen = rng.choice(days, len(days), replace=True)
        units = np.asarray([by_day[day][2] for day in chosen], dtype=float)
        gains = np.asarray([by_day[day][0] - by_day[day][1] for day in chosen])
        differences.append(float(np.average(gains, weights=units)))
    return {"n_day_clusters": len(days),
            "brier_improvement_95pct_interval":
            [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            "method": method}


def _split_fold(rows, sessions, start, test_length=3, embargo=1):
    test_days = set(sessions[start:start+test_length])
    candidates = [r for r in rows if r["session"] in test_days]
    if not candidates:
        return [], [], test_days
    first_test = min(pd.Timestamp(r["cut_at_utc"]) for r in candidates)
    train_days = set(sessions[:max(0, start-embargo)])
    train = [r for r in rows if r["session"] in train_days and
             pd.Timestamp(r["label_at_utc"]) < first_test]
    return train, candidates, test_days


def run():
    protocol = read_signed(PROTOCOL)
    if protocol["version"] != VERSION:
        raise ValueError("Protocolo de otra versión.")
    rows = _rows(protocol)
    sessions = protocol["sessions"]
    predictions = defaultdict(list)
    folds = []
    for start in protocol["fold_starts"]:
        if start >= len(sessions):
            continue
        train, test, test_days = _split_fold(
            rows, sessions, start, protocol["fold_test_sessions"],
            protocol["embargo_sessions"])
        if not train or not test or len({r["y"] for r in train}) < 2:
            folds.append({"test_days": sorted(test_days), "status": "INSUFFICIENT_TRAIN_CLASSES"})
            continue
        weights = _group_weights(train)
        prior = float(np.average([r["y"] for r in train], weights=weights))
        if not 0 < prior < 1:
            raise ValueError("Prior sin ambas clases.")
        p_base = np.full(len(test), prior)
        predictions["TRAIN_FREQUENCY"].extend(zip(test, p_base))
        metrics = {"TRAIN_FREQUENCY": _score(test, p_base)}
        for name, news in (("RIDGE_TECH", False), ("RIDGE_TECH_NEWS", True)):
            x, t, _, _, _ = _preprocess(train, test, news)
            model = _fit(x, np.asarray([r["y"] for r in train]), weights)
            p = _predict(t, model)
            predictions[name].extend(zip(test, p))
            metrics[name] = _score(test, p)
        folds.append({"test_days": sorted(test_days), "status": "SCORED",
                      "train_cuts": len(train), "train_symbol_sessions":
                      len(set((r["symbol"], r["session"]) for r in train)),
                      "test_cuts": len(test), "train_prior_tp": prior,
                      "metrics": metrics})
    results = {}
    for name, values in predictions.items():
        evaluation_rows = [r for r, _ in values]
        results[name] = _score(evaluation_rows, [p for _, p in values])
    base = results.get("TRAIN_FREQUENCY", {}).get("brier")
    for name in ("RIDGE_TECH", "RIDGE_TECH_NEWS"):
        if name in results and base:
            results[name]["brier_skill_vs_frequency"] = 1 - results[name]["brier"] / base
    by_symbol = {}
    for symbol in ("SMCI", "NVDA"):
        by_symbol[symbol] = {}
        for name, values in predictions.items():
            selected = [(r, p) for r, p in values if r["symbol"] == symbol]
            if selected:
                by_symbol[symbol][name] = _score(
                    [r for r, _ in selected], [p for _, p in selected])
    paired_uncertainty = {}
    if predictions.get("RIDGE_TECH_NEWS"):
        for name in ("TRAIN_FREQUENCY", "RIDGE_TECH"):
            paired_uncertainty["RIDGE_TECH_NEWS_vs_" + name] = _paired_bootstrap(
                predictions, name, "RIDGE_TECH_NEWS")
    # Fit a frozen research artifact on all labels known at run time. It is not
    # deployed and its probabilities are expressly NOT a live trading signal.
    final = {}
    if rows and len({r["y"] for r in rows}) == 2:
        for name, news in (("RIDGE_TECH", False), ("RIDGE_TECH_NEWS", True)):
            x, _, medians, means, scales = _preprocess(rows, rows, news)
            coeff = _fit(x, np.asarray([r["y"] for r in rows]), _group_weights(rows))
            final[name] = {"features": list(TECH_FEATURES + NEWS_FEATURES if news else TECH_FEATURES),
                           "coefficients": coeff.tolist(), "medians": medians,
                           "means": means.tolist(), "scales": scales.tolist(),
                           "training_cuts": len(rows), "research_only": True}
    outcomes = Counter(r["outcome"] for r in rows)
    report = {
        "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(protocol), "data_source": DATA_SOURCE,
        "population": "LONG, 1D, resolved and OHLC-possible fills; not broker executions",
        "trainable_cuts": len(rows),
        "symbol_sessions": len(set((r["symbol"], r["session"]) for r in rows)),
        "sessions": len(set(r["session"] for r in rows)),
        "outcomes": dict(outcomes),
        "finbert_identity": {
            "model_version": signed_payload(ROOT / protocol["snapshot_manifest"][0]["path"])["model_version"],
            "weights_manifest_sha256": signed_payload(ROOT / protocol["snapshot_manifest"][0]["path"])["finbert_manifest_sha256"],
        },
        "missing_news_4h_cuts": sum(r["news"][0] is None for r in rows),
        "folds": folds, "walk_forward": results,
        "walk_forward_by_symbol": by_symbol,
        "paired_uncertainty": paired_uncertainty,
        "final_models": final,
        "interpretation": "EXPLORATORY; V2 numerical stability fix follows inspected V1 results. Existing dates/outcomes inspected; no virgin holdout, no calibrated/live probability, no buy/sell authorization.",
        "training_performed": True, "production_model_changed": False,
        "portfolio_db_accessed": False,
    }
    path = OUT / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_signed(path, report)
    print(json.dumps({"report": str(path), "trainable_cuts": len(rows),
                      "symbol_sessions": report["symbol_sessions"],
                      "outcomes": report["outcomes"], "walk_forward": results,
                      "by_symbol": by_symbol,
                      "paired_uncertainty": paired_uncertainty,
                      "folds": [{k:v for k,v in fold.items() if k != "metrics"} for fold in folds]},
                     ensure_ascii=False, indent=2))
    return report


def _apply_frozen(rows, model):
    """Score rows with frozen coefficients; train-time medians/means/scales only."""
    with_news = len(model["features"]) == len(TECH_FEATURES) + len(NEWS_FEATURES)
    if list(model["features"]) != list(TECH_FEATURES + (NEWS_FEATURES if with_news else ())):
        raise ValueError("Features del modelo congelado no coinciden con el contrato.")
    values = []
    for r in rows:
        vector = r["tech"] + (r["news"] if with_news else [])
        values.append([m if v is None else float(v) for v, m in zip(vector, model["medians"])])
    x = np.asarray(values, dtype=float).reshape(len(rows), len(model["features"]))
    z = np.clip((x - np.asarray(model["means"])) / np.asarray(model["scales"]),
                -STANDARDIZED_CLIP, STANDARDIZED_CLIP)
    return _predict(z, np.asarray(model["coefficients"], dtype=float))


def _frozen_run(protocol):
    """Latest complete signed run for this protocol; forward cuts must postdate it."""
    protocol_sha = sha(protocol)
    for path in sorted(OUT.glob("run_*.json"), reverse=True):
        payload = read_signed(path)
        if (payload.get("protocol_sha256") == protocol_sha and payload.get("final_models")
                and payload.get("finbert_identity")):
            return path, payload
    raise ValueError("No existe un run firmado con modelos finales para este protocolo.")


def forward():
    """Genuine out-of-sample test: frozen V2 models on sessions after the protocol.

    Nothing is refit. Sessions after the preregistered cohort were unknown when
    the models were frozen, so this is the first evaluation that is not an
    already-inspected cohort. It still authorizes no trade.
    """
    protocol = read_signed(PROTOCOL)
    if protocol["version"] != VERSION:
        raise ValueError("Protocolo de otra versión.")
    run_path, frozen = _frozen_run(protocol)
    frozen_at = pd.Timestamp(frozen["generated_at_utc"])
    # Labels written after the freeze (e.g. the last protocol session) were not
    # seen by the frozen model; they are neither training nor forward evidence.
    train = [r for r in _rows(protocol) if pd.Timestamp(r["label_at_utc"]) <= frozen_at]
    if len(train) != frozen["trainable_cuts"]:
        raise ValueError("Los cortes de entrenamiento ya no coinciden con el run congelado.")
    prior = float(np.average([r["y"] for r in train], weights=_group_weights(train)))
    identity = (frozen["finbert_identity"]["model_version"],
                frozen["finbert_identity"]["weights_manifest_sha256"])
    last_session = max(protocol["sessions"])
    rows, skipped = [], Counter()
    for path in _paths():
        snapshot = signed_payload(path)
        if snapshot["session_ny"] <= last_session:
            continue
        if (snapshot.get("synthetic_example")
                or snapshot.get("data_source", DATA_SOURCE) != protocol["source"]
                or snapshot.get("training_performed") is not False
                or snapshot["horizon"] != "1D"):
            skipped["other_population"] += 1
            continue
        if (snapshot["model_version"], snapshot["finbert_manifest_sha256"]) != identity:
            skipped["finbert_changed"] += 1
            continue
        if pd.Timestamp(snapshot["cut_at_utc"]) <= frozen_at:
            skipped["cut_before_freeze"] += 1
            continue
        row = _snapshot_row(path, snapshot, protocol["source"])
        if row is None:
            skipped["not_labelled_long_fill"] += 1
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r["cut_at_utc"], r["symbol"]))
    report = {
        "version": VERSION + "_FORWARD", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(protocol), "frozen_run": run_path.name,
        "frozen_run_sha256": sha(frozen), "train_prior_tp": prior,
        "forward_cuts": len(rows),
        "forward_symbol_sessions": len({(r["symbol"], r["session"]) for r in rows}),
        "forward_sessions": sorted({r["session"] for r in rows}),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "skipped": dict(skipped),
        "refit_performed": False, "production_model_changed": False,
        "portfolio_db_accessed": False,
        "interpretation": ("Out-of-sample on sessions unknown at freeze time. "
                           "Small sample: evidence, not a calibrated live probability "
                           "nor a buy/sell authorization."),
    }
    if rows:
        predictions = {"TRAIN_FREQUENCY": list(zip(rows, np.full(len(rows), prior)))}
        for name in ("RIDGE_TECH", "RIDGE_TECH_NEWS"):
            predictions[name] = list(zip(rows, _apply_frozen(rows, frozen["final_models"][name])))
        results = {name: _score(rows, [p for _, p in values]) for name, values in predictions.items()}
        base = results["TRAIN_FREQUENCY"]["brier"]
        for name in ("RIDGE_TECH", "RIDGE_TECH_NEWS"):
            results[name]["brier_skill_vs_frequency"] = 1 - results[name]["brier"] / base
        report["forward"] = results
        report["observed_tp_rate"] = float(np.average([r["y"] for r in rows],
                                                      weights=_group_weights(rows)))
        report["paired_uncertainty"] = {
            "RIDGE_TECH_NEWS_vs_" + name: _paired_bootstrap(
                predictions, name, "RIDGE_TECH_NEWS",
                method="paired day-cluster bootstrap; forward sessions unseen at freeze")
            for name in ("TRAIN_FREQUENCY", "RIDGE_TECH")
        }
    path = OUT / ("forward_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write_signed(path, report)
    print(json.dumps({"report": str(path), **{k: v for k, v in report.items()
                                              if k not in ("interpretation",)}},
                     ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "forward"))
    args = parser.parse_args()
    if args.command == "prepare":
        p = prepare()
        print(json.dumps({"protocol": str(PROTOCOL), "sha256": sha(p),
                          "snapshots": len(p["snapshot_manifest"]),
                          "sessions": len(p["sessions"])}))
    elif args.command == "forward":
        forward()
    else:
        run()


if __name__ == "__main__":
    main()
