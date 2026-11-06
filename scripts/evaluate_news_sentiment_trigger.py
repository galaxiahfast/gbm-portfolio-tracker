"""Preregistered offline 1D ablation: 19 base features vs 19 + PIT news.

Fail closed until all 598 11:00 snapshots have genuinely point-in-time news
coverage. Never reads the previously opened final holdout or portfolio.db.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import exchange_calendars as xcals
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.calibrated_trigger_research import (
    apply_platt, count_selection, fit_platt, frozen_quantile_threshold, net_ev,
)
from portfolio_tracker.analytics.horizon_models import TARGET_CLASSES
from portfolio_tracker.analytics.learned_trigger_research import embargoed_training, score_labels
from portfolio_tracker.analytics.news_sentiment_model_research import (
    NEWS_FEATURE_NAMES, fit_augmented_model, predict_augmented,
)
from portfolio_tracker.analytics.news_sentiment_replay import attach_sentiment_feature
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.evaluate_calibrated_trigger import (
    SOURCE, OUTPUT as BASE_OUTPUT, _folds, digest,
)


OUTPUT = ROOT / "output/historical_replay/news_sentiment_v1"
PREREG = OUTPUT / "ablation_preregistration_v2.json"
RESULT = OUTPUT / "ablation_development.json"


def preregister():
    if PREREG.exists():
        return read_signed(PREREG)
    base = read_signed(BASE_OUTPUT / "preregistration.json")
    protocol = {
        "version": "PIT_NEWS_ABLATION_1D_V2",
        "supersedes": "PIT_NEWS_ABLATION_1D_V1, never evaluated: a late-scored article was incorrectly classed as no-news instead of N/D. No provider data or labels were inspected before this correction.",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_development_sha256": base["source_development_sha256"],
        "source_base_validation_sha256": digest(BASE_OUTPUT / "development_validation.json"),
        "source_base_protocol_sha256": sha(base),
        "horizon": "1 Día", "symbols": ["SMCI", "NVDA"],
        "fixed_cuts": "299 XNYS sessions per symbol, 11:00 NY",
        "required_pit_coverage": "598/598 before training; N/D is never imputed as 0",
        "feature_names": list(NEWS_FEATURE_NAMES),
        "news_feature": "Mean of eligible NEWS document sentiment in trailing 24h; empty complete snapshot=0; missing provenance=N/D",
        "model": "Same L2 multinomial softmax and preprocessing; fixed L2=1.0",
        "validation": "Same three expanding development folds, base 70% then disjoint Platt 30%, one XNYS-session embargo and label-availability purge",
        "rules": ["TOP_5_PERCENT", "TOP_10_PERCENT", "TOP_20_PERCENT", "NET_EV_POSITIVE"],
        "extreme_sentiment_predeclared": "absolute score >= 0.5; descriptive subgroup only, not selection rule",
        "costs": "Existing net class payoffs only; no new commission/spread/slippage assumptions",
        "final_holdout": "ALREADY_OPENED; NEVER REUSED; OOS RESULT N/D",
        "promotion": "NONE; production and portfolio.db unchanged",
        "module_sha256": digest(ROOT / "portfolio_tracker/analytics/news_sentiment_model_research.py"),
        "sidecar_module_sha256": digest(ROOT / "portfolio_tracker/analytics/news_sentiment_replay.py"),
        "runner_sha256": digest(Path(__file__)),
    }
    write_signed(PREREG, protocol)
    return protocol


def _validated_sidecar(path, development_rows):
    coverage = read_signed(path)
    if (coverage["cuts_total"] != 598 or coverage["valid_pit_scores"] != 598
            or coverage["training_allowed"] is not True):
        raise ValueError("PIT news coverage is incomplete; training forbidden.")
    points = {point["cut_id"]: point for point in coverage["points"]}
    if len(points) != 598:
        raise ValueError("Duplicate/missing news cut id.")
    for row in development_rows:
        attach_sentiment_feature(row, points[row["cut_id"]])
    return points


def evaluate(coverage_path):
    p = preregister()
    if RESULT.exists():
        return read_signed(RESULT)
    if (digest(ROOT / "portfolio_tracker/analytics/news_sentiment_model_research.py") != p["module_sha256"]
            or digest(ROOT / "portfolio_tracker/analytics/news_sentiment_replay.py") != p["sidecar_module_sha256"]
            or digest(Path(__file__)) != p["runner_sha256"]):
        raise ValueError("Research code changed after preregistration.")
    if digest(SOURCE / "development.json") != p["source_development_sha256"]:
        raise ValueError("Signed development replay changed.")
    if digest(BASE_OUTPUT / "development_validation.json") != p["source_base_validation_sha256"]:
        raise ValueError("Baseline validation changed.")
    base_validation = read_signed(BASE_OUTPUT / "development_validation.json")
    prior = read_signed(SOURCE / "preregistration.json")
    development = read_signed(SOURCE / "development.json")
    rows = [row for row in development["rows"] if row["horizon"] == "1 Día"]
    points = _validated_sidecar(coverage_path, rows)
    augmented = [attach_sentiment_feature(row, points[row["cut_id"]]) for row in rows]
    sessions = [date.date().isoformat() for date in xcals.get_calendar("XNYS").sessions_in_range(
        prior["development_start"], prior["development_end"])]
    if len(sessions) != 239:
        raise ValueError("Development XNYS calendar changed.")
    folds = []
    for train_sessions, test_sessions in _folds(sessions):
        start = int(len(train_sessions) * 0.70)
        base_train = embargoed_training(augmented, train_sessions[start], sessions=sessions)
        calibration = [row for row in augmented if row["session_ny"] in set(train_sessions[start:])]
        cal_labeled = [row for row in calibration if row["outcome"] in TARGET_CLASSES]
        test = [row for row in augmented if row["session_ny"] in set(test_sessions)]
        model = fit_augmented_model(base_train, l2=1.0)
        calibrator = fit_platt(predict_augmented(model, cal_labeled),
                              [row["outcome"] for row in cal_labeled])
        cal_ev = net_ev(apply_platt(calibrator, predict_augmented(model, calibration)),
                        model["payoff_by_class_net_return"])
        test_p = apply_platt(calibrator, predict_augmented(model, test))
        test_ev = net_ev(test_p, model["payoff_by_class_net_return"])
        selections = {f"TOP_{int(frac*100)}_PERCENT":
                      test_ev >= frozen_quantile_threshold(cal_ev, frac)
                      for frac in (0.05, 0.10, 0.20)}
        selections["NET_EV_POSITIVE"] = test_ev > 0
        extreme = np.asarray([abs(row["features"][-1]) >= 0.5 for row in test])
        folds.append({"test_start": test_sessions[0], "test_end": test_sessions[-1],
                      "base_train_n": len(base_train), "calibration_n": len(cal_labeled),
                      "test_long_plans": len(test), "test_extreme_sentiment": int(extreme.sum()),
                      "metrics": score_labels(test, test_p),
                      "trades": {name: count_selection(test, selected)
                                 for name, selected in selections.items()},
                      "extreme_trades": {name: count_selection(test, np.asarray(selected) & extreme)
                                         for name, selected in selections.items()}})
    total = sum(fold["metrics"]["n"] for fold in folds)
    augmented_metrics = {"n": total,
                         "brier": sum(fold["metrics"]["brier"] * fold["metrics"]["n"]
                                      for fold in folds) / total,
                         "log_loss": sum(fold["metrics"]["log_loss"] * fold["metrics"]["n"]
                                         for fold in folds) / total}
    base_oof = base_validation["development_oof"]
    if total != base_oof["calibrated"]["n"]:
        raise ValueError("Ablation test sample count changed.")
    result = {"version": p["version"], "protocol_sha256": sha(p),
              "source_coverage_sha256": digest(coverage_path),
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "base_calibrated": base_oof["calibrated"],
              "news_calibrated": augmented_metrics,
              "frequency": base_oof["frequency"],
              "ema_conditioned_frequency": base_oof["ema_conditioned_frequency"],
              "folds": folds, "holdout": "NOT_EVALUATED_ALREADY_OPENED",
              "production_changed": False, "portfolio_db_accessed": False,
              "model_approved": False}
    write_signed(RESULT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preregister", "evaluate"))
    parser.add_argument("--coverage", type=Path)
    args = parser.parse_args()
    if args.phase == "preregister":
        print("PREREGISTERED", sha(preregister()))
    else:
        if args.coverage is None:
            parser.error("--coverage signed sidecar is required")
        print(evaluate(args.coverage))
