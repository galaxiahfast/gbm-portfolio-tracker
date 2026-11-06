"""Preregistered development-only 1D FinBERT ablation on matched news cuts.

The 299-session archive remains intact; N/D cuts are excluded from *both*
model arms, never neutral-imputed. The old final holdout was already opened in
earlier research and is not reopened here. No production module or portfolio
database is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.calibrated_trigger_research import (
    apply_platt, count_selection, fit_platt, frozen_quantile_threshold, net_ev,
)
from portfolio_tracker.analytics.headline_finbert_model_research import (
    fit_headline_model, predict_headline,
)
from portfolio_tracker.analytics.headline_sentiment_replay import FEATURE_VERSION
from portfolio_tracker.analytics.horizon_models import TARGET_CLASSES
from portfolio_tracker.analytics.learned_trigger_research import (
    embargoed_training, fit_model, predict, score_labels,
)
from scripts.collect_benzinga_headlines import OUTPUT as NEWS_OUTPUT
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.evaluate_calibrated_trigger import SOURCE, OUTPUT as BASE_OUTPUT, _folds, digest


OUTPUT = NEWS_OUTPUT
PREREG = OUTPUT / "ablation_preregistration.json"
RESULT = OUTPUT / "development_ablation.json"


def preregister():
    if PREREG.exists():
        return read_signed(PREREG)
    base = read_signed(BASE_OUTPUT / "preregistration.json")
    protocol = {
        "version": "FROZEN_FINBERT_HEADLINE_ABLATION_1D_V1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_development_sha256": digest(SOURCE / "development.json"),
        "source_base_protocol_sha256": sha(base),
        "feature_version": FEATURE_VERSION,
        "source": "Benzinga title and creation timestamp; edited-after-cut excluded; PIT proxy, not certified delivery",
        "model_repository": "ProsusAI/finbert",
        "model_revision": "4556d13015211d73dccd3fdd39d39232506f3e43",
        "symbols": ["SMCI", "NVDA"], "horizon": "1 Día",
        "fixed_cuts": "299 XNYS sessions per symbol at 11:00 NY; previous 60-session final holdout not reused",
        "news_windows_hours": [4, 24],
        "no_news": "N/D; no zero imputation. Both 19-feature and 20-feature arms fit/evaluate on the same headline-present rows for each window.",
        "selection_bias_warning": "Complete-case subset is conditional on having news; not representative of all 299 cuts.",
        "model": "Existing L2 multinomial logistic regression, l2=1.0, same features/targets/costs plus one FinBERT score",
        "calibration": "Same three expanding outer development folds, train 70%, separate Platt 30%; purge labels unavailable before test boundary; one-session train embargo",
        "decision_rules": ["TOP_5_PERCENT", "TOP_10_PERCENT", "TOP_20_PERCENT", "NET_EV_POSITIVE"],
        "comparators": ["MATCHED_19_FEATURES", "TRAIN_ONLY_CLASS_FREQUENCY", "EMA9_21", "CURRENT_AND"],
        "final_holdout": "ALREADY_OPENED; NEVER REUSED; OOS RESULT N/D",
        "promotion": "NONE; production and portfolio.db unchanged",
        "research_module_sha256": digest(ROOT / "portfolio_tracker/analytics/headline_finbert_model_research.py"),
        "selector_module_sha256": digest(ROOT / "portfolio_tracker/analytics/headline_sentiment_replay.py"),
        "runner_sha256": digest(Path(__file__)),
    }
    write_signed(PREREG, protocol)
    return protocol


def join_news(rows, sidecar, hours):
    if hours not in (4, 24) or sidecar.get("cuts") != 598:
        raise ValueError("News sidecar is not the fixed 598-cut archive")
    points = {(point["symbol"], point["session_ny"]): point for point in sidecar["points"]}
    if len(points) != 598:
        raise ValueError("News sidecar duplicate/missing cut")
    paired = []
    missing = Counter()
    for row in rows:
        point = points.get((row["symbol"], row["session_ny"]))
        if point is None or point["cut_at"] != pd.Timestamp(row["observed_at"]).isoformat():
            raise ValueError("News/replay cut mismatch")
        news = point["windows"][str(hours)]
        if news["status"] == "N/D" and news["score"] is None:
            missing[row["symbol"]] += 1
            continue
        if news["status"] != "AVAILABLE" or news["score"] is None:
            raise ValueError("Invalid news availability state")
        paired.append((row, {**row, "features": [*row["features"], float(news["score"])],
                             "headline_count": news["article_count"]}))
    return paired, dict(missing)


def _weighted(folds, key):
    n = sum(f["metrics"][key]["n"] for f in folds)
    if not n:
        return {"n": 0, "brier": None, "log_loss": None}
    return {"n": n, "brier": sum(f["metrics"][key]["brier"] * f["metrics"][key]["n"]
                                   for f in folds) / n,
            "log_loss": sum(f["metrics"][key]["log_loss"] * f["metrics"][key]["n"]
                            for f in folds) / n}


def _total_counts(items):
    return {"long_signals": sum(item["long_signals"] for item in items),
            "possible_ohlc_fills": sum(item["possible_ohlc_fills"] for item in items),
            "outcomes": {category: sum(item["outcomes"][category] for item in items)
                         for category in TARGET_CLASSES},
            "net_return_sum_hypothetical": sum(item["net_return_sum_hypothetical"]
                                               for item in items)}


def _fold_evaluation(paired, sessions, train_sessions, test_sessions, hours):
    by_base = [base for base, _ in paired]
    by_news = [news for _, news in paired]
    calibration_start = train_sessions[int(len(train_sessions) * 0.70)]
    base_train = embargoed_training(by_base, calibration_start, sessions=sessions)
    news_train = embargoed_training(by_news, calibration_start, sessions=sessions)
    cal_set = set(train_sessions[int(len(train_sessions) * 0.70):])
    test_set = set(test_sessions)
    boundary = pd.Timestamp(test_sessions[0] + "T11:00:00", tz="America/New_York").tz_convert("UTC")
    base_cal = [r for r in by_base if r["session_ny"] in cal_set]
    news_cal = [r for r in by_news if r["session_ny"] in cal_set]
    base_cal_labeled = [r for r in base_cal if r["outcome"] in TARGET_CLASSES
                        and pd.Timestamp(r["label_available_at"]) <= boundary]
    news_cal_labeled = [r for r in news_cal if r["outcome"] in TARGET_CLASSES
                        and pd.Timestamp(r["label_available_at"]) <= boundary]
    base_test = [r for r in by_base if r["session_ny"] in test_set]
    news_test = [r for r in by_news if r["session_ny"] in test_set]
    if not (len(base_train) == len(news_train)
            and len(base_cal_labeled) == len(news_cal_labeled)
            and len(base_test) == len(news_test)):
        raise ValueError("Matched base/news populations diverged")
    base_model = fit_model(base_train, l2=1.0)
    news_model = fit_headline_model(news_train, hours=hours, l2=1.0)
    base_calibrator = fit_platt(predict(base_model, base_cal_labeled),
                                [r["outcome"] for r in base_cal_labeled])
    news_calibrator = fit_platt(predict_headline(news_model, news_cal_labeled, hours=hours),
                                [r["outcome"] for r in news_cal_labeled])
    base_test_p = apply_platt(base_calibrator, predict(base_model, base_test))
    news_test_p = apply_platt(news_calibrator, predict_headline(news_model, news_test, hours=hours))
    prior = np.repeat(np.asarray(base_model["prior"]).reshape(1, -1), len(base_test), axis=0)
    ema_prior = np.asarray([base_model["ema_prior"][str(bool(r["ema_rule"]))] for r in base_test])
    metrics = {"base": score_labels(base_test, base_test_p),
               "news": score_labels(news_test, news_test_p),
               "frequency": score_labels(base_test, prior),
               "ema_frequency": score_labels(base_test, ema_prior)}
    decisions = {}
    for name, model, calibrator, cal_rows, test_rows, test_p, predictor in (
        ("base", base_model, base_calibrator, base_cal, base_test, base_test_p,
         lambda m, r: predict(m, r)),
        ("news", news_model, news_calibrator, news_cal, news_test, news_test_p,
         lambda m, r: predict_headline(m, r, hours=hours)),
    ):
        cal_ev = net_ev(apply_platt(calibrator, predictor(model, cal_rows)),
                        model["payoff_by_class_net_return"])
        test_ev = net_ev(test_p, model["payoff_by_class_net_return"])
        masks = {f"TOP_{int(frac*100)}_PERCENT":
                 test_ev >= frozen_quantile_threshold(cal_ev, frac)
                 for frac in (0.05, 0.10, 0.20)}
        masks["NET_EV_POSITIVE"] = test_ev > 0
        decisions[name] = {rule: count_selection(test_rows, mask) for rule, mask in masks.items()}
    decisions["CURRENT_AND"] = count_selection(base_test,
                                                [bool(r["current_and"]) for r in base_test])
    decisions["EMA9_21"] = count_selection(base_test,
                                           [bool(r["ema_rule"]) for r in base_test])
    extreme = np.asarray([abs(r["features"][-1]) >= 0.5 for r in news_test])
    subgroup = {}
    for name, mask in (("extreme_sentiment", extreme),
                       ("non_extreme_sentiment", ~extreme),
                       ("trend_regime", np.asarray([bool(r["features"][12]) for r in base_test])),
                       ("range_regime", np.asarray([bool(r["features"][13]) for r in base_test]))):
        base_group = [r for r, chosen in zip(base_test, mask) if chosen]
        news_group = [r for r, chosen in zip(news_test, mask) if chosen]
        subgroup[name] = {"base": score_labels(base_group, base_test_p[mask]),
                          "news": score_labels(news_group, news_test_p[mask])}
    return {"test_start": test_sessions[0], "test_end": test_sessions[-1],
            "matched_train": len(base_train), "matched_calibration": len(base_cal_labeled),
            "matched_test_plans": len(base_test), "metrics": metrics,
            "decisions": decisions, "subgroups_descriptive_only": subgroup}


def evaluate(sidecar_path):
    protocol = preregister()
    if RESULT.exists():
        return read_signed(RESULT)
    if any((digest(path) != protocol[key]) for path, key in (
        (ROOT / "portfolio_tracker/analytics/headline_finbert_model_research.py", "research_module_sha256"),
        (ROOT / "portfolio_tracker/analytics/headline_sentiment_replay.py", "selector_module_sha256"),
        (Path(__file__), "runner_sha256"),
        (SOURCE / "development.json", "source_development_sha256"),
    )):
        raise ValueError("Research code or source changed after preregistration")
    sidecar = read_signed(sidecar_path)
    if sidecar["version"] != FEATURE_VERSION:
        raise ValueError("Wrong frozen FinBERT sidecar version")
    prior = read_signed(SOURCE / "preregistration.json")
    source = read_signed(SOURCE / "development.json")
    rows = [r for r in source["rows"] if r["horizon"] == "1 Día"]
    sessions = [s.date().isoformat() for s in xcals.get_calendar("XNYS").sessions_in_range(
        prior["development_start"], prior["development_end"])]
    result = {"version": protocol["version"], "protocol_sha256": sha(protocol),
              "sidecar_sha256": digest(sidecar_path), "development_candidates": len(rows),
              "full_archive_cuts": sidecar["cuts"], "windows": {},
              "final_holdout": "NOT_EVALUATED_ALREADY_OPENED",
              "production_changed": False, "portfolio_db_accessed": False,
              "model_approved": False,
              "generated_at": datetime.now(timezone.utc).isoformat()}
    for hours in (4, 24):
        paired, missing = join_news(rows, sidecar, hours)
        window = {"matched_candidate_rows": len(paired), "missing_by_symbol": missing,
                  "folds": [], "status": "EVIDENCIA_INSUFICIENTE"}
        try:
            for train_sessions, test_sessions in _folds(sessions):
                window["folds"].append(_fold_evaluation(paired, sessions, train_sessions,
                                                        test_sessions, hours))
        except ValueError as exc:
            window["reason"] = str(exc)
        else:
            window["metrics"] = {key: _weighted(window["folds"], key)
                                 for key in ("base", "news", "frequency", "ema_frequency")}
            reference = window["metrics"]["frequency"]["brier"]
            if not reference or any(window["metrics"][key]["n"] == 0
                                    for key in window["metrics"]):
                window["reason"] = "No labeled matched test rows or invalid frequency reference"
            else:
                window["status"] = "EXPLORATORY_DEVELOPMENT_ONLY"
                window["brier_skill_vs_frequency"] = {
                    key: (1 - window["metrics"][key]["brier"] / reference)
                    for key in ("base", "news")}
                window["trade_summary"] = {
                    arm: {rule: _total_counts([fold["decisions"][arm][rule]
                                               for fold in window["folds"]])
                          for rule in window["folds"][0]["decisions"][arm]}
                    if arm in ("base", "news") else
                    _total_counts([fold["decisions"][arm] for fold in window["folds"]])
                    for arm in ("base", "news", "CURRENT_AND", "EMA9_21")}
        result["windows"][str(hours)] = window
    write_signed(RESULT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preregister", "evaluate"))
    parser.add_argument("--sidecar", type=Path)
    args = parser.parse_args()
    if args.phase == "preregister":
        print("PREREGISTERED", sha(preregister()))
    else:
        if args.sidecar is None:
            parser.error("--sidecar signed_cut_sidecar.json is required")
        result = evaluate(args.sidecar)
        print({k: v for k, v in result.items() if k != "windows"})
