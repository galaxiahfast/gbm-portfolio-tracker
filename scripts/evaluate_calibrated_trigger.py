"""Preregister then evaluate a calibrated, offline 1D LONG decision rule.

Consumes only the already signed *development* replay. The prior final block
has been opened and is deliberately not loaded or relabeled here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import exchange_calendars as xcals
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.calibrated_trigger_research import (
    CALIBRATION_VERSION, apply_platt, count_selection, fit_platt,
    frozen_quantile_threshold, net_ev,
)
from portfolio_tracker.analytics.horizon_models import TARGET_CLASSES
from portfolio_tracker.analytics.learned_trigger_research import (
    embargoed_training, fit_model, predict, score_labels,
)
from scripts.download_massive_history import read_signed, sha, write_signed


SOURCE = ROOT / "output/historical_replay/learned_trigger_v1"
OUTPUT = ROOT / "output/historical_replay/calibrated_trigger_rule_v2"
PROTOCOL_PATH = OUTPUT / "preregistration.json"
RESULT_PATH = OUTPUT / "development_validation.json"
FIXED_L2 = 1.0
CALIBRATION_FRACTION = 0.30
TOP_FRACTIONS = (0.05, 0.10, 0.20)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregister():
    if PROTOCOL_PATH.exists():
        return read_signed(PROTOCOL_PATH)
    prior = read_signed(SOURCE / "preregistration.json")
    if not (SOURCE / "holdout_opening.json").exists():
        raise ValueError("El estado anterior del holdout es inesperado.")
    protocol = {
        "version": "CALIBRATED_LONG_EV_RESEARCH_V2",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_development_sha256": digest(SOURCE / "development.json"),
        "source_protocol_sha256": sha(prior),
        "development_start": prior["development_start"],
        "development_end": prior["development_end"],
        "horizon": "1 Día",
        "excluded_horizon": "1 Hora",
        "exclusion_reason": "63/68 possible final fills were TIMEOUT (92.6%); changing 1h barriers would change its target.",
        "fixed_model_l2": FIXED_L2,
        "model_source": "Existing L2 multinomial logit, fixed before this run; no fit on AND entries alone.",
        "calibration": CALIBRATION_VERSION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "minimum_calibration_class_support": 3,
        "outer_folds": "Three expanding XNYS-session folds from existing replay; earliest 70% of each prior train span for base model; later 30% for Platt and decision thresholds.",
        "purge_and_embargo": "Label available by boundary; skip one XNYS session before calibration and test.",
        "decision_rules": ["TOP_5_PERCENT", "TOP_10_PERCENT", "TOP_20_PERCENT", "NET_EV_POSITIVE"],
        "quantile_policy": "Threshold fitted on pre-test calibration candidate EVs; no ranking over test/future candidates.",
        "ev_policy": "Training-only class mean net returns already include assumed 25bps commission/side, 5bps slippage/side, 4bps full spread. NET_EV_POSITIVE is equivalent to gross EV > modeled cost; no double charge.",
        "comparators": ["CURRENT_AND", "EMA9_21", "TRAIN_ONLY_CLASS_FREQUENCY", "TRAIN_ONLY_EMA_CONDITIONED_FREQUENCY"],
        "prior_internal_holdout": "ALREADY_OPENED_AND_PREVIOUSLY_INSPECTED; NOT REUSED, NOT FRESH OOS",
        "approval": "NONE; exploratory development validation only; production unchanged",
        "source_module_sha256": digest(ROOT / "portfolio_tracker/analytics/calibrated_trigger_research.py"),
        "source_runner_sha256": digest(Path(__file__)),
    }
    write_signed(PROTOCOL_PATH, protocol)
    return protocol


def _folds(sessions):
    # Same calendar boundaries as the existing learned-trigger pilot.
    n = len(sessions)
    start = max(100, n // 2)
    tail = n - start
    if tail < 60:
        raise ValueError("Desarrollo insuficiente para tres folds.")
    limits = (start, start + tail // 3, start + 2 * tail // 3, n)
    return [(sessions[:limits[i]], sessions[limits[i]:limits[i + 1]]) for i in range(3)]


def _metrics(rows, probabilities, model):
    prior = np.repeat(np.asarray(model["prior"]).reshape(1, -1), len(rows), axis=0)
    ema_prior = np.asarray([model["ema_prior"][str(bool(r["ema_rule"]))] for r in rows])
    return {"calibrated": score_labels(rows, probabilities),
            "frequency": score_labels(rows, prior),
            "ema_conditioned_frequency": score_labels(rows, ema_prior)}


def _aggregate(folds):
    result = {"folds_scored": len(folds)}
    for comparator in ("calibrated", "frequency", "ema_conditioned_frequency"):
        samples = [(f["metrics"][comparator], f) for f in folds]
        n = sum(metric["n"] for metric, _ in samples)
        result[comparator] = {"n": n,
                              "brier": sum(metric["brier"] * metric["n"] for metric, _ in samples) / n if n else None,
                              "log_loss": sum(metric["log_loss"] * metric["n"] for metric, _ in samples) / n if n else None}
    for comparator in ("frequency", "ema_conditioned_frequency"):
        denominator = result[comparator]["brier"]
        result["brier_skill_vs_" + comparator] = (
            1 - result["calibrated"]["brier"] / denominator
            if denominator and result["calibrated"]["brier"] is not None else None)
    trades = {}
    for arm in ("TOP_5_PERCENT", "TOP_10_PERCENT", "TOP_20_PERCENT", "NET_EV_POSITIVE", "CURRENT_AND", "EMA9_21"):
        entries = [f["trades"][arm] for f in folds]
        trades[arm] = {"long_signals": sum(e["long_signals"] for e in entries),
                       "possible_ohlc_fills": sum(e["possible_ohlc_fills"] for e in entries),
                       "outcomes": {c: sum(e["outcomes"][c] for e in entries) for c in TARGET_CLASSES},
                       "net_return_sum_hypothetical": sum(e["net_return_sum_hypothetical"] for e in entries)}
    result["trades"] = trades
    return result


def evaluate():
    p = preregister()
    if RESULT_PATH.exists():
        return read_signed(RESULT_PATH)
    if digest(SOURCE / "development.json") != p["source_development_sha256"]:
        raise ValueError("Datos de desarrollo cambiaron tras preregistro.")
    if (digest(ROOT / "portfolio_tracker/analytics/calibrated_trigger_research.py") != p["source_module_sha256"]
            or digest(Path(__file__)) != p["source_runner_sha256"]):
        raise ValueError("Código experimental cambió tras preregistro.")
    if p["prior_internal_holdout"] != "ALREADY_OPENED_AND_PREVIOUSLY_INSPECTED; NOT REUSED, NOT FRESH OOS":
        raise ValueError("Política de holdout alterada.")
    development = read_signed(SOURCE / "development.json")
    if development["protocol_sha256"] != p["source_protocol_sha256"]:
        raise ValueError("Replay de otro protocolo.")
    rows = [r for r in development["rows"] if r["horizon"] == p["horizon"]]
    sessions = [day.date().isoformat() for day in xcals.get_calendar("XNYS").sessions_in_range(
        p["development_start"], p["development_end"])]
    if len(sessions) != 239 or any(row["session_ny"] not in set(sessions) for row in rows):
        raise ValueError("Calendario de desarrollo desalineado.")
    folds = []
    for train_sessions, test_sessions in _folds(sessions):
        calibration_start_index = int(len(train_sessions) * (1 - CALIBRATION_FRACTION))
        calibration_start = train_sessions[calibration_start_index]
        base = embargoed_training(rows, calibration_start, sessions=sessions)
        calibration = [r for r in rows if r["session_ny"] in set(train_sessions[calibration_start_index:])]
        calibration_labeled = [r for r in calibration if r["outcome"] in TARGET_CLASSES]
        test = [r for r in rows if r["session_ny"] in set(test_sessions)]
        fold = {"test_start": test_sessions[0], "test_end": test_sessions[-1],
                "base_train_n": len(base), "calibration_n": len(calibration_labeled),
                "calibration_class_counts": dict(Counter(r["outcome"] for r in calibration_labeled)),
                "test_long_plans": len(test)}
        try:
            model = fit_model(base, FIXED_L2)
            cal_p = predict(model, calibration_labeled)
            calibrator = fit_platt(cal_p, [r["outcome"] for r in calibration_labeled])
            # Class payoffs and quantile thresholds are frozen before the test span.
            cal_all_p = apply_platt(calibrator, predict(model, calibration))
            cal_ev = net_ev(cal_all_p, model["payoff_by_class_net_return"])
            thresholds = {f"TOP_{int(x*100)}_PERCENT": frozen_quantile_threshold(cal_ev, x)
                          for x in TOP_FRACTIONS}
            test_p = apply_platt(calibrator, predict(model, test))
            test_ev = net_ev(test_p, model["payoff_by_class_net_return"])
            selection = {name: test_ev >= threshold for name, threshold in thresholds.items()}
            selection["NET_EV_POSITIVE"] = test_ev > 0.0
            selection["CURRENT_AND"] = [bool(r["current_and"]) for r in test]
            selection["EMA9_21"] = [bool(r["ema_rule"]) for r in test]
            fold.update(status="SCORED", calibration=calibrator,
                        thresholds_net_return=thresholds,
                        test_ev_min=float(np.min(test_ev)) if len(test_ev) else None,
                        test_ev_max=float(np.max(test_ev)) if len(test_ev) else None,
                        metrics=_metrics(test, test_p, model),
                        trades={name: count_selection(test, selected)
                                for name, selected in selection.items()})
        except ValueError as exc:
            fold.update(status="NOT_SCORED", reason=str(exc))
        folds.append(fold)
    scored = [fold for fold in folds if fold["status"] == "SCORED"]
    result = {"version": p["version"], "protocol_sha256": sha(p),
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "horizon": "1 Día", "excluded_1h": True,
              "development_long_candidates": len(rows),
              "development_possible_ohlc_fills": sum(r["outcome"] in TARGET_CLASSES for r in rows),
              "folds": folds, "development_oof": _aggregate(scored) if scored else None,
              "final_holdout": "NOT_EVALUATED_ALREADY_OPENED",
              "production_changed": False, "portfolio_db_accessed": False,
              "model_approved": False}
    write_signed(RESULT_PATH, result)
    render_report(result)
    return result


def render_report(result):
    aggregate = result["development_oof"]
    lines = ["# Regla calibrada LONG 1D · ensayo de desarrollo, no producción", "",
             "El bloque final anterior ya estaba abierto y sus fechas se habían inspeccionado; no se reutilizó. "
             "Esto es **walk-forward de desarrollo**, no una nueva prueba independiente ni una aprobación.",
             "Se excluyó 1h antes de correr: 63/68 fills posibles previos fueron timeout (92.6%). "
             "No se modificaron barreras de 1h.",
             "Calibración Platt uno-contra-resto (normalizada a tres clases), ajustada solo en el tramo "
             "cronológico posterior al entrenamiento base. Umbrales de cuantiles fijados con esa calibración, "
             "no con el test. Purga de etiquetas y embargo XNYS de una sesión.",
             "EV calculado con pagos netos por clase del entrenamiento. Las comisiones, spread y slippage "
             "son **supuestos** ya incluidos; exigir EV neto > coste otra vez los contaría dos veces.", "",
             "| Regla | Señales LONG | Fills OHLC posibles | TP/SL/timeout | Retorno neto hipotético sumado | Brier Skill vs frecuencia / EMA |",
             "|---|---:|---:|---|---:|---|" ]
    if aggregate:
        skill_frequency = aggregate["brier_skill_vs_frequency"]
        skill_ema = aggregate["brier_skill_vs_ema_conditioned_frequency"]
        skill = f"{skill_frequency:+.4f} / {skill_ema:+.4f}" if skill_frequency is not None and skill_ema is not None else "N/D"
        for name, trade in aggregate["trades"].items():
            outcomes = trade["outcomes"]
            lines.append(f"| {name} | {trade['long_signals']} | {trade['possible_ohlc_fills']} | "
                         f"{outcomes['TP_FIRST']}/{outcomes['SL_FIRST']}/{outcomes['TIMEOUT']} | "
                         f"{trade['net_return_sum_hypothetical']:+.4f} | "
                         f"{skill if name.startswith('TOP_') or name == 'NET_EV_POSITIVE' else 'N/A (regla, no probabilidad)'} |")
        lines += ["", f"Brier calibrado={aggregate['calibrated']['brier']:.4f}; "
                  f"frecuencia base={aggregate['frequency']['brier']:.4f}; "
                  f"frecuencia condicionada por EMA={aggregate['ema_conditioned_frequency']['brier']:.4f}.",
                  f"Log loss calibrado={aggregate['calibrated']['log_loss']:.4f}; "
                  f"base={aggregate['frequency']['log_loss']:.4f}; "
                  f"EMA={aggregate['ema_conditioned_frequency']['log_loss']:.4f}.",
                  "Brier y log loss se miden sobre todos los fills etiquetados de test, no solo señales seleccionadas. "
                  "La regla EMA9/21 por sí sola no emite probabilidades; para Brier se usa la frecuencia "
                  "por estado EMA estimada exclusivamente en entrenamiento."]
    else:
        lines.append("| Ninguna | 0 | 0 | N/D | N/D | N/D |")
    lines += ["", "## Folds", ""]
    for fold in result["folds"]:
        lines.append(f"- {fold['test_start']}–{fold['test_end']}: {fold['status']}; "
                     f"entrenamiento={fold['base_train_n']}, calibración={fold['calibration_n']}, "
                     f"planes test={fold['test_long_plans']}; "
                     f"clases cal={fold['calibration_class_counts']}" +
                     (f"; causa={fold['reason']}" if "reason" in fold else ""))
    lines += ["", "Ningún fill de broker fue observado. El AND actual no cambió. "
              "No se abrió el holdout ni se consultó `portfolio.db`. "
              "Sin bloque futuro nuevo, no es válido afirmar cuántas señales habrá en un holdout independiente."]
    (OUTPUT / "reporte.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preregister", "evaluate"))
    phase = parser.parse_args().phase
    if phase == "preregister":
        print("PREREGISTERED", sha(preregister()))
    else:
        outcome = evaluate()
        print("DEVELOPMENT_EVALUATED", outcome["development_oof"])


if __name__ == "__main__":
    main()
