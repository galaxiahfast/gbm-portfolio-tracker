"""Offline gate audit and preregistered clock experiment; NEVER train or write SQL.

The signal/label/fill code is frozen, including all thresholds. A research-only
clock adapter allows comparing the requested hours without changing the live
11:00 contract. Research authorization is NOT a live/broker authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_historical_replay import load_dataset
from scripts.download_massive_history import read_signed, sha, write_signed
from portfolio_tracker.analytics.causal_core import trigger_momentum
from portfolio_tracker.analytics.closed_bars import NY, _calendar, select_last_closed_bar
from portfolio_tracker.analytics.temporal_contract import BAR, ANCHOR_VERSION
from portfolio_tracker.analytics.technical_probability import add_intraday_indicators
from portfolio_tracker.analytics.technical_validity import current_indicator_suffix
from portfolio_tracker.analytics.causal_core import INTRADAY_READY
from portfolio_tracker.analytics.historical_replay import (
    DEFAULT_PARAMETERS, _cross_asset, _operational_label, validate_historical_replay,
)
from portfolio_tracker.analytics.replay import evaluate_replay_cut
from portfolio_tracker.services.model_execution_record import code_fingerprints, build_replay_snapshot

AUDIT_VERSION = "TRIGGER_CLOCK_DIAGNOSTIC_V1"
INPUT = ROOT / "output/historical_replay/input/massive_400_2026-09-25"
BASELINE = ROOT / "output/historical_replay/pilot_1h_1d"
OUTPUT = ROOT / "output/historical_replay/trigger_audit"
TIMES = ("10:00", "11:00", "12:30", "14:00", "15:30")


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indicator_prefix(dataset, cut):
    start = (cut.tz_convert(NY) - pd.DateOffset(months=1)).normalize()
    bars = select_last_closed_bar(dataset.intraday.loc[
        (dataset.intraday.index >= start) & (dataset.intraday.index <= cut)], "5m", cut)
    return current_indicator_suffix(add_intraday_indicators(bars),
                                    [*INTRADAY_READY, "RSI14"], "5m")


def gate_diagnostics(features, bars, *, plan_side=None, full_trigger=None):
    """Independent marginals, NOT an order-dependent attribution of failures.

    'Inside zone' and 'bullish candle' are descriptive diagnostics, not gates
    of TriggerEngine. The actual closed-bar price gate is previous-high break.
    """
    p, q = bars.iloc[-2], bars.iloc[-1]
    up = bool(p.StochRSI_K <= p.StochRSI_D and q.StochRSI_K > q.StochRSI_D)
    down = bool(p.StochRSI_K >= p.StochRSI_D and q.StochRSI_K < q.StochRSI_D)
    long_momentum, short_momentum = trigger_momentum(bars)
    # Frozen hierarchy text contains the actual SetupEngine votes. Do not use
    # daily EMA features as a substitute for the 4h/1h tactical vote.
    votes = re.search(r"Estructura 4h/1h=\[(-?\d+), (-?\d+)\]", features["hierarchy_detail"])
    if not votes:
        raise ValueError("Snapshot sin votos tácticos verificables.")
    setup = tuple(map(int, votes.groups()))
    buy = features["buy_levels"]
    permission = features["macro_permission"]
    long_core = permission == "LONG_ONLY" and all(v >= 0 for v in setup) and long_momentum
    short_core = permission == "SHORT_ONLY" and all(v <= 0 for v in setup) and short_momentum
    return {
        "inside_buy_zone_descriptive": bool(buy["entry_low"] <= q.Close <= buy["entry_high"]),
        "bullish_closed_candle_descriptive": bool(q.Close > q.Open),
        "previous_high_break": bool(q.Close >= p.High),
        "volume_above_previous20": bool(q.Volume > bars.Volume.iloc[-21:-1].mean() > 0),
        "macd_long": bool(q.MACD >= q.MACD_signal),
        "stoch_cross_up": up,
        "stoch_not_overbought": bool(max(q.StochRSI_K, q.StochRSI_D) <= 80),
        "long_momentum_all": bool(long_momentum),
        "regime_long_only": permission == "LONG_ONLY",
        "setup_long": all(v >= 0 for v in setup),
        "risk_veto_inactive": not features["risk_veto"],
        "fundamental_veto_observed": False,  # NOT available historically
        "long_core_complete": bool(long_core),
        "short_core_complete": bool(short_core),
        "final_trigger_any_direction": bool(features["activation_trigger_met"] if full_trigger is None else full_trigger),
        "stoch_cross_down": down,
        "previous_low_break": bool(q.Close <= p.Low),
        "macd_short": bool(q.MACD <= q.MACD_signal),
        "setup_short": all(v <= 0 for v in setup),
        "regime_short_only": permission == "SHORT_ONLY",
        "plan_side": plan_side,
        "macro_permission": permission,
        "inputs": {"previous_k": float(p.StochRSI_K), "previous_d": float(p.StochRSI_D),
                   "k": float(q.StochRSI_K), "d": float(q.StochRSI_D),
                   "close": float(q.Close), "previous_high": float(p.High),
                   "previous_low": float(p.Low), "volume": float(q.Volume),
                   "previous20_volume_mean": float(bars.Volume.iloc[-21:-1].mean()),
                   "macd": float(q.MACD), "macd_signal": float(q.MACD_signal),
                   "setup_votes": setup},
    }


GATE_NAMES = (
    "inside_buy_zone_descriptive", "bullish_closed_candle_descriptive",
    "previous_high_break", "volume_above_previous20", "macd_long",
    "stoch_cross_up", "stoch_not_overbought", "long_momentum_all",
    "regime_long_only", "setup_long", "risk_veto_inactive",
    "long_core_complete", "final_trigger_any_direction",
)


def gate_counts(records):
    gates = [r["gates"] for r in records]
    return {name: {"fail": sum(not g[name] for g in gates),
                   "pass": sum(g[name] for g in gates)} for name in GATE_NAMES}


def joint_counts(records):
    names = ("volume_above_previous20", "stoch_cross_up", "stoch_not_overbought",
             "macd_long", "previous_high_break")
    gates = [r["gates"] for r in records]
    return {
        "all_momentum_pass": sum(all(g[k] for k in names) for g in gates),
        "sole_missing_momentum_gate": {k: sum(not g[k] and all(g[n] for n in names if n != k)
                                             for g in gates) for k in names},
        "pair_pass_counts": {f"{a} AND {b}": sum(g[a] and g[b] for g in gates)
                             for i, a in enumerate(names) for b in names[i+1:]},
        "long_momentum_when_macro_allows": sum(g["long_momentum_all"] and g["regime_long_only"] for g in gates),
    }


def audit_existing(output=OUTPUT):
    results = {}
    for symbol in ("SMCI", "NVDA"):
        path = next(BASELINE.glob(f"{symbol}_*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_historical_replay(payload)
        dataset = load_dataset(INPUT, symbol)
        rows = []
        for index, item in enumerate(payload["observations"], 1):
            cut = pd.Timestamp(item["observed_at"])
            bars = indicator_prefix(dataset, cut)
            tail = item["feature_snapshot"]["indicator_tail"]["intraday_indicators"]
            if pd.Timestamp(tail["at"]) != bars.index[-1]:
                raise ValueError("La reconstrucción cambió la vela fuente.")
            for key in ("Close", "MACD", "MACD_signal", "StochRSI_K", "StochRSI_D", "Volume", "Volume_MA20"):
                if abs(float(tail["values"][key]) - float(bars.iloc[-1][key])) > 1e-9:
                    raise ValueError(f"Paridad de snapshot falló: {symbol} {cut} {key}")
            contract = item["horizons"][0]["operational_contract"]
            rows.append({"observed_at": item["observed_at"],
                         "gates": gate_diagnostics(item["feature_snapshot"]["features"], bars,
                                                   plan_side=(contract or {}).get("side"))})
            if index % 50 == 0:
                print(f"Auditoría {symbol}: {index}/{len(payload['observations'])}", flush=True)
        results[symbol] = {"cuts": len(rows), "counts": gate_counts(rows), "rows": rows,
                           "baseline_file_sha256": digest_file(path),
                           "baseline_content_sha256": payload["content_sha256"],
                           "macro_permissions": dict(Counter(r["gates"]["macro_permission"] for r in rows)),
                           "code_sha256": payload["observations"][0]["feature_snapshot"]["code_sha256"]}
    audit = {"version": AUDIT_VERSION, "scope": "SAVED_1100_CUTS_NO_NEW_REPLAY", "symbols": results,
             "created_at": datetime.now(timezone.utc).isoformat(),
             "fundamental_news_gate": "UNOBSERVED_NO_POINT_IN_TIME_DATA"}
    write_signed(output / "baseline_gate_audit.json", audit)
    for symbol, data in results.items():
        print(symbol, json.dumps(data["counts"], ensure_ascii=False), flush=True)
    return audit


def preregister(output=OUTPUT):
    """Must follow completed baseline audit and precede every new cut."""
    audit = read_signed(output / "baseline_gate_audit.json")
    cohorts = [{"id": f"intraday_{t.replace(':', '')}", "time_ny": t,
                "frequency": "every_session", "horizons": [60, 1440]} for t in TIMES]
    # One daily observation at 15:30 and one weekly at the LAST XNYS session
    # of each completed exchange week, 15:30. Early-close sessions are skipped,
    # never shifted to a more favorable time. Daily overlaps intraday15:30/1D:
    # it is a view of the SAME event, never a second independent observation.
    cohorts += [{"id": "daily_1D", "time_ny": "15:30", "frequency": "every_session", "horizons": [1440]},
                {"id": "weekly_1S", "time_ny": "15:30", "frequency": "last_session_of_completed_week", "horizons": [10080]}]
    protocol = {
        "version": AUDIT_VERSION, "registered_at": datetime.now(timezone.utc).isoformat(),
        "start": "2025-07-21", "end": "2026-09-25", "symbols": ["SMCI", "NVDA"],
        "baseline_audit_sha256": sha(audit), "cohorts": cohorts,
        "parameters": DEFAULT_PARAMETERS,
        "input_manifest_sha256": sha(read_signed(INPUT / "manifest.json")),
        "dataset_sha256": {s: load_dataset(INPUT, s).fingerprint(s) for s in ("SMCI", "NVDA")},
        "engine_code_sha256": code_fingerprints(),
        "ledger_before_sha256": digest_file(ROOT / "data/portfolio.db"),
        "research_code_sha256": {"scripts/audit_trigger_replay.py": digest_file(Path(__file__)),
                                 "scripts/research_clock_adapter.py": digest_file(ROOT / "scripts/research_clock_adapter.py")},
        "live_anchor_unchanged": ANCHOR_VERSION,
        "research_clock": "EXACT_PREREGISTERED_CLOSED_BAR_NEXT_FULL_5M_V1",
        "actionability_scope": "RESEARCH_TECHNICAL_ONLY_NOT_LIVE_OR_BROKER_AUTHORIZATION",
        "deduplication": "shared_symbol_cut_side_entry_stop_target_is_one_event_across_horizons_and_views",
        "overlap_policy": "chronological_full_timeout_embargo_no_outcome_selection",
        "training": False, "parameter_tuning": False,
        "missing_early_close_cut_policy": "SKIP_AND_REPORT_NOT_RESCHEDULE",
        "historical_fundamental_veto": "UNOBSERVED_NOT_ASSUMED_SAFE",
    }
    # Exact source archive, verified against the preregistration. No trading
    # code edits are needed to run this experiment.
    for name, expected in {**protocol["engine_code_sha256"], **protocol["research_code_sha256"]}.items():
        source = (ROOT / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("Fuente cambió durante preregistro.")
        destination = output / "source_archive" / expected / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source:
            raise ValueError("Archivo fuente archivado inconsistente.")
        destination.write_bytes(source)
    write_signed(output / "preregistration.json", protocol)
    print("PREREGISTRADO", sha(protocol), json.dumps(cohorts, ensure_ascii=False), flush=True)
    return protocol


def planned_cuts(dataset, protocol):
    schedule = _calendar(2024, 2027).schedule
    selected = schedule.loc[protocol["start"]:protocol["end"]]
    week_last = {}
    for day in schedule.index:
        iso = day.isocalendar()
        week_last[(iso.year, iso.week)] = day
    available = set(dataset.intraday.index)
    tasks, skipped = [], []
    for day, session in selected.iterrows():
        iso = day.isocalendar()
        weekly = week_last[(iso.year, iso.week)] == day
        for clock in TIMES:
            cut = pd.Timestamp(f"{day.date()} {clock}", tz=NY).tz_convert("UTC")
            cohorts = [f"intraday_{clock.replace(':', '')}"]
            horizons = [60, 1440]
            if clock == "15:30":
                cohorts.append("daily_1D")
                if weekly:
                    cohorts.append("weekly_1S")
                    horizons.append(10080)
            row = {"cut_at": cut.isoformat(), "time_ny": clock,
                   "cohorts": cohorts, "horizons": horizons}
            if cut >= session.close or cut - BAR not in available:
                skipped.append({**row, "reason": "MARKET_CLOSED_OR_SOURCE_BAR_MISSING"})
            else:
                tasks.append(row)
    return tasks, skipped


def validate_frozen_environment(protocol):
    if code_fingerprints() != protocol["engine_code_sha256"]:
        raise ValueError("El motor cambió tras el preregistro; se prohíbe mezclar versiones.")
    for name, expected in protocol["research_code_sha256"].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError("El protocolo de investigación cambió tras el preregistro.")
    if sha(read_signed(INPUT / "manifest.json")) != protocol["input_manifest_sha256"]:
        raise ValueError("El histórico cambió tras el preregistro.")


_WORKER = {}


def worker_initialize(symbol, protocol):
    global _WORKER
    from portfolio_tracker.analytics.cross_correlation import PEERS
    _WORKER = {"symbol": symbol, "protocol": protocol,
               "dataset": load_dataset(INPUT, symbol),
               "peer": load_dataset(INPUT, PEERS[symbol])}
    validate_frozen_environment(protocol)


def worker_cut(task):
    """Every feature precedes the cut; only labelers see subsequent bars."""
    from portfolio_tracker.analytics.operational_target import make_operational_contract
    from scripts.research_clock_adapter import research_contract, assess_research_execution
    symbol, protocol = _WORKER["symbol"], _WORKER["protocol"]
    dataset, peer = _WORKER["dataset"], _WORKER["peer"]
    cut = pd.Timestamp(task["cut_at"])
    analysis = evaluate_replay_cut(dataset, cut, symbol=symbol,
                                  atr_stop_multiple=protocol["parameters"]["stop_atr_multiple"])
    analysis = _cross_asset(analysis, symbol, dataset, peer, cut)
    snapshot = build_replay_snapshot(analysis, observed_at=cut.to_pydatetime(),
                                     protocol=AUDIT_VERSION)
    eligible = bool(analysis.activation_trigger_met and not analysis.execution_plan_conditional
                    and not analysis.risk_veto and not analysis.signal_rejected)
    rows = []
    horizons = {p.label: p for p in analysis.horizon_projections}
    from portfolio_tracker.services.directional_collection import HORIZON_MINUTES
    for minutes in task["horizons"]:
        label = next(k for k, v in HORIZON_MINUTES.items() if v == minutes)
        try:
            origin = make_operational_contract(symbol, horizons[label], minutes, analysis,
                                               cut, {**protocol["parameters"],
                                               "observation_protocol": AUDIT_VERSION})
        except ValueError as exc:
            if str(exc) != "TP/entrada/SL no encierran correctamente el precio de referencia.":
                raise
            rows.append({"horizon_minutes": minutes, "operational_contract": None,
                         "operational_result": {"status": "NOT_APPLICABLE_NO_PLAN", "outcome": None},
                         "execution_result": {"status": "NO_VALID_FIRST_PASSAGE_PLAN", "outcome": None}})
            continue
        contract = research_contract(origin, eligible=eligible, protocol=protocol, cut=cut)
        rows.append({"horizon_minutes": minutes, "operational_contract": contract,
                     "operational_result": _operational_label(origin, dataset),
                     "execution_result": assess_research_execution(contract, dataset, protocol)})
    gates = gate_diagnostics(snapshot["features"], analysis.intraday_indicators,
                             plan_side=analysis.execution_levels.direction)
    gates.update({"signal_not_rejected": not analysis.signal_rejected,
                  "plan_not_conditional": not analysis.execution_plan_conditional,
                  "score_threshold65_met": analysis.operation_probability >= 65,
                  "technical_eligible": eligible})
    return {"version": AUDIT_VERSION, "symbol": symbol, **task,
            "protocol_sha256": protocol["protocol_sha256"],
            "dataset_sha256": dataset.fingerprint(symbol),
            "feature_snapshot": snapshot, "gates": gates, "horizons": rows,
            "computed_at": datetime.now(timezone.utc).isoformat()}


def baseline_record(symbol, task, old, audit, protocol):
    """Reuse immutable already-computed 11:00 cuts, not retrospective reissuance."""
    return {"version": AUDIT_VERSION, "symbol": symbol, **task,
            "protocol_sha256": protocol["protocol_sha256"],
            "dataset_sha256": protocol["dataset_sha256"][symbol],
            "feature_snapshot": old["feature_snapshot"], "gates": audit["gates"],
            "horizons": [{key: row[key] for key in ("horizon_minutes", "operational_contract",
                                                    "operational_result", "execution_result")}
                         for row in old["horizons"]],
            "provenance": "EXACT_FROZEN_BASELINE_REUSED_NO_NEW_SIGNAL"}


def run_research(symbol, output=OUTPUT, workers=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    protocol = read_signed(output / "preregistration.json")
    validate_frozen_environment(protocol)
    protocol = {**protocol, "protocol_sha256": sha(protocol)}
    dataset = load_dataset(INPUT, symbol)
    if dataset.fingerprint(symbol) != protocol["dataset_sha256"][symbol]:
        raise ValueError("Firma de dataset discrepante.")
    tasks, skipped = planned_cuts(dataset, protocol)
    folder = output / "cuts" / symbol
    baseline = json.loads(next(BASELINE.glob(f"{symbol}_*.json")).read_text(encoding="utf-8"))
    validate_historical_replay(baseline)
    old = {r["observed_at"]: r for r in baseline["observations"]}
    audit = {r["observed_at"]: r for r in read_signed(output / "baseline_gate_audit.json")["symbols"][symbol]["rows"]}
    pending = []
    def destination(task):
        return folder / (pd.Timestamp(task["cut_at"]).strftime("%Y%m%d_%H%M") + ".json")
    for task in tasks:
        path = destination(task)
        if path.exists():
            row = read_signed(path)
            if row["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError("Checkpoint pertenece a otro protocolo.")
        elif task["time_ny"] == "11:00":
            write_signed(path, baseline_record(symbol, task, old[task["cut_at"]],
                                               audit[task["cut_at"]], protocol))
        else:
            pending.append(task)
    print(f"{symbol}: {len(tasks)} cortes; {len(skipped)} ausentes por cierre; {len(pending)} por ejecutar", flush=True)
    completed = len(tasks) - len(pending)
    errors = []
    # Checkpoints are persisted by the parent after each completed task, so a
    # power-off never loses already measured cuts or reselects good outcomes.
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_initialize,
                             initargs=(symbol, protocol)) as executor:
        futures = {executor.submit(worker_cut, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                write_signed(destination(task), future.result())
            except Exception as exc:
                errors.append({**task, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {symbol} {task['cut_at']}: {exc}", flush=True)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"{symbol}: {completed}/{len(tasks)}; errores={len(errors)}", flush=True)
    write_signed(output / f"{symbol}_completion.json", {
        "protocol_sha256": protocol["protocol_sha256"], "symbol": symbol,
        "candidate_cuts": len(tasks), "skipped": skipped, "errors": errors,
        "records": [{"file": str(destination(t).relative_to(output)),
                     "file_sha256": digest_file(destination(t))}
                    for t in tasks if destination(t).exists()]})
    if errors:
        raise RuntimeError("Corrida incompleta; errores explícitos guardados, no entrenar.")


def count_cohort(records, cohort, horizon):
    selected = [r for r in records if cohort in r["cohorts"]]
    rows = [h for r in selected for h in r["horizons"] if h["horizon_minutes"] == horizon]
    eligible = [h for h in rows if (h.get("operational_contract") or {}).get("eligible_at_emission")]
    outcomes = Counter(h["execution_result"].get("outcome") for h in eligible)
    statuses = Counter(h["execution_result"]["status"] for h in eligible)
    hypotheses = Counter(h["operational_result"].get("outcome") for h in rows)
    from scripts.report_historical_pilot import nonoverlapping
    return {"cuts": len(rows), "long": sum(h["operational_contract"]["side"] == "LONG" for h in eligible),
            "short": sum(h["operational_contract"]["side"] == "SHORT" for h in eligible),
            "possible_fills": sum(h["execution_result"].get("fill_price") is not None for h in eligible),
            **{k: outcomes[k] for k in ("TP_FIRST", "SL_FIRST", "TIMEOUT")},
            "ambiguous": statuses["AMBIGUOUS_NO_TRADE"],
            "pending": statuses["PENDING_ENTRY"] + statuses["FILLED_PENDING"],
            "missing_evidence": statuses["EVIDENCE_INCOMPLETE"],
            "no_fill": sum(n for k, n in statuses.items() if k.startswith("NO_FILL")),
            "no_plan": sum(h["operational_contract"] is None for h in rows),
            "nonoverlapping_eligible": len(nonoverlapping(eligible)),
            "hypothetical": {k: hypotheses[k] for k in ("TP_FIRST", "SL_FIRST", "TIMEOUT")},
            "hypothetical_pending": sum(h["operational_result"]["status"] == "RIGHT_CENSORED" for h in rows),
            "gate_counts": gate_counts(selected), "joint_counts": joint_counts(selected)}


def report(output=OUTPUT):
    protocol = read_signed(output / "preregistration.json")
    validate_frozen_environment(protocol)
    audit = read_signed(output / "baseline_gate_audit.json")
    symbols = {}
    for symbol in protocol["symbols"]:
        completion = read_signed(output / f"{symbol}_completion.json")
        if completion["errors"] or len(completion["records"]) != completion["candidate_cuts"]:
            raise ValueError("No se permite un reporte completo con cortes perdidos.")
        records = []
        for item in completion["records"]:
            path = output / item["file"]
            if digest_file(path) != item["file_sha256"]:
                raise ValueError("Checkpoint alterado después de completar.")
            records.append(read_signed(path))
        symbols[symbol] = {"cohorts": {c["id"]: {str(h): count_cohort(records, c["id"], h)
                          for h in c["horizons"]} for c in protocol["cohorts"]},
                           "skipped": completion["skipped"], "unique_cuts": len(records),
                           "permissions": dict(Counter(r["gates"]["macro_permission"] for r in records)),
                           "baseline_joint_counts": joint_counts(audit["symbols"][symbol]["rows"])}
    payload = {"version": AUDIT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": sha(protocol), "symbols": symbols,
               "baseline_audit_sha256": sha(audit),
               "ledger_before_sha256": protocol["ledger_before_sha256"],
               "ledger_after_sha256": digest_file(ROOT / "data/portfolio.db")}
    if payload["ledger_before_sha256"] != payload["ledger_after_sha256"]:
        raise ValueError("La base cambió durante la auditoría; no afirmar inmutabilidad.")
    write_signed(output / "report.json", payload)
    lines = ["# Auditoría del gatillo y cortes fijos", "", f"Fecha UTC: {payload['created_at']}",
             "", "Sin entrenamiento, ajuste de parámetros, Brier ni estimación de precisión.",
             "", "## Fallos independientes en 299 cortes de las 11:00 por activo", "",
             "| Condición LONG | SMCI fallos | NVDA fallos | % SMCI / NVDA |",
             "|---|---:|---:|---|"
             ]
    for name in GATE_NAMES:
        a, b = (audit["symbols"][s]["counts"][name]["fail"] for s in ("SMCI", "NVDA"))
        lines.append(f"| {name} | {a} | {b} | {100*a/299:.1f}% / {100*b/299:.1f}% |")
    lines += ["", "Los porcentajes no se suman: un corte puede fallar múltiples condiciones.",
              "Zona y vela alcista son descriptivos; no gates finales. Rebote no tiene una puerta final propia.",
              "El régimen exige LONG_ONLY/SHORT_ONLY: BOTH_REDUCED no autoriza entradas.",
              "Veto fundamental/noticioso histórico: NO OBSERVADO; cero vetos técnicos no demuestra ausencia de riesgo.",
              "", "## Conteo por horario y horizonte", "",
              "Autorización = autorización técnica del experimento preregistrado. NO es permiso live, préstamo SHORT ni fill real.",
              "El contrato vivo sigue permitiendo únicamente 11:00. Sus reglas e indicadores no se modificaron.",
              "1D vence al cierre de la siguiente sesión XNYS, no al cierre del mismo día. 1S = cinco sesiones siguientes.",
              "", "| Activo | Cohorte NY | Horizonte | Cortes | LONG | SHORT | Fills posibles | TP | SL | Timeout | Ambiguos | Pendientes | Sin fill |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for symbol, data in symbols.items():
        for cohort, horizons in data["cohorts"].items():
            for horizon, row in horizons.items():
                lines.append(f"| {symbol} | {cohort} | {horizon} min | " + " | ".join(str(row[k]) for k in
                    ("cuts", "long", "short", "possible_fills", "TP_FIRST", "SL_FIRST", "TIMEOUT", "ambiguous", "pending", "no_fill")) + " |")
    lines += ["", "## Barreras hipotéticas (población separada, NO entradas)", "",
              "| Activo | Cohorte | Horizonte | TP primero | SL primero | Timeout | Pendientes |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for symbol, data in symbols.items():
        for cohort, horizons in data["cohorts"].items():
            for horizon, row in horizons.items():
                hypo = row["hypothetical"]
                lines.append(f"| {symbol} | {cohort} | {horizon} | {hypo['TP_FIRST']} | {hypo['SL_FIRST']} | {hypo['TIMEOUT']} | {row['hypothetical_pending']} |")
    lines += ["", "## Interpretación y límites", "",
              "- El corte diario quedó fijado a 15:30; el semanal a 15:30 de la última sesión XNYS de cada semana completa.",
              "- Cierre anticipado: se omiten las horas fuera de sesión, sin mover el corte. Detalle en report.json.",
              "- daily_1D y intraday_1530/1D son el mismo evento, NO dos observaciones. Los horizontes de una misma entrada comparten evento.",
              "- Los conteos brutos por horario NO se suman como evidencia independiente; se informa embargo por timeout en JSON.",
              "- Falta permiso de cuenta, posición histórica, noticias punto-en-tiempo y confirmación de broker. SHORT técnico no significa ejecutable.",
              "- La comparación no selecciona horarios por ganancias: todos quedaron firmados antes del replay.",
              "- Para evaluar si un filtro es excesivo: preregistrar población de candidatos, medir cobertura por régimen/activo y por condición,",
              "  registrar spread/fills y comparar cambios aislados en desarrollo con embargo; nunca usar estos resultados para retocar y declarar validación.",
              "- Ningún resultado permite declarar precisión. No se entrenó ni se escribió en bases live o contables.",
              "", f"Preregistro SHA-256: `{sha(protocol)}`", f"Reporte SHA-256: `{sha(payload)}`",
              f"portfolio.db SHA-256 (lectura solamente): `{payload['ledger_after_sha256']}`"]
    lines += ["", "## Fallos de gatillo por horario (marginales)", "",
              "| Activo | Hora | Volumen | Cruce Stoch arriba | Ruptura máximo previo | MACD LONG | LONG_ONLY | Impulso LONG completo |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for symbol, data in symbols.items():
        for clock in TIMES:
            counts = data["cohorts"][f"intraday_{clock.replace(':', '')}"]["60"]["gate_counts"]
            lines.append(f"| {symbol} | {clock} | " + " | ".join(str(counts[k]["fail"]) for k in
                ("volume_above_previous20", "stoch_cross_up", "previous_high_break", "macd_long", "regime_long_only", "long_momentum_all")) + " |")
    target = output / "report.md"
    if target.exists():
        raise ValueError("Reporte Markdown ya existente; no sobrescribir auditorías.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({s: v["cohorts"] for s, v in symbols.items()}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit-existing", "preregister", "run", "report"])
    parser.add_argument("--symbol", choices=["SMCI", "NVDA"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.action == "audit-existing":
        audit_existing(args.output)
    elif args.action == "preregister":
        preregister(args.output)
    elif args.action == "run":
        if not args.symbol:
            parser.error("run requiere --symbol")
        if not 1 <= args.workers <= 12:
            parser.error("--workers fuera de rango 1..12")
        run_research(args.symbol, args.output, workers=args.workers)
    else:
        report(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
