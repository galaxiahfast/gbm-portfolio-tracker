"""Run isolated one-gate replay arms; development first, sealed holdout LAST."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT as AUDIT, INPUT, TIMES, gate_diagnostics, digest_file
from scripts.build_historical_replay import load_dataset
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.measure_trigger_selectivity import verify_tail
from scripts.research_clock_adapter import research_contract, signed_hash, assess_research_execution
from scripts.seal_trigger_experiment_holdout import OUTPUT, preregister, decrypt_after_opening
from scripts.trigger_experiment_rules import VARIANTS, clock_reference, evaluate_variants, nonoverlapping_events, statistics
from portfolio_tracker.analytics.causal_core import INTRADAY_READY
from portfolio_tracker.analytics.historical_replay import _cross_asset
from portfolio_tracker.analytics.replay import ReplayDataset, evaluate_replay_cut
from portfolio_tracker.analytics.technical_probability import add_intraday_indicators
from portfolio_tracker.analytics.technical_validity import current_indicator_suffix
from portfolio_tracker.analytics.closed_bars import _calendar
from portfolio_tracker.analytics.operational_target import make_operational_contract
from portfolio_tracker.services.model_execution_record import build_replay_snapshot
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES

_STATE = {}


def validate_environment(protocol):
    for name, value in {**protocol["engine_code_sha256"], **protocol["experiment_source_sha256"]}.items():
        if digest_file(ROOT / name) != value:
            raise ValueError(f"Frozen source changed: {name}")
    if digest_file(ROOT / "data/portfolio.db") != protocol["ledger_sha256"]:
        raise ValueError("portfolio.db changed; stop without opening it.")


def unclock(contract):
    origin = deepcopy(contract)
    if "research_clock" in origin:
        clock = origin.pop("research_clock")
        origin["eligible_at_emission"] = clock["live_eligible_at_emission"]
        origin["contract_sha256"] = signed_hash(origin)
        if origin["contract_sha256"] != clock["production_contract_sha256"]:
            raise ValueError("Original contract altered.")
    return origin


def worker_initialize(symbol, protocol, stage, holdout_data=None):
    global _STATE
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if stage == "development":
        data = {s: load_dataset(INPUT, s) for s in protocol["symbols"]}
    else:
        data = {}
        for s in protocol["symbols"]:
            obj = holdout_data[s]
            data[s] = ReplayDataset(pd.read_json(StringIO(obj["5m"]), orient="table"),
                                    pd.read_json(StringIO(obj["daily"]), orient="table"), obj["as_of"])
    other = next(s for s in protocol["symbols"] if s != symbol)
    _STATE = {"symbol":symbol, "protocol":{**protocol, "protocol_sha256":sha(protocol)}, "stage":stage,
              "dataset":data[symbol], "peer":data[other], "clock_p50":clock_reference(data[symbol].intraday)}
    validate_environment(protocol)


def native_cut(task):
    state = _STATE
    p, dataset, symbol = state["protocol"], state["dataset"], state["symbol"]
    cut = pd.Timestamp(task["cut_at"])
    analysis = evaluate_replay_cut(dataset, cut, symbol=symbol, atr_stop_multiple=p["parameters"]["stop_atr_multiple"])
    analysis = _cross_asset(analysis, symbol, dataset, state["peer"], cut)
    snapshot = build_replay_snapshot(analysis, observed_at=cut.to_pydatetime(), protocol=p["version"])
    eligible = bool(analysis.activation_trigger_met and not analysis.execution_plan_conditional and not analysis.risk_veto and not analysis.signal_rejected)
    horizons = []
    projected = {x.label:x for x in analysis.horizon_projections}
    for minutes in (60, 1440):
        label = next(k for k, v in HORIZON_MINUTES.items() if v == minutes)
        try:
            origin = make_operational_contract(symbol, projected[label], minutes, analysis, cut, p["parameters"])
        except ValueError as exc:
            if str(exc) != "TP/entrada/SL no encierran correctamente el precio de referencia.":
                raise
            horizons.append({"horizon_minutes":minutes, "operational_contract":None})
            continue
        horizons.append({"horizon_minutes":minutes, "operational_contract":research_contract(origin, eligible=eligible, protocol=p, cut=cut)})
    gates = gate_diagnostics(snapshot["features"], analysis.intraday_indicators, plan_side=analysis.execution_levels.direction)
    gates["signal_not_rejected"] = not analysis.signal_rejected
    return {"symbol":symbol, **task, "feature_snapshot":snapshot, "gates":gates, "horizons":horizons}, analysis.intraday_indicators


def apply_to_cut(record, bars):
    s = _STATE
    protocol, dataset = s["protocol"], s["dataset"]
    source = pd.Timestamp(record["cut_at"]) - pd.Timedelta(minutes=5)
    median = float(s["clock_p50"].loc[source])
    gates = evaluate_variants(record, bars, median)
    results = {}
    for variant, gate in gates.items():
        horizons = []
        if gate["eligible"]:
            for h in record["horizons"]:
                if h["horizon_minutes"] not in (60, 1440):
                    continue
                origin = unclock(h["operational_contract"])
                contract = research_contract(origin, eligible=True, protocol=protocol, cut=record["cut_at"])
                execution = assess_research_execution(contract, dataset, protocol)
                if variant == "CURRENT" and s["stage"] == "development" and execution != h["execution_result"]:
                    raise ValueError("Current control fill/outcome differs from frozen replay.")
                horizons.append({"horizon_minutes":h["horizon_minutes"], "contract":contract, "execution":execution})
        results[variant] = {**gate, "horizons":horizons}
    return {"symbol":s["symbol"], "cut_at":record["cut_at"], "time_ny":record["time_ny"],
            "source_bar_at":source.isoformat(), "plan_same_across_all_variants":True, "variants":results,
            "original_features_sha256":sha(record["feature_snapshot"]),
            "native_holdout_snapshot":record if s["stage"] == "holdout" else None}


def worker_day(tasks):
    s, measured = _STATE, []
    if s["stage"] == "development":
        records = [read_signed(AUDIT / task["source_file"]) for task in tasks]
        last_cut = max(pd.Timestamp(r["cut_at"]) for r in records)
        start = (last_cut.tz_convert("America/New_York") - pd.DateOffset(months=1)).normalize()
        source = s["dataset"].intraday
        raw = source.loc[(source.index >= start) & (source.index + pd.Timedelta(minutes=5) <= last_cut)]
        indicators = add_intraday_indicators(raw)
        for record in records:
            cut = pd.Timestamp(record["cut_at"])
            bars = current_indicator_suffix(indicators.loc[indicators.index + pd.Timedelta(minutes=5) <= cut], [*INTRADAY_READY,"RSI14"],"5m")
            verify_tail(record, bars)
            measured.append(apply_to_cut(record, bars))
    else:
        for task in tasks:
            record, bars = native_cut(task)
            measured.append(apply_to_cut(record, bars))
    return measured


def holdout_tasks(protocol):
    schedule = _calendar(2023,2025).schedule.loc[protocol["holdout"]["start"]:protocol["holdout"]["end"]]
    tasks, skipped = [], []
    for day, session in schedule.iterrows():
        for clock in TIMES:
            cut = pd.Timestamp(f"{day.date()} {clock}", tz="America/New_York").tz_convert("UTC")
            task = {"cut_at":cut.isoformat(), "time_ny":clock}
            if cut >= session.close:
                skipped.append(task)
            else:
                tasks.append(task)
    return tasks, skipped


def summarize(measured):
    result = {}
    for variant in VARIANTS:
        rows = [{"cut_at":r["cut_at"], "side":r["variants"][variant]["side"], "exposure":r["variants"][variant]["exposure"],
                 "horizons":r["variants"][variant]["horizons"]} for r in measured if r["variants"][variant]["eligible"]]
        independent = nonoverlapping_events(rows)
        result[variant] = {"eligible_distinct_cuts_raw":len(rows), "independent_events":len(independent),
                           "embargo_excluded":len(rows)-len(independent),
                           "raw":{str(h):statistics(rows,h) for h in (60,1440)},
                           "independent":{str(h):{**statistics(independent,h),
                               "LONG_only":statistics(independent,h,"LONG"), "SHORT_only_NOT_GBM":statistics(independent,h,"SHORT")} for h in (60,1440)},
                           "selected_cuts":[r["cut_at"] for r in independent]}
    return result


def run_stage(stage, workers=4):
    protocol = preregister()
    validate_environment(protocol)
    seal = read_signed(OUTPUT / "holdout_seal.json")
    if seal["protocol_sha256"] != sha(protocol):
        raise ValueError("Seal/protocol mismatch.")
    holdout = None
    if stage == "holdout":
        development = read_signed(OUTPUT / "development_complete.json")
        if development["protocol_sha256"] != sha(protocol) or development["errors"]:
            raise ValueError("Development not closed under exact same protocol.")
        if not (OUTPUT / "tests_passed.json").exists():
            raise ValueError("Missing tests checkpoint before holdout opening.")
        from portfolio_tracker.analytics.holdout_registry import reserve_holdout_opening
        tasks, _ = holdout_tasks(protocol)
        receipt_path = OUTPUT / "holdout_opening.json"
        if receipt_path.exists():
            receipt = read_signed(receipt_path)
            if receipt["protocol_sha256"] != sha(protocol) or receipt["development_sha256"] != sha(development):
                raise ValueError("Cannot reuse holdout for different experiment.")
            if (OUTPUT / "holdout_complete.json").exists():
                raise ValueError("Holdout already evaluated; read closed report, cannot reopen.")
        else:
            openings = {}
            for symbol in protocol["symbols"]:
                openings[symbol] = reserve_holdout_opening(registry_path=OUTPUT.parent / "trigger_experiment_holdout_openings.sqlite",
                    dataset_sha256=seal["audits"][symbol]["sealed_data_sha256"],symbol=symbol,horizon="TRIGGER_EXPERIMENT_1H_1D_V1",
                    protocol_sha256=sha(protocol), commitment_sha256=sha(seal),
                    members=tuple((f"{symbol}:{t['cut_at']}",t["cut_at"]) for t in tasks))
            receipt = {"protocol_sha256":sha(protocol), "development_sha256":sha(development), "openings":openings,
                       "opened_at":datetime.now(timezone.utc).isoformat(), "stage":"ALL_VARIANTS_ONCE_NO_INTERMEDIATE_TUNING"}
            write_signed(receipt_path,receipt)
        holdout = decrypt_after_opening(protocol,receipt["openings"])
        print("Holdout opened after development closure; protocol unchanged.",flush=True)
    all_symbols, errors = {}, []
    for symbol in protocol["symbols"]:
        if stage == "development":
            complete = read_signed(AUDIT / f"{symbol}_completion.json")
            tasks = []
            for item in complete["records"]:
                if digest_file(AUDIT / item["file"]) != item["file_sha256"]:
                    raise ValueError("Frozen cut file altered.")
                r = read_signed(AUDIT / item["file"])
                tasks.append({"cut_at":r["cut_at"],"time_ny":r["time_ny"],"source_file":item["file"]})
            skipped = complete["skipped"]
        else:
            tasks, skipped = holdout_tasks(protocol)
        groups = defaultdict(list)
        for task in tasks:
            groups[pd.Timestamp(task["cut_at"]).tz_convert("America/New_York").date().isoformat()].append(task)
        folder = OUTPUT / stage / symbol
        pending = []
        for day, rows in sorted(groups.items()):
            dest = folder / f"{day}.json"
            if dest.exists():
                prior = read_signed(dest)
                if prior["protocol_sha256"] != sha(protocol):
                    raise ValueError("Day checkpoint from different experiment.")
            else:
                pending.append((day,rows))
        done = len(groups)-len(pending)
        print(stage,symbol,"days",len(groups),"pending",len(pending),flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_initialize, initargs=(symbol,protocol,stage,holdout)) as pool:
            futures = {pool.submit(worker_day, rows):day for day,rows in pending}
            for future in as_completed(futures):
                day = futures[future]
                try:
                    write_signed(folder / f"{day}.json",{"protocol_sha256":sha(protocol),"stage":stage,"symbol":symbol,"day":day,"cuts":future.result()})
                except Exception as exc:
                    errors.append({"stage":stage,"symbol":symbol,"day":day,"error":f"{type(exc).__name__}: {exc}"})
                    print("ERROR",errors[-1],flush=True)
                done += 1
                if done % 20 == 0 or done == len(groups):
                    print(stage,symbol,done,"/",len(groups),"errors",len(errors),flush=True)
        measured = []
        files = []
        for day in sorted(groups):
            path = folder / f"{day}.json"
            if path.exists():
                rows = read_signed(path)["cuts"]
                measured.extend(rows)
                files.append({"path":str(path.relative_to(OUTPUT)),"sha256":digest_file(path)})
        all_symbols[symbol] = {"processed_cuts":len(measured),"expected_cuts":len(tasks),"skipped":skipped,
                               "variants":summarize(measured),"checkpoint_files":files}
    payload = {"stage":stage,"protocol_sha256":sha(protocol),"seal_sha256":sha(seal),"errors":errors,"symbols":all_symbols,
               "completed_at":datetime.now(timezone.utc).isoformat(),"ledger_sha256":digest_file(ROOT / "data/portfolio.db")}
    validate_environment(protocol)
    if errors or any(v["processed_cuts"] != v["expected_cuts"] for v in all_symbols.values()):
        write_signed(OUTPUT / f"{stage}_failed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",payload)
        raise RuntimeError("Stage incomplete; holdout/report blocked.")
    if stage == "development":
        for symbol, target in (("SMCI",6),("NVDA",0)):
            if all_symbols[symbol]["variants"]["CURRENT"]["eligible_distinct_cuts_raw"] != target:
                raise ValueError("Current control does not reproduce prior audit.")
    write_signed(OUTPUT / f"{stage}_complete.json",payload)
    print("Stage CLOSED",stage,sha(payload),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["development","holdout"])
    parser.add_argument("--workers",type=int,default=4,choices=range(1,9))
    args=parser.parse_args()
    run_stage(args.stage,args.workers)
