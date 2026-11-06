"""Build versioned, non-training EDGAR event features and descriptive 1D audit.

Reads only signed research artifacts. SEC acceptance is not certified first
availability. No model fit, live recommendation, or portfolio database access.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import json
import math
from pathlib import Path
import sys

import exchange_calendars as xcals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.historical_replay import validate_historical_replay
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.pilot_sec_edgar_events import (
    OUT as SEC_PILOT, REPLAY_FILES, SYMBOLS,
    eligible_events, parse_utc, previous_session_close,
)


FEATURE_VERSION = "SEC_EDGAR_SPARSE_EVENTS_V1"
AUDIT_VERSION = "SEC_EDGAR_1D_DESCRIPTIVE_AUDIT_V1"
OUT = ROOT / "output/historical_replay/sec_edgar_event_sidecar_v1"
POLICIES = {
    "previous_session": "PREVIOUS_XNYS_CLOSE_V1",
    "five_minute_lag": "ACCEPTANCE_PLUS_5M_V1",
}


def source_files(symbol: str) -> tuple[Path, Path, Path]:
    return (SEC_PILOT / f"{symbol}_sec_submissions.json",
            SEC_PILOT / f"{symbol}_filings.json",
            SEC_PILOT / f"{symbol}_coverage.json")


def build_cut_feature(cut: str, events: list[dict], calendar, policy: str,
                      source_hashes: dict) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unknown EDGAR feature policy")
    cut_at = parse_utc(cut)
    prior_close = previous_session_close(cut_at, calendar)
    known_at = prior_close if policy == "previous_session" else cut_at - timedelta(minutes=5)
    in_24h = eligible_events(events, cut_at, 24, prior_close, policy)
    in_48h = eligible_events(events, cut_at, 48, prior_close, policy)
    latest = max((item for item in events if parse_utc(item["accepted_at_utc"]) <= known_at),
                 key=lambda item: item["accepted_at_utc"], default=None)
    last_at = parse_utc(latest["accepted_at_utc"]) if latest else None
    return {
        "feature_version": FEATURE_VERSION,
        "cut_at_utc": cut_at.isoformat(),
        "policy": POLICIES[policy],
        "known_at_utc": known_at.isoformat(),
        "event_count_24h": len(in_24h),
        "event_count_48h": len(in_48h),
        # Form flags refer to the 48h window, not all historical filings.
        "has_8k": any(item["form"] == "8-K" for item in in_48h),
        "has_10q": any(item["form"] == "10-Q" for item in in_48h),
        "has_10k": any(item["form"] == "10-K" for item in in_48h),
        "days_since_last_event": round((cut_at - last_at).total_seconds() / 86400, 6)
        if last_at else None,
        "last_event_accession": latest["accession"] if latest else None,
        "event_accessions_48h": [item["accession"] for item in in_48h],
        "source_hashes": source_hashes,
    }


def build_sidecar(symbol: str, policy: str, calendar) -> dict:
    if symbol not in SYMBOLS:
        raise ValueError("Unknown symbol")
    submissions_path, filings_path, coverage_path = source_files(symbol)
    submissions = read_signed(submissions_path)
    filings = read_signed(filings_path)
    coverage = read_signed(coverage_path)
    replay = json.loads(REPLAY_FILES[symbol].read_text(encoding="utf-8"))
    validate_historical_replay(replay)
    if (replay["symbol"] != symbol or len(replay["observations"]) != 299
            or filings["symbol"] != symbol or coverage["symbol"] != symbol):
        raise ValueError("Mismatched SEC/replay symbol or cut scope")
    if filings["submissions_sha256"] != sha(submissions["response"]):
        raise ValueError("SEC source history hash mismatch")
    if coverage["replay_artifact_sha256"] != replay["observations"][0]["dataset_sha256"]:
        raise ValueError("SEC pilot used a different market dataset")
    source_hashes = {
        "sec_submissions_payload_sha256": sha(submissions),
        "sec_filings_payload_sha256": sha(filings),
        "sec_coverage_payload_sha256": sha(coverage),
        "market_replay_artifact_sha256": replay["artifact_sha256"],
    }
    events = filings["events"]
    rows = [build_cut_feature(cut["observed_at"], events, calendar, policy, source_hashes)
            for cut in replay["observations"]]
    if len({row["cut_at_utc"] for row in rows}) != 299:
        raise ValueError("Duplicate or missing EDGAR cuts")
    # Fail closed if the recomputed event windows differ from the prior signed pilot.
    for row, pilot in zip(rows, coverage["coverage"]["cuts"]):
        if row["cut_at_utc"] != pilot["cut_at_utc"]:
            raise ValueError("EDGAR and pilot cut timestamp mismatch")
        for hours in (24, 48):
            previous = pilot["windows"][f"{policy}_{hours}h"]
            if (row[f"event_count_{hours}h"] != previous["event_count"]
                    or (hours == 48 and row["event_accessions_48h"] != previous["accessions"])):
                raise ValueError("Recomputed EDGAR coverage changed")
    payload = {"feature_version": FEATURE_VERSION, "symbol": symbol,
               "policy": POLICIES[policy], "horizon_independent": True,
               "source_hashes": source_hashes, "rows": rows,
               "trained": False, "production_changed": False,
               "portfolio_db_accessed": False,
               "pit_status": "ACCEPTANCE_PROXY_NOT_FIRST_PUBLIC_AVAILABILITY"}
    write_signed(OUT / f"{symbol}_{policy}_sidecar.json", payload)
    return payload


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    divisor = 1 + z*z/total
    center = (p + z*z/(2*total)) / divisor
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / divisor
    return [round(100*(center-radius), 2), round(100*(center+radius), 2)]


def group_stats(cases: list[dict]) -> dict:
    outcomes = Counter(case["outcome"] for case in cases if case["status"] == "RESOLVED")
    resolved = sum(outcomes.values())
    return {
        "cuts": len(cases), "resolved_hypothetical": resolved,
        "tp": outcomes["TP_FIRST"], "sl": outcomes["SL_FIRST"],
        "timeout": outcomes["TIMEOUT"],
        "unresolved_or_no_plan": len(cases) - resolved,
        "tp_rate_resolved_pct": round(100*outcomes["TP_FIRST"]/resolved, 2) if resolved else None,
        "tp_rate_wilson_95_pct": wilson(outcomes["TP_FIRST"], resolved),
        "authorized_entries": sum(case["authorized"] for case in cases),
        "possible_fills": sum(case["filled"] for case in cases),
    }


def compare(cases: list[dict], *, strata: tuple[str, ...]) -> list[dict]:
    keys = sorted({tuple(case[key] for key in strata) for case in cases})
    rows = []
    for key in keys:
        matched = [case for case in cases if tuple(case[name] for name in strata) == key]
        no_event = group_stats([case for case in matched if not case["event"]])
        event = group_stats([case for case in matched if case["event"]])
        base = no_event["tp_rate_resolved_pct"]
        for label, stats in (("event", event), ("no_event", no_event)):
            rows.append({**dict(zip(strata, key)), "group": label, **stats,
                         "delta_tp_vs_no_event_pp": round(stats["tp_rate_resolved_pct"]-base, 2)
                         if base is not None and stats["tp_rate_resolved_pct"] is not None else None})
    return rows


def audit_policy(sidecars: dict[str, dict], policy: str) -> dict:
    cases = []
    for symbol, sidecar in sidecars.items():
        replay = json.loads(REPLAY_FILES[symbol].read_text(encoding="utf-8"))
        validate_historical_replay(replay)
        feature_by_cut = {row["cut_at_utc"]: row for row in sidecar["rows"]}
        for cut in replay["observations"]:
            time = parse_utc(cut["observed_at"]).isoformat()
            feature = feature_by_cut[time]
            horizon = next(row for row in cut["horizons"] if row["horizon_minutes"] == 1440)
            contract = horizon.get("operational_contract") or {}
            hypothetical = horizon["operational_result"]
            execution = horizon["execution_result"]
            cases.append({
                "symbol": symbol, "cut_at_utc": time,
                "event": feature["event_count_48h"] > 0,
                "market_regime": cut["feature_snapshot"]["features"].get("market_regime") or "UNKNOWN",
                "side": contract.get("side") or "NO_PLAN",
                "status": hypothetical["status"],
                "outcome": hypothetical.get("outcome"),
                "authorized": bool(contract.get("eligible_at_emission")),
                "filled": execution.get("fill_price") is not None,
                "execution_status": execution["status"],
                "execution_outcome": execution.get("outcome"),
            })
    if len(cases) != 598 or len({(x["symbol"], x["cut_at_utc"]) for x in cases}) != 598:
        raise ValueError("Expected exactly 598 distinct asset-cuts")
    return {"audit_version": AUDIT_VERSION, "feature_version": FEATURE_VERSION,
            "policy": POLICIES[policy], "horizon_minutes": 1440,
            "label_population": "HYPOTHETICAL_BARRIER_NOT_EXECUTED",
            "group_definition": "At least one SEC filing in prior 48 wall-clock hours under policy",
            "by_symbol": compare(cases, strata=("symbol",)),
            "by_symbol_side": compare(cases, strata=("symbol", "side")),
            "by_symbol_regime_side": compare(cases, strata=("symbol", "market_regime", "side")),
            "execution_statuses": dict(Counter(case["execution_status"] for case in cases)),
            "trained": False, "portfolio_db_accessed": False,
            "interpretation_limit": "Event presence is not directional sentiment; small groups and serial dependence."}


def build() -> dict:
    calendar = xcals.get_calendar("XNYS")
    summary = {}
    for policy in POLICIES:
        sidecars = {symbol: build_sidecar(symbol, policy, calendar) for symbol in SYMBOLS}
        audit = audit_policy(sidecars, policy)
        write_signed(OUT / f"audit_{policy}.json", audit)
        summary[policy] = {"by_symbol": audit["by_symbol"],
                           "execution_statuses": audit["execution_statuses"]}
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
