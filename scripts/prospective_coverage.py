"""Read-only coverage of signed prospective cuts and signed outcome sidecars.

Counts raw cuts, resolved barriers, possible fills and distinct session groups
separately. Five overlapping cuts in one day are NOT five independent trades.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import exchange_calendars as xcals
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_massive_history import sha
from scripts.prospective_archive_common import (
    ARCHIVE, CUT_TIMES, DATA_SOURCE, GRACE, HORIZONS, NY, SYMBOLS,
    label_path, signed_payload, utc, write_payload,
)


REPORT_VERSION = "PROSPECTIVE_COVERAGE_V2"
MINIMUM = 300


def _future_session_date(start_date, n_sessions):
    if n_sessions <= 0:
        return start_date.isoformat()
    calendar = xcals.get_calendar("XNYS", start=f"{start_date.year-1}-01-01",
                                  end=f"{start_date.year+4}-12-31")
    sessions = calendar.sessions_in_range(start_date + timedelta(days=1),
                                          start_date + timedelta(days=4*366))
    return sessions[n_sessions - 1].date().isoformat() if len(sessions) >= n_sessions else None


def _estimate_date(count, observed_sessions, first_day, last_day, today):
    if count >= MINIMUM:
        return last_day
    if len(observed_sessions) < 5 or first_day is None or last_day is None:
        return None
    calendar = xcals.get_calendar("XNYS", start=f"{first_day.year-1}-01-01",
                                  end=f"{today.year+4}-12-31")
    elapsed = len(calendar.sessions_in_range(first_day, last_day))
    if elapsed <= 0:
        return None
    rate = count / elapsed
    if rate <= 0:
        return None
    projected = int((MINIMUM - count) / rate + 0.999999)
    return _future_session_date(today, projected)


def build_report(*, clock=None, save=True):
    now = utc(clock)
    snapshots_root = ARCHIVE / "snapshots"
    paths = sorted(snapshots_root.glob("*/*.json")) if snapshots_root.exists() else []
    groups = defaultdict(list)
    scheduled_seen = set()
    daily = defaultdict(lambda: {"snapshots": 0, "with_news": 0,
                                "resolved": 0, "pending": 0})
    for path in paths:
        item = signed_payload(path)
        if item.get("synthetic_example"):
            continue
        symbol, horizon = item.get("symbol"), item.get("horizon")
        if symbol not in SYMBOLS or horizon not in HORIZONS:
            raise ValueError("Identidad inesperada en archivo prospectivo.")
        if (item.get("training_performed") is not False
                or utc(datetime.fromisoformat(item["cut_at_utc"])) > now):
            raise ValueError("Snapshot prospectivo inválido o futuro.")
        # Legacy signed V1 files cannot be rewritten; their signed policy
        # identifies IEX. Refuse a mixed-source cohort rather than averaging it.
        source = item.get("data_source", DATA_SOURCE if item.get("policy", {}).get("market_feed") == "Alpaca IEX" else None)
        if source != DATA_SOURCE:
            raise ValueError("Fuentes mixtas o procedencia ausente en el archivo prospectivo.")
        related_label = label_path(path)
        label = signed_payload(related_label) if related_label.exists() else None
        if label is not None and (label.get("parent_snapshot_sha256") != sha(item)
                                   or label.get("snapshot_id") != item["snapshot_id"]
                                   or label.get("data_source", DATA_SOURCE) != DATA_SOURCE
                                   or label.get("outcome") not in {"TP_FIRST", "SL_FIRST", "TIMEOUT"}):
            raise ValueError("Etiqueta no coincide con su emisión PIT firmada.")
        has_news = item["sentiment_24h"]["status"] == "AVAILABLE"
        groups[(symbol, horizon)].append((item, label, has_news))
        scheduled_seen.add((item["scheduled_cut_at_utc"], symbol, horizon))
        key = (item["session_ny"], symbol, horizon)
        daily[key]["snapshots"] += 1
        daily[key]["with_news"] += int(has_news)
        daily[key]["resolved" if label else "pending"] += 1
    rows = []
    today = now.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date()
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            cases = groups[(symbol, horizon)]
            total = len(cases)
            news = sum(has_news for _, _, has_news in cases)
            possibly_truncated = sum(bool(item.get("news_provider_metadata", {}).get("possible_truncation"))
                                       for item, _, _ in cases)
            resolved = sum(label is not None for _, label, _ in cases)
            valid_plans = sum(item["operational_contract"] is not None for item, _, _ in cases)
            authorized = sum(item["production_actionable"] for item, _, _ in cases)
            possible_fills = sum(label is not None and label["fill_possible"] is True
                                 for _, label, _ in cases)
            session_days = sorted({item["session_ny"] for item, _, _ in cases})
            resolved_days = sorted({item["session_ny"] for item, label, _ in cases if label})
            first = datetime.fromisoformat(session_days[0]).date() if session_days else None
            last = datetime.fromisoformat(session_days[-1]).date() if session_days else None
            rows.append({
                "symbol": symbol, "horizon": horizon, "data_source": DATA_SOURCE,
                "snapshots": total,
                "with_news": news, "without_news": total - news,
                "news_pct": round(100*news/total, 2) if total else None,
                "possibly_truncated_news_snapshots": possibly_truncated,
                "resolved": resolved, "pending": total - resolved,
                "valid_plans": valid_plans,
                "no_valid_plan": total - valid_plans,
                "pending_with_valid_plan": valid_plans - resolved,
                "production_anchor_authorized": authorized,
                "possible_fills_ohlc_not_real": possible_fills,
                "distinct_session_groups": len(session_days),
                "resolved_session_groups": len(resolved_days),
                "missing_to_300_raw_snapshots": max(0, MINIMUM - total),
                "missing_to_300_resolved_snapshots": max(0, MINIMUM - resolved),
                "missing_to_300_possible_fills": max(0, MINIMUM - possible_fills),
                "estimated_300_raw_snapshot_date": _estimate_date(
                    total, session_days, first, last, today),
                "estimated_300_resolved_date": _estimate_date(
                    resolved, resolved_days, first, last, today),
                "estimated_300_possible_fill_date": _estimate_date(
                    possible_fills, resolved_days, first, last, today),
            })
    backfills = []
    for path in sorted((ARCHIVE / "backfills").glob("*/*.json")):
        payload = signed_payload(path)
        if payload.get("backfilled") is True:
            if (payload.get("pit_quality") != "degraded"
                    or payload.get("data_source") != DATA_SOURCE
                    or payload.get("training_eligible") is not False):
                raise ValueError("Backfill degradado inválido o mezclado con otra fuente.")
            backfills.append({"historical_cut_at_utc": payload["historical_cut_at_utc"],
                              "symbol": payload["symbol"], "horizon": payload["horizon"],
                              "pit_quality": "degraded", "training_eligible": False,
                              "path": str(path.relative_to(ARCHIVE))})
    missing_cuts = []
    first_cut = min((datetime.fromisoformat(cut) for cut, _, _ in scheduled_seen), default=None)
    if first_cut is not None:
        first_local = first_cut.astimezone(NY).date()
        last_local = now.astimezone(NY).date()
        calendar = xcals.get_calendar("XNYS", start=f"{first_local.year-1}-01-01",
                                      end=f"{last_local.year+1}-12-31")
        for session_day in calendar.sessions_in_range(first_local, last_local):
            session = calendar.schedule.loc[session_day]
            for cut_label in CUT_TIMES:
                scheduled = datetime.fromisoformat(
                    f"{session_day.date().isoformat()}T{cut_label}:00").replace(tzinfo=NY).astimezone(timezone.utc)
                if (scheduled < first_cut or scheduled + GRACE > now
                        or not session.open.to_pydatetime() <= scheduled < session.close.to_pydatetime()):
                    continue
                absent = [(symbol, horizon) for symbol in SYMBOLS for horizon in HORIZONS
                          if (scheduled.isoformat(), symbol, horizon) not in scheduled_seen]
                if absent:
                    missing_cuts.append({
                        "scheduled_cut_at_utc": scheduled.isoformat(),
                        "session_ny": session_day.date().isoformat(),
                        "cut_ny": cut_label, "missing_symbol_horizons":
                        [{"symbol": symbol, "horizon": horizon} for symbol, horizon in absent],
                        "backfill_is_not_prospective_recovery": any(
                            b["historical_cut_at_utc"] == scheduled.isoformat() for b in backfills),
                    })
    report = {
        "report_version": REPORT_VERSION, "generated_at_utc": now.isoformat(),
        "data_source": DATA_SOURCE,
        "source_policy": "Signed V1 snapshots derive source only from their immutable Alpaca IEX policy; mixed feeds are rejected.",
        "minimum_per_horizon": MINIMUM, "rows": rows,
        "missing_prospective_cuts": missing_cuts,
        "backfilled_degraded_research_only": backfills,
        "backfill_count_excluded_from_denominators": len(backfills),
        "by_day_symbol_horizon": [
            {"session_ny": day, "symbol": symbol, "horizon": horizon,
             "data_source": DATA_SOURCE, **values}
            for (day, symbol, horizon), values in sorted(daily.items())],
        "independence_warning": "Five same-day cuts and overlapping 1D horizons are not independent events; session groups are a proxy, not proof of independence.",
        "eta_policy": "N/D until at least five actual session groups; projection uses observed XNYS-session rate, not 5 guaranteed independent events/day.",
        "example_excluded": True, "training_performed": False,
        "portfolio_db_accessed": False,
    }
    if save:
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        write_payload(ARCHIVE / "reports" / f"coverage_{stamp}.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(save=not args.no_save)
    print(json.dumps({"generated_at_utc": report["generated_at_utc"],
                      "rows": report["rows"],
                      "missing_prospective_cuts": report["missing_prospective_cuts"],
                      "backfill_count_excluded_from_denominators": report["backfill_count_excluded_from_denominators"],
                      "independence_warning": report["independence_warning"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
