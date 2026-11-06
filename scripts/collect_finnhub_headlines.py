"""Research-only Finnhub company-news pilot for fixed 11:00 NY replay cuts.

Never put an API key in this file or a CLI argument. Use FINNHUB_API_KEY in the
process environment or --prompt-key (hidden console input). A successful HTTP
response does not certify point-in-time availability: Finnhub documents
``datetime`` as publication time, not first-seen/ingestion time.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import getpass
import os
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_massive_history import read_signed, sha, write_signed


OUTPUT = ROOT / "output/historical_replay/headline_finbert_finnhub_v1"
URL = "https://finnhub.io/api/v1/company-news"
SYMBOLS = ("SMCI", "NVDA")
FIXED_DATES = ("2025-08-05", "2025-09-15", "2026-09-24")
RANDOM_SEED = 20260929


def pilot_dates():
    calendar = xcals.get_calendar("XNYS")
    pool = [session.date().isoformat() for session in calendar.sessions_in_range(
        "2025-10-01", "2026-09-18")]
    selected = sorted(random.Random(RANDOM_SEED).sample(pool, 3))
    return (*FIXED_DATES, *selected)


def preregister():
    path = OUTPUT / "pilot_preregistration.json"
    if path.exists():
        return read_signed(path)
    protocol = {
        "version": "FINNHUB_COMPANY_NEWS_PILOT_V1",
        "source": "Finnhub company-news",
        "symbols": list(SYMBOLS), "dates": list(pilot_dates()),
        "random_seed": RANDOM_SEED,
        "window": "(cut - 24h, cut], cut=11:00 America/New_York",
        "timestamp_field": "datetime is published Unix seconds, not certified first-seen",
        "required_for_full_replay": "all 299 fixed sessions per symbol; never silently shorten the historical cohort",
        "no_training_before_pilot": True,
        "portfolio_db_accessed": False,
    }
    write_signed(path, protocol)
    return protocol


def _cut(day):
    local = datetime.combine(date.fromisoformat(day),
                             datetime.min.time().replace(hour=11),
                             tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(timezone.utc)


def _safe_message(value, key):
    return str(value).replace(key, "[REDACTED]")[:240]


def fetch(session, key, symbol, day):
    """Return response metadata without ever logging a URL containing token."""
    beginning = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    try:
        response = session.get(URL, params={"symbol": symbol, "from": beginning,
                                            "to": day, "token": key}, timeout=30)
    except requests.RequestException as exc:
        return {"http_status": None, "error": type(exc).__name__, "articles": []}
    try:
        data = response.json()
    except ValueError:
        return {"http_status": response.status_code, "error": "NON_JSON_RESPONSE",
                "articles": []}
    if response.status_code != 200:
        message = data.get("error", data.get("message", "HTTP_ERROR")) if isinstance(data, dict) else "HTTP_ERROR"
        return {"http_status": response.status_code,
                "error": _safe_message(message, key), "articles": []}
    if not isinstance(data, list):
        message = data.get("error", data.get("message", "UNEXPECTED_SCHEMA")) if isinstance(data, dict) else "UNEXPECTED_SCHEMA"
        return {"http_status": response.status_code,
                "error": _safe_message(message, key), "articles": []}
    return {"http_status": response.status_code, "error": None, "articles": data}


def audit_articles(data, symbol, day):
    cut = _cut(day)
    lower = cut - timedelta(hours=24)
    valid, invalid, after_cut, in_window, outside_dates, ticker_mismatch = [], 0, 0, 0, 0, 0
    requested_dates = {day, (date.fromisoformat(day) - timedelta(days=1)).isoformat()}
    for item in data:
        if not isinstance(item, dict):
            invalid += 1
            continue
        stamp = item.get("datetime")
        title = item.get("headline")
        if (isinstance(stamp, bool) or not isinstance(stamp, int)
                or stamp <= 0 or not isinstance(title, str) or not title.strip()):
            invalid += 1
            continue
        try:
            published = datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            invalid += 1
            continue
        # Finnhub's date filters are date-based; audit the provider-returned
        # timestamp in UTC. A 00:00 UTC article can belong to the previous
        # NY evening and is not evidence of a malformed Unix timestamp.
        if published.date().isoformat() not in requested_dates:
            outside_dates += 1
        related = item.get("related")
        if isinstance(related, str) and related.strip():
            if symbol not in {part.strip().upper() for part in related.split(",")}:
                ticker_mismatch += 1
        if published > cut:
            after_cut += 1
        if lower < published <= cut:
            in_window += 1
        valid.append({"id": str(item.get("id", "")), "symbol_requested": symbol,
                      "published_at": published.isoformat(),
                      "headline": title.strip(), "related": item.get("related"),
                      "source": item.get("source")})
    return {"raw_count": len(data), "valid_timestamps": len(valid),
            "invalid_timestamps_or_headlines": invalid,
            "timestamp_outside_requested_dates": outside_dates,
            "related_ticker_mismatch": ticker_mismatch,
            "published_after_cut": after_cut,
            "eligible_24h_before_cut": in_window,
            "articles": valid}


def run_pilot(key):
    protocol = preregister()
    report_path = OUTPUT / "pilot_report.json"
    if report_path.exists():
        return read_signed(report_path)
    http = requests.Session()
    rows = []
    for day in protocol["dates"]:
        for symbol in SYMBOLS:
            response = fetch(http, key, symbol, day)
            audit = audit_articles(response["articles"], symbol, day)
            rows.append({"day": day, "symbol": symbol,
                         "http_status": response["http_status"],
                         "error": response["error"],
                         "raw_response_sha256": sha(response["articles"]), **audit})
    passes = all(row["http_status"] == 200 and row["error"] is None
                 and row["eligible_24h_before_cut"] > 0
                 and row["invalid_timestamps_or_headlines"] == 0
                 and row["timestamp_outside_requested_dates"] == 0
                 and row["related_ticker_mismatch"] == 0 for row in rows)
    report = {"version": protocol["version"], "protocol_sha256": sha(protocol),
              "source": "Finnhub company-news", "provenance": "PUBLISHED_TIME_PROXY_NOT_CERTIFIED_PIT",
              "rows": rows, "pilot_pass": passes,
              "full_archive_permitted": passes,
              "production_changed": False, "portfolio_db_accessed": False}
    write_signed(report_path, report)
    return report


def run_archive(key):
    pilot = read_signed(OUTPUT / "pilot_report.json")
    if pilot["pilot_pass"] is not True:
        raise ValueError("Finnhub pilot failed; archive and training are forbidden")
    if (OUTPUT / "archive_manifest.json").exists():
        return read_signed(OUTPUT / "archive_manifest.json")
    prior = read_signed(ROOT / "output/historical_replay/learned_trigger_v1/preregistration.json")
    sessions = xcals.get_calendar("XNYS").sessions_in_range(
        prior["development_start"], prior["internal_holdout_end"])
    if len(sessions) != 299:
        raise ValueError("Expected 299 fixed replay sessions")
    http = requests.Session()
    coverage = []
    for session in sessions:
        day = session.date().isoformat()
        for symbol in SYMBOLS:
            path = OUTPUT / "archive" / f"{symbol}_{day}.json"
            if path.exists():
                row = read_signed(path)
            else:
                response = fetch(http, key, symbol, day)
                if response["http_status"] != 200 or response["error"] is not None:
                    raise RuntimeError(f"Finnhub archive failed {symbol} {day}: "
                                       f"HTTP {response['http_status']} {response['error']}")
                audit = audit_articles(response["articles"], symbol, day)
                row = {"source": "Finnhub company-news", "symbol": symbol, "day": day,
                       "http_status": 200, "raw_response_sha256": sha(response["articles"]),
                       **audit}
                write_signed(path, row)
                time.sleep(1.1)
            if row["symbol"] != symbol or row["day"] != day:
                raise ValueError("Archive identity mismatch")
            if row["invalid_timestamps_or_headlines"] or row["timestamp_outside_requested_dates"]:
                raise ValueError(f"Finnhub timestamp audit failed for {symbol} {day}")
            coverage.append({"symbol": symbol, "session_ny": day,
                             "raw_count": row["raw_count"],
                             "eligible_24h_before_cut": row["eligible_24h_before_cut"],
                             "sha256": sha(row)})
    result = {"version": "FINNHUB_COMPANY_NEWS_ARCHIVE_V1",
              "source": "Finnhub company-news", "provenance": "PUBLISHED_TIME_PROXY_NOT_CERTIFIED_PIT",
              "pilot_sha256": sha(pilot), "sessions_per_symbol": len(sessions),
              "cuts": len(coverage), "coverage": coverage,
              "model_trained": False, "portfolio_db_accessed": False}
    write_signed(OUTPUT / "archive_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preregister", "pilot", "archive"))
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if args.phase == "preregister":
        print({"dates": preregister()["dates"], "key_needed": False})
        return
    key = getpass.getpass("New Finnhub key (hidden): ") if args.prompt_key else os.environ.get("FINNHUB_API_KEY")
    if not key:
        parser.error("Set FINNHUB_API_KEY locally or use --prompt-key. Never paste it in chat.")
    if args.phase == "pilot":
        result = run_pilot(key)
        print({"pilot_pass": result["pilot_pass"],
               "rows": [{k: row[k] for k in ("day", "symbol", "http_status", "error",
                                             "raw_count", "valid_timestamps",
                                             "eligible_24h_before_cut", "published_after_cut",
                                             "timestamp_outside_requested_dates", "related_ticker_mismatch")}
                        for row in result["rows"]]})
    else:
        result = run_archive(key)
        print({"cuts": result["cuts"],
               "sessions_per_symbol": result["sessions_per_symbol"]})


if __name__ == "__main__":
    main()
