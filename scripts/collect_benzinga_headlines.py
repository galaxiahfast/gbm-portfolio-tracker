"""Acquire an immutable *research-only* Benzinga headline archive.

Pilot first. The key is read only from BENZINGA_API_KEY and is never written or
printed. ``created`` is not a certified delivery timestamp: outputs are marked
PIT_PROXY, and historical title revisions after a cut are excluded downstream.
This script does not read portfolio.db, infer model outcomes, or train anything.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import getpass
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.headline_sentiment_replay import (
    canonical_sha256, normalize_benzinga, select_at_cut,
)
from scripts.download_massive_history import read_signed, write_signed


OUTPUT = ROOT / "output/historical_replay/headline_finbert_v1"
URL = "https://api.benzinga.com/api/v2/news"
SYMBOLS = ("SMCI", "NVDA")
PILOT_DATES = ("2025-08-05", "2025-09-15", "2026-09-24")
PAGE_SIZE = 100


def _get_page(session, key, symbol, day, page):
    try:
        response = session.get(URL, headers={"accept": "application/json"}, params={
            "token": key, "tickers": symbol, "dateFrom": str(day - timedelta(days=1)),
            "dateTo": str(day + timedelta(days=1)), "page": page, "pageSize": PAGE_SIZE,
            "displayOutput": "headline",
        }, timeout=30)
    except requests.RequestException as exc:
        # Requests exceptions include the full URL, which contains the token.
        raise RuntimeError(f"Benzinga network failure for {symbol} {day} page {page}") from None
    if response.status_code == 429:
        raise RuntimeError(f"Benzinga rate limit for {symbol} {day} page {page}")
    if response.status_code != 200:
        raise RuntimeError(f"Benzinga HTTP {response.status_code} for {symbol} {day} page {page}")
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"Benzinga invalid JSON for {symbol} {day} page {page}") from None
    if not isinstance(data, list):
        raise RuntimeError(f"Benzinga schema not a list for {symbol} {day} page {page}")
    return data


def collect_day(session, key, symbol, day, *, folder=OUTPUT):
    day = date.fromisoformat(str(day))
    path = folder / "raw" / f"{symbol}_{day}.json"
    if path.exists():
        saved = read_signed(path)
        if saved["symbol"] != symbol or saved["day"] != day.isoformat():
            raise ValueError("Signed headline archive identity mismatch")
        return saved
    all_items = []
    for page in range(50):
        items = _get_page(session, key, symbol, day, page)
        all_items.extend(items)
        if len(items) < PAGE_SIZE:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Benzinga pagination cap reached for {symbol} {day}")
    raw_hash = canonical_sha256(all_items)
    articles = {}
    exclusions = Counter()
    for item in all_items:
        try:
            row = normalize_benzinga(item)
        except (KeyError, TypeError, ValueError) as exc:
            exclusions[type(exc).__name__] += 1
            continue
        if symbol not in row["tickers"]:
            exclusions["not_tagged_to_symbol"] += 1
            continue
        if row["id"] in articles and articles[row["id"]] != row:
            raise ValueError("Conflicting article versions in response")
        articles[row["id"]] = row
    payload = {
        "source": "BENZINGA_NEWS_HEADLINE", "provenance": "PIT_PROXY_NOT_CERTIFIED",
        "symbol": symbol, "day": day.isoformat(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "query_from": (day - timedelta(days=1)).isoformat(),
        "query_to": (day + timedelta(days=1)).isoformat(),
        "pages_complete": True, "raw_response_sha256": raw_hash,
        "raw_count": len(all_items), "normalized_count": len(articles),
        "exclusions": dict(exclusions), "articles": list(articles.values()),
    }
    write_signed(path, payload)
    return payload


def _cut(day):
    return datetime.combine(date.fromisoformat(str(day)),
                            datetime.min.time().replace(hour=11),
                            tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def pilot(session, key):
    summaries = []
    for symbol in SYMBOLS:
        for day in PILOT_DATES:
            payload = collect_day(session, key, symbol, day)
            cut = _cut(day)
            at_cut = select_at_cut(payload["articles"], symbol, cut, 24)
            summaries.append({
                "symbol": symbol, "day": day,
                "raw_count": payload["raw_count"],
                "normalized_count": payload["normalized_count"],
                "at_11_ny_24h": len(at_cut),
                "excluded_revised_after_cut": sum(
                    1 for article in payload["articles"]
                    if article["created_at"] <= cut.isoformat()
                    and article["updated_at"] > cut.isoformat()),
                "raw_response_sha256": payload["raw_response_sha256"],
                "archive_sha256": canonical_sha256(payload),
            })
    old = [row for row in summaries if row["day"] < "2026-01-01"]
    access_seen = all(any(row["at_11_ny_24h"] > 0 for row in old
                          if row["symbol"] == symbol)
                      for symbol in SYMBOLS)
    result = {"version": "BENZINGA_HEADLINE_PILOT_V1", "dates": list(PILOT_DATES),
              "coverage_access_seen_for_both_old_symbols": access_seen,
              "pit_proven": False, "rows": summaries,
              "next_step": "Archive only if old-date entitlement and timestamps pass."}
    write_signed(OUTPUT / "pilot_report.json", result)
    return result


def archive(session, key):
    report = read_signed(OUTPUT / "pilot_report.json")
    if report["coverage_access_seen_for_both_old_symbols"] is not True:
        raise ValueError("Historical entitlement not demonstrated by pilot")
    prior = read_signed(ROOT / "output/historical_replay/learned_trigger_v1/preregistration.json")
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(prior["development_start"],
                                          prior["internal_holdout_end"])
    if len(sessions) != 299:
        raise ValueError("Expected exactly 299 fixed replay sessions")
    days = [session.date() for session in sessions]
    coverage = []
    for symbol in SYMBOLS:
        for day in days:
            row = collect_day(session, key, symbol, day)
            selected = select_at_cut(row["articles"], symbol, _cut(day), 24)
            coverage.append({"symbol": symbol, "session_ny": day.isoformat(),
                             "article_count_24h": len(selected),
                             "archive_sha256": canonical_sha256(row)})
    result = {"version": "BENZINGA_299_CUT_ARCHIVE_V1", "source": "Benzinga",
              "provenance": "PIT_PROXY_NOT_CERTIFIED", "cuts": len(coverage),
              "sessions_per_symbol": len(days), "coverage": coverage,
              "model_trained": False, "portfolio_db_accessed": False}
    write_signed(OUTPUT / "archive_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "archive"))
    parser.add_argument("--prompt-key", action="store_true",
                        help="Read a replacement key privately without storing it")
    args = parser.parse_args()
    key = getpass.getpass("New Benzinga key (hidden): ") if args.prompt_key else os.environ.get("BENZINGA_API_KEY")
    if not key:
        parser.error("Set BENZINGA_API_KEY locally or use --prompt-key; do not paste it in chat.")
    http = requests.Session()
    result = pilot(http, key) if args.phase == "pilot" else archive(http, key)
    print({k: v for k, v in result.items() if k not in {"rows", "coverage"}})


if __name__ == "__main__":
    main()
