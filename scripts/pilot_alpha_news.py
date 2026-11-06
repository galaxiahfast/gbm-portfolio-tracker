"""Research-only Alpha Vantage raw-headline pilot for fixed replay cuts.

Reads ALPHAVANTAGE_API_KEY or a hidden prompt; never stores the key. The API's
time_published is not a certified first-available timestamp, so this pilot
cannot by itself authorize PIT model training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import getpass
import os
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_finnhub_headlines import pilot_dates
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.pilot_massive_news import SYMBOLS, cut_utc


OUT = ROOT / "output/historical_replay/headline_finbert_alpha_v1"
URL = "https://www.alphavantage.co/query"


def preregister() -> dict:
    path = OUT / "pilot_preregistration.json"
    if path.exists():
        return read_signed(path)
    protocol = {
        "version": "ALPHA_VANTAGE_NEWS_PILOT_V1",
        "symbols": list(SYMBOLS), "dates": list(pilot_dates()),
        "cut": "11:00 America/New_York",
        "eligible_window": "(cut - 24h, cut]",
        "retrieval_window": "[cut - 24h, cut + 6h) for after-cut audit",
        "timestamp": "time_published is publication time, not first-seen",
        "limit": 1000, "requests_expected": 12,
        "training_before_PIT_certification": False,
        "portfolio_db_accessed": False,
    }
    write_signed(path, protocol)
    return protocol


def _parse_published(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def audit_articles(articles: list, symbol: str, day: str) -> dict:
    cut = cut_utc(day)
    lower = cut - timedelta(hours=24)
    upper = cut + timedelta(hours=6)
    valid = eligible = after = invalid = outside = 0
    normalized = []
    for item in articles:
        if not isinstance(item, dict):
            invalid += 1
            continue
        published = _parse_published(item.get("time_published"))
        title = item.get("title")
        if published is None or not isinstance(title, str) or not title.strip():
            invalid += 1
            continue
        valid += 1
        if not lower <= published < upper:
            outside += 1
        if lower < published <= cut:
            eligible += 1
        elif cut < published < upper:
            after += 1
        normalized.append({"symbol_requested": symbol, "published_utc": published.isoformat(),
                           "title": title.strip(), "url": item.get("url")})
    return {"raw_count": len(articles), "valid_timestamps_and_titles": valid,
            "eligible_24h_before_cut": eligible, "published_after_cut": after,
            "invalid_timestamps_or_titles": invalid,
            "outside_retrieval_window": outside, "articles": normalized}


def fetch_articles(http: requests.Session, key: str, symbol: str, day: str) -> dict:
    cut = cut_utc(day)
    lower = cut - timedelta(hours=24)
    upper = cut + timedelta(hours=6)
    params = {"function": "NEWS_SENTIMENT", "tickers": symbol,
              "time_from": lower.strftime("%Y%m%dT%H%M"),
              "time_to": upper.strftime("%Y%m%dT%H%M"),
              "sort": "EARLIEST", "limit": 1000, "apikey": key}
    try:
        response = http.get(URL, params=params, timeout=40)
    except requests.RequestException as exc:
        return {"http_status": None, "error": type(exc).__name__,
                "articles": [], "complete": False}
    try:
        payload = response.json()
    except ValueError:
        return {"http_status": response.status_code, "error": "NON_JSON_RESPONSE",
                "articles": [], "complete": False}
    if response.status_code != 200:
        return {"http_status": response.status_code,
                "error": f"HTTP_{response.status_code}", "articles": [], "complete": False}
    if not isinstance(payload, dict):
        return {"http_status": 200, "error": "UNEXPECTED_SCHEMA",
                "articles": [], "complete": False}
    feed = payload.get("feed")
    if not isinstance(feed, list):
        message = payload.get("Information") or payload.get("Note") or payload.get("Error Message")
        message = str(message or "NO_FEED_FIELD").replace(key, "[REDACTED]")[:240]
        return {"http_status": 200, "error": message, "articles": [], "complete": False}
    if len(feed) >= 1000:
        return {"http_status": 200, "error": "POSSIBLY_TRUNCATED_AT_1000",
                "articles": feed, "complete": False}
    return {"http_status": 200, "error": None, "articles": feed, "complete": True}


def run_pilot(key: str) -> dict:
    protocol = preregister()
    report_path = OUT / "pilot_report.json"
    if report_path.exists():
        return read_signed(report_path)
    http = requests.Session()
    rows = []
    for day in protocol["dates"]:
        for symbol in SYMBOLS:
            if rows:
                time.sleep(12.5)
            response = fetch_articles(http, key, symbol, day)
            audit = audit_articles(response["articles"], symbol, day)
            rows.append({"day": day, "symbol": symbol,
                         "http_status": response["http_status"],
                         "error": response["error"], "complete": response["complete"],
                         "raw_response_sha256": sha(response["articles"]), **audit})
    coverage_pass = all(row["http_status"] == 200 and row["error"] is None
                        and row["complete"] and row["eligible_24h_before_cut"] > 0
                        and row["invalid_timestamps_or_titles"] == 0
                        and row["outside_retrieval_window"] == 0 for row in rows)
    report = {"version": protocol["version"], "protocol_sha256": sha(protocol),
              "coverage_pass": coverage_pass, "PIT_availability_certified": False,
              "training_permitted": False,
              "reason": "time_published does not certify first availability",
              "rows": rows, "production_changed": False, "portfolio_db_accessed": False}
    write_signed(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preregister", "pilot"))
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if args.phase == "preregister":
        print({"dates": preregister()["dates"], "key_needed": False})
        return
    key = getpass.getpass("Alpha Vantage key (hidden): ") if args.prompt_key else os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        parser.error("Set ALPHAVANTAGE_API_KEY locally or use --prompt-key; never paste it in chat.")
    report = run_pilot(key)
    print({"coverage_pass": report["coverage_pass"],
           "PIT_availability_certified": report["PIT_availability_certified"],
           "rows": [{k: row[k] for k in ("day", "symbol", "http_status", "error",
                                         "raw_count", "valid_timestamps_and_titles",
                                         "eligible_24h_before_cut", "published_after_cut")}
                    for row in report["rows"]]})


if __name__ == "__main__":
    main()
