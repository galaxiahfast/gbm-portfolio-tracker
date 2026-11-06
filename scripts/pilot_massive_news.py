"""Research-only Massive news pilot for the fixed 11:00 New York replay cuts.

The API key is read from MASSIVE_API_KEY or a hidden prompt. No accounting,
production, training, or live market files are read or written. A publication
timestamp is *not* a certified first-available timestamp.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import getpass
import os
from pathlib import Path
import sys
import time
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_finnhub_headlines import pilot_dates
from scripts.download_massive_history import read_signed, sha, write_signed


OUT = ROOT / "output/historical_replay/headline_finbert_massive_v1"
URL = "https://api.massive.com/v2/reference/news"
SYMBOLS = ("SMCI", "NVDA")
NY = ZoneInfo("America/New_York")
MAX_PAGES = 20


def cut_utc(day: str) -> datetime:
    return datetime.combine(date.fromisoformat(day), datetime.min.time(),
                            tzinfo=NY).replace(hour=11).astimezone(timezone.utc)


def preregister() -> dict:
    path = OUT / "pilot_preregistration.json"
    if path.exists():
        return read_signed(path)
    protocol = {
        "version": "MASSIVE_HEADLINE_PILOT_V1",
        "symbols": list(SYMBOLS),
        "dates": list(pilot_dates()),
        "cut": "11:00 America/New_York",
        "eligible_window": "(cut - 24h, cut]",
        "retrieval_window": "[cut - 24h, cut + 6h) for after-cut audit",
        "timestamp": "published_utc is publication time, not first-seen",
        "page_size": 1000,
        "max_pages": MAX_PAGES,
        "training_before_PIT_certification": False,
        "portfolio_db_accessed": False,
    }
    write_signed(path, protocol)
    return protocol


def _next_params(next_url: str) -> dict:
    parts = urlsplit(next_url)
    if parts.scheme != "https" or parts.hostname != "api.massive.com" or parts.path != "/v2/reference/news":
        raise ValueError("Unsafe Massive pagination URL")
    # Never retain a credential returned as part of the provider's next_url.
    return {k: v for k, v in parse_qsl(parts.query) if k.lower() != "apikey"}


def _parse_published(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def audit_articles(articles: list, symbol: str, day: str) -> dict:
    cut = cut_utc(day)
    lower = cut - timedelta(hours=24)
    upper = cut + timedelta(hours=6)
    valid = eligible = after = invalid = ticker_mismatch = outside_window = 0
    normalized = []
    for item in articles:
        if not isinstance(item, dict):
            invalid += 1
            continue
        published = _parse_published(item.get("published_utc"))
        title = item.get("title")
        if published is None or not isinstance(title, str) or not title.strip():
            invalid += 1
            continue
        valid += 1
        if not lower <= published < upper:
            outside_window += 1
        if lower < published <= cut:
            eligible += 1
        elif cut < published < upper:
            after += 1
        tickers = item.get("tickers")
        if isinstance(tickers, list) and symbol not in tickers:
            ticker_mismatch += 1
        normalized.append({
            "id": str(item.get("id", "")), "symbol_requested": symbol,
            "published_utc": published.isoformat(), "title": title.strip(),
            "tickers": tickers, "publisher": item.get("publisher", {}).get("name")
            if isinstance(item.get("publisher"), dict) else None,
        })
    return {
        "raw_count": len(articles), "valid_timestamps_and_titles": valid,
        "eligible_24h_before_cut": eligible, "published_after_cut": after,
        "invalid_timestamps_or_titles": invalid,
        "ticker_mismatch": ticker_mismatch, "outside_retrieval_window": outside_window,
        "articles": normalized,
    }


def fetch_articles(http: requests.Session, key: str, symbol: str, day: str,
                   *, sleep=time.sleep) -> dict:
    cut = cut_utc(day)
    lower = cut - timedelta(hours=24)
    upper = cut + timedelta(hours=6)
    params = {
        "ticker": symbol, "published_utc.gte": lower.isoformat().replace("+00:00", "Z"),
        "published_utc.lt": upper.isoformat().replace("+00:00", "Z"),
        "sort": "published_utc", "order": "asc", "limit": 1000,
    }
    articles = []
    for page in range(1, MAX_PAGES + 1):
        if page > 1:
            sleep(12.5)  # Massive Basic documents 5 requests/minute.
        try:
            response = http.get(URL, params={**params, "apiKey": key}, timeout=40)
        except requests.RequestException as exc:
            return {"http_status": None, "error": type(exc).__name__,
                    "articles": articles, "pages": page - 1, "complete": False}
        status = response.status_code
        if status != 200:
            return {"http_status": status, "error": f"HTTP_{status}",
                    "articles": articles, "pages": page, "complete": False}
        try:
            payload = response.json()
        except ValueError:
            return {"http_status": status, "error": "NON_JSON_RESPONSE",
                    "articles": articles, "pages": page, "complete": False}
        if not isinstance(payload, dict):
            return {"http_status": status, "error": "UNEXPECTED_SCHEMA",
                    "articles": articles, "pages": page, "complete": False}
        if payload.get("status") not in (None, "OK", "DELAYED"):
            reason = str(payload.get("error") or payload.get("message") or payload["status"])
            return {"http_status": status, "error": reason.replace(key, "[REDACTED]")[:240],
                    "articles": articles, "pages": page, "complete": False}
        if not isinstance(payload.get("results", []), list):
            return {"http_status": status, "error": "UNEXPECTED_SCHEMA",
                    "articles": articles, "pages": page, "complete": False}
        articles.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if not next_url:
            return {"http_status": status, "error": None,
                    "articles": articles, "pages": page, "complete": True}
        try:
            params = _next_params(next_url)
        except ValueError:
            return {"http_status": status, "error": "UNSAFE_PAGINATION_URL",
                    "articles": articles, "pages": page, "complete": False}
    return {"http_status": 200, "error": "PAGINATION_LIMIT",
            "articles": articles, "pages": MAX_PAGES, "complete": False}


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
            rows.append({
                "day": day, "symbol": symbol,
                "http_status": response["http_status"], "error": response["error"],
                "pages": response["pages"], "complete": response["complete"],
                "raw_response_sha256": sha(response["articles"]), **audit,
            })
    coverage_pass = all(row["http_status"] == 200 and row["error"] is None
                        and row["complete"] and row["eligible_24h_before_cut"] > 0
                        and row["invalid_timestamps_or_titles"] == 0
                        and row["ticker_mismatch"] == 0
                        and row["outside_retrieval_window"] == 0 for row in rows)
    report = {
        "version": protocol["version"], "protocol_sha256": sha(protocol),
        "coverage_pass": coverage_pass,
        "PIT_availability_certified": False,
        "training_permitted": False,
        "reason": "published_utc does not certify first availability",
        "rows": rows, "production_changed": False, "portfolio_db_accessed": False,
    }
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
    key = getpass.getpass("Massive key (hidden): ") if args.prompt_key else os.environ.get("MASSIVE_API_KEY")
    if not key:
        parser.error("Set MASSIVE_API_KEY locally or use --prompt-key; never paste it in chat.")
    report = run_pilot(key)
    print({"coverage_pass": report["coverage_pass"],
           "PIT_availability_certified": report["PIT_availability_certified"],
           "rows": [{k: row[k] for k in ("day", "symbol", "http_status", "error",
                                         "raw_count", "valid_timestamps_and_titles",
                                         "eligible_24h_before_cut", "published_after_cut")}
                    for row in report["rows"]]})


if __name__ == "__main__":
    main()
