"""Read-only EDGAR event-coverage pilot over the existing signed 11:00 NY replay.

SEC acceptance time is *not* certified first public availability. The primary
feature uses only filings accepted before the previous XNYS session close;
the five-minute-lag view is a less conservative sensitivity check. No model
training, live decision, or accounting database access occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.historical_replay import validate_historical_replay
from scripts.download_massive_history import read_signed, sha, write_signed


VERSION = "SEC_EDGAR_EVENT_COVERAGE_V1"
SYMBOLS = ("SMCI", "NVDA")
FORMS = frozenset(("8-K", "10-Q", "10-K"))
REPLAY_FILES = {
    "SMCI": ROOT / "output/historical_replay/pilot_1h_1d/SMCI_0524e9f9f0c28c5c.json",
    "NVDA": ROOT / "output/historical_replay/pilot_1h_1d/NVDA_8dc898dc3a5ab7ab.json",
}
OUT = ROOT / "output/historical_replay/sec_edgar_event_pilot_v1"
NY = ZoneInfo("America/New_York")


def parse_utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("EDGAR/replay timestamp without timezone")
    return result.astimezone(timezone.utc)


def contact_email() -> str:
    value = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not value:
        result = subprocess.run(["git", "config", "user.email"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        value = result.stdout.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise ValueError("Set SEC_CONTACT_EMAIL or git config user.email for SEC User-Agent")
    return value


class SecClient:
    def __init__(self, email: str) -> None:
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": f"STOCK-EDGAR-Research/0.1 ({email})",
                                  "Accept-Encoding": "gzip, deflate"})
        self.last_request = 0.0

    def get_json(self, url: str) -> dict:
        # Well below the SEC's ten-requests-per-second ceiling.
        time.sleep(max(0.0, 0.3 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        response = self.http.get(url, timeout=40)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected SEC response type: {url}")
        return payload

    def header_acceptance(self, url: str) -> str | None:
        time.sleep(max(0.0, 0.3 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        with self.http.get(url, timeout=40, stream=True) as response:
            response.raise_for_status()
            prefix = b""
            for chunk in response.iter_content(chunk_size=8192):
                prefix += chunk
                found = re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", prefix)
                if found:
                    return found.group(1).decode("ascii")
                if len(prefix) > 262144:
                    break
        return None


def load_replays() -> dict[str, list[dict]]:
    results = {}
    for symbol, path in REPLAY_FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_historical_replay(payload)
        cuts = payload["observations"]
        if payload["symbol"] != symbol or len(cuts) != 299:
            raise ValueError(f"Unexpected replay scope: {symbol}")
        results[symbol] = cuts
    if [x["observed_at"] for x in results["SMCI"]] != [
            x["observed_at"] for x in results["NVDA"]]:
        raise ValueError("Replay cuts differ across symbols")
    return results


def cik_lookup(tickers: dict, symbol: str) -> int:
    matches = [entry for entry in tickers.values()
               if isinstance(entry, dict) and entry.get("ticker") == symbol]
    if len(matches) != 1:
        raise ValueError(f"Official SEC CIK lookup not unique for {symbol}")
    return int(matches[0]["cik_str"])


def filing_records(submissions: dict, cik: int) -> list[dict]:
    recent = submissions["filings"]["recent"]
    count = len(recent["form"])
    required = ("accessionNumber", "filingDate", "acceptanceDateTime", "form",
                "primaryDocument", "items")
    if any(len(recent[key]) != count for key in required):
        raise ValueError("Misaligned EDGAR recent columns")
    records = []
    for index, form in enumerate(recent["form"]):
        if form not in FORMS:
            continue
        accession = recent["accessionNumber"][index]
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Invalid SEC accession")
        accepted = parse_utc(recent["acceptanceDateTime"][index])
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        records.append({
            "cik": f"{cik:010d}", "accession": accession, "form": form,
            "filing_date": recent["filingDate"][index],
            "accepted_at_utc": accepted.isoformat(),
            "items": recent["items"][index] or "",
            "header_url": f"{base}/{accession}.txt",
            "index_url": f"{base}/{accession}-index.htm",
            "primary_document": recent["primaryDocument"][index],
        })
    records.sort(key=lambda row: (row["accepted_at_utc"], row["accession"]))
    return records


def previous_session_close(cut: datetime, calendar) -> datetime:
    session = pd.Timestamp(cut.astimezone(NY).date())
    prior = calendar.previous_session(session)
    return calendar.session_close(prior).to_pydatetime().astimezone(timezone.utc)


def eligible_events(events: list[dict], cut: datetime, hours: int,
                    previous_close: datetime, policy: str) -> list[dict]:
    if hours not in (24, 48) or policy not in ("previous_session", "five_minute_lag"):
        raise ValueError("Unknown preregistered EDGAR window/policy")
    latest = previous_close if policy == "previous_session" else cut - timedelta(minutes=5)
    start = cut - timedelta(hours=hours)
    return [row for row in events if start < parse_utc(row["accepted_at_utc"]) <= latest]


def coverage(cuts: list[dict], events: list[dict], calendar) -> dict:
    rows = []
    for item in cuts:
        cut = parse_utc(item["observed_at"])
        prior_close = previous_session_close(cut, calendar)
        windows = {}
        for policy in ("previous_session", "five_minute_lag"):
            for hours in (24, 48):
                selected = eligible_events(events, cut, hours, prior_close, policy)
                windows[f"{policy}_{hours}h"] = {
                    "event_count": len(selected),
                    "accessions": [row["accession"] for row in selected],
                }
        rows.append({"cut_at_utc": cut.isoformat(), "session_ny": cut.astimezone(NY).date().isoformat(),
                     "previous_xnys_close_utc": prior_close.isoformat(), "windows": windows})
    summary = {}
    for key in rows[0]["windows"]:
        covered = [row for row in rows if row["windows"][key]["event_count"] > 0]
        unique_accessions = {acc for row in covered for acc in row["windows"][key]["accessions"]}
        summary[key] = {"cuts_with_event": len(covered), "cuts_without_event": len(rows)-len(covered),
                        "unique_filings_seen": len(unique_accessions)}
    return {"cuts": rows, "summary": summary}


def build() -> dict:
    replays = load_replays()
    client = SecClient(contact_email())
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    tickers_path = OUT / "sec_company_tickers.json"
    if tickers_path.exists():
        tickers = read_signed(tickers_path)["response"]
    else:
        tickers = client.get_json(tickers_url)
        write_signed(tickers_path, {"source": tickers_url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "response": tickers})

    calendar = xcals.get_calendar("XNYS")
    report = {"version": VERSION, "symbols": {}, "trained": False,
              "production_changed": False, "portfolio_db_accessed": False,
              "availability_warning": "SEC acceptance is not first-public-availability"}
    for symbol in SYMBOLS:
        cik = cik_lookup(tickers, symbol)
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        raw_path = OUT / f"{symbol}_sec_submissions.json"
        if raw_path.exists():
            submissions = read_signed(raw_path)["response"]
        else:
            submissions = client.get_json(url)
            write_signed(raw_path, {"source": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "response": submissions})
        if int(submissions["cik"]) != cik:
            raise ValueError(f"CIK mismatch: {symbol}")
        events = filing_records(submissions, cik)
        first_cut = parse_utc(replays[symbol][0]["observed_at"])
        if not events or parse_utc(events[0]["accepted_at_utc"]) > first_cut - timedelta(hours=48):
            raise ValueError(f"Insufficient SEC history for replay: {symbol}")
        # Header check is a sample, not a claim that all old files were first seen at acceptance.
        candidate = next((row for row in events if "2025-07" <= row["filing_date"] <= "2025-09"), None)
        header_check = {"accession": candidate["accession"] if candidate else None,
                        "retrieved": False, "matches_acceptance_local": None}
        if candidate:
            try:
                value = client.header_acceptance(candidate["header_url"])
                expected = parse_utc(candidate["accepted_at_utc"]).astimezone(NY).strftime("%Y%m%d%H%M%S")
                header_check.update({"retrieved": value is not None,
                                     "matches_acceptance_local": value == expected if value else None})
            except requests.RequestException as exc:
                header_check["error_type"] = type(exc).__name__
        measured = coverage(replays[symbol], events, calendar)
        write_signed(OUT / f"{symbol}_filings.json", {"symbol": symbol, "cik": f"{cik:010d}",
                                                      "submissions_sha256": sha(submissions), "events": events})
        write_signed(OUT / f"{symbol}_coverage.json", {"symbol": symbol, "cik": f"{cik:010d}",
                                                       "replay_artifact_sha256": replays[symbol][0]["dataset_sha256"],
                                                       "coverage": measured})
        report["symbols"][symbol] = {
            "cik": f"{cik:010d}", "company": submissions["name"],
            "replay_cuts": len(replays[symbol]), "relevant_filings_in_sec_history": len(events),
            "forms_in_history": dict(Counter(row["form"] for row in events)),
            "history_first_filing_date": submissions["filings"]["recent"]["filingDate"][-1],
            "header_check": header_check,
            "coverage": measured["summary"],
        }
    write_signed(OUT / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", nargs="?", choices=("pilot",), default="pilot")
    parser.parse_args()
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
