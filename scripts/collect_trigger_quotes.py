"""Read-only SIP quote audit for the six frozen events. Never call trading APIs.

Credentials stay in process memory, entered with getpass or environment variables.
No keys, authorization headers or provider error bodies are written to artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.measure_trigger_selectivity import DEFAULT_OUTPUT


def run(attempt):
    report = read_signed(DEFAULT_OUTPUT / "measurement_report.json")
    key = os.environ.get("APCA_API_KEY_ID") or getpass.getpass("Alpaca key (hidden): ")
    secret = os.environ.get("APCA_API_SECRET_KEY") or getpass.getpass("Alpaca secret (hidden): ")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    evidence = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "measurement_report_sha256": sha(report), "provider": "ALPACA", "feed": "sip",
                "endpoint": "https://data.alpaca.markets/v2/stocks/SMCI/quotes",
                "read_only": True, "no_purchase": True, "quotes": [], "failures": [],
                "method": "LAST QUOTE <= REFERENCE TIME, MAX AGE 10s; reference fill/exit times are simulated 5m bar opens, NOT broker fills."}
    try:
        for event in report["fill_evidence"]:
            references = [("cut", event["cut_at"])]
            for h in event["resolutions"]:
                references.extend([(f"possible_fill_{h['horizon_minutes']}", h["fill_at_possible_OHLC"]),
                                   (f"simulated_exit_{h['horizon_minutes']}", h["exit_at_simulated"])])
            for label, at in references:
                timestamp = pd.Timestamp(at)
                params = {"feed": "sip", "start": (timestamp - pd.Timedelta(seconds=10)).isoformat(),
                          "end": timestamp.isoformat(), "sort": "desc", "limit": 1}
                response = session.get(evidence["endpoint"], params=params, timeout=25)
                if response.status_code != 200:
                    evidence["failures"].append({"event_id": event["event_id"], "reference": label,
                        "reference_at": at, "http_status": response.status_code,
                        "error": "AUTH_OR_ENTITLEMENT" if response.status_code in (401, 403) else "PROVIDER_HTTP_ERROR"})
                    if response.status_code in (401, 403):
                        print("Quote access rejected. HTTP", response.status_code, flush=True)
                        return
                    continue
                quotes = response.json().get("quotes", []) or []
                row = {"event_id": event["event_id"], "side": event["side"], "reference": label,
                       "reference_at": at, "quote": quotes[0] if quotes else None}
                if quotes:
                    q = quotes[0]
                    age = (timestamp - pd.Timestamp(q["t"])).total_seconds()
                    if not 0 <= age <= 10:
                        raise ValueError("Quote falls outside backward window.")
                    row["age_seconds"] = age
                    row["spread_observed"] = float(q["ap"]) - float(q["bp"])
                    row["valid_positive_uncrossed_market"] = 0 < float(q["bp"]) <= float(q["ap"])
                evidence["quotes"].append(row)
            print("Quoted event", event["event_id"][:12], flush=True)
    except requests.RequestException:
        evidence["failures"].append({"error": "NETWORK_ERROR_DETAILS_REDACTED"})
        print("Quote download network error; no credentials logged.", flush=True)
    finally:
        session.close()
        del key, secret
        path = DEFAULT_OUTPUT / "quote_access" / f"{attempt}.json"
        write_signed(path, evidence)
        print("Evidence:", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, choices=["provided_text_pair", "provided_screenshot_pair"])
    run(parser.parse_args().attempt)
