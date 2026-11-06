"""Audit real SIP quotes over fixed simulated fill windows, without changing replay."""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT, digest_file
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.measure_trigger_selectivity import DEFAULT_OUTPUT


def inspect_quotes(quotes, side, limit, start, end):
    """Marketable quote is possible execution evidence, never broker fill proof."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    valid, compatible, excluded = [], [], 0
    previous = None
    for q in quotes:
        timestamp = pd.Timestamp(q["t"])
        if timestamp < start or timestamp > end or (previous is not None and timestamp < previous):
            raise ValueError("Quote outside fixed time interval / ordering mismatch.")
        previous = timestamp
        bid, ask = float(q["bp"]), float(q["ap"])
        if not (0 < bid <= ask and q.get("bs", 0) > 0 and q.get("as", 0) > 0):
            excluded += 1
            continue
        valid.append(q)
        if (side == "LONG" and ask <= limit) or (side == "SHORT" and bid >= limit):
            compatible.append(q)
    spreads = np.array([float(q["ap"]) - float(q["bp"]) for q in valid])
    return {"quotes_total": len(quotes), "valid_positive_uncrossed": len(valid), "excluded_invalid_quotes": excluded,
            "compatible_with_limit_count": len(compatible), "first_compatible_quote": compatible[0] if compatible else None,
            "spread_p25_median_p75_quote_weighted": np.quantile(spreads, [.25, .5, .75]).tolist() if len(spreads) else [None]*3,
            "fill_possible_from_quotes": bool(compatible), "broker_fill_confirmed": False,
            "quote_size_units_not_interpreted_as_broker_capacity": True}


def run():
    output = DEFAULT_OUTPUT / "quote_access"
    diagnosis = read_signed(OUTPUT / "diagnosis.json")
    snapshot = read_signed(output / "provided_screenshot_pair.json")
    ledger_before = digest_file(ROOT / "data/portfolio.db")
    method = {"version": "QUOTE_WINDOW_DIAGNOSTIC_V1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_script_sha256": digest_file(Path(__file__)), "provider": "ALPACA", "feed": "sip",
              "rule": "All quotes from frozen evaluation start through possible OHLC fill bar +5m, inclusive. LONG ask<=limit; SHORT bid>=limit. Positive, uncrossed, both sizes positive. No edits to fill engine.",
              "diagnosis_sha256": sha(diagnosis), "snapshot_quote_sha256": sha(snapshot),
              "windows": [{"event_id": e["event_id"], "start": e["evaluation_starts_at"],
                            "end": (pd.Timestamp(next(x for x in snapshot["quotes"] if x["event_id"] == e["event_id"] and x["reference"] == "possible_fill_60")["reference_at"]) + pd.Timedelta(minutes=5)).isoformat()}
                           for e in diagnosis["symbols"]["SMCI"]["events"]]}
    write_signed(output / "quote_window_method.json", method)  # Fixed BEFORE downloading path.
    key = os.environ.get("APCA_API_KEY_ID") or getpass.getpass("Alpaca key (hidden): ")
    secret = os.environ.get("APCA_API_SECRET_KEY") or getpass.getpass("Alpaca secret (hidden): ")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    results = []
    try:
        for event, window in zip(diagnosis["symbols"]["SMCI"]["events"], method["windows"]):
            params = {"feed": "sip", "start": window["start"], "end": window["end"], "sort": "asc", "limit": 10000}
            quotes, pages, seen_tokens = [], 0, set()
            while True:
                response = session.get("https://data.alpaca.markets/v2/stocks/SMCI/quotes", params=params, timeout=30)
                if response.status_code != 200:
                    raise RuntimeError(f"Quotes HTTP {response.status_code}; details redacted")
                data = response.json()
                quotes.extend(data.get("quotes", []) or [])
                pages += 1
                token = data.get("next_page_token")
                if not token:
                    break
                if token in seen_tokens:
                    raise RuntimeError("Repeated pagination token.")
                seen_tokens.add(token)
                params["page_token"] = token
            raw = {"method_sha256": sha(method), "event_id": event["event_id"], "start": window["start"], "end": window["end"],
                   "pages": pages, "quotes": quotes}
            raw_path = output / "windows" / f"{event['event_id']}.json"
            write_signed(raw_path, raw)
            summary = {"event_id": event["event_id"], "side": event["side"], "cut_at": event["cut_at"], "limit": event["entry_price"],
                       "raw_file": str(raw_path.relative_to(DEFAULT_OUTPUT)), "raw_sha256": sha(raw),
                       **window, **inspect_quotes(quotes, event["side"], event["entry_price"], window["start"], window["end"])}
            results.append(summary)
            print("Audit", event["event_id"][:12], "quotes", len(quotes), "compatible", summary["compatible_with_limit_count"], flush=True)
    finally:
        session.close()
        del key, secret
    if digest_file(ROOT / "data/portfolio.db") != ledger_before:
        raise ValueError("Ledger changed; do not certify unchanged.")
    write_signed(output / "quote_window_audit.json", {"method_sha256": sha(method), "events": results,
                 "ledger_before_and_after_sha256": ledger_before, "no_parameters_or_filters_changed": True})


if __name__ == "__main__":
    run()
