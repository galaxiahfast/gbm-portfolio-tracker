"""Two-date, two-symbol provenance pilot for SentiSense's documented range API.

Set SENTISENSE_API_KEY locally. Never pass a key as a CLI argument and never
print it. This pilot does not join scores to replay or train a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_massive_history import read_signed, write_signed

OUTPUT = ROOT / "output/historical_replay/news_sentiment_v1/pilot"
SYMBOLS = ("SMCI", "NVDA")
DATES = ("2026-05-05", "2026-09-25")
URL = "https://app.sentisense.ai/api/v1/documents/ticker/{symbol}/range"
AVAILABILITY_FIELDS = ("sentimentAvailableAt", "sentiment_available_at",
                       "scoredAt", "scored_at", "firstSeenAt", "first_seen_at")


def run():
    key = os.environ.get("SENTISENSE_API_KEY")
    if not key:
        raise RuntimeError("Configure SENTISENSE_API_KEY locally; do not paste it in chat.")
    session = requests.Session()
    reports = []
    for symbol in SYMBOLS:
        for day in DATES:
            response = session.get(URL.format(symbol=symbol),
                                   headers={"X-SentiSense-API-Key": key},
                                   params={"startDate": day, "endDate": day,
                                           "source": "NEWS", "limit": 200},
                                   timeout=20)
            if response.status_code != 200:
                # Do not log response text: vendor errors may echo credentials.
                raise RuntimeError(f"SentiSense HTTP {response.status_code} for {symbol} {day}.")
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
                raise RuntimeError(f"Unexpected SentiSense schema for {symbol} {day}.")
            docs = data["documents"]
            total = data.get("totalCount")
            if total is not None and (not isinstance(total, int) or total < len(docs)):
                raise RuntimeError(f"Invalid totalCount for {symbol} {day}.")
            raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
            raw_hash = hashlib.sha256(raw.encode()).hexdigest()
            available_fields = sorted({name for doc in docs for name in AVAILABILITY_FIELDS
                                       if name in doc and doc[name] is not None})
            payload = {"source": "SENTISENSE_DOCUMENTS_NEWS", "symbol": symbol,
                       "day_ny": day, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                       "response_sha256": raw_hash, "total_count": total,
                       "documents_returned": len(docs),
                       "coverage_complete": total == len(docs) if total is not None else False,
                       "score_availability_fields": available_fields,
                       "pit_proven_for_historical_11_ny": False,
                       "raw_response": data}
            OUTPUT.mkdir(parents=True, exist_ok=True)
            destination = OUTPUT / f"{symbol}_{day}_{raw_hash[:12]}.json"
            if destination.exists():
                previous = read_signed(destination)
                if previous["response_sha256"] != raw_hash or previous["raw_response"] != data:
                    raise RuntimeError("Previously signed provider response differs.")
                payload = previous
            else:
                write_signed(destination, payload)
            reports.append({name: payload[name] for name in (
                "symbol", "day_ny", "response_sha256", "total_count",
                "documents_returned", "coverage_complete", "score_availability_fields",
                "pit_proven_for_historical_11_ny")})
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return reports


if __name__ == "__main__":
    run()
