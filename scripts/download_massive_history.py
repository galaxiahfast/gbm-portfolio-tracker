"""Acquire a fixed historical pilot, never live data or accounting records.

Only run after pilot_massive_sample.py has passed. Raw responses are retained
with SHA-256 and resumable requests; no API key is written to disk. No training.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import getpass
import hashlib
import json
import os
from pathlib import Path
import time

import exchange_calendars as xcals
import pandas as pd
import requests

from scripts.pilot_massive_sample import ROOT, SYMBOLS, _get_json


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_signed(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_signed(path) != payload:
            raise ValueError(f"Artefacto inmutable ya existente: {path.name}")
        return
    envelope = {"payload": payload, "sha256": sha(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def read_signed(path):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("sha256") != sha(envelope.get("payload")):
        raise ValueError(f"SHA-256 inválido: {path.name}")
    return envelope["payload"]


class Client:
    def __init__(self, key, folder):
        self.key, self.folder = key, folder
        self.http = requests.Session()
        self.last_call = 0.0

    def get(self, name, path, params):
        destination = self.folder / f"{name}.json"
        query = {"path": path, "params": params}
        if destination.exists():
            cached = read_signed(destination)
            if cached["query"] != query:
                raise ValueError("No se permite reemplazar una consulta histórica distinta.")
            return cached["response"]
        time.sleep(max(0, 12.5 - (time.monotonic() - self.last_call)))
        self.last_call = time.monotonic()
        response = _get_json(self.http, path, params, self.key)
        write_signed(destination, {
            "query": query, "response": response,
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        print(f"Adquirido: {name} ({len(response.get('results') or [])} filas)", flush=True)
        return response


def audit_intraday(responses, schedule, symbol=None):
    if any(response.get("adjusted") is not True
           or (symbol is not None and response.get("ticker") != symbol)
           for response in responses):
        raise ValueError("Respuesta histórica sin identidad/ajuste confirmado.")
    bars = [row for response in responses for row in response.get("results", [])]
    frame = pd.DataFrame(bars).rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
    })
    frame.index = pd.to_datetime(frame.pop("t"), unit="ms", utc=True)
    expected = pd.DatetimeIndex([])
    for row in schedule.itertuples():
        expected = expected.union(pd.date_range(row.open, row.close, freq="5min", inclusive="left"))
    regular = frame.loc[frame.index.isin(expected), ["Open", "High", "Low", "Close", "Volume"]]
    duplicates = int(regular.index.duplicated().sum())
    missing = expected.difference(regular.index)
    import numpy as np
    invalid = (~np.isfinite(regular.to_numpy())).any(axis=1)
    invalid |= (regular[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    invalid |= regular.Volume <= 0
    invalid |= regular.High < regular[["Open", "Close", "Low"]].max(axis=1)
    invalid |= regular.Low > regular[["Open", "Close", "High"]].min(axis=1)
    audit = {
        "sessions": len(schedule), "expected_bars": len(expected), "regular_bars": len(regular),
        "missing_count": len(missing), "missing_ny": [x.tz_convert("America/New_York").isoformat() for x in missing],
        "duplicate_count": duplicates, "invalid_ohlcv_count": int(invalid.sum()),
        "excluded_nonregular_bars": len(frame) - len(regular),
        "calendar": "XNYS", "timezone": "America/New_York", "timestamp_semantics": "bar_open",
    }
    if duplicates or len(missing) or invalid.any():
        raise ValueError(f"Auditoría 5m rechazada: {json.dumps(audit)}")
    return regular.sort_index(), audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-session", type=date.fromisoformat, required=True)
    parser.add_argument("--sessions", type=int, default=400, choices=range(350, 401))
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args(argv)
    key = getpass.getpass("Clave Massive (oculta): ") if args.prompt_key else os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("Falta MASSIVE_API_KEY.")
    # Basic's rolling two-year entitlement. Leave two calendar days at the
    # boundary rather than silently retry an older, non-entitled date.
    first = pd.Timestamp.now(tz="America/New_York").date().replace(
        year=pd.Timestamp.now(tz="America/New_York").year - 2
    ) + timedelta(days=2)
    calendar = xcals.get_calendar("XNYS", start=first.isoformat(), end=args.end_session.isoformat())
    daily_schedule = calendar.schedule
    schedule = daily_schedule.tail(args.sessions)
    if len(schedule) != args.sessions:
        raise ValueError("La ventana Basic no contiene las sesiones solicitadas.")
    start, end = schedule.index[0].date(), schedule.index[-1].date()
    folder = ROOT / "output" / "historical_acquisition" / f"massive_{args.sessions}_{end}"
    protocol = {
        "provider": "Massive", "plan": "Stocks Basic", "advertised_cost_usd_month": 0,
        "symbols": list(SYMBOLS), "intraday_start": str(start), "end": str(end),
        "daily_start": str(daily_schedule.index[0].date()), "requested_sessions": args.sessions,
        "cut": "11:00 America/New_York", "horizons": [60, 1440], "no_training": True,
    }
    write_signed(folder / "acquisition_protocol.json", protocol)
    client = Client(key, folder / "raw")
    summaries = {}
    for symbol in SYMBOLS:
        splits = client.get(f"{symbol}_splits", "/v3/reference/splits", {
            "ticker": symbol, "execution_date.gte": protocol["daily_start"],
            "execution_date.lte": str(end), "limit": 100,
        }).get("results", [])
        if any(str(start) <= row["execution_date"] <= str(end) for row in splits):
            raise ValueError(f"{symbol}: split durante el replay; requiere normalización punto-en-tiempo.")
        responses, cursor = [], start
        while cursor <= end:
            stop = min(cursor + timedelta(days=44), end)
            responses.append(client.get(
                f"{symbol}_5m_{cursor}_{stop}",
                f"/v2/aggs/ticker/{symbol}/range/5/minute/{cursor}/{stop}",
                {"adjusted": "true", "sort": "asc", "limit": 50000},
            ))
            cursor = stop + timedelta(days=1)
        frame, audit = audit_intraday(responses, schedule, symbol)
        daily = {}
        for adjusted in (True, False):
            daily[str(adjusted).lower()] = client.get(
                f"{symbol}_daily_adjusted_{adjusted}",
                f"/v2/aggs/ticker/{symbol}/range/1/day/{protocol['daily_start']}/{end}",
                {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
            )
        audit["splits"] = splits
        audit["all_splits_before_intraday_start"] = True
        audit["regular_5m_frame"] = frame.to_json(orient="table", date_format="iso")
        audit["daily_adjusted_response"] = daily["true"]
        audit["daily_raw_response"] = daily["false"]
        write_signed(folder / f"{symbol}_acquired.json", audit)
        summaries[symbol] = {k: v for k, v in audit.items() if not k.endswith("response") and k != "regular_5m_frame"}
        print(f"AUDITADO {symbol}: {len(schedule)} sesiones, {len(frame)} velas regulares, cero huecos.", flush=True)
    write_signed(folder / "acquisition_summary.json", summaries)
    print(f"ADQUISICIÓN COMPLETA: {folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
