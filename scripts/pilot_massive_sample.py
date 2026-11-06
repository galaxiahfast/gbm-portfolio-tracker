"""Audit five real XNYS sessions per symbol before any bulk historical import.

Requires MASSIVE_API_KEY in the process environment. This script never touches
SQLite, the live market cache, or the portfolio ledger. A successful run writes
only signed sample artifacts under output/historical_provider_samples/.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.massive.com"
NY = ZoneInfo("America/New_York")
SYMBOLS = ("SMCI", "NVDA")


def _sessions(end_session: date | None = None) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    today_ny = pd.Timestamp.now(tz=NY).date()
    end = end_session or (today_ny - timedelta(days=1))
    if end >= today_ny:
        raise ValueError("La muestra debe terminar antes de la sesión NY actual (plan Basic EOD).")
    calendar = xcals.get_calendar(
        "XNYS", start=(end - timedelta(days=30)).isoformat(), end=end.isoformat()
    )
    schedule = calendar.schedule
    sessions = list(schedule.index[-5:])
    if len(sessions) != 5:
        raise ValueError("El calendario XNYS no ofrece cinco sesiones completas.")
    return schedule, sessions


def _get_json(session: requests.Session, path: str, params: dict, api_key: str) -> dict:
    # Do not put the URL or exception text in error messages: they may contain the key.
    try:
        response = session.get(
            f"{API_ROOT}{path}", params={**params, "apiKey": api_key}, timeout=40
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Massive no responde ({type(exc).__name__}).") from None
    if response.status_code != 200:
        raise RuntimeError(f"Massive devolvió HTTP {response.status_code}; no se importó la muestra.")
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("Massive devolvió JSON inválido.") from None
    if not isinstance(payload, dict) or payload.get("status") not in ("OK", "DELAYED"):
        raise RuntimeError("Massive no confirmó la consulta de datos.")
    if payload.get("next_url"):
        raise RuntimeError("Muestra paginada inesperadamente; no se acepta una descarga parcial.")
    return payload


def audit_symbol(
    symbol: str,
    schedule: pd.DataFrame,
    sessions: list[pd.Timestamp],
    http: requests.Session,
    api_key: str,
) -> dict:
    start, end = sessions[0].date().isoformat(), sessions[-1].date().isoformat()
    aggregates = _get_json(
        http,
        f"/v2/aggs/ticker/{symbol}/range/5/minute/{start}/{end}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
        api_key,
    )
    if aggregates.get("ticker") != symbol or aggregates.get("adjusted") is not True:
        raise ValueError(f"{symbol}: respuesta sin identidad o ajuste por splits confirmado.")
    splits = _get_json(
        http,
        "/v3/reference/splits",
        {"ticker": symbol, "execution_date.gte": start, "execution_date.lte": end, "limit": 100},
        api_key,
    )
    split_rows = splits.get("results") or []
    if not isinstance(split_rows, list):
        raise ValueError(f"{symbol}: listado de splits inválido.")
    # If a split falls inside the pilot, adjusted=true alone is insufficient to
    # verify its factor. Require a separate raw/adjusted reconciliation first.
    if split_rows:
        raise ValueError(f"{symbol}: split dentro de la muestra; requiere cotejo raw/adjusted.")

    expected = set()
    for day in sessions:
        market_open = pd.Timestamp(schedule.loc[day, "open"])
        market_close = pd.Timestamp(schedule.loc[day, "close"])
        expected.update(pd.date_range(market_open, market_close, freq="5min", inclusive="left"))

    bars = []
    observed = set()
    excluded_extended_hours = 0
    for item in aggregates.get("results") or []:
        if not isinstance(item, dict):
            raise ValueError(f"{symbol}: vela no estructurada.")
        stamp = pd.Timestamp(int(item["t"]), unit="ms", tz="UTC")
        if stamp not in expected:
            excluded_extended_hours += 1
            continue
        if stamp in observed:
            raise ValueError(f"{symbol}: vela duplicada en {stamp.isoformat()}.")
        observed.add(stamp)
        try:
            o, h, l, c, v = (float(item[key]) for key in ("o", "h", "l", "c", "v"))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{symbol}: OHLCV incompleto en {stamp.isoformat()}.") from None
        if not all(math.isfinite(value) for value in (o, h, l, c, v)):
            raise ValueError(f"{symbol}: OHLCV no finito en {stamp.isoformat()}.")
        if not (0 < l <= min(o, c) <= max(o, c) <= h and v > 0):
            raise ValueError(f"{symbol}: OHLCV inconsistente en {stamp.isoformat()}.")
        local = stamp.tz_convert(NY)
        bars.append({
            "timestamp_utc": stamp.isoformat(),
            "timestamp_ny": local.isoformat(),
            "session": local.date().isoformat(),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })

    missing = sorted(expected - observed)
    if missing:
        sample = ", ".join(ts.tz_convert(NY).isoformat() for ts in missing[:5])
        raise ValueError(f"{symbol}: {len(missing)} huecos 5m regulares; primeros: {sample}.")
    bars.sort(key=lambda bar: bar["timestamp_utc"])
    counts = {day.date().isoformat(): 0 for day in sessions}
    for bar in bars:
        counts[bar["session"]] += 1
    payload = {
        "source": "Massive Stocks Basic",
        "source_endpoint": "/v2/aggs/ticker/{symbol}/range/5/minute/{from}/{to}",
        "symbol": symbol,
        "calendar": "XNYS",
        "timezone": "America/New_York",
        "timestamp_semantics": "bar_open",
        "split_policy": "adjusted=true; no corporate split in the sampled sessions",
        "sessions": counts,
        "excluded_extended_hours": excluded_extended_hours,
        "bars": bars,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "payload": payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "provider_request_ids": {
            "aggregates": aggregates.get("request_id"),
            "splits": splits.get("request_id"),
        },
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-session", type=date.fromisoformat)
    parser.add_argument("--prompt-key", action="store_true", help="Read the key without echoing or storing it.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output" / "historical_provider_samples",
    )
    args = parser.parse_args(argv)
    api_key = (getpass.getpass("Clave Massive (oculta): ") if args.prompt_key
               else os.environ.get("MASSIVE_API_KEY", "")).strip()
    if not api_key:
        print("FALTA MASSIVE_API_KEY: no se descargó ni firmó ningún dato.", file=sys.stderr)
        return 2
    try:
        schedule, sessions = _sessions(args.end_session)
        with requests.Session() as http:
            envelopes = {
                symbol: audit_symbol(symbol, schedule, sessions, http, api_key)
                for symbol in SYMBOLS
            }
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"MUESTRA RECHAZADA: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for symbol, envelope in envelopes.items():
        target = args.output_dir / f"massive_{symbol}_{sessions[0].date()}_{sessions[-1].date()}.json"
        target.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = envelope["payload"]["bars"]
        print(json.dumps({
            "symbol": symbol,
            "artifact": str(target.resolve()),
            "sessions": envelope["payload"]["sessions"],
            "bars": len(rows),
            "first_bar": rows[0],
            "last_bar": rows[-1],
            "sha256": envelope["sha256"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
