"""Research-only, append-only prospective archive primitives.

No portfolio database, order API, Streamlit, or retrospective emission is used.
Provider credentials must come from the process environment and are never saved.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests

from scripts.download_massive_history import read_signed, sha, write_signed


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output" / "prospective_archive"
SYMBOLS = ("SMCI", "NVDA")
HORIZONS = {"1h": 60, "1D": 1_440}
CUT_TIMES = ("10:00", "11:00", "12:30", "14:00", "15:30")
NY = ZoneInfo("America/New_York")
POLICY_VERSION = "XNYS_FIVE_CUT_ALPACA_IEX_RESEARCH_V1"
FEATURE_VERSION = "PROSPECTIVE_TECHNICAL_FINBERT_IEX_V1"
SNAPSHOT_VERSION = "SIGNED_PROSPECTIVE_SNAPSHOT_V1"
LABEL_VERSION = "SIGNED_PROSPECTIVE_TRIPLE_BARRIER_V1"
FEED = "iex"
ADJUSTMENT = "split"
DATA_SOURCE = "ALPACA_IEX_SPLIT_ADJUSTED"
GRACE = timedelta(minutes=5)
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks"


def utc(value=None) -> datetime:
    stamp = datetime.now(timezone.utc) if value is None else value
    if isinstance(stamp, pd.Timestamp):
        stamp = stamp.to_pydatetime()
    if not isinstance(stamp, datetime) or stamp.tzinfo is None:
        raise ValueError("Se requiere timestamp con zona horaria.")
    return stamp.astimezone(timezone.utc)


def active_cut(at=None) -> tuple[datetime, str] | None:
    """Return an XNYS cut only during its prospective five-minute grace."""
    clock = utc(at)
    local = clock.astimezone(NY)
    calendar = xcals.get_calendar("XNYS", start=f"{local.year-1}-01-01",
                                  end=f"{local.year+1}-12-31")
    day = pd.Timestamp(local.date())
    if day not in calendar.schedule.index:
        return None
    session = calendar.schedule.loc[day]
    for label in CUT_TIMES:
        hour, minute = map(int, label.split(":"))
        cut = datetime.combine(local.date(), time(hour, minute), NY).astimezone(timezone.utc)
        if (session.open.to_pydatetime() <= cut < session.close.to_pydatetime()
                and cut <= clock < cut + GRACE):
            return cut, label
    return None


def snapshot_path(symbol: str, cut: datetime, horizon: str) -> Path:
    symbol = symbol.upper()
    if symbol not in SYMBOLS or horizon not in HORIZONS:
        raise ValueError("Identidad de snapshot no soportada.")
    local = utc(cut).astimezone(NY)
    label = local.strftime("%Y%m%d_%H%M")
    return ARCHIVE / "snapshots" / local.strftime("%Y-%m-%d") / f"{label}_{symbol}_{horizon}.json"


def label_path(snapshot: Path) -> Path:
    relative = snapshot.relative_to(ARCHIVE / "snapshots")
    return ARCHIVE / "labels" / relative


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_bars(rows: list[dict], timeframe: str) -> pd.DataFrame:
    """Validate Alpaca OHLCV; timestamps denote bar OPEN."""
    if timeframe not in {"5Min", "1Day"} or not isinstance(rows, list) or not rows:
        raise ValueError("Respuesta de velas vacía o temporalidad desconocida.")
    frame = pd.DataFrame(rows)
    if not {"t", "o", "h", "l", "c", "v"} <= set(frame.columns):
        raise ValueError("Alpaca devolvió OHLCV incompleto.")
    stamps = pd.to_datetime(frame["t"], utc=True, errors="raise")
    if timeframe == "1Day":
        index = pd.DatetimeIndex(stamps.dt.date)
    else:
        index = pd.DatetimeIndex(stamps)
        if any(index.minute % 5) or any(index.second) or any(index.microsecond):
            raise ValueError("Vela 5m fuera de la rejilla.")
    result = frame.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                   "c": "Close", "v": "Volume"})
    result = result.loc[:, ["Open", "High", "Low", "Close", "Volume"]].astype(float)
    result.index = index
    result = result.sort_index()
    if result.index.has_duplicates or not np.isfinite(result.to_numpy()).all():
        raise ValueError("Velas duplicadas o no finitas.")
    if ((result[["Open", "High", "Low", "Close"]] <= 0).any().any()
            or (result.Volume <= 0).any()
            or (result.High < result[["Open", "Close", "Low"]].max(axis=1)).any()
            or (result.Low > result[["Open", "Close", "High"]].min(axis=1)).any()):
        raise ValueError("OHLCV inconsistente.")
    return result


class AlpacaBars:
    """Read-only IEX research feed; never uses the trading/order endpoint.

    This account returned SIP HTTP 403 and IEX HTTP 200 on 2026-09-30.
    SIP would require a separate versioned cohort, never silent mixing.
    """

    def __init__(self, key: str, secret: str, http=None):
        if not key or not secret:
            raise ValueError("Faltan credenciales de Alpaca.")
        self.http = http or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime):
        if symbol not in SYMBOLS or timeframe not in {"5Min", "1Day"}:
            raise ValueError("Consulta de barras no registrada.")
        params = {"feed": FEED, "adjustment": ADJUSTMENT, "timeframe": timeframe,
                  "start": utc(start).isoformat(), "end": utc(end).isoformat(),
                  "limit": 10000, "sort": "asc"}
        rows, pages, seen = [], [], set()
        while True:
            try:
                response = self.http.get(f"{ALPACA_DATA_URL}/{symbol}/bars",
                                         params=params, timeout=40)
            except requests.RequestException as exc:
                raise RuntimeError(f"Alpaca no respondió ({type(exc).__name__}); credenciales ocultas.") from None
            if response.status_code != 200:
                raise RuntimeError(f"Alpaca IEX HTTP {response.status_code}; sin fallback a SIP/datos atrasados.")
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError("Alpaca devolvió JSON inválido.") from None
            if not isinstance(payload, dict) or payload.get("symbol") != symbol:
                raise ValueError("Respuesta Alpaca sin identidad del activo.")
            part = payload.get("bars") or []
            if not isinstance(part, list):
                raise ValueError("Formato de barras Alpaca inválido.")
            rows.extend(part)
            pages.append({"row_count": len(part), "received_at_utc": utc().isoformat()})
            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen:
                raise ValueError("Token de paginación Alpaca repetido.")
            seen.add(token)
            params["page_token"] = token
        return clean_bars(rows, timeframe), {"provider": "Alpaca", "feed": FEED,
            "adjustment": ADJUSTMENT, "timeframe": timeframe,
            "retrieved_at_utc": utc().isoformat(), "pages": pages,
            "raw_rows_sha256": sha(rows), "row_count": len(rows)}


def fetch_current_headlines(http, key: str, symbol: str, at=None):
    """Stamp *our receipt time* after a successful HTTP response; never backdate."""
    if not key or symbol not in SYMBOLS:
        raise ValueError("Finnhub key/activo no disponible.")
    clock = utc(at)
    first_day = (clock - timedelta(hours=24)).date()
    last_day = clock.date()
    parts, rows = [], []
    day = first_day
    while day <= last_day:
        try:
            response = http.get(FINNHUB_URL,
                                params={"symbol": symbol, "from": day.isoformat(),
                                        "to": day.isoformat(), "token": key}, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"Finnhub no respondió ({type(exc).__name__}); clave oculta.") from None
        received = utc()
        if response.status_code != 200:
            raise RuntimeError(f"Finnhub HTTP {response.status_code}; no se equipara error con ausencia de noticias.")
        try:
            raw = response.json()
        except ValueError:
            raise RuntimeError("Finnhub devolvió JSON inválido.") from None
        if not isinstance(raw, list):
            raise ValueError("Finnhub no devolvió una lista de noticias.")
        parts.append({"day_utc": day.isoformat(), "raw_count": len(raw),
                      "received_at_utc": received.isoformat(),
                      "raw_sha256": sha(raw), "possible_provider_cap": len(raw) >= 250})
        rows.extend((item, received) for item in raw)
        day += timedelta(days=1)
    final_received = utc()
    selected = {}
    for item, received in rows:
        if not isinstance(item, dict):
            continue
        stamp, title = item.get("datetime"), item.get("headline")
        if isinstance(stamp, bool) or not isinstance(stamp, int) or not isinstance(title, str) or not title.strip():
            continue
        try:
            published = datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            continue
        related = item.get("related")
        if not isinstance(related, str) or symbol not in {x.strip().upper() for x in related.split(",")}:
            continue
        if not final_received - timedelta(hours=24) < published <= received:
            continue
        identity = str(item.get("id", ""))
        if not identity:
            continue
        article = {"id": identity, "symbol": symbol, "headline": title.strip(),
                   "published_at_utc": published.isoformat(),
                   "collected_at_utc": received.isoformat(),
                   "source": str(item.get("source") or ""),
                   "related": related,
                   "availability_policy": "FIRST_OBSERVED_BY_OUR_COLLECTOR"}
        if identity in selected and selected[identity]["headline"] != article["headline"]:
            raise ValueError("Finnhub devolvió dos versiones de un titular en la misma captura.")
        if identity not in selected:
            selected[identity] = article  # earliest receipt from these partitioned requests
    return sorted(selected.values(), key=lambda x: (x["published_at_utc"], x["id"])), {
        "provider": "Finnhub company-news", "retrieved_at_utc": final_received.isoformat(),
        "raw_count": len(rows), "retained_count": len(selected),
        "requests": parts, "possible_truncation": any(part["possible_provider_cap"] for part in parts),
        "raw_response_sha256": sha(parts),
        "availability_policy": "FIRST_OBSERVED_BY_OUR_COLLECTOR"}


def signed_payload(path: Path):
    return read_signed(path)


def write_payload(path: Path, payload):
    write_signed(path, payload)
