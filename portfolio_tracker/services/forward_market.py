"""Raw provider snapshots for forward testing; no reconstructed past forecasts."""
from __future__ import annotations

from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from threading import RLock

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..analytics.closed_bars import OHLCV, select_last_closed_bar, utc
from .quant_market_data import normalize_symbol
from .zone_forward import canonical, digest

_DOWNLOAD_LOCK = RLock()


def _download(symbol, **kwargs):
    import yfinance as yf
    with _DOWNLOAD_LOCK:
        cache = DATA_DIR / "yfinance_cache"
        cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache))
        frame = yf.download(symbol, auto_adjust=False, prepost=False, threads=False,
                            progress=False, timeout=20, multi_level_index=False,
                            keepna=True, **kwargs)
    if frame is None or frame.empty:
        raise ValueError(f"Sin datos verificables de Yahoo para {symbol}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs(symbol, axis=1, level=-1)
    return frame.loc[:, list(OHLCV)].copy()


def daily_context(frame, now=None):
    """Closed daily/weekly/monthly EMA with explicit warm-up, daily Wilder ADX."""
    from ..analytics.technical_probability import _add_adx
    clock = utc(now)
    frame = select_last_closed_bar(frame, "1d", clock)
    contexts = {}
    for name, rule, timeframe in (("Diario", None, "1d"), ("Semanal", "W-FRI", "1wk"), ("Mensual", "ME", "1mo")):
        sampled = frame.copy() if rule is None else frame.resample(rule).agg(OHLCV).dropna()
        sampled = select_last_closed_bar(sampled, timeframe, clock)
        last = {"bars": len(sampled), "as_of": str(sampled.index[-1]) if len(sampled) else None}
        for span in (9, 21, 50, 200):
            ema = sampled.Close.ewm(span=span, adjust=False, min_periods=span).mean()
            last[f"EMA{span}"] = None if ema.empty or pd.isna(ema.iloc[-1]) else float(ema.iloc[-1])
        if name == "Diario":
            _add_adx(sampled)
            value = sampled["ADX14"].iloc[-1] if len(sampled) else np.nan
            last["ADX14"] = float(value) if pd.notna(value) and np.isfinite(value) else None
        contexts[name] = last
    return contexts


def download_daily_history(symbol="SMCI", *, now=None, cache_dir=None, force=False):
    """Cache signed closed daily bars since 2024. Never silently reuse stale data."""
    symbol = normalize_symbol(symbol)
    clock = utc(now)
    folder = Path(cache_dir) if cache_dir else DATA_DIR / "forward_market"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{symbol}_daily_2024.json"
    with _DOWNLOAD_LOCK:
        if target.exists() and not force:
            envelope = json.loads(target.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            age = clock - utc(payload["fetched_at"])
            if digest(payload) == envelope["sha256"] and pd.Timedelta(0) <= age < pd.Timedelta(hours=6):
                frame = pd.read_json(StringIO(payload["frame"]), orient="table")
                return frame, payload
        raw = _download(symbol, start="2024-01-01",
                        end=(clock + pd.Timedelta(days=1)).date().isoformat(), interval="1d")
        frame = select_last_closed_bar(raw, "1d", clock)
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Histórico diario vacío o con datos no finitos.")
        payload = {"symbol": symbol, "source": "yfinance/raw-unadjusted",
                   "fetched_at": clock.isoformat(), "requested_start": "2024-01-01",
                   "first_session": str(frame.index[0]), "last_session": str(frame.index[-1]),
                   "bars": len(frame), "frame": frame.to_json(orient="table", date_format="iso"),
                   "context": daily_context(frame, clock)}
        envelope = {"sha256": digest(payload), "payload": payload}
        # Atomic replacement, no partial file when multiple sessions refresh.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as temp:
            temp.write(canonical(envelope))
            temporary = temp.name
        os.replace(temporary, target)
        return frame, payload


def resolution_frames(symbol, day):
    """Exact session request; no current-spot fallback or downsampled daily proxy."""
    symbol = normalize_symbol(symbol)
    first = pd.Timestamp(day)
    end = (first + pd.Timedelta(days=1)).date().isoformat()
    return (_download(symbol, start=day, end=end, interval="5m"),
            _download(symbol, start=day, end=end, interval="1d"))
