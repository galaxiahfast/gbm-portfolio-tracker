"""Incremental input cache; calls existing normalization/closed-bar rules unchanged."""
from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar, select_last_closed_bar, utc
from portfolio_tracker.services.forward_market import _download
from portfolio_tracker.services.quant_market_data import _normalize_frame
from portfolio_tracker.services.zone_forward import digest
from scripts.autopilot_runtime import atomic_json


class MarketCache:
    def __init__(self, folder, download=None):
        self.folder = Path(folder)
        self.download = download or _download

    def _read(self, path):
        if not path.exists():
            return None, None
        content = json.loads(path.read_text(encoding="utf-8"))
        if digest(content["payload"]) != content["sha256"]:
            raise ValueError(f"Caché alterada: {path.name}; se requiere revisión, no se utiliza.")
        return pd.read_json(StringIO(content["payload"]["frame"]), orient="table"), content["payload"]

    def _write(self, path, frame, now, request):
        payload = {"frame": frame.to_json(orient="table", date_format="iso"),
                   "fetched_at": utc(now).isoformat(), "request": request,
                   "source": "yfinance/raw-unadjusted"}
        envelope = {"payload": payload, "sha256": digest(payload)}
        # Immutable acquisition archive + atomic current pointer/data file.
        archive = path.parent / "archive" / f"{path.stem}_{envelope['sha256']}.json"
        if not archive.exists():
            atomic_json(archive, envelope)
        atomic_json(path, envelope)

    def artifact_refs(self, symbol):
        """Verify and describe the immutable inputs used by this collection cut.

        These references are stored in the signed forecast snapshot. A fully
        mocked cache with no files remains feature-only; a partial or altered
        production cache fails closed rather than claiming replay evidence.
        """
        root = self.folder / symbol
        names = {"5m": "5m", "1d": "daily"}
        paths = {key: root / f"{stem}.json" for key, stem in names.items()}
        if not any(path.exists() for path in paths.values()):
            return {}
        if not all(path.exists() for path in paths.values()):
            raise ValueError(f"Archivos de mercado incompletos para {symbol}.")
        refs = {}
        for key, stem in names.items():
            current = json.loads(paths[key].read_text(encoding="utf-8"))
            sha = current["sha256"]
            if digest(current["payload"]) != sha:
                raise ValueError(f"Caché de mercado alterada para {symbol}/{key}.")
            archive = root / "archive" / f"{stem}_{sha}.json"
            if not archive.is_file():
                raise ValueError(f"Falta el archivo inmutable para {symbol}/{key}.")
            saved = json.loads(archive.read_text(encoding="utf-8"))
            if saved != current or digest(saved["payload"]) != sha:
                raise ValueError(f"Archivo inmutable alterado para {symbol}/{key}.")
            frame = pd.read_json(StringIO(saved["payload"]["frame"]), orient="table")
            refs[key] = {
                "archive_path": archive.relative_to(root).as_posix(),
                "sha256": sha,
                "source": saved["payload"]["source"],
                "fetched_at": saved["payload"]["fetched_at"],
                "rows": len(frame),
            }
        return refs

    def verify_artifact_refs(self, symbol, refs):
        """Recheck archived acquisition files from a later signed execution.

        Unlike artifact_refs, this does not inspect moving `5m.json` or
        `daily.json` pointers, so historical cuts remain independently
        verifiable after the next market-data refresh.
        """
        root = (self.folder / symbol).resolve()
        if set(refs) != {"5m", "1d"}:
            return False
        for key, ref in refs.items():
            try:
                sha = ref["sha256"]
                archive = (root / ref["archive_path"]).resolve()
                if (not isinstance(sha, str) or len(sha) != 64
                        or any(char not in "0123456789abcdef" for char in sha)
                        or not archive.is_relative_to(root)
                        or archive.parent != root / "archive"
                        or archive.name != f"{'5m' if key == '5m' else 'daily'}_{sha}.json"):
                    return False
                saved = json.loads(archive.read_text(encoding="utf-8"))
                if saved["sha256"] != sha or digest(saved["payload"]) != sha:
                    return False
                frame = pd.read_json(StringIO(saved["payload"]["frame"]), orient="table")
                if (len(frame) != ref["rows"] or saved["payload"]["source"] != ref["source"]
                        or saved["payload"]["fetched_at"] != ref["fetched_at"]):
                    return False
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                return False
        return True

    def frames(self, symbol, now):
        clock = utc(now)
        local = clock.tz_convert(NY)
        month_start = (local - pd.DateOffset(months=1)).normalize()
        root = self.folder / symbol
        intra_path, daily_path = root / "5m.json", root / "daily.json"
        cached, _ = self._read(intra_path)
        request = {"interval": "5m"}
        if cached is None or cached.empty:
            request["period"] = "1mo"  # Exact UI bootstrap window.
            fresh = self.download(symbol, **request)
            combined = _normalize_frame(fresh, symbol)
        else:
            cached = cached.copy()
            cached.index = pd.DatetimeIndex(cached.index).tz_convert(NY)
            cached = cached.loc[cached.index >= month_start]
            if cached.empty:
                request["period"] = "1mo"
            else:
                # Re-fetch overlap to complete the previously forming candle and
                # revise provider corrections. Also repair earlier missing buckets.
                first = max(month_start, cached.index[-1] - pd.Timedelta(days=1))
                schedule = _calendar(month_start.year - 1, clock.year + 1).schedule
                expected = []
                for day, session in schedule.loc[str(month_start.date()):str(local.date())].iterrows():
                    end = min(utc(session["close"]), clock.floor("5min"))
                    if end > utc(session["open"]):
                        expected.extend(pd.date_range(session["open"], end, freq="5min", inclusive="left"))
                missing = pd.DatetimeIndex(expected).difference(cached.index.tz_convert("UTC"))
                if len(missing):
                    first = min(first, missing[0].tz_convert(NY))
                request["start"] = first.to_pydatetime()
                request["end"] = clock.to_pydatetime()
            fresh = _normalize_frame(self.download(symbol, **request), symbol)
            fresh.index = pd.DatetimeIndex(fresh.index).tz_convert(NY)
            overlap = cached.index.intersection(fresh.index)
            if len(overlap) and ((fresh.loc[overlap, "Close"] / cached.loc[overlap, "Close"] - 1).abs() > .01).any():
                # Large historical revision/split: do not mix price bases in cache.
                request = {"period": "1mo", "interval": "5m"}
                combined = _normalize_frame(self.download(symbol, **request), symbol)
            else:
                combined = pd.concat([cached, fresh])
            combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
        combined.index = pd.DatetimeIndex(combined.index).tz_convert(NY)
        combined = combined.loc[combined.index >= month_start]
        closed = select_last_closed_bar(combined, "5m", clock)
        self._write(intra_path, closed, clock, {k: str(v) for k, v in request.items()})
        daily, meta = self._read(daily_path)
        schedule = _calendar(local.year - 1, local.year + 1).schedule
        completed = schedule.loc[schedule["close"] <= clock]
        if completed.empty:
            raise ValueError("No hay sesión diaria XNYS cerrada para validar el contexto.")
        required_day = completed.index[-1].date()
        cached_day = pd.Timestamp(daily.index[-1]).date() if daily is not None and not daily.empty else None
        # Five years like the UI, not the shorter validation-context cache.
        if (daily is None or cached_day != required_day
                or utc(meta["fetched_at"]).tz_convert(NY).date() != local.date()):
            downloaded = _normalize_frame(self.download(symbol, period="5y", interval="1d"), symbol)
            daily = select_last_closed_bar(downloaded, "1d", clock)
            newest_day = pd.Timestamp(daily.index[-1]).date() if not daily.empty else None
            if newest_day != required_day:
                # A provider lag is not a valid daily context. Do not mark the
                # partial fetch as today's cache; the next scheduled attempt
                # must actually fetch again, still before its 11:05 deadline.
                raise ValueError(
                    f"Contexto 1d incompleto: último {newest_day}; requerido {required_day}."
                )
            self._write(daily_path, daily, clock, {"period": "5y", "interval": "1d"})
        return closed, select_last_closed_bar(daily, "1d", clock)
