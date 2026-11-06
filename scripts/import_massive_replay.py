"""Verify acquired responses and import isolated, signed replay input frames."""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_massive_history import ROOT, SYMBOLS, read_signed, sha, write_signed
from portfolio_tracker.analytics.closed_bars import _calendar


def verify_daily(acquired, expected_sessions):
    adjusted = acquired["daily_adjusted_response"]
    raw = acquired["daily_raw_response"]
    if adjusted.get("adjusted") is not True or raw.get("adjusted") is not False:
        raise ValueError("El proveedor no confirmó ambas bases de precios diarios.")
    if len(adjusted["results"]) != len(raw["results"]):
        raise ValueError("Cobertura diaria raw/adjusted diferente.")
    raw_by_time = {row["t"]: row for row in raw["results"]}
    rows, labels, maximum_error = [], [], 0.0
    for row in adjusted["results"]:
        other = raw_by_time[row["t"]]
        day = pd.Timestamp(row["t"], unit="ms", tz="UTC").tz_convert("America/New_York").date()
        factor = np.prod([
            event["split_to"] / event["split_from"]
            for event in acquired["splits"] if str(day) < event["execution_date"]
        ])
        for key in ("o", "h", "l", "c"):
            error = abs(other[key] / factor - row[key])
            maximum_error = max(maximum_error, error)
            if not np.isclose(other[key] / factor, row[key], rtol=2e-6, atol=1e-4):
                raise ValueError(f"Factor de split incompatible en {day}, {key}.")
        if not np.isclose(other["v"] * factor, row["v"], rtol=2e-6, atol=1e-3):
            raise ValueError(f"Volumen sin ajuste coherente en {day}.")
        labels.append(pd.Timestamp(day))
        rows.append([row[key] for key in ("o", "h", "l", "c", "v")])
    daily = pd.DataFrame(rows, index=pd.DatetimeIndex(labels),
                         columns=["Open", "High", "Low", "Close", "Volume"]).sort_index()
    if daily.index.duplicated().any() or not daily.index.equals(expected_sessions):
        raise ValueError("Cobertura diaria incompatible con XNYS o timestamps duplicados.")
    if not np.isfinite(daily.to_numpy()).all():
        raise ValueError("Datos diarios no finitos.")
    if ((daily[["Open", "High", "Low", "Close"]] <= 0).any().any()
            or (daily.Volume <= 0).any()
            or (daily.High < daily[["Open", "Close", "Low"]].max(axis=1)).any()
            or (daily.Low > daily[["Open", "Close", "High"]].min(axis=1)).any()):
        raise ValueError("OHLCV diario fuera de dominio.")
    return daily, {"daily_sessions": len(daily), "maximum_split_price_rounding_error": maximum_error,
                   "price_and_volume_split_factors_verified": True}


def import_acquisition(folder, output):
    protocol = read_signed(folder / "acquisition_protocol.json")
    summary = read_signed(folder / "acquisition_summary.json")
    schedule = _calendar(2024, 2027).schedule.loc[protocol["daily_start"]:protocol["end"]]
    manifest = {"role": "HISTORICAL_REPLAY_INPUT_NOT_LIVE", "protocol": protocol, "symbols": {}}
    for symbol in SYMBOLS:
        acquired = read_signed(folder / f"{symbol}_acquired.json")
        if (symbol not in summary or acquired["missing_count"] or acquired["duplicate_count"]
                or acquired["invalid_ohlcv_count"] or not acquired["all_splits_before_intraday_start"]):
            raise ValueError(f"{symbol}: adquisición no aprobada.")
        intraday = pd.read_json(StringIO(acquired["regular_5m_frame"]), orient="table")
        intraday.index = intraday.index.tz_convert("America/New_York")
        daily, audit = verify_daily(acquired, schedule.index)
        # Use the exact regular-session 5m path for the covered daily OHLCV.
        # Older daily bars are indicator warm-up only, never hypothetical fills.
        grouped = intraday.groupby(pd.Index(intraday.index.date)).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
        })
        grouped.index = pd.DatetimeIndex(grouped.index)
        differences = (daily.loc[grouped.index, "Close"] - grouped.Close).abs()
        audit["provider_daily_vs_regular_close_max_difference"] = float(differences.max())
        daily.loc[grouped.index] = grouped
        available_dates = sorted(set(intraday.index.date))
        usable = [day for day in available_dates if int((daily.index.date < day).sum()) >= 200]
        if not usable:
            raise ValueError("Sin cortes con 200 sesiones diarias previas cerradas.")
        first_cut = str(usable[0])
        as_of = pd.Timestamp(schedule.iloc[-1].close).isoformat()
        source_hash = sha(acquired)
        for timeframe, frame in (("5m", intraday), ("daily", daily)):
            write_signed(output / symbol / f"{timeframe}.json", {
                "frame": frame.to_json(orient="table", date_format="iso"),
                "fetched_at": as_of,
                "source": "Massive Stocks Basic",
                "artifact_role": "HISTORICAL_REPLAY_INPUT_NOT_LIVE",
                "acquisition_sha256": source_hash,
                "split_policy": "All splits precede first replay cut; adjusted price/share basis",
                "daily_policy": "regular 5m aggregation when covered; provider daily for prior warm-up",
            })
        raw_files = sorted((folder / "raw").glob(f"{symbol}_*.json"))
        raw_hashes = {path.name: sha(read_signed(path)) for path in raw_files}
        manifest["symbols"][symbol] = {
            **audit, "intraday_sessions": len(available_dates), "regular_bars": len(intraday),
            "first_replay_session": first_cut, "eligible_candidate_sessions_after_warmup": len(usable),
            "warmup_excluded_sessions": len(available_dates) - len(usable),
            "as_of": as_of, "acquisition_sha256": source_hash, "raw_response_sha256": raw_hashes,
            "splits": acquired["splits"],
        }
    write_signed(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(ROOT / "output" / "historical_replay" / "input"):
        raise ValueError("La importación debe permanecer en output/historical_replay/input, nunca en live/SQLite.")
    manifest = import_acquisition(args.acquisition.resolve(), args.output.resolve())
    for symbol, row in manifest["symbols"].items():
        print(f"{symbol}: {row['regular_bars']} velas; {row['daily_sessions']} días; "
              f"primer corte con calentamiento: {row['first_replay_session']}; "
              f"{row['eligible_candidate_sessions_after_warmup']} cortes candidatos.")
    print(f"Manifiesto SHA-256: {sha(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
