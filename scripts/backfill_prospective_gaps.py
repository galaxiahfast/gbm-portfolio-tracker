"""Research reconstruction of the two missed 10:00 NY cuts; never PIT evidence.

Records live in backfills/, not snapshots/, and cannot enter the prospective
labeler, coverage denominator or training. Historical bars are fetched NOW;
news receipt and broker fills at the missed times cannot be reconstructed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.closed_bars import select_last_closed_bar
from portfolio_tracker.analytics.cross_correlation import apply_cross_context, build_cross_context, unavailable
from portfolio_tracker.analytics.multi_timeframe import SCORE_MAP_VERSION
from portfolio_tracker.analytics.technical_probability import analyze_probability
from portfolio_tracker.services.model_execution_record import build_replay_snapshot, prediction_snapshot
from scripts.download_massive_history import sha
from scripts.prospective_archive_common import (
    ARCHIVE, DATA_SOURCE, FEATURE_VERSION, HORIZONS, SYMBOLS, AlpacaBars,
    signed_payload, utc, write_payload,
)

NY = ZoneInfo("America/New_York")
MISSED_DAYS = ("2026-10-01", "2026-10-05")
VERSION = "DEGRADED_PROSPECTIVE_GAP_RECONSTRUCTION_V1"


def _cut(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T10:00:00").replace(tzinfo=NY).astimezone(timezone.utc)


def _target(day: str, symbol: str, horizon: str) -> Path:
    return ARCHIVE / "backfills" / day / f"{day.replace('-', '')}_1000_{symbol}_{horizon}.json"


def _market_at(client: AlpacaBars, symbol: str, first_cut: datetime, retrieval: datetime):
    intraday, five_meta = client.fetch(symbol, "5Min", first_cut - timedelta(days=32), retrieval)
    daily, day_meta = client.fetch(symbol, "1Day", first_cut - timedelta(days=800), retrieval)
    return intraday, daily, {"5m": five_meta, "1d": day_meta}


def reconstruct(*, bars_client=None, clock=None):
    retrieved = utc(clock)
    all_targets = [_target(day, symbol, horizon) for day in MISSED_DAYS
                   for symbol in SYMBOLS for horizon in HORIZONS]
    if all(path.exists() for path in all_targets):
        for path in all_targets:
            item = signed_payload(path)
            if (item.get("backfilled") is not True or item.get("pit_quality") != "degraded"
                    or item.get("training_eligible") is not False):
                raise ValueError("Backfill existente no cumple la política degradada.")
        return {"version": VERSION, "data_source": DATA_SOURCE,
                "records_saved": 0, "already_present": len(all_targets), "analysis_unavailable": []}
    if bars_client is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Faltan credenciales de mercado para reconstrucción IEX.")
        bars_client = AlpacaBars(key, secret)
    first = _cut(MISSED_DAYS[0])
    market = {symbol: _market_at(bars_client, symbol, first, retrieved) for symbol in SYMBOLS}
    result = {"version": VERSION, "data_source": DATA_SOURCE,
              "records_saved": 0, "already_present": 0, "analysis_unavailable": []}
    for day in MISSED_DAYS:
        scheduled = _cut(day)
        if retrieved <= scheduled:
            raise ValueError("No se puede reconstruir un corte futuro.")
        capped = {}
        for symbol in SYMBOLS:
            five, daily, metadata = market[symbol]
            # Retain only bars available by the historical cut. The provider's
            # current response may contain later bars; they never enter analysis.
            capped[symbol] = (
                select_last_closed_bar(five, "5m", scheduled + timedelta(seconds=1)),
                select_last_closed_bar(daily, "1d", scheduled + timedelta(seconds=1)),
                metadata,
            )
        analyses, errors = {}, {}
        for symbol in SYMBOLS:
            five, daily, _ = capped[symbol]
            try:
                analysis = analyze_probability(symbol, five, daily,
                    as_of_time=scheduled + timedelta(seconds=1), require_fresh=False)
                if utc(analysis.source_bar_closed_at) != scheduled:
                    raise ValueError("Falta la vela cerrada exactamente a las 10:00 NY.")
                analyses[symbol] = analysis
            except (ValueError, RuntimeError) as exc:
                errors[symbol] = f"{type(exc).__name__}: {exc}"
        if len(analyses) == 2:
            for symbol in SYMBOLS:
                peer = "NVDA" if symbol == "SMCI" else "SMCI"
                try:
                    context = build_cross_context(symbol, capped[symbol][1], capped[peer][1],
                        capped[symbol][0], capped[peer][0], as_of=scheduled + timedelta(seconds=1))
                except (ValueError, RuntimeError) as exc:
                    context = unavailable(symbol, type(exc).__name__)
                analyses[symbol] = apply_cross_context(analyses[symbol], context)
        for symbol in SYMBOLS:
            five, daily, metadata = capped[symbol]
            input_payload = {
                "version": VERSION, "symbol": symbol, "data_source": DATA_SOURCE,
                "historical_cut_at_utc": scheduled.isoformat(),
                "retrieved_at_utc": retrieved.isoformat(),
                "five_minute_frame": five.to_json(orient="table", date_format="iso"),
                "daily_frame": daily.to_json(orient="table", date_format="iso"),
                "provider_metadata": metadata,
            }
            input_hash = sha(input_payload)
            input_path = ARCHIVE / "backfills" / day / f"{symbol}_market_{input_hash}.json"
            if not input_path.exists():
                write_payload(input_path, input_payload)
            analysis = analyses.get(symbol)
            replay = None
            if analysis is not None:
                replay = build_replay_snapshot(analysis, observed_at=retrieved,
                    protocol=VERSION, input_artifacts={"5m": sha(input_payload["five_minute_frame"]),
                                                       "1d": sha(input_payload["daily_frame"])})
            else:
                result["analysis_unavailable"].append({"day": day, "symbol": symbol,
                                                       "reason": errors.get(symbol, "N/D")})
            for horizon, label in (("1h", "1 Hora"), ("1D", "1 Día")):
                path = _target(day, symbol, horizon)
                if path.exists():
                    existing = signed_payload(path)
                    if (existing.get("historical_cut_at_utc"), existing.get("symbol"), existing.get("horizon")) != (
                            scheduled.isoformat(), symbol, horizon):
                        raise ValueError("Colisión de identidad en backfill firmado.")
                    result["already_present"] += 1
                    continue
                projection = (next((item for item in analysis.horizon_projections
                                    if item.label == label), None) if analysis else None)
                payload = {
                    "backfill_version": VERSION, "backfilled": True,
                    "pit_quality": "degraded", "training_eligible": False,
                    "production_actionable": False, "label_eligible": False,
                    "portfolio_db_accessed": False, "training_performed": False,
                    "symbol": symbol, "horizon": horizon,
                    "horizon_minutes": HORIZONS[horizon], "data_source": DATA_SOURCE,
                    "historical_cut_at_utc": scheduled.isoformat(),
                    "cut_at_utc": retrieved.isoformat(),  # actual reconstruction, never backdated
                    "features": replay["features"] if replay else None,
                    "model_features": replay["model_features"] if replay else None,
                    "indicator_tail": replay["indicator_tail"] if replay else None,
                    "feature_version": FEATURE_VERSION,
                    "score_map_version": SCORE_MAP_VERSION,
                    "prediction": prediction_snapshot(projection) if projection else None,
                    "analysis_error": errors.get(symbol),
                    "sentiment_4h": {"status": "N/D", "score": None},
                    "sentiment_24h": {"status": "N/D", "score": None},
                    "headlines": [], "outcome": "UNLABELLED_DEGRADED_BACKFILL",
                    "market_input_archive_path": str(input_path.relative_to(ARCHIVE)),
                    "market_input_archive_sha256": input_hash,
                    "source_bar_closed_at_utc": (
                        utc(analysis.source_bar_closed_at).isoformat() if analysis else None),
                    "pit_limitation": (
                        "Barras históricas consultadas después del corte; posibles revisiones. "
                        "No existe timestamp de recepción de titulares al corte ni fill real. "
                        "Excluido de evidencia prospectiva, calibración, entrenamiento y órdenes."
                    ),
                }
                write_payload(path, payload)
                result["records_saved"] += 1
    return result


def main(argv=None):
    if argv:
        raise ValueError("Este backfill solo acepta los dos cortes ausentes fijados.")
    print(json.dumps(reconstruct(), ensure_ascii=False))


if __name__ == "__main__":
    main()
