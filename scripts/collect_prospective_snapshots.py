"""Collect five fixed XNYS research cuts, without UI, orders or accounting writes.

Run once per scheduler invocation. The script itself enforces the NY calendar,
five registered cut times, a five-minute prospective collection window and
one immutable signed snapshot per symbol/horizon/cut. Missed cuts are NEVER
reconstructed later. Credentials come only from environment variables.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.cross_correlation import (
    apply_cross_context, build_cross_context, unavailable,
)
from portfolio_tracker.analytics.headline_sentiment_replay import (
    FINBERT_REPOSITORY, FINBERT_REVISION, FrozenFinBert,
)
from portfolio_tracker.analytics.operational_target import make_operational_contract
from portfolio_tracker.analytics.technical_probability import analyze_probability
from portfolio_tracker.services.model_execution_record import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_replay_snapshot, prediction_snapshot,
)
from portfolio_tracker.analytics.horizon_models import MODEL_FEATURE_VERSION
from portfolio_tracker.analytics.multi_timeframe import SCORE_MAP_VERSION
from scripts.download_massive_history import sha
from scripts.prospective_archive_common import (
    ARCHIVE, DATA_SOURCE, FEATURE_VERSION, GRACE, HORIZONS, POLICY_VERSION,
    SNAPSHOT_VERSION, SYMBOLS, AlpacaBars, active_cut,
    fetch_current_headlines, hash_file, snapshot_path, utc, write_payload,
)


FINBERT_FOLDER = (ROOT / "output/historical_replay"
                  / "headline_finbert_alpha_exploratory_v1/finbert_snapshot")
FINBERT_REFERENCE_MANIFEST = (ROOT / "output/historical_replay"
                              / "headline_finbert_alpha_exploratory_v1/model_manifest.json")
REQUIRED_ASSETS = ("config.json", "pytorch_model.bin", "special_tokens_map.json",
                   "tokenizer_config.json", "vocab.txt")
PROTOCOL = "PROSPECTIVE_FIVE_CUT_RESEARCH_ONLY_V1"


def _logger():
    folder = ARCHIVE / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    result = logging.getLogger("prospective_collector")
    if not result.handlers:
        result.setLevel(logging.INFO)
        handler = logging.FileHandler(folder / "collector.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        result.addHandler(handler)
    return result


def verified_finbert_manifest():
    """Use only the already-frozen model/tokenizer; never download a revision."""
    from scripts.download_massive_history import read_signed
    reference = read_signed(FINBERT_REFERENCE_MANIFEST)
    if (reference.get("repository") != FINBERT_REPOSITORY
            or reference.get("revision") != FINBERT_REVISION):
        raise ValueError("FinBERT no coincide con la revisión congelada.")
    hashes = {}
    for name in REQUIRED_ASSETS:
        asset = FINBERT_FOLDER / name
        if not asset.is_file():
            raise ValueError(f"Falta activo FinBERT congelado: {name}.")
        hashes[name] = hash_file(asset)
    if hashes != reference.get("files_sha256"):
        raise ValueError("Pesos/tokenizador FinBERT alterados; captura cancelada.")
    return {"repository": FINBERT_REPOSITORY, "revision": FINBERT_REVISION,
            "files_sha256": hashes, "scoring": "positive_probability-minus-negative_probability"}


def sentiment_window(headlines: list[dict], at: datetime, hours: int) -> dict:
    lower = utc(at) - timedelta(hours=hours)
    selected = [item for item in headlines
                if lower < datetime.fromisoformat(item["published_at_utc"]) <= utc(at)
                and datetime.fromisoformat(item["collected_at_utc"]) <= utc(at)]
    if not selected:
        return {"status": "N/D", "score": None, "article_count": 0,
                "headline_ids": [], "window_hours": hours}
    return {"status": "AVAILABLE", "score": sum(item["finbert_score"] for item in selected) / len(selected),
            "article_count": len(selected), "headline_ids": [item["id"] for item in selected],
            "window_hours": hours}


def _fetch_market(client: AlpacaBars, symbol: str, cut: datetime):
    # The same split-adjusted IEX basis is required for both timeframes.
    request_at = utc()
    intraday, intraday_meta = client.fetch(symbol, "5Min", cut - timedelta(days=32), request_at)
    daily, daily_meta = client.fetch(symbol, "1Day", cut - timedelta(days=800), utc())
    if len(intraday) < 300 or len(daily) < 200:
        raise ValueError("Warm-up insuficiente: se requieren 300 velas 5m y 200 diarias.")
    return {"intraday": intraday, "daily": daily,
            "intraday_meta": intraday_meta, "daily_meta": daily_meta}


def _snapshot(symbol: str, horizon_key: str, scheduled: datetime, cut_at: datetime,
              analysis, market, headlines, news_meta, model_manifest,
              market_archive_path: Path, market_archive_sha256: str):
    label = "1 Hora" if horizon_key == "1h" else "1 Día"
    horizon = next(item for item in analysis.horizon_projections if item.label == label)
    artifacts = {"5m": sha(market["intraday"].to_json(orient="table", date_format="iso")),
                 "1d": sha(market["daily"].to_json(orient="table", date_format="iso"))}
    replay = build_replay_snapshot(analysis, observed_at=cut_at, protocol=PROTOCOL,
                                   input_artifacts=artifacts)
    no_plan_reason = None
    try:
        contract = make_operational_contract(symbol, horizon, HORIZONS[horizon_key],
                                             analysis, cut_at, {"research_protocol": PROTOCOL})
    except ValueError as exc:
        if str(exc) != "TP/entrada/SL no encierran correctamente el precio de referencia.":
            raise
        contract = None
        no_plan_reason = str(exc)
    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_id": sha({"symbol": symbol, "scheduled_cut_at_utc": scheduled.isoformat(),
                            "horizon": horizon_key, "policy": POLICY_VERSION}),
        "symbol": symbol, "horizon": horizon_key,
        "data_source": DATA_SOURCE,
        "horizon_minutes": HORIZONS[horizon_key],
        "session_ny": scheduled.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        "scheduled_cut_at_utc": scheduled.isoformat(),
        "cut_at_utc": cut_at.isoformat(),  # actual completed collection, never scheduled/backdated
        "news_collected_at_utc": news_meta["retrieved_at_utc"],
        "source_bar_closed_at_utc": pd.Timestamp(analysis.source_bar_closed_at).tz_convert("UTC").isoformat(),
        "features": replay["features"], "model_features": replay["model_features"],
        "indicator_tail": replay["indicator_tail"],
        "feature_version": FEATURE_VERSION,
        "score_map_version": SCORE_MAP_VERSION,
        "technical_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "operational_feature_version": MODEL_FEATURE_VERSION,
        "model_version": f"{FINBERT_REPOSITORY}@{FINBERT_REVISION}",
        "finbert_manifest_sha256": sha(model_manifest),
        "sentiment_4h": sentiment_window(headlines, cut_at, 4),
        "sentiment_24h": sentiment_window(headlines, cut_at, 24),
        "headlines": headlines,
        "prediction": prediction_snapshot(horizon),
        "operational_contract": contract, "no_plan_reason": no_plan_reason,
        "policy": {"version": POLICY_VERSION, "market_feed": "Alpaca IEX",
                   "adjustment": "split", "news_source": "Finnhub company-news",
                   "news_available_from": "our_completed_fetch_timestamp",
                   "news_publication_time_only_used_for_trailing_windows": True,
                   "emission_grace_minutes": 5,
                   "production_anchor_preserved": "XNYS 11:00 only",
                   "off_anchor_actionable": False,
                   "research_feed_not_comparable_to_historical_sip": True,
                   "labeler": "existing first-passage operational_target"},
        "market_input_sha256": sha(artifacts),
        "market_input_refs": artifacts,
        "market_input_archive_path": str(market_archive_path.relative_to(ARCHIVE)),
        "market_input_archive_sha256": market_archive_sha256,
        "market_provider_metadata": {"5m": market["intraday_meta"], "1d": market["daily_meta"]},
        "news_provider_metadata": news_meta,
        "analysis_code_sha256": replay["code_sha256"],
        "production_actionable": bool(contract and contract["eligible_at_emission"]),
        "outcome": "PENDING", "label_at_utc": None,
        "mfe": None, "mae": None, "fill_possible": None,
        "training_performed": False, "portfolio_db_accessed": False,
    }
    return snapshot


def collect_once(*, clock=None, bars_client=None, news_http=None,
                 finbert=None, manifest=None, finnhub_key=None):
    logger = _logger()
    active = active_cut(clock)
    if active is None:
        return {"status": "OUTSIDE_XNYS_CUT", "saved": 0, "failed": []}
    scheduled, label = active
    due = scheduled + GRACE
    remaining = {symbol: [h for h in HORIZONS if not snapshot_path(symbol, scheduled, h).exists()]
                 for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            path = snapshot_path(symbol, scheduled, horizon)
            if path.exists():
                from scripts.download_massive_history import read_signed
                existing = read_signed(path)
                if (existing.get("symbol"), existing.get("horizon"), existing.get("scheduled_cut_at_utc")) != (
                        symbol, horizon, scheduled.isoformat()):
                    raise ValueError("Colisión de identidad del snapshot firmado.")
    if all(not values for values in remaining.values()):
        return {"status": "ALREADY_COLLECTED", "saved": 0, "failed": []}
    if bars_client is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Faltan APCA_API_KEY_ID/APCA_API_SECRET_KEY vigentes en el entorno.")
        bars_client = AlpacaBars(key, secret)
    if news_http is None:
        news_http = requests.Session()
    finnhub_key = finnhub_key or os.environ.get("FINNHUB_API_KEY")
    if not finnhub_key:
        raise RuntimeError("Falta FINNHUB_API_KEY vigente en el entorno.")
    manifest = manifest or verified_finbert_manifest()
    inputs, failures = {}, []
    for symbol, needed in remaining.items():
        if not needed:
            continue
        try:
            market = _fetch_market(bars_client, symbol, scheduled)
            headlines, news_meta = fetch_current_headlines(news_http, finnhub_key, symbol)
            inputs[symbol] = (market, headlines, news_meta)
        except Exception as exc:
            failures.append(symbol)
            logger.error("%s %s adquisición falló: %s", label, symbol, type(exc).__name__)
    if not inputs:
        return {"status": "NO_DATA", "saved": 0, "failed": failures}
    if finbert is None and any(headlines for _, headlines, _ in inputs.values()):
        cache = ARCHIVE / "runtime_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
        os.environ["TRANSFORMERS_CACHE"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        finbert = FrozenFinBert(FINBERT_FOLDER)
    analyses = {}
    for symbol, (market, headlines, _) in inputs.items():
        try:
            if headlines:
                scores = finbert.score([item["headline"] for item in headlines])
                if len(scores) != len(headlines):
                    raise ValueError("FinBERT devolvió número de scores inesperado.")
                for item, score in zip(headlines, scores):
                    item["finbert_score"] = float(score)
            # The closed source candle must be exactly the scheduled cut, even
            # if the provider returned a newer open or the script ran late.
            now = utc()
            if not scheduled <= now < due:
                raise ValueError("Ventana de corte vencida; prohibido reconstruir una predicción retrospectiva.")
            analysis = analyze_probability(symbol, market["intraday"], market["daily"],
                                           as_of_time=now, require_fresh=True)
            if utc(analysis.source_bar_closed_at) != scheduled:
                raise ValueError("Falta la vela 5m cerrada exactamente al corte.")
            analyses[symbol] = analysis
        except Exception as exc:
            failures.append(symbol)
            logger.error("%s %s análisis falló: %s", label, symbol, type(exc).__name__)
    if len(analyses) == 2:
        for symbol in tuple(analyses):
            peer = "NVDA" if symbol == "SMCI" else "SMCI"
            try:
                context = build_cross_context(symbol, inputs[symbol][0]["daily"],
                    inputs[peer][0]["daily"], inputs[symbol][0]["intraday"],
                    inputs[peer][0]["intraday"], as_of=utc())
            except Exception as exc:
                context = unavailable(symbol, type(exc).__name__)
            analyses[symbol] = apply_cross_context(analyses[symbol], context)
    saved = 0
    for symbol, analysis in analyses.items():
        market, headlines, news_meta = inputs[symbol]
        try:
            cut_at = utc()
            if not scheduled <= cut_at < due:
                raise ValueError("La firma terminaría fuera de la ventana; se omite el corte.")
            market_archive = {
                "version": "PROSPECTIVE_ALPACA_IEX_INPUTS_V1", "symbol": symbol,
                "data_source": DATA_SOURCE,
                "scheduled_cut_at_utc": scheduled.isoformat(),
                "intraday_frame": market["intraday"].to_json(orient="table", date_format="iso"),
                "daily_frame": market["daily"].to_json(orient="table", date_format="iso"),
                "metadata": {"5m": market["intraday_meta"], "1d": market["daily_meta"]},
            }
            market_hash = sha(market_archive)
            market_path = (ARCHIVE / "market_inputs" / scheduled.astimezone(
                ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
                / f"{symbol}_{scheduled.strftime('%H%M')}_{market_hash}.json")
            write_payload(market_path, market_archive)
            for horizon in remaining[symbol]:
                payload = _snapshot(symbol, horizon, scheduled, cut_at, analysis,
                                    market, headlines, news_meta, manifest,
                                    market_path, market_hash)
                write_payload(snapshot_path(symbol, scheduled, horizon), payload)
                saved += 1
            logger.info("%s %s guardó %s horizontes; %s titulares; corte real %s",
                        label, symbol, len(remaining[symbol]), len(headlines), cut_at.isoformat())
        except Exception as exc:
            failures.append(symbol)
            logger.error("%s %s firma falló: %s", label, symbol, type(exc).__name__)
    return {"status": "COLLECTED" if saved else "NO_SNAPSHOTS", "saved": saved,
            "failed": sorted(set(failures)), "scheduled_cut_at_utc": scheduled.isoformat()}


def write_example():
    """A visibly synthetic schema example, excluded from all coverage counts."""
    example = {
        "snapshot_version": SNAPSHOT_VERSION, "synthetic_example": True,
        "snapshot_id": "EXAMPLE_NOT_A_MARKET_OBSERVATION", "symbol": "SMCI",
        "data_source": DATA_SOURCE,
        "horizon": "1D", "horizon_minutes": 1440,
        "session_ny": "EXAMPLE", "scheduled_cut_at_utc": None,
        "cut_at_utc": None, "news_collected_at_utc": None,
        "source_bar_closed_at_utc": None,
        "features": {name: None for name in FEATURE_NAMES},
        "model_features": {name: None for name in FEATURE_NAMES},
        "indicator_tail": {}, "feature_version": FEATURE_VERSION,
        "score_map_version": SCORE_MAP_VERSION,
        "technical_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "operational_feature_version": MODEL_FEATURE_VERSION,
        "model_version": f"{FINBERT_REPOSITORY}@{FINBERT_REVISION}",
        "finbert_manifest_sha256": None,
        "sentiment_4h": {"status": "N/D", "score": None, "article_count": 0,
                         "headline_ids": [], "window_hours": 4},
        "sentiment_24h": {"status": "N/D", "score": None, "article_count": 0,
                          "headline_ids": [], "window_hours": 24},
        "headlines": [], "prediction": None, "operational_contract": None,
        "no_plan_reason": "Ejemplo sintético, no se etiqueta",
        "policy": {"version": POLICY_VERSION, "market_feed": "Alpaca IEX",
                   "news_source": "Finnhub company-news",
                   "news_available_from": "our_completed_fetch_timestamp",
                   "production_anchor_preserved": "XNYS 11:00 only"},
        "market_input_sha256": None, "market_input_refs": {},
        "market_input_archive_path": None, "market_input_archive_sha256": None,
        "market_provider_metadata": {}, "news_provider_metadata": {},
        "analysis_code_sha256": {}, "production_actionable": False,
        "outcome": "PENDING", "label_at_utc": None, "mfe": None, "mae": None,
        "fill_possible": None, "training_performed": False,
        "portfolio_db_accessed": False,
    }
    target = ARCHIVE / "examples" / "synthetic_schema_snapshot_iex_v1.json"
    write_payload(target, example)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="store_true",
                        help="Write only a clearly synthetic signed schema example")
    args = parser.parse_args(argv)
    if args.example:
        print(write_example())
    else:
        print(json.dumps(collect_once(), ensure_ascii=False))


if __name__ == "__main__":
    main()
