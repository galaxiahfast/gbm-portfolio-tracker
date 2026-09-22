"""Historical fixed-cut replay with causal features and first-passage labels.

The analyzer sees only bars that were closed at each 11:00 New York cut.  Data
after the cut is used solely by the labelers.  Results are deliberately kept
separate from ``live_model_observations`` so replay evidence can never inflate
the live/OOS sample.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from .closed_bars import NY, _calendar, utc
from .cross_correlation import PEERS, apply_cross_context, build_cross_context, unavailable
from .operational_target import (
    IncompleteOperationalEvidence,
    resolve_operational_outcome,
)
from .replay import ReplayDataset, evaluate_replay_cut
from ..services.directional_collection import HORIZON_MINUTES, cut_forecasts
from ..services.model_observations import (
    canonical,
    exact_closed_prices,
    exact_daily_closed_prices,
    maturity,
)
from ..services.scenario_calibration import outcome_class


HISTORICAL_REPLAY_CONTRACT = "XNYS_1100_HISTORICAL_CAUSAL_REPLAY_V1"
DEFAULT_PARAMETERS = {
    "minimum_probability": 0.55,
    "stop_atr_multiple": 2.25,
    "risk_per_trade_pct": 1.0,
}
SCENARIO_CLASSES = ("UP", "RANGE", "DOWN")


def _sha(payload) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _day(value, fallback) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(fallback).normalize().tz_localize(None)
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Fecha de replay inválida.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(NY).tz_localize(None)
    return stamp.normalize()


def fixed_historical_cuts(dataset: ReplayDataset, *, start=None, end=None) -> tuple[pd.Timestamp, ...]:
    """Return exact 11:00 NY cuts that have a source bar closed at the cut."""

    first_day = dataset.intraday.index[0].tz_convert(NY).date()
    last_day = dataset.as_of.tz_convert(NY).date()
    start_day = _day(start, first_day)
    end_day = _day(end, last_day)
    if end_day < start_day:
        raise ValueError("El final del replay precede al inicio.")
    schedule = _calendar(start_day.year - 1, end_day.year + 1).schedule.loc[start_day:end_day]
    available = pd.DatetimeIndex(dataset.intraday.index).tz_convert("UTC")
    cuts = []
    for session in schedule.itertuples():
        cut = utc(session.open) + pd.Timedelta(minutes=90)
        source_open = cut - pd.Timedelta(minutes=5)
        if cut <= dataset.as_of and cut <= utc(session.close) and source_open in available:
            cuts.append(cut)
    return tuple(cuts)


def _scenario_label(contract, due, intraday_closes, daily_closes, as_of):
    key = due.isoformat()
    source = intraday_closes if int(contract["model"]["horizon_minutes"]) <= 390 else daily_closes
    if key in source:
        price = float(source[key])
        label = SCENARIO_CLASSES[outcome_class(price, contract["range_low"], contract["range_high"])]
        return {
            "status": "RESOLVED",
            "available_at": key,
            "close_price": price,
            "outcome": label,
            "source": "5m_exact_close" if source is intraday_closes else "1d_exact_session_close",
        }
    return {
        "status": "RIGHT_CENSORED" if due > as_of else "MISSING_EXACT_CLOSE",
        "available_at": key,
        "close_price": None,
        "outcome": None,
        "source": None,
    }


def _operational_label(contract, dataset):
    starts = pd.Timestamp(contract["evaluation_starts_at"])
    timeout = pd.Timestamp(contract["timeout_at"])
    cutoff = min(dataset.as_of, timeout)
    intraday = dataset.intraday.loc[
        (dataset.intraday.index >= starts) & (dataset.intraday.index < cutoff)
    ]
    first_day = starts.tz_convert(NY).date()
    last_day = cutoff.tz_convert(NY).date()
    daily_days = pd.Index([pd.Timestamp(value).date() for value in dataset.daily.index])
    daily = dataset.daily.loc[(daily_days >= first_day) & (daily_days <= last_day)]
    try:
        result = resolve_operational_outcome(
            contract,
            intraday,
            dataset.as_of,
            daily_bars=daily,
        )
    except IncompleteOperationalEvidence as exc:
        return {
            "status": "MISSING_CAUSAL_PATH",
            "outcome": None,
            "exit_price": None,
            "exit_at": None,
            "exit_source": None,
            "evidence_sha256": None,
            "detail": str(exc),
        }
    if result is None:
        return {
            "status": "RIGHT_CENSORED" if dataset.as_of < timeout else "MISSING_CAUSAL_PATH",
            "outcome": None,
            "exit_price": None,
            "exit_at": None,
            "exit_source": None,
            "evidence_sha256": None,
            "detail": "El histórico termina antes del evento o no contiene el cierre exacto.",
        }
    return {
        "status": "RESOLVED",
        "outcome": result.outcome.value,
        "exit_price": float(result.exit_price),
        "exit_at": result.exit_at,
        "exit_source": result.exit_source,
        "evidence_sha256": result.evidence_sha256,
        "detail": "",
    }


def _cross_asset(analysis, symbol, dataset, peer_dataset, cut):
    if peer_dataset is None:
        return apply_cross_context(analysis, unavailable(symbol, "peer histórico no suministrado"))
    try:
        context = build_cross_context(
            symbol,
            dataset.daily,
            peer_dataset.daily,
            dataset.intraday,
            peer_dataset.intraday,
            as_of=cut,
        )
    except Exception as exc:  # a missing peer cut must not destroy the primary replay
        context = unavailable(symbol, str(exc))
    return apply_cross_context(analysis, context)


def _cut_record(
    symbol,
    dataset,
    peer_dataset,
    cut,
    parameters,
    dataset_sha256,
    intraday_closes,
    daily_closes,
):
    analysis = evaluate_replay_cut(
        dataset,
        cut,
        atr_stop_multiple=float(parameters["stop_atr_multiple"]),
        symbol=symbol,
    )
    analysis = _cross_asset(analysis, symbol, dataset, peer_dataset, cut)
    forecasts = cut_forecasts(
        analysis,
        parameters,
        cut.to_pydatetime(),
        protocol=HISTORICAL_REPLAY_CONTRACT,
    )
    horizons = []
    shared_replay = None
    for row in forecasts:
        metadata = json.loads(row["parameters_json"])
        replay = metadata["replay"]
        if shared_replay is None:
            shared_replay = replay
        elif canonical(replay) != canonical(shared_replay):
            raise ValueError("Los seis horizontes no comparten el mismo snapshot causal.")
        minutes = int(row["horizon_minutes"])
        due = maturity(cut, minutes)
        scenario_contract = metadata["scenario_contract"]
        operational_contract = metadata["operational_contract"]
        horizons.append({
            "label": next(label for label, value in HORIZON_MINUTES.items() if value == minutes),
            "horizon_minutes": minutes,
            "prediction": metadata["prediction_snapshot"],
            "scenario_contract": scenario_contract,
            "scenario_result": _scenario_label(
                scenario_contract,
                due,
                intraday_closes,
                daily_closes,
                dataset.as_of,
            ),
            "operational_contract": operational_contract,
            "operational_result": _operational_label(operational_contract, dataset),
        })
    cut_payload = {
        "cut_id": _sha({
            "symbol": symbol,
            "observed_at": cut.isoformat(),
            "protocol": HISTORICAL_REPLAY_CONTRACT,
            "dataset_sha256": dataset_sha256,
        }),
        "symbol": symbol,
        "observed_at": cut.isoformat(),
        "source_bar_closed_at": analysis.source_bar_closed_at.isoformat(),
        "dataset_sha256": dataset_sha256,
        "feature_snapshot": shared_replay,
        "horizons": horizons,
        "limitations": [
            "NO_HISTORICAL_FUNDAMENTAL_NEWS",
            "NO_ACCOUNT_OR_POSITION_STATE",
            "HISTORICAL_REPLAY_NOT_LIVE_OOS",
        ],
    }
    cut_payload["cut_sha256"] = _sha(cut_payload)
    return cut_payload


def build_historical_replay(
    symbol: str,
    dataset: ReplayDataset,
    *,
    peer_dataset: ReplayDataset | None = None,
    parameters: Mapping | None = None,
    start=None,
    end=None,
    max_cuts: int | None = None,
):
    """Replay fixed historical cuts and label their future path causally.

    The returned artifact is research evidence.  It is never inserted into the
    live forward tables and must be split chronologically before calibration.
    """

    symbol = str(symbol).strip().upper()
    if not symbol:
        raise ValueError("Símbolo obligatorio.")
    values = dict(DEFAULT_PARAMETERS if parameters is None else parameters)
    required = set(DEFAULT_PARAMETERS)
    if set(values) != required:
        raise ValueError(f"Parámetros requeridos: {sorted(required)}.")
    try:
        values = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Los parámetros del replay deben ser numéricos.") from exc
    if not 2.0 <= values["stop_atr_multiple"] <= 2.5:
        raise ValueError("stop_atr_multiple debe estar entre 2.0 y 2.5.")
    if max_cuts is not None and (isinstance(max_cuts, bool) or int(max_cuts) <= 0):
        raise ValueError("max_cuts debe ser positivo.")

    dataset_sha256 = dataset.fingerprint(symbol)
    intraday_closes = exact_closed_prices(dataset.intraday, dataset.as_of)
    daily_closes = exact_daily_closed_prices(dataset.daily, dataset.as_of)
    cuts = fixed_historical_cuts(dataset, start=start, end=end)
    if max_cuts is not None:
        cuts = cuts[-int(max_cuts):]
    observations = []
    rejected = Counter()
    for cut in cuts:
        try:
            observations.append(
                _cut_record(
                    symbol,
                    dataset,
                    peer_dataset,
                    cut,
                    values,
                    dataset_sha256,
                    intraday_closes,
                    daily_closes,
                )
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            rejected[str(exc)] += 1

    deterministic = {
        "contract": HISTORICAL_REPLAY_CONTRACT,
        "symbol": symbol,
        "dataset_sha256": dataset_sha256,
        "peer_dataset_sha256": (
            peer_dataset.fingerprint(PEERS.get(symbol, "PEER")) if peer_dataset else None
        ),
        "parameters": values,
        "requested_start": None if start is None else str(start),
        "requested_end": None if end is None else str(end),
        "max_cuts": None if max_cuts is None else int(max_cuts),
        "candidate_cuts": len(cuts),
        "observations": observations,
        "rejected": dict(sorted(rejected.items())),
        "separation_policy": "REPLAY_NEVER_COUNTS_AS_LIVE_OOS",
    }
    payload = {
        **deterministic,
        "replay_id": _sha({
            "contract": HISTORICAL_REPLAY_CONTRACT,
            "symbol": symbol,
            "dataset_sha256": dataset_sha256,
            "parameters": values,
            "requested_start": deterministic["requested_start"],
            "requested_end": deterministic["requested_end"],
            "max_cuts": deterministic["max_cuts"],
        }),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["content_sha256"] = _sha(deterministic)
    payload["artifact_sha256"] = _sha(payload)
    validate_historical_replay(payload)
    return payload


def validate_historical_replay(payload) -> bool:
    """Fail closed when a cut, outcome, feature snapshot or manifest changed."""

    if not isinstance(payload, dict) or payload.get("contract") != HISTORICAL_REPLAY_CONTRACT:
        raise ValueError("Contrato de replay histórico desconocido.")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("Observaciones históricas inválidas.")
    for item in observations:
        if not isinstance(item, dict) or item.get("cut_sha256") != _sha({
            key: value for key, value in item.items() if key != "cut_sha256"
        }):
            raise ValueError("Firma de corte histórico inválida.")
        if len(item.get("horizons", ())) != len(HORIZON_MINUTES):
            raise ValueError("Un corte histórico debe contener seis horizontes.")
    deterministic_keys = (
        "contract", "symbol", "dataset_sha256", "peer_dataset_sha256",
        "parameters", "requested_start", "requested_end", "max_cuts",
        "candidate_cuts", "observations", "rejected", "separation_policy",
    )
    deterministic = {key: payload.get(key) for key in deterministic_keys}
    if payload.get("content_sha256") != _sha(deterministic):
        raise ValueError("Firma del replay histórico inválida.")
    expected_replay_id = _sha({
        "contract": HISTORICAL_REPLAY_CONTRACT,
        "symbol": payload.get("symbol"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "parameters": payload.get("parameters"),
        "requested_start": payload.get("requested_start"),
        "requested_end": payload.get("requested_end"),
        "max_cuts": payload.get("max_cuts"),
    })
    if payload.get("replay_id") != expected_replay_id:
        raise ValueError("Identidad del replay histórico inválida.")
    if payload.get("artifact_sha256") != _sha({
        key: value for key, value in payload.items() if key != "artifact_sha256"
    }):
        raise ValueError("Firma integral del artefacto histórico inválida.")
    if payload.get("candidate_cuts") != len(observations) + sum(payload.get("rejected", {}).values()):
        raise ValueError("Conteo de cortes histórico inconsistente.")
    return True


def write_historical_replay(payload, path) -> Path:
    """Atomically persist one verified JSON artifact; no SQLite side effects."""

    validate_historical_replay(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(canonical(payload), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
