"""Resolve immutable prospective plans after their XNYS horizon has matured.

The original PIT snapshot is NEVER rewritten. A signed label sidecar names its
parent snapshot SHA-256, preserving the exact features/headlines known at the
cut. Missing market evidence remains PENDING; no current-price substitution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

import exchange_calendars as xcals
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.execution_path import assess_execution_path
from portfolio_tracker.analytics.operational_target import (
    IncompleteOperationalEvidence, resolve_operational_outcome,
    validate_operational_contract,
)
from scripts.download_massive_history import sha
from scripts.prospective_archive_common import (
    ARCHIVE, DATA_SOURCE, LABEL_VERSION, AlpacaBars, label_path, signed_payload, utc, write_payload,
)


def _logger():
    folder = ARCHIVE / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    result = logging.getLogger("prospective_labeler")
    if not result.handlers:
        result.setLevel(logging.INFO)
        handler = logging.FileHandler(folder / "labeler.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        result.addHandler(handler)
    return result


def _complete_5m_path(frame: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame | None:
    """Return exact post-cut bars only if every XNYS 5m bucket is present."""
    start, end = pd.Timestamp(utc(start)), pd.Timestamp(utc(end))
    if end <= start:
        return None
    calendar = xcals.get_calendar("XNYS", start=f"{start.year-1}-01-01",
                                  end=f"{end.year+1}-12-31")
    expected = pd.DatetimeIndex([], tz="UTC")
    for session in calendar.schedule.loc[str(start.date()):str(end.date())].itertuples():
        first = max(session.open, start)
        last = min(session.close, end)
        if last > first:
            expected = expected.append(pd.date_range(first, last - pd.Timedelta(minutes=5),
                                                     freq="5min"))
    actual = frame.loc[(frame.index >= start) & (frame.index < end)]
    return actual if len(expected) and actual.index.equals(expected) else None


def excursions(contract: dict, frame: pd.DataFrame, exit_at: str) -> dict:
    """MFE/MAE in USD/share on the fully observed path through first exit."""
    path = _complete_5m_path(frame, datetime.fromisoformat(contract["evaluation_starts_at"]),
                             datetime.fromisoformat(exit_at))
    if path is None:
        return {"mfe": None, "mae": None, "mfe_pct": None, "mae_pct": None,
                "excursion_status": "N/D_INCOMPLETE_5M_PATH"}
    entry = float(contract["entry_price"])
    if contract["side"] == "LONG":
        favorable = float(path.High.max()) - entry
        adverse = entry - float(path.Low.min())
    else:
        favorable = entry - float(path.Low.min())
        adverse = float(path.High.max()) - entry
    return {"mfe": max(0.0, favorable), "mae": max(0.0, adverse),
            "mfe_pct": max(0.0, favorable) / entry * 100,
            "mae_pct": max(0.0, adverse) / entry * 100,
            "excursion_status": "COMPLETE_5M_REFERENCE_PRICE_NOT_CONFIRMED_FILL"}


def research_fill_possible(contract: dict, frame: pd.DataFrame, exit_at: str):
    """Conservative, *research-only* OHLC entry reach at any registered cut.

    The live execution assessor intentionally denies off-11:00 cuts. This
    separate diagnostic never upgrades `eligible_at_emission` or executes an
    order; it only answers whether the frozen reference could have filled.
    """
    deadline = min(datetime.fromisoformat(exit_at),
                   datetime.fromisoformat(contract["execution_terms"]["entry_deadline_at"]))
    path = _complete_5m_path(frame, datetime.fromisoformat(contract["evaluation_starts_at"]),
                             deadline)
    if path is None:
        return None, "N/D_INCOMPLETE_ENTRY_PATH"
    entry = float(contract["entry_price"])
    stop = float(contract["stop_loss"])
    target = float(contract["take_profit"])
    for row in path.itertuples():
        opening, high, low = float(row.Open), float(row.High), float(row.Low)
        if contract["side"] == "LONG":
            if opening <= stop:
                return False, "INVALIDATED_BEFORE_FILL"
            if opening >= target or (high >= target and low > entry):
                return False, "TARGET_BEFORE_FILL"
            if low <= entry:
                if high >= target and low > stop:
                    return False, "AMBIGUOUS_SAME_BAR_NO_TRADE"
                return True, "OHLC_LIMIT_REACH_NOT_BROKER_FILL"
        else:
            if opening >= stop:
                return False, "INVALIDATED_BEFORE_FILL"
            if opening <= target or (low <= target and high < entry):
                return False, "TARGET_BEFORE_FILL"
            if high >= entry:
                if low <= target and high < stop:
                    return False, "AMBIGUOUS_SAME_BAR_NO_TRADE"
                return True, "OHLC_LIMIT_REACH_NOT_BROKER_FILL"
    return False, "ENTRY_NOT_REACHED"


def _pending_paths():
    base = ARCHIVE / "snapshots"
    if not base.exists():
        return []
    return [path for path in sorted(base.glob("*/*.json")) if not label_path(path).exists()]


def label_once(*, clock=None, bars_client=None) -> dict:
    now = utc(clock)
    logger = _logger()
    pending = _pending_paths()
    mature = []
    for path in pending:
        snapshot = signed_payload(path)
        if snapshot.get("synthetic_example") or snapshot.get("training_performed") is not False:
            raise ValueError("Snapshot prospectivo inválido.")
        contract = snapshot.get("operational_contract")
        if contract is None:
            continue  # directional/no-plan cut, never invent barriers
        validate_operational_contract(contract)
        if (contract["model"]["horizon_minutes"] != snapshot["horizon_minutes"]
                or contract["observed_at"] != snapshot["cut_at_utc"]):
            raise ValueError("Contrato no coincide con el corte inmutable.")
        timeout = utc(datetime.fromisoformat(contract["timeout_at"]))
        if now >= timeout + timedelta(minutes=15):
            mature.append((path, snapshot, contract, timeout))
    if not mature:
        return {"pending_snapshots": len(pending), "mature": 0, "labeled": 0,
                "errors": 0, "no_plan": len(pending) - sum(
                    signed_payload(path).get("operational_contract") is not None for path in pending)}
    if bars_client is None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Faltan APCA_API_KEY_ID/APCA_API_SECRET_KEY vigentes.")
        bars_client = AlpacaBars(key, secret)
    labeled, errors = 0, 0
    for path, snapshot, contract, timeout in mature:
        symbol = snapshot["symbol"]
        try:
            if (snapshot.get("policy", {}).get("market_feed") != "Alpaca IEX"
                    or snapshot.get("data_source", DATA_SOURCE) != DATA_SOURCE):
                raise ValueError("Feed del snapshot distinto del etiquetador IEX versionado.")
            start = datetime.fromisoformat(contract["evaluation_starts_at"])
            intraday, bars_meta = bars_client.fetch(symbol, "5Min", start - timedelta(minutes=5),
                                                     timeout + timedelta(minutes=1))
            daily, daily_meta = bars_client.fetch(symbol, "1Day", start - timedelta(days=1),
                                                   timeout + timedelta(days=1))
            result = resolve_operational_outcome(contract, intraday, now, daily_bars=daily)
            if result is None:
                continue
            execution = assess_execution_path(contract, intraday, now, daily_bars=daily)
            fill_possible, fill_status = research_fill_possible(contract, intraday, result.exit_at)
            label = {"label_version": LABEL_VERSION,
                     "data_source": DATA_SOURCE,
                     "snapshot_id": snapshot["snapshot_id"],
                     "parent_snapshot_sha256": sha(snapshot),
                     "symbol": symbol, "horizon": snapshot["horizon"],
                     "cut_at_utc": snapshot["cut_at_utc"],
                     "outcome": result.outcome.value,
                     "label_at_utc": now.isoformat(),
                     "exit_at_utc": result.exit_at,
                     "exit_price": float(result.exit_price),
                     "exit_source": result.exit_source,
                     "evidence_sha256": result.evidence_sha256,
                     **excursions(contract, intraday, result.exit_at),
                     "fill_possible": fill_possible,
                     "fill_status": fill_status,
                     "production_anchor_authorized": bool(contract["eligible_at_emission"]),
                     "production_execution_status": execution.status,
                     "production_possible_fill": execution.fill_price is not None,
                     "production_fill_price": execution.fill_price,
                     "market_provider_metadata": {"5m": bars_meta, "1d": daily_meta},
                     "label_policy": "TP_FIRST_SL_FIRST_TIMEOUT_V2; SL_FIRST same-bar; gap at open",
                     "fill_policy": "research OHLC limit reach; not a broker-confirmed fill; does not authorize off-anchor cuts",
                     "training_performed": False, "portfolio_db_accessed": False}
            write_payload(label_path(path), label)
            labeled += 1
            logger.info("%s %s %s etiquetado %s; parent=%s", symbol,
                        snapshot["session_ny"], snapshot["horizon"],
                        result.outcome.value, sha(snapshot))
        except (IncompleteOperationalEvidence, ValueError) as exc:
            errors += 1
            logger.warning("%s %s evidencia pendiente: %s", symbol,
                           snapshot["snapshot_id"], type(exc).__name__)
        except Exception as exc:
            errors += 1
            logger.error("%s %s etiquetado falló: %s", symbol,
                         snapshot["snapshot_id"], type(exc).__name__)
    return {"pending_snapshots": len(pending), "mature": len(mature),
            "labeled": labeled, "errors": errors}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(label_once(), ensure_ascii=False))


if __name__ == "__main__":
    main()
