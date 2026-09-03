"""Causal, versioned contracts for live forecasts; no accounting or network I/O.

observed_at = emission time; available_at = target candle CLOSE (maturity).
Horizons are wall-clock minutes rounded UP to the next 5m close. A target
outside XNYS regular hours is invalid, never rolled to another session.
SHA-256 detects tampering, not authenticity against rewriting all hashes.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation

import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar

VERSION = 2
POLICY = "WALL_CLOCK_CEIL_5M_XNYS_V2"
FORECAST_FIELDS = (
    "symbol", "observed_at", "available_at", "horizon_minutes",
    "reference_price", "raw_probability_up", "predicted_direction",
    "parameters_json", "source_bar_at", "horizon_policy", "integrity_version",
    "created_at",
)
RESOLUTION_FIELDS = (
    "outcome_price", "outcome_up", "successful", "resolved_at",
    "outcome_bar_at", "outcome_source", "resolution_status",
)


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("El timestamp debe incluir zona horaria explícita.")
    return stamp.tz_convert("UTC")


def maturity(observed_at, horizon_minutes: int) -> pd.Timestamp:
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int) or not 0 < horizon_minutes <= 525_600:
        raise ValueError("Horizonte inválido: usar minutos enteros entre 1 y 525600.")
    return (utc_timestamp(observed_at) + pd.Timedelta(minutes=horizon_minutes)).ceil("5min")


def is_regular_close(value) -> bool:
    stamp = utc_timestamp(value)
    day = pd.Timestamp(stamp.tz_convert(NY).date())
    schedule = _calendar(day.year, day.year).schedule
    if day not in schedule.index:
        return False
    session = schedule.loc[day]
    return bool(session.open < stamp <= session.close and
                (stamp - session.open) % pd.Timedelta(minutes=5) == pd.Timedelta(seconds=0))


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def forecast_digest(row) -> str:
    return hashlib.sha256(canonical({key: row[key] for key in FORECAST_FIELDS}).encode()).hexdigest()


def resolution_digest(row) -> str:
    payload = {key: row[key] for key in (*FORECAST_FIELDS, *RESOLUTION_FIELDS)}
    payload["observation_sha256"] = row["observation_sha256"]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def valid_observation(row) -> bool:
    """Fail closed for legacy, malformed, incomplete or altered observations."""
    try:
        if row["integrity_version"] != VERSION or row["horizon_policy"] != POLICY:
            return False
        observed = utc_timestamp(row["observed_at"])
        due = utc_timestamp(row["available_at"])
        if due != maturity(observed, row["horizon_minutes"]):
            return False
        source = utc_timestamp(row["source_bar_at"])
        if source > observed or not is_regular_close(source):
            return False
        price, probability = Decimal(row["reference_price"]), Decimal(row["raw_probability_up"])
        if not price.is_finite() or price <= 0 or not probability.is_finite() or not 0 <= probability <= 1:
            return False
        direction = "UP" if probability >= Decimal("0.5") else "DOWN"
        if row["predicted_direction"] != direction or forecast_digest(row) != row["observation_sha256"]:
            return False
        if row["resolution_status"] == "PENDING":
            return row["resolution_sha256"] is None and all(
                row[key] is None for key in RESOLUTION_FIELDS if key != "resolution_status")
        if row["resolution_sha256"] != resolution_digest(row):
            return False
        if utc_timestamp(row["resolved_at"]) < due:
            return False
        if row["resolution_status"] == "INVALID_MARKET_CLOSED":
            return not is_regular_close(due) and all(row[k] is None for k in (
                "outcome_price", "outcome_up", "successful", "outcome_bar_at", "outcome_source"))
        if row["resolution_status"] != "RESOLVED" or not is_regular_close(due):
            return False
        if utc_timestamp(row["outcome_bar_at"]) != due or not row["outcome_source"]:
            return False
        outcome = Decimal(row["outcome_price"])
        up = int(outcome > price)
        return bool(outcome.is_finite() and outcome > 0 and row["outcome_up"] == up
                    and row["successful"] == int((direction == "UP") == bool(up)))
    except (KeyError, IndexError, TypeError, ValueError, InvalidOperation, OverflowError):
        return False


def exact_closed_prices(history: pd.DataFrame | None, as_of) -> dict[str, str]:
    """Raw 5m OHLCV, OPEN-labelled. No nearest/forward fill or spot fallback."""
    if history is None or history.empty:
        return {}
    columns = ("Open", "High", "Low", "Close", "Volume")
    if not all(c in history.columns for c in columns):
        raise ValueError("La resolución requiere OHLCV histórico de 5 minutos.")
    index = pd.DatetimeIndex(history.index)
    if index.tz is None:
        raise ValueError("Las velas históricas deben tener zona horaria explícita.")
    now = utc_timestamp(as_of)
    prices = {}
    duplicates = index.duplicated(keep=False)
    for duplicate, start, values in zip(duplicates, index, history.loc[:, columns].itertuples(index=False, name=None)):
        if duplicate:
            continue
        end = utc_timestamp(start) + pd.Timedelta(minutes=5)
        if end > now or not is_regular_close(end):
            continue
        try:
            o, h, l, c, v = map(float, values)
            if not all(math.isfinite(x) for x in (o,h,l,c,v)) or min(o,h,l,c) <= 0 or v < 0:
                continue
            if l > min(o,c) or h < max(o,c) or l > h:
                continue
            prices[end.isoformat()] = str(Decimal(str(values[3])))
        except (ValueError, TypeError, InvalidOperation):
            continue
    return prices
