"""Causal, versioned contracts for live forecasts; no accounting or network I/O.

``observed_at`` is the emission instant and ``available_at`` is the exact XNYS
bar/session close at which the target becomes knowable.  Version 3 counts only
regular-market time: intraday horizons skip nights, weekends, holidays and
early-close gaps, while daily/weekly/monthly horizons mature after a fixed
number of *future exchange sessions*.  Version 2 remains verifiable for audit,
but is never mixed into the current calibration cohort.

SHA-256 detects tampering, not authenticity against rewriting all hashes.
"""
from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from decimal import Decimal, InvalidOperation

import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar

LEGACY_VERSION = 2
LEGACY_POLICY = "WALL_CLOCK_CEIL_5M_XNYS_V2"
VERSION = 3
POLICY = "XNYS_TRADING_MINUTES_AND_FUTURE_SESSIONS_V3"

# Public identifiers are intentionally unchanged so historical reports and
# scenario contracts remain readable.  Their interpretation is now explicit.
SESSION_HORIZONS = {
    1_440: 1,      # next XNYS session close
    10_080: 5,     # fifth future XNYS session close
    43_200: 21,    # twenty-first future XNYS session close
    259_200: 126,  # one trading half-year
}
MAX_INTRADAY_MINUTES = 390
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


def _schedule_for(stamp: pd.Timestamp):
    """Return enough XNYS schedule for all supported future horizons."""

    return _calendar(stamp.year - 1, stamp.year + 2).schedule


def _session_row(stamp: pd.Timestamp):
    local_day = pd.Timestamp(stamp.tz_convert(NY).date())
    schedule = _schedule_for(stamp)
    if local_day not in schedule.index:
        raise ValueError("La observación no pertenece a una sesión XNYS.")
    return local_day, schedule, schedule.loc[local_day]


def _intraday_maturity(observed: pd.Timestamp, horizon_minutes: int) -> pd.Timestamp:
    # Emissions refer to the latest known 5m close.  Flooring avoids adding an
    # artificial extra bar merely because the write occurred a few seconds
    # after that close.
    anchor = observed.floor("5min")
    day, schedule, session = _session_row(anchor)
    if not session.open < anchor <= session.close:
        raise ValueError("La observación intradía no parte de un cierre regular XNYS.")

    remaining = int(math.ceil(horizon_minutes / 5.0) * 5)
    position = int(schedule.index.get_loc(day))
    cursor = anchor
    while position < len(schedule):
        current = schedule.iloc[position]
        start = cursor if position == int(schedule.index.get_loc(day)) else current.open
        available = max(0, int((current.close - start) / pd.Timedelta(minutes=1)))
        if remaining <= available:
            return utc_timestamp(start + pd.Timedelta(minutes=remaining))
        remaining -= available
        position += 1
        if position < len(schedule):
            cursor = schedule.iloc[position].open
    raise ValueError("El calendario XNYS no cubre el vencimiento solicitado.")


def _future_session_maturity(observed: pd.Timestamp, sessions: int) -> pd.Timestamp:
    day, schedule, _ = _session_row(observed)
    # The current session is observation time, never outcome time.  Session 1
    # therefore means the next tradable XNYS session, including holiday skips.
    position = bisect_right(schedule.index, day) + sessions - 1
    if position >= len(schedule):
        raise ValueError("El calendario XNYS no cubre el vencimiento solicitado.")
    return utc_timestamp(schedule.iloc[position].close)


def maturity(observed_at, horizon_minutes: int, *, policy: str = POLICY) -> pd.Timestamp:
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int) or not 0 < horizon_minutes <= 525_600:
        raise ValueError("Horizonte inválido: usar minutos enteros entre 1 y 525600.")
    observed = utc_timestamp(observed_at)
    if policy == LEGACY_POLICY:
        return (observed + pd.Timedelta(minutes=horizon_minutes)).ceil("5min")
    if policy != POLICY:
        raise ValueError("Política temporal desconocida.")
    if horizon_minutes in SESSION_HORIZONS:
        return _future_session_maturity(observed, SESSION_HORIZONS[horizon_minutes])
    if horizon_minutes <= MAX_INTRADAY_MINUTES:
        return _intraday_maturity(observed, horizon_minutes)
    raise ValueError(
        "Horizonte sin contrato XNYS: usa hasta 390 minutos de mercado o "
        "1D/1S/1M/6M versionados."
    )


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
        version_policy = (row["integrity_version"], row["horizon_policy"])
        if version_policy not in {
            (LEGACY_VERSION, LEGACY_POLICY),
            (VERSION, POLICY),
        }:
            return False
        observed = utc_timestamp(row["observed_at"])
        due = utc_timestamp(row["available_at"])
        if due != maturity(
            observed,
            row["horizon_minutes"],
            policy=row["horizon_policy"],
        ):
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


def exact_daily_closed_prices(history: pd.DataFrame | None, as_of) -> dict[str, str]:
    """Map raw daily OHLCV session labels to their exact XNYS close.

    Daily bars are required for session-based horizons because public 5-minute
    feeds do not retain enough history for 1M/6M outcomes.  No nearest-session,
    forward-fill or current-price fallback is allowed.
    """

    if history is None or history.empty:
        return {}
    columns = ("Open", "High", "Low", "Close", "Volume")
    if not all(column in history.columns for column in columns):
        raise ValueError("La resolución diaria requiere OHLCV histórico.")
    now = utc_timestamp(as_of)
    source = history.sort_index()
    index = pd.DatetimeIndex(source.index)
    duplicate_days = pd.Index([pd.Timestamp(value).date() for value in index]).duplicated(keep=False)
    prices: dict[str, str] = {}
    for duplicate, label, values in zip(
        duplicate_days,
        index,
        source.loc[:, columns].itertuples(index=False, name=None),
    ):
        if duplicate:
            continue
        day = pd.Timestamp(pd.Timestamp(label).date())
        calendar = _calendar(day.year - 1, day.year + 1)
        if day not in calendar.schedule.index:
            continue
        close_at = utc_timestamp(calendar.schedule.loc[day, "close"])
        if close_at > now:
            continue
        try:
            o, h, l, c, v = map(float, values)
            if not all(math.isfinite(x) for x in (o, h, l, c, v)) or min(o, h, l, c) <= 0 or v < 0:
                continue
            if l > min(o, c) or h < max(o, c) or l > h:
                continue
            prices[close_at.isoformat()] = str(Decimal(str(values[3])))
        except (ValueError, TypeError, InvalidOperation):
            continue
    return prices
