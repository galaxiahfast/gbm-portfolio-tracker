"""Single, versioned XNYS clock for live emission, replay and execution.

The 11:00 NY cut is the close of the 10:55–11:00 source candle.  Emission
may finish during that 5-minute bucket, but its OHLC is not an outcome: the
first *fully prospective* candle opens at 11:05.  Both the directional close
and TP/SL/timeout therefore use the same 11:05 start and 12:05 one-hour end.
An off-cut analysis can be displayed, never promoted to an actionable entry.
"""
from __future__ import annotations

import math
from bisect import bisect_right

import pandas as pd

from .closed_bars import NY, _calendar

ANCHOR_VERSION = "XNYS_1100_NEXT_FULL_5M_V1"
SOURCE_CUT_AFTER_OPEN = pd.Timedelta(minutes=90)
BAR = pd.Timedelta(minutes=5)
ACTIONABLE_GRACE = BAR
SESSION_HORIZONS = {1_440: 1, 10_080: 5, 43_200: 21, 259_200: 126}
MAX_INTRADAY_MINUTES = 390


def utc_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("El timestamp debe incluir zona horaria explícita.")
    return stamp.tz_convert("UTC")


def _session(stamp):
    stamp = utc_timestamp(stamp)
    day = pd.Timestamp(stamp.tz_convert(NY).date())
    schedule = _calendar(day.year - 1, day.year + 2).schedule
    if day not in schedule.index:
        raise ValueError("La observación no pertenece a una sesión XNYS.")
    return day, schedule, schedule.loc[day]


def scheduled_cut(value) -> pd.Timestamp:
    """The one replay/live source-bar close for this XNYS session."""
    _, _, session = _session(value)
    cut = utc_timestamp(session.open) + SOURCE_CUT_AFTER_OPEN
    if cut >= utc_timestamp(session.close):
        raise ValueError("La sesión no cubre el corte de las 11:00 NY.")
    return cut


def is_actionable_emission(observed_at, source_bar_at) -> bool:
    """Only a fresh 11:00 source within its five-minute emission grace qualifies."""
    try:
        observed = utc_timestamp(observed_at)
        cut = scheduled_cut(observed)
        return bool(cut <= observed < cut + ACTIONABLE_GRACE
                    and utc_timestamp(source_bar_at) == cut)
    except (TypeError, ValueError, OverflowError):
        return False


def first_evaluable_open(observed_at) -> pd.Timestamp:
    """Never include the candle whose range may contain pre-emission ticks."""
    observed = utc_timestamp(observed_at)
    _session(observed)
    return observed.floor("5min") + BAR


def _trading_minute_maturity(start: pd.Timestamp, minutes: int) -> pd.Timestamp:
    day, schedule, session = _session(start)
    if not session.open < start <= session.close:
        raise ValueError("El anclaje intradía debe seguir una vela regular cerrada.")
    remaining = int(math.ceil(minutes / 5.0) * 5)
    position = int(schedule.index.get_loc(day))
    while position < len(schedule):
        current = schedule.iloc[position]
        cursor = start if position == int(schedule.index.get_loc(day)) else current.open
        available = max(0, int((current.close - cursor) / pd.Timedelta(minutes=1)))
        if remaining <= available:
            return utc_timestamp(cursor + pd.Timedelta(minutes=remaining))
        remaining -= available
        position += 1
    raise ValueError("El calendario XNYS no cubre el vencimiento solicitado.")


def maturity(observed_at, horizon_minutes: int) -> pd.Timestamp:
    """V1 target close shared by directional and first-passage labels."""
    if (isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int)
            or not 0 < horizon_minutes <= 525_600):
        raise ValueError("Horizonte inválido.")
    observed = utc_timestamp(observed_at)
    day, schedule, _ = _session(observed)
    if horizon_minutes in SESSION_HORIZONS:
        index = bisect_right(schedule.index, day) + SESSION_HORIZONS[horizon_minutes] - 1
        if index >= len(schedule):
            raise ValueError("El calendario XNYS no cubre el vencimiento solicitado.")
        return utc_timestamp(schedule.iloc[index].close)
    if horizon_minutes <= MAX_INTRADAY_MINUTES:
        return _trading_minute_maturity(first_evaluable_open(observed), horizon_minutes)
    raise ValueError("Horizonte sin contrato XNYS versionado.")
