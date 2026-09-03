"""Point-in-time OHLCV: timestamps denote candle OPEN, never candle close.

NYSE calendar includes DST, holidays and early closes. Naive intraday timestamps
are interpreted in New York; naive daily timestamps are session labels.
"""
from functools import lru_cache

import pandas as pd
import exchange_calendars as xcals

NY = "America/New_York"
OHLCV = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def utc(value=None):
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@lru_cache(maxsize=8)
def _calendar(first_year, last_year):
    return xcals.get_calendar("XNYS", start=f"{first_year}-01-01", end=f"{last_year}-12-31")


def select_last_closed_bar(frame, timeframe, as_of=None):
    """Return ALL closed bars up to as_of, excluding the forming bucket.

    For 1h/4h inputs, labels must be session-anchored (09:30 NY). End-of-day
    shortened buckets close at the exchange close, not at an artificial 17:30.
    Weekly/monthly labels follow pandas W-FRI/ME; holiday final sessions work.
    """
    if frame.empty:
        return frame.copy()
    source = frame.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")].copy()
    idx = pd.DatetimeIndex(source.index)
    now = utc(as_of)
    calendar = _calendar(idx.min().year - 1, max(idx.max().year, now.year) + 1)
    schedule = calendar.schedule
    valid = []
    for stamp in idx:
        local = stamp.tz_localize(NY) if stamp.tzinfo is None else stamp.tz_convert(NY)
        day = pd.Timestamp(stamp.date() if timeframe in ("1d", "1wk", "1mo") else local.date())
        if timeframe in ("1wk", "1mo"):
            period = day.to_period("W-FRI" if timeframe == "1wk" else "M")
            sessions = schedule.loc[period.start_time.normalize():period.end_time.normalize()]
            end = sessions.iloc[-1]["close"] if not sessions.empty else None
        elif day not in schedule.index:
            end = None
        else:
            session = schedule.loc[day]
            if timeframe == "1d":
                end = session["close"]
            else:
                duration = pd.Timedelta(minutes={"5m": 5, "1h": 60, "4h": 240}[timeframe])
                start = local.tz_convert("UTC")
                end = min(start + duration, session["close"]) if session["open"] <= start < session["close"] else None
        valid.append(end is not None and utc(end) <= now)
    return source.loc[valid].copy()


def resample_closed(frame, timeframe, as_of=None):
    """Build 1h/4h from closed 5m with complete coverage, separately per session."""
    source = select_last_closed_bar(frame, "5m", as_of)
    if source.empty:
        return source
    source.index = pd.DatetimeIndex(source.index)
    source.index = source.index.tz_localize(NY) if source.index.tz is None else source.index.tz_convert(NY)
    parts = []
    for _, session in source.groupby(source.index.date):
        origin = session.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
        grouped = session.resample(timeframe, origin=origin)
        bars = grouped.agg(OHLCV).dropna(subset=["Open", "Close"])
        bars = select_last_closed_bar(bars, timeframe, as_of)
        # Missing 5m observations must not masquerade as a complete higher bar.
        cal = _calendar(origin.year - 1, origin.year + 1)
        close = cal.schedule.loc[pd.Timestamp(origin.date()), "close"]
        counts = grouped["Close"].count()
        expected = [int((min(s + pd.Timedelta(timeframe), close) - s) / pd.Timedelta(minutes=5)) for s in bars.index]
        bars = bars.loc[[counts.loc[s] == n for s, n in zip(bars.index, expected)]]
        parts.append(bars)
    return pd.concat(parts).sort_index() if parts else source.iloc[:0]
