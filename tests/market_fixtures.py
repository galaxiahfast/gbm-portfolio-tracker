"""Synthetic observations on actual regular-session timestamps (not overnight)."""
import pandas as pd


def intraday_index(periods):
    parts = [pd.date_range(f"{day.date()} 13:30", periods=78, freq="5min", tz="UTC")
             for day in pd.bdate_range("2026-08-20", periods=10)]
    index = parts[0]
    for part in parts[1:]:
        index = index.append(part)
    return index[:periods]
