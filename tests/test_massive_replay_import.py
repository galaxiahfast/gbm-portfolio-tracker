import pandas as pd
import pytest

from scripts.import_massive_replay import verify_daily


def _source():
    days = pd.DatetimeIndex(["2024-09-30", "2024-10-01"])
    adjusted = []
    raw = []
    for index, day in enumerate(days):
        stamp = day.tz_localize("America/New_York").tz_convert("UTC")
        row = {"t": int(stamp.timestamp() * 1000), "o": 40.0, "h": 41.0,
               "l": 39.0, "c": 40.5, "v": 10000.0}
        adjusted.append(row)
        other = dict(row)
        if index == 0:
            for key in ("o", "h", "l", "c"):
                other[key] *= 10
            other["v"] /= 10
        raw.append(other)
    return days, {"splits": [{"execution_date": "2024-10-01", "split_from": 1, "split_to": 10}],
                  "daily_adjusted_response": {"adjusted": True, "results": adjusted},
                  "daily_raw_response": {"adjusted": False, "results": raw}}


def test_replay_import_verifies_split_units_for_price_and_volume():
    days, source = _source()
    daily, audit = verify_daily(source, days)
    assert len(daily) == 2
    assert audit["price_and_volume_split_factors_verified"] is True


def test_replay_import_rejects_wrong_split_factor():
    days, source = _source()
    source["daily_raw_response"]["results"][0]["c"] = 40.5
    with pytest.raises(ValueError, match="Factor de split"):
        verify_daily(source, days)


def test_replay_import_rejects_missing_session():
    days, source = _source()
    with pytest.raises(ValueError, match="XNYS"):
        verify_daily(source, days.append(pd.DatetimeIndex(["2024-10-02"])))
