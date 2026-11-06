"""Audit gates for the isolated provider acquisition, no network or SQLite."""
import pandas as pd
import pytest

from scripts.download_massive_history import audit_intraday, read_signed, write_signed
from portfolio_tracker.analytics.closed_bars import _calendar


def _sample():
    # Thanksgiving early close: the correct grid is 42 bars, not 78.
    schedule = _calendar(2025, 2025).schedule.loc["2025-11-28":"2025-11-28"]
    row = schedule.iloc[0]
    bars = [{"t": int(ts.timestamp() * 1000), "o": 10, "h": 11,
             "l": 9, "c": 10, "v": 500}
            for ts in pd.date_range(row.open, row.close, freq="5min", inclusive="left")]
    return schedule, {"ticker": "SMCI", "adjusted": True, "results": bars}


def test_history_audits_early_close_without_filling_missing_bars():
    schedule, response = _sample()
    frame, audit = audit_intraday([response], schedule, "SMCI")
    assert len(frame) == audit["expected_bars"] == 42
    assert audit["missing_count"] == 0
    response["results"].pop()
    with pytest.raises(ValueError, match="rechazada"):
        audit_intraday([response], schedule, "SMCI")


def test_history_rejects_duplicate_and_wrong_asset():
    schedule, response = _sample()
    with pytest.raises(ValueError, match="identidad"):
        audit_intraday([response], schedule, "NVDA")
    response["results"].append(response["results"][0])
    with pytest.raises(ValueError, match="rechazada"):
        audit_intraday([response], schedule, "SMCI")


def test_acquisition_is_signed_and_immutable(tmp_path):
    path = tmp_path / "sample.json"
    write_signed(path, {"prices": [1, 2]})
    assert read_signed(path) == {"prices": [1, 2]}
    write_signed(path, {"prices": [1, 2]})
    with pytest.raises(ValueError, match="inmutable"):
        write_signed(path, {"prices": [9, 2]})
