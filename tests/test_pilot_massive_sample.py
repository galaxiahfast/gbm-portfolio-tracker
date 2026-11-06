"""Provider-pilot guards; synthetic bars here are never counted as evidence."""
from __future__ import annotations

from datetime import date
import hashlib
import json

import pandas as pd
import pytest

from scripts import pilot_massive_sample as pilot


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _FakeSession:
    def __init__(self, schedule, sessions, missing=False, split=False):
        self.schedule = schedule
        self.sessions = sessions
        self.missing = missing
        self.split = split

    def get(self, url, *, params, timeout):
        assert params["apiKey"] == "test-key"
        if url.endswith("/v3/reference/splits"):
            rows = [{"execution_date": str(self.sessions[2].date())}] if self.split else []
            return _Response({"status": "OK", "results": rows})
        bars = []
        for day in self.sessions:
            market_open = self.schedule.loc[day, "open"]
            market_close = self.schedule.loc[day, "close"]
            for stamp in pd.date_range(market_open, market_close, freq="5min", inclusive="left"):
                bars.append({
                    "t": int(stamp.timestamp() * 1000),
                    "o": 40.0, "h": 40.5, "l": 39.5, "c": 40.1, "v": 1000,
                })
        if self.missing:
            bars.pop()
        return _Response({
            "status": "OK", "ticker": "SMCI", "adjusted": True, "results": bars,
        })


def test_sample_audit_checks_full_xnys_grid_and_hash():
    schedule, sessions = pilot._sessions(date(2025, 9, 24))
    envelope = pilot.audit_symbol(
        "SMCI", schedule, sessions, _FakeSession(schedule, sessions), "test-key"
    )
    payload = envelope["payload"]
    assert len(payload["sessions"]) == 5
    assert len(payload["bars"]) == sum(payload["sessions"].values())
    assert all(bar["timestamp_ny"].endswith("-04:00") for bar in payload["bars"])
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert envelope["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_sample_audit_rejects_missing_regular_bar():
    schedule, sessions = pilot._sessions(date(2025, 9, 24))
    with pytest.raises(ValueError, match="huecos 5m"):
        pilot.audit_symbol(
            "SMCI", schedule, sessions,
            _FakeSession(schedule, sessions, missing=True), "test-key",
        )


def test_sample_audit_requires_split_reconciliation():
    schedule, sessions = pilot._sessions(date(2025, 9, 24))
    with pytest.raises(ValueError, match="split dentro de la muestra"):
        pilot.audit_symbol(
            "SMCI", schedule, sessions,
            _FakeSession(schedule, sessions, split=True), "test-key",
        )


def test_no_key_does_not_create_artifact(monkeypatch, tmp_path):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    assert pilot.main(["--output-dir", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []
