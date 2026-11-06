"""Offline tests: no credentials, production database or market network."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts import collect_prospective_snapshots as collector
from scripts import label_prospective_snapshots as labeler
from scripts import prospective_archive_common as common
from scripts import prospective_coverage as coverage
from scripts import backfill_prospective_gaps as backfill
from scripts.download_massive_history import read_signed, write_signed


def test_only_five_exact_xnys_cut_windows_are_accepted():
    accepted = datetime(2026, 9, 30, 15, 1, tzinfo=timezone.utc)  # 11:01 NY
    assert common.active_cut(accepted)[1] == "11:00"
    assert common.active_cut(datetime(2026, 9, 30, 15, 5, tzinfo=timezone.utc)) is None
    assert common.active_cut(datetime(2026, 9, 30, 16, 31, tzinfo=timezone.utc))[1] == "12:30"
    assert common.active_cut(datetime(2026, 10, 3, 15, 1, tzinfo=timezone.utc)) is None


def test_provider_bars_are_complete_and_open_labeled():
    rows = [{"t": "2026-09-30T15:00:00Z", "o": 40, "h": 41,
             "l": 39, "c": 40.5, "v": 10}]
    frame = common.clean_bars(rows, "5Min")
    assert str(frame.index[0]) == "2026-09-30 15:00:00+00:00"
    assert frame.Close.iloc[0] == 40.5
    with pytest.raises(ValueError, match="OHLCV"):
        common.clean_bars([{**rows[0], "h": 39}], "5Min")


def test_finnhub_receipt_is_ours_and_later_headlines_are_excluded(monkeypatch):
    received = datetime(2026, 9, 30, 15, 1, tzinfo=timezone.utc)
    original_utc = common.utc
    monkeypatch.setattr(common, "utc", lambda value=None: received if value is None else original_utc(value))
    class Response:
        status_code = 200

        def json(self):
            return [
                {"id": 1, "headline": "Earlier filing", "datetime": int((received - timedelta(minutes=3)).timestamp()),
                 "related": "SMCI", "source": "provider"},
                {"id": 2, "headline": "Later article", "datetime": int((received + timedelta(minutes=1)).timestamp()),
                 "related": "SMCI"},
                {"id": 3, "headline": "Wrong ticker", "datetime": int((received - timedelta(minutes=3)).timestamp()),
                 "related": "NVDA"},
            ]
    class HTTP:
        def get(self, *args, **kwargs):
            assert "token" in kwargs["params"]
            return Response()
    headlines, meta = common.fetch_current_headlines(HTTP(), "hidden", "SMCI", received)
    assert [item["id"] for item in headlines] == ["1"]
    assert headlines[0]["collected_at_utc"] == received.isoformat()
    assert meta["raw_count"] == 6 and meta["retained_count"] == 1


def test_no_news_is_missing_not_neutral():
    cut = datetime(2026, 9, 30, 15, 1, tzinfo=timezone.utc)
    assert collector.sentiment_window([], cut, 4) == {
        "status": "N/D", "score": None, "article_count": 0,
        "headline_ids": [], "window_hours": 4}


def test_research_fill_is_separate_from_production_anchor():
    index = pd.date_range("2026-09-30T15:05:00Z", periods=2, freq="5min")
    bars = pd.DataFrame({"Open": [40.5, 40.1], "High": [40.6, 40.4],
                         "Low": [40.2, 39.9], "Close": [40.3, 40.0],
                         "Volume": [100, 100]}, index=index)
    contract = {"evaluation_starts_at": "2026-09-30T15:05:00+00:00",
                "execution_terms": {"entry_deadline_at": "2026-09-30T20:00:00+00:00"},
                "entry_price": 40.0, "stop_loss": 39.0,
                "take_profit": 41.0, "side": "LONG", "eligible_at_emission": False}
    possible, status = labeler.research_fill_possible(contract, bars, "2026-09-30T15:15:00+00:00")
    assert possible is True and status == "OHLC_LIMIT_REACH_NOT_BROKER_FILL"
    assert contract["eligible_at_emission"] is False


def test_no_plan_remains_pending_and_example_never_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ARCHIVE", tmp_path)
    monkeypatch.setattr(labeler, "ARCHIVE", tmp_path)
    monkeypatch.setattr(coverage, "ARCHIVE", tmp_path)
    monkeypatch.setattr(common, "ARCHIVE", tmp_path)
    example_path = collector.write_example()
    assert read_signed(example_path)["synthetic_example"] is True
    report = coverage.build_report(save=False)
    assert all(row["snapshots"] == 0 for row in report["rows"])
    path = tmp_path / "snapshots" / "2026-09-30" / "20260930_1100_SMCI_1D.json"
    write_signed(path, {"symbol": "SMCI", "horizon": "1D", "scheduled_cut_at_utc":
                        "2026-09-30T15:00:00+00:00", "operational_contract": None,
                        "training_performed": False})
    result = labeler.label_once(clock=datetime(2026, 10, 1, 22, tzinfo=timezone.utc))
    assert result["no_plan"] == 1 and result["labeled"] == 0


def test_degraded_backfill_does_not_increase_prospective_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "ARCHIVE", tmp_path)
    target = tmp_path / "backfills" / "2026-10-01" / "20261001_1000_SMCI_1h.json"
    write_signed(target, {"backfilled": True, "pit_quality": "degraded",
                          "training_eligible": False, "data_source": common.DATA_SOURCE,
                          "historical_cut_at_utc": "2026-10-01T14:00:00+00:00",
                          "symbol": "SMCI", "horizon": "1h"})
    report = coverage.build_report(clock=datetime(2026, 10, 5, tzinfo=timezone.utc), save=False)
    assert report["backfill_count_excluded_from_denominators"] == 1
    assert all(row["snapshots"] == 0 for row in report["rows"])


def test_repeated_backfill_checks_signed_records_without_calling_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "ARCHIVE", tmp_path)
    for day in backfill.MISSED_DAYS:
        for symbol in common.SYMBOLS:
            for horizon in common.HORIZONS:
                write_signed(backfill._target(day, symbol, horizon), {
                    "backfilled": True, "pit_quality": "degraded",
                    "training_eligible": False})
    result = backfill.reconstruct(clock=datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert result["records_saved"] == 0
    assert result["already_present"] == 8
