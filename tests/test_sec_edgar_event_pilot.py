"""Pure offline checks for the isolated SEC event-coverage pilot."""
from datetime import datetime, timezone

import exchange_calendars as xcals

from scripts.pilot_sec_edgar_events import (
    cik_lookup, coverage, eligible_events, filing_records, parse_utc,
    previous_session_close,
)


def test_sec_cik_lookup_and_original_header_link():
    tickers = {"0": {"cik_str": 1375365, "ticker": "SMCI"},
               "1": {"cik_str": 1045810, "ticker": "NVDA"}}
    assert cik_lookup(tickers, "SMCI") == 1375365
    assert cik_lookup(tickers, "NVDA") == 1045810
    submissions = {"filings": {"recent": {
        "form": ["8-K", "8-K/A"],
        "accessionNumber": ["0001375365-25-000023", "0001375365-25-000024"],
        "filingDate": ["2025-08-05", "2025-08-06"],
        "acceptanceDateTime": ["2025-08-05T20:08:29.000Z", "2025-08-06T20:00:00.000Z"],
        "primaryDocument": ["smci-20250805.htm", "amendment.htm"],
        "items": ["2.02", ""],
    }}}
    events = filing_records(submissions, 1375365)
    assert len(events) == 1  # Exact forms only; amendments are not silently merged.
    assert events[0]["header_url"].endswith("/000137536525000023/0001375365-25-000023.txt")


def test_ny_cut_and_previous_session_policy_excludes_same_day_event():
    calendar = xcals.get_calendar("XNYS")
    cut = datetime(2025, 8, 5, 15, 0, tzinfo=timezone.utc)  # 11:00 NY
    previous_close = previous_session_close(cut, calendar)
    assert previous_close == datetime(2025, 8, 4, 20, 0, tzinfo=timezone.utc)
    events = [
        {"accepted_at_utc": "2025-08-04T19:59:00+00:00", "accession": "old"},
        {"accepted_at_utc": "2025-08-05T14:54:00+00:00", "accession": "new"},
        {"accepted_at_utc": "2025-08-05T14:56:00+00:00", "accession": "too-late"},
    ]
    assert [x["accession"] for x in eligible_events(events, cut, 24, previous_close,
                                                      "previous_session")] == ["old"]
    assert [x["accession"] for x in eligible_events(events, cut, 24, previous_close,
                                                      "five_minute_lag")] == ["old", "new"]


def test_coverage_counts_independent_cuts_not_duplicate_filings():
    calendar = xcals.get_calendar("XNYS")
    cuts = [{"observed_at": "2025-08-05T15:00:00+00:00"},
            {"observed_at": "2025-08-06T15:00:00+00:00"}]
    events = [{"accepted_at_utc": "2025-08-04T19:00:00+00:00", "accession": "one"}]
    result = coverage(cuts, events, calendar)
    assert result["summary"]["previous_session_24h"]["cuts_with_event"] == 1
    assert result["summary"]["previous_session_24h"]["cuts_without_event"] == 1
    assert result["summary"]["previous_session_48h"]["unique_filings_seen"] == 1
    assert parse_utc(result["cuts"][0]["cut_at_utc"]).tzinfo is not None
