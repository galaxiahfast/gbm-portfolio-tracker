"""Offline causal-boundary and denominator checks for EDGAR research sidecar."""

import exchange_calendars as xcals

from scripts.build_sec_edgar_sidecar import build_cut_feature, compare, group_stats


def test_sparse_features_use_only_events_known_under_policy():
    calendar = xcals.get_calendar("XNYS")
    cut = "2025-08-05T15:00:00+00:00"  # 11:00 NY
    events = [
        {"accepted_at_utc": "2025-08-04T19:55:00+00:00", "accession": "prior",
         "form": "8-K"},
        {"accepted_at_utc": "2025-08-05T14:54:00+00:00", "accession": "today",
         "form": "10-Q"},
        {"accepted_at_utc": "2025-08-05T14:56:00+00:00", "accession": "too-late",
         "form": "10-K"},
    ]
    source_hashes = {"sec_submissions_payload_sha256": "a" * 64}
    strict = build_cut_feature(cut, events, calendar, "previous_session", source_hashes)
    assert strict["policy"] == "PREVIOUS_XNYS_CLOSE_V1"
    assert strict["event_count_24h"] == 1
    assert strict["event_count_48h"] == 1
    assert strict["has_8k"] and not strict["has_10q"] and not strict["has_10k"]
    assert strict["event_accessions_48h"] == ["prior"]
    assert strict["source_hashes"] == source_hashes
    assert strict["days_since_last_event"] > 0

    sensitivity = build_cut_feature(cut, events, calendar, "five_minute_lag", source_hashes)
    assert sensitivity["event_count_24h"] == 2
    assert sensitivity["has_10q"] and not sensitivity["has_10k"]
    assert sensitivity["event_accessions_48h"] == ["prior", "today"]


def test_censored_cut_is_not_counted_as_tp_sl_or_timeout():
    cases = [{"status": "RESOLVED", "outcome": "TP_FIRST", "authorized": False, "filled": False},
             {"status": "RIGHT_CENSORED", "outcome": None, "authorized": False, "filled": False}]
    result = group_stats(cases)
    assert result["cuts"] == 2
    assert result["resolved_hypothetical"] == 1
    assert (result["tp"], result["sl"], result["timeout"]) == (1, 0, 0)
    assert result["unresolved_or_no_plan"] == 1
    assert result["tp_rate_resolved_pct"] == 100.0


def test_event_delta_uses_same_symbol_and_side_baseline():
    cases = [
        {"symbol": "SMCI", "side": "LONG", "event": True, "status": "RESOLVED",
         "outcome": "TP_FIRST", "authorized": False, "filled": False},
        {"symbol": "SMCI", "side": "LONG", "event": False, "status": "RESOLVED",
         "outcome": "SL_FIRST", "authorized": False, "filled": False},
        {"symbol": "NVDA", "side": "LONG", "event": False, "status": "RESOLVED",
         "outcome": "TP_FIRST", "authorized": False, "filled": False},
    ]
    rows = compare(cases, strata=("symbol", "side"))
    smci_event = next(x for x in rows if x["symbol"] == "SMCI" and x["group"] == "event")
    assert smci_event["delta_tp_vs_no_event_pp"] == 100.0
    nvda_event = next(x for x in rows if x["symbol"] == "NVDA" and x["group"] == "event")
    assert nvda_event["cuts"] == 0 and nvda_event["delta_tp_vs_no_event_pp"] is None
