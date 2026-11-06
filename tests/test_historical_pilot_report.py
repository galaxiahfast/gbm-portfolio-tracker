from copy import deepcopy

from scripts.report_historical_pilot import event_key, nonoverlapping, summarize


def _row(start="2025-12-31T16:05:00+00:00", timeout="2025-12-31T17:05:00+00:00", eligible=True):
    return {"horizon_minutes": 60,
            "operational_contract": {"side": "LONG", "entry_price": 100, "stop_loss": 98,
                                     "take_profit": 104, "evaluation_starts_at": start,
                                     "timeout_at": timeout, "eligible_at_emission": eligible},
            "operational_result": {"status": "RESOLVED", "outcome": "TP_FIRST"},
            "execution_result": {"status": "SIMULATED_RESOLVED" if eligible else "NOT_AUTHORIZED",
                                 "outcome": "TP_FIRST" if eligible else None,
                                 "fill_price": 100 if eligible else None}}


def test_report_never_counts_hypothetical_win_as_authorized_entry():
    report = summarize({"observations": [{"horizons": [_row(eligible=False)]}]}, 60)
    assert report["hypothetical_barrier_outcomes"]["TP_FIRST"] == 1
    assert report["technical_authorized_entries"] == 0
    assert report["executable_resolved_events"] == 0


def test_shared_barriers_are_one_event_across_horizons():
    hour = _row()
    day = deepcopy(hour)
    day["horizon_minutes"] = 1440
    day["operational_contract"]["timeout_at"] = "2026-01-02T21:00:00+00:00"
    assert event_key(hour["operational_contract"]) == event_key(day["operational_contract"])


def test_nonoverlap_uses_declared_timeout_not_winning_exit():
    first = _row(timeout="2026-01-02T21:00:00+00:00")
    overlapping = _row(start="2026-01-02T16:05:00+00:00", timeout="2026-01-02T17:05:00+00:00")
    later = _row(start="2026-01-05T16:05:00+00:00", timeout="2026-01-05T17:05:00+00:00")
    assert nonoverlapping([first, overlapping, later]) == [first, later]
