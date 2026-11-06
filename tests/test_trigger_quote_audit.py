"""Quote compatibility is not proof of broker execution."""
import pytest

from scripts.audit_trigger_quote_windows import inspect_quotes


def quote(t, bid, ask, size=1):
    return {"t": t, "bp": bid, "ap": ask, "bs": size, "as": size}


@pytest.mark.parametrize("side,bid,ask,possible", [
    ("LONG", 99.99, 100.01, False), ("LONG", 99.99, 100., True),
    ("SHORT", 99.99, 100.01, False), ("SHORT", 100., 100.01, True),
])
def test_bid_ask_not_midpoint_controls_possible_limit(side, bid, ask, possible):
    result = inspect_quotes([quote("2025-09-19T14:05:01Z", bid, ask)], side, 100,
                            "2025-09-19T14:05:00Z", "2025-09-19T14:10:00Z")
    assert result["fill_possible_from_quotes"] is possible
    assert result["broker_fill_confirmed"] is False


def test_crossed_or_empty_liquidity_not_execution_evidence():
    quotes = [quote("2025-09-19T14:05:01Z", 101, 99), quote("2025-09-19T14:05:02Z", 99, 100, size=0)]
    result = inspect_quotes(quotes, "LONG", 100, "2025-09-19T14:05:00Z", "2025-09-19T14:10:00Z")
    assert result["excluded_invalid_quotes"] == 2
    assert result["first_compatible_quote"] is None


def test_future_quote_cannot_certify_possible_fill():
    with pytest.raises(ValueError, match="outside"):
        inspect_quotes([quote("2025-09-19T14:10:01Z", 99, 100)], "LONG", 100,
                       "2025-09-19T14:05:00Z", "2025-09-19T14:10:00Z")
