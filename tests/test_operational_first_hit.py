"""Contract tests for the operational TP/SL first-passage target.

Assumptions made explicit by this suite:

* ``bars_5m`` uses OPEN-labelled, timezone-aware regular-session candles; the
  outcome timestamp is the candle close (label + five minutes), when its OHLC
  is first knowable.  The first evaluable candle opens at
  ``observed_at.ceil("5min")`` so no OHLC range can include pre-emission ticks.
* The prediction freezes direction, reference price, TP, SL and expiry before
  any outcome candle is read.
* Opening gaps are evaluated before unknown intrabar extremes.  If both TP and
  SL occur inside one candle, the unobservable order is labelled ``SL_FIRST``.
* A timeout exists only when the exact candle closing at ``expires_at`` is
  available.  Ambiguous or discontinuous input must fail closed rather than
  using a nearest candle or a later quote.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_tracker.analytics.operational_target import (
    OperationalContract,
    OperationalOutcome,
    resolve_operational_outcome,
    scan_operational_outcome,
)


UTC = timezone.utc
SOURCE_CLOSE = datetime(2026, 9, 3, 15, 0, tzinfo=UTC)  # 11:00 NY (EDT)
OBSERVED_AT = SOURCE_CLOSE + timedelta(seconds=10)
EVALUATION_START = datetime(2026, 9, 3, 15, 5, tzinfo=UTC)
EXPIRY = datetime(2026, 9, 3, 15, 20, tzinfo=UTC)


def contract(direction: str = "LONG", **overrides) -> OperationalContract:
    levels = (
        {"reference_price": Decimal("100"), "take_profit": Decimal("105"),
         "stop_loss": Decimal("95")}
        if direction == "LONG"
        else {"reference_price": Decimal("100"), "take_profit": Decimal("95"),
              "stop_loss": Decimal("105")}
    )
    values = {
        "symbol": "SMCI",
        "direction": direction,
        "observed_at": OBSERVED_AT,
        "source_bar_at": SOURCE_CLOSE,
        "expires_at": EXPIRY,
        **levels,
        **overrides,
    }
    return OperationalContract(**values)


def candles(*rows: tuple[str, float, float, float, float]) -> pd.DataFrame:
    """Build valid 5m OHLCV. Timestamps are candle-open labels."""

    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": [1_000.0] * len(rows),
        },
        index=pd.DatetimeIndex([row[0] for row in rows]),
    )


def decimal(value) -> Decimal:
    return Decimal(str(value))


def assert_failed_closed(call) -> None:
    """Invalid market evidence may be rejected or left unresolved, never used."""

    try:
        result = call()
    except ValueError:
        return
    assert result is None


def test_incremental_evidence_hash_matches_one_shot_path():
    first = candles(("2026-09-03T15:05:00Z", 100, 101, 99, 100.5))
    second = candles(("2026-09-03T15:10:00Z", 100.5, 106, 100, 105.5))
    all_bars = pd.concat([first, second])
    whole = scan_operational_outcome(
        contract(), all_bars, datetime(2026, 9, 3, 15, 15, tzinfo=UTC),
    )
    checkpoint = scan_operational_outcome(
        contract(), first, datetime(2026, 9, 3, 15, 10, tzinfo=UTC),
    )
    resumed = scan_operational_outcome(
        contract(), second, datetime(2026, 9, 3, 15, 15, tzinfo=UTC),
        resume_at=checkpoint.scanned_through,
        previous_evidence_sha256=checkpoint.evidence_sha256,
        previous_evidence_count=checkpoint.evidence_count,
    )
    assert checkpoint.result is None
    assert resumed.result == whole.result
    assert resumed.evidence_sha256 == whole.evidence_sha256
    assert resumed.evidence_count == whole.evidence_count == 2


@pytest.mark.parametrize(
    "direction,outcome,row,expected_exit",
    [
        ("LONG", OperationalOutcome.TP_FIRST,
         ("2026-09-03T15:05:00Z", 100, 105, 99, 104), Decimal("105")),
        ("LONG", OperationalOutcome.SL_FIRST,
         ("2026-09-03T15:05:00Z", 100, 101, 95, 96), Decimal("95")),
        ("SHORT", OperationalOutcome.TP_FIRST,
         ("2026-09-03T15:05:00Z", 100, 101, 95, 96), Decimal("95")),
        ("SHORT", OperationalOutcome.SL_FIRST,
         ("2026-09-03T15:05:00Z", 100, 105, 99, 104), Decimal("105")),
    ],
)
def test_long_and_short_resolve_the_exact_first_5m_hit(
    direction, outcome, row, expected_exit,
):
    result = resolve_operational_outcome(
        contract(direction), candles(row), EVALUATION_START + timedelta(minutes=5),
    )

    assert result is not None
    assert result.outcome is outcome
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp("2026-09-03T15:10:00Z")
    assert decimal(result.exit_price) == expected_exit


def test_timestamp_order_not_input_order_determines_which_level_was_first():
    # The later SL row is intentionally supplied first.  A resolver that trusts
    # DataFrame row order instead of market timestamps will mislabel this case.
    reversed_rows = candles(
        ("2026-09-03T15:10:00Z", 104, 104.5, 94, 96),  # later SL
        ("2026-09-03T15:05:00Z", 100, 105, 99, 104),   # earlier TP
    )

    result = resolve_operational_outcome(
        contract(), reversed_rows, EVALUATION_START + timedelta(minutes=10),
    )

    assert result is not None
    assert result.outcome is OperationalOutcome.TP_FIRST
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp("2026-09-03T15:10:00Z")


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_same_candle_tp_and_sl_is_conservatively_sl_first(direction):
    result = resolve_operational_outcome(
        contract(direction),
        candles(("2026-09-03T15:05:00Z", 100, 106, 94, 100)),
        EVALUATION_START + timedelta(minutes=5),
    )

    assert result is not None
    assert result.outcome is OperationalOutcome.SL_FIRST
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp("2026-09-03T15:10:00Z")


@pytest.mark.parametrize(
    "direction,opening,high,low,expected",
    [
        # Even though the opposite level also appears in the later H/L range,
        # the observed open is causally earlier and therefore wins.
        ("LONG", 106, 107, 94, OperationalOutcome.TP_FIRST),
        ("LONG", 94, 106, 93, OperationalOutcome.SL_FIRST),
        ("SHORT", 94, 106, 93, OperationalOutcome.TP_FIRST),
        ("SHORT", 106, 107, 94, OperationalOutcome.SL_FIRST),
    ],
)
def test_opening_gap_precedes_intrabar_high_low(
    direction, opening, high, low, expected,
):
    result = resolve_operational_outcome(
        contract(direction),
        candles(("2026-09-03T15:05:00Z", opening, high, low, 100)),
        EVALUATION_START + timedelta(minutes=5),
    )

    assert result is not None
    assert result.outcome is expected
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp("2026-09-03T15:10:00Z")


def test_timeout_uses_only_the_exact_expiry_candle_close():
    bars = candles(
        # This straddles observation time and therefore is never evidence.
        ("2026-09-03T15:00:00Z", 100, 110, 90, 101),
        ("2026-09-03T15:05:00Z", 101, 103, 99, 102),
        ("2026-09-03T15:10:00Z", 102, 104, 99, 103),
        ("2026-09-03T15:15:00Z", 103, 104, 98, 101.25),
        # A later quote must not replace the preregistered timeout close.
        ("2026-09-03T15:20:00Z", 101.25, 110, 90, 109),
    )

    assert resolve_operational_outcome(
        contract(), bars, EXPIRY - timedelta(microseconds=1),
    ) is None
    result = resolve_operational_outcome(
        contract(), bars, EXPIRY + timedelta(minutes=5),
    )

    assert result is not None
    assert result.outcome is OperationalOutcome.TIMEOUT
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp(EXPIRY)
    assert decimal(result.exit_price) == Decimal("101.25")


def test_level_hit_in_expiry_candle_precedes_timeout():
    bars = candles(
        ("2026-09-03T15:05:00Z", 100, 102, 98, 101),
        ("2026-09-03T15:10:00Z", 101, 103, 99, 102),
        ("2026-09-03T15:15:00Z", 102, 105, 99, 104),
    )

    result = resolve_operational_outcome(contract(), bars, EXPIRY)

    assert result is not None
    assert result.outcome is OperationalOutcome.TP_FIRST
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp(EXPIRY)


def test_pre_contract_and_post_expiry_touches_are_not_labels():
    bars = candles(
        # This is the source candle: its close is the contract's source_bar_at.
        ("2026-09-03T14:55:00Z", 100, 110, 90, 100),
        # This candle straddles observed_at=15:00:10 and is also ineligible.
        ("2026-09-03T15:00:00Z", 100, 110, 90, 101),
        ("2026-09-03T15:05:00Z", 101, 103, 99, 102),
        ("2026-09-03T15:10:00Z", 102, 104, 99, 103),
        ("2026-09-03T15:15:00Z", 103, 104, 98, 101),
        # Opens exactly at expiry, so its extremes become knowable afterwards.
        ("2026-09-03T15:20:00Z", 101, 110, 90, 109),
    )

    result = resolve_operational_outcome(
        contract(), bars, EXPIRY + timedelta(minutes=5),
    )

    assert result is not None
    assert result.outcome is OperationalOutcome.TIMEOUT
    assert pd.Timestamp(result.outcome_bar_at) == pd.Timestamp(EXPIRY)


def test_unclosed_future_candle_is_not_visible_to_resolution():
    result = resolve_operational_outcome(
        contract(),
        candles(("2026-09-03T15:05:00Z", 100, 106, 99, 105)),
        EVALUATION_START + timedelta(minutes=4, seconds=59),
    )

    assert result is None


def test_missing_exact_expiry_bar_never_uses_nearest_or_later_close():
    bars = candles(
        ("2026-09-03T15:05:00Z", 100, 102, 98, 101),
        ("2026-09-03T15:10:00Z", 101, 103, 99, 102),
        # 15:15 OPEN-labelled candle is deliberately absent.
        ("2026-09-03T15:20:00Z", 103, 104, 99, 102),
    )

    assert_failed_closed(
        lambda: resolve_operational_outcome(
            contract(), bars, EXPIRY + timedelta(minutes=5),
        )
    )


def test_duplicate_or_non_5m_evidence_fails_closed():
    duplicate = candles(
        ("2026-09-03T15:05:00Z", 100, 105, 99, 104),
        ("2026-09-03T15:05:00Z", 100, 101, 95, 96),
    )
    off_grid = candles(
        ("2026-09-03T15:06:00Z", 100, 106, 99, 105),
    )

    for bars in (duplicate, off_grid):
        assert_failed_closed(
            lambda bars=bars: resolve_operational_outcome(
                contract(), bars, EXPIRY,
            )
        )


def test_gap_in_known_path_cannot_claim_a_later_first_hit():
    bars = candles(
        ("2026-09-03T15:05:00Z", 100, 102, 98, 101),
        # Missing 15:10 candle: its unobserved range could have hit either side.
        ("2026-09-03T15:15:00Z", 101, 106, 99, 105),
    )

    assert_failed_closed(
        lambda: resolve_operational_outcome(
            contract(), bars, EXPIRY,
        )
    )


def test_contract_levels_and_direction_are_frozen_at_prediction_time():
    frozen = contract()

    with pytest.raises(FrozenInstanceError):
        frozen.take_profit = Decimal("110")
    with pytest.raises(FrozenInstanceError):
        frozen.stop_loss = Decimal("90")
    with pytest.raises(FrozenInstanceError):
        frozen.reference_price = Decimal("101")
    with pytest.raises(FrozenInstanceError):
        frozen.direction = "SHORT"


@pytest.mark.parametrize(
    "direction,tp,sl",
    [
        ("LONG", Decimal("99"), Decimal("95")),
        ("LONG", Decimal("105"), Decimal("101")),
        ("SHORT", Decimal("101"), Decimal("105")),
        ("SHORT", Decimal("95"), Decimal("99")),
    ],
)
def test_contract_rejects_levels_inconsistent_with_fixed_direction(direction, tp, sl):
    with pytest.raises(ValueError):
        contract(direction, take_profit=tp, stop_loss=sl)
