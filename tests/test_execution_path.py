"""A barrier win without a possible entry fill is not a trading win."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from portfolio_tracker.analytics.execution_path import (
    EXECUTION_VERSION, advance_execution_checkpoint, assess_execution_path,
    validate_execution_checkpoint,
)
from portfolio_tracker.analytics.operational_target import make_operational_contract
from tests.test_model_execution_record import _analysis


CUT = datetime(2026, 9, 3, 15, 0, 10, tzinfo=timezone.utc)


def _contract():
    analysis = _analysis()
    return make_operational_contract(
        "SMCI", analysis.horizon_projections[0], 60, analysis, CUT, {},
    )


def _bars(*rows):
    index = pd.date_range("2026-09-03T15:05:00Z", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=index)


def test_frozen_execution_costs_and_deadline_are_signed():
    contract = _contract()
    terms = contract["execution_terms"]
    assert terms["version"] == EXECUTION_VERSION
    assert terms["entry_deadline_at"] == "2026-09-03T20:00:00+00:00"
    assert terms["fill_semantics"] == "POSSIBLE_NOT_BROKER_CONFIRMED"
    assert len(contract["contract_sha256"]) == 64


def test_target_before_limit_entry_is_not_counted_as_success():
    bars = _bars((100.5, 106.0, 100.2, 105.0, 1000.0))
    result = assess_execution_path(_contract(), bars, "2026-09-03T15:10:00Z")
    assert result.status == "NO_FILL_TARGET_PASSED"
    assert result.outcome is None and result.net_pnl_per_share is None


def test_possible_fill_then_target_is_costed_sequentially():
    bars = _bars(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 106.0, 99.0, 104.0, 1000.0),
    )
    result = assess_execution_path(_contract(), bars, "2026-09-03T15:15:00Z")
    assert result.status == "SIMULATED_RESOLVED"
    assert result.outcome == "TP_FIRST"
    assert result.fill_at == "2026-09-03T15:10:00+00:00"
    assert result.exit_at == "2026-09-03T15:15:00+00:00"
    assert result.net_pnl_per_share == pytest.approx(5 - 0.0032 * 205)


def test_gap_stop_uses_real_open_not_ideal_stop():
    bars = _bars(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (90.0, 92.0, 88.0, 90.0, 1000.0),
    )
    result = assess_execution_path(_contract(), bars, "2026-09-03T15:15:00Z")
    assert result.status == "SIMULATED_RESOLVED"
    assert result.outcome == "SL_FIRST"
    assert result.exit_price == 90.0
    assert result.net_pnl_per_share < -10.0


def test_same_candle_entry_and_target_is_ambiguous_not_a_win():
    bars = _bars((100.0, 106.0, 99.0, 104.0, 1000.0))
    result = assess_execution_path(_contract(), bars, "2026-09-03T15:10:00Z")
    assert result.status == "AMBIGUOUS_NO_TRADE"
    assert result.outcome is None


def test_missing_closed_bar_fails_closed():
    bars = _bars((100.0, 101.0, 99.0, 100.0, 1000.0))
    result = assess_execution_path(_contract(), bars, "2026-09-03T15:15:00Z")
    assert result.status == "EVIDENCE_INCOMPLETE"


def test_old_checkpoint_without_fill_state_is_never_backfilled_from_later_bars():
    contract = _contract()
    next_bar = (
        pd.Timestamp("2026-09-03T15:10:00Z"), pd.Timestamp("2026-09-03T15:15:00Z"),
        100.0, 106.0, 99.0, 104.0, 1000.0, "5m",
    )
    state = advance_execution_checkpoint(
        contract, (next_bar,), next_bar[1],
        previous_scanned_through="2026-09-03T15:10:00Z",
    )
    assert state["status"] == "LEGACY_CHECKPOINT_UNASSESSED"
    assert state["fill_price"] is None and state["outcome"] is None


def test_checkpoint_rejects_inconsistent_fill_and_watermark():
    contract = _contract()
    bar = (
        pd.Timestamp("2026-09-03T15:05:00Z"), pd.Timestamp("2026-09-03T15:10:00Z"),
        100.0, 101.0, 99.0, 100.0, 1000.0, "5m",
    )
    state = advance_execution_checkpoint(contract, (bar,), bar[1])
    assert state["status"] == "FILLED_PENDING"
    with pytest.raises(ValueError, match="desincronizado"):
        validate_execution_checkpoint({**state, "scanned_through": "2026-09-03T15:15:00Z"},
                                      contract, bar[1])
    with pytest.raises(ValueError, match="fill incoherente"):
        validate_execution_checkpoint({**state, "fill_price": None}, contract, bar[1])
