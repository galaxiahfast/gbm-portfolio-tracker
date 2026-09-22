"""Operational selection must use net first-hit EV, never directional rank."""
from __future__ import annotations

import pytest

from portfolio_tracker.analytics.net_expectation import (
    TradingCostPolicy,
    evaluate_long_opportunity,
    select_highest_net_expectation,
)
from portfolio_tracker.services.operational_model_registry import _approved_result


def _opportunity(**overrides):
    arguments = dict(
        horizon="1 Día", entry=100, stop=95, take_profit=110,
        tp_first=0.8, sl_first=0.1, timeout=0.1,
        capital=10_000, cash=10_000,
    )
    arguments.update(overrides)
    return evaluate_long_opportunity(**arguments)


def test_net_expectation_deducts_both_sides_and_stresses_timeout():
    item = _opportunity()
    assert item.net_tp_per_share == pytest.approx(9.37)
    assert item.net_sl_per_share == pytest.approx(-5.585)
    assert item.net_ev_per_share == pytest.approx(6.379)
    assert item.shares == 30  # Concentration is tighter than the risk cap.
    assert item.net_ev_total == pytest.approx(item.net_ev_per_share * item.shares)
    assert item.monetary_risk <= item.risk_budget
    assert item.eligible is True


def test_higher_raw_tp_probability_does_not_win_if_net_portfolio_ev_is_lower():
    high_probability = _opportunity(horizon="1 Hora", entry=100, stop=98,
                                    take_profit=105, tp_first=0.8, sl_first=0.1, timeout=0.1)
    higher_net_ev = _opportunity(horizon="1 Semana", entry=100, stop=98,
                                 take_profit=110, tp_first=0.7, sl_first=0.2, timeout=0.1)
    assert high_probability.tp_first > higher_net_ev.tp_first
    assert high_probability.net_ev_total < higher_net_ev.net_ev_total
    assert select_highest_net_expectation((high_probability, higher_net_ev)) == higher_net_ev


def test_negative_ev_or_cost_damaged_rr_blocks_selection():
    negative = _opportunity(tp_first=0.2, sl_first=0.1, timeout=0.7)
    gross_only = _opportunity(entry=100, stop=98, take_profit=103)
    assert negative.net_ev_per_share < 0
    assert not negative.eligible
    assert gross_only.net_reward_risk < 1.5
    assert not gross_only.eligible
    assert select_highest_net_expectation((negative, gross_only)) is None


def test_position_size_includes_costs_in_cash_and_stop_risk():
    item = _opportunity(capital=1_000, cash=100.1, current_market_value=0)
    assert item.shares == 0  # One share requires $100.30 including entry costs.
    assert item.eligible is False
    with pytest.raises(ValueError, match="sumen uno"):
        _opportunity(tp_first=0.7, sl_first=0.1, timeout=0.1)
    with pytest.raises(ValueError, match="costes"):
        TradingCostPolicy(commission_bps_per_side=-1)


def test_registry_rechecks_both_oos_baselines_before_promotion():
    metrics = {
        "samples": 80, "brier": 0.20, "raw_brier": 0.25,
        "baseline_brier": 0.35, "log_loss": 0.40,
        "raw_log_loss": 0.45, "baseline_log_loss": 0.60,
    }
    validation = {
        "raw": {"brier": 0.27, "log_loss": 0.48},
        "calibrated": {"brier": 0.22, "log_loss": 0.43},
        "baseline": {"brier": 0.38, "log_loss": 0.62},
    }
    record = {
        "status": "APPROVED_SEALED_HOLDOUT_CALIBRATED",
        "promotable": True,
        "score_semantics": "HISTORICAL_OOS_CALIBRATED_PRELIMINARY",
        "final_holdout": {"status": "OPENED_ONCE_AFTER_PROTOCOL_FREEZE", "metrics": metrics},
        "calibration": {"approved": True, "validation_metrics": validation},
    }
    assert _approved_result(record)
    assert not _approved_result({**record, "final_holdout": {
        "status": "OPENED_ONCE_AFTER_PROTOCOL_FREEZE",
        "metrics": {**metrics, "brier": 0.40},
    }})
    assert not _approved_result({**record, "calibration": {
        "approved": True,
        "validation_metrics": {**validation, "raw": {"brier": 0.4, "log_loss": 0.7}},
    }})
