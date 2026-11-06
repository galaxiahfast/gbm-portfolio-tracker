"""Diagnostic tooling must never widen the production emission window."""
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_trigger_replay import (
    gate_diagnostics, joint_counts, planned_cuts,
)
from scripts.research_clock_adapter import (
    research_contract, validate_research_contract, assess_research_execution,
)
from portfolio_tracker.analytics.operational_target import (
    make_operational_contract, validate_operational_contract,
)
from portfolio_tracker.analytics.execution_path import assess_execution_path


def fixture_contract(clock="12:30"):
    cut = pd.Timestamp(f"2026-09-24 {clock}", tz="America/New_York").tz_convert("UTC")
    horizon = SimpleNamespace(engine_name="test", probability_up=70,
                              probability_down=10, probability_range=20)
    analysis = SimpleNamespace(last_price=100, source_bar_closed_at=cut,
                               activation_trigger_met=True, execution_plan_conditional=False,
                               risk_veto=False, signal_rejected=False,
                               execution_levels=SimpleNamespace(direction="LONG", stop_loss=98, take_profit_1=104))
    contract = make_operational_contract("SMCI", horizon, 60, analysis, cut, {})
    protocol = {"protocol_sha256": "a" * 64, "start": "2025-07-21", "end": "2026-09-25",
                "cohorts": [{"time_ny": "11:00"}, {"time_ny": "12:30"}]}
    return contract, protocol, cut


def bars_for(contract):
    start = pd.Timestamp(contract["evaluation_starts_at"])
    due = pd.Timestamp(contract["timeout_at"])
    index = pd.date_range(start, due - pd.Timedelta(minutes=5), freq="5min")
    bars = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.}, index=index)
    bars.iloc[1, bars.columns.get_loc("High")] = 105
    bars.iloc[1, bars.columns.get_loc("Close")] = 104
    return SimpleNamespace(intraday=bars, as_of=due)


def test_research_clock_does_not_authorize_off_anchor_live():
    origin, protocol, cut = fixture_contract()
    assert origin["eligible_at_emission"] is False
    research = research_contract(origin, eligible=True, protocol=protocol, cut=cut)
    validate_research_contract(research, protocol)
    with pytest.raises(ValueError, match="fuera de corte"):
        validate_operational_contract(research)
    assert origin["eligible_at_emission"] is False
    assert validate_operational_contract.__module__.endswith("operational_target")


def test_private_clock_execution_reuses_same_rules_and_does_not_patch_globals():
    from portfolio_tracker.analytics import execution_path
    validator = execution_path.validate_operational_contract
    origin, protocol, cut = fixture_contract()
    research = research_contract(origin, eligible=True, protocol=protocol, cut=cut)
    result = assess_research_execution(research, bars_for(origin), protocol)
    assert result["status"] == "SIMULATED_RESOLVED"
    assert result["outcome"] == "TP_FIRST"
    assert result["fill_price"] == 100
    assert execution_path.validate_operational_contract is validator
    assert assess_execution_path(origin, bars_for(origin).intraday, bars_for(origin).as_of).status == "NOT_AUTHORIZED"


def test_1100_original_and_research_fill_outcomes_identical():
    origin, protocol, cut = fixture_contract("11:00")
    dataset = bars_for(origin)
    research = research_contract(origin, eligible=True, protocol=protocol, cut=cut)
    actual = assess_research_execution(research, dataset, protocol)
    baseline = assess_execution_path(origin, dataset.intraday, dataset.as_of).as_record()
    assert actual == baseline


def test_research_cannot_change_any_price_or_cost():
    origin, protocol, cut = fixture_contract()
    research = research_contract(origin, eligible=True, protocol=protocol, cut=cut)
    research["stop_loss"] = 97
    from scripts.research_clock_adapter import signed_hash
    research["contract_sha256"] = signed_hash(research)
    with pytest.raises(ValueError, match="algo más"):
        validate_research_contract(research, protocol)


def test_zone_and_bullish_body_are_not_hard_trigger_conditions():
    index = pd.date_range("2026-09-24 09:10", periods=22, freq="5min", tz="America/New_York")
    bars = pd.DataFrame({"Open": 100., "High": 100., "Low": 99., "Close": 100., "Volume": 100.,
                         "StochRSI_K": 30., "StochRSI_D": 40., "MACD": 1., "MACD_signal": 0.}, index=index)
    bars.iloc[-1, bars.columns.get_loc("Volume")] = 200
    bars.iloc[-1, bars.columns.get_loc("StochRSI_K")] = 50
    features = {"hierarchy_detail": "Estructura 4h/1h=[1, 1]", "macro_permission": "LONG_ONLY",
                "buy_levels": {"entry_low": 95, "entry_high": 96}, "risk_veto": False,
                "activation_trigger_met": True}
    gates = gate_diagnostics(features, bars, plan_side="LONG")
    assert not gates["inside_buy_zone_descriptive"]
    assert not gates["bullish_closed_candle_descriptive"]
    assert gates["long_core_complete"]
    assert joint_counts([{"gates": gates}])["all_momentum_pass"] == 1


def test_early_close_times_are_skipped_and_daily_is_same_event():
    # NYSE closes 13:00 on the Friday following Thanksgiving.
    index = pd.date_range("2025-11-28 09:30", "2025-11-28 12:55", freq="5min", tz="America/New_York").tz_convert("UTC")
    dataset = SimpleNamespace(intraday=pd.DataFrame(index=index))
    tasks, skipped = planned_cuts(dataset, {"start": "2025-11-28", "end": "2025-11-28"})
    assert [t["time_ny"] for t in tasks] == ["10:00", "11:00", "12:30"]
    assert [t["time_ny"] for t in skipped] == ["14:00", "15:30"]
    assert skipped[-1]["cohorts"] == ["intraday_1530", "daily_1D", "weekly_1S"]
