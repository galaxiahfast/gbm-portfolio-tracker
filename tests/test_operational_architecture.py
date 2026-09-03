from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_tracker.analytics.closed_bars import select_last_closed_bar, resample_closed
from portfolio_tracker.analytics.decision_engines import (
    adx_hysteresis, RegimeEngine, SetupEngine, TriggerEngine, Regime, Setup, Permission,
)
from portfolio_tracker.db import Database
from portfolio_tracker.models import TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.operational_state import synchronize_position, read_state, macro_memory, audit_operational_history
from portfolio_tracker.services.pdf_report import executive_decision
from tests.test_pdf_report import _analysis


def bars(index, close=100):
    return pd.DataFrame(dict(Open=close, High=close + 1, Low=close - 1, Close=close, Volume=1000), index=index)


def test_closed_5m_includes_exact_boundary_despite_volume():
    source = bars(pd.date_range("2026-09-01 13:30", periods=3, freq="5min", tz="UTC"))
    source.iloc[0, source.columns.get_loc("Volume")] = 0
    closed = select_last_closed_bar(source, "5m", "2026-09-01 13:40Z")
    assert len(closed) == 2
    assert closed.Volume.iloc[0] == 0
    assert len(select_last_closed_bar(source, "5m", "2026-09-01 13:39:59Z")) == 1


def test_daily_holiday_early_close_and_dst():
    source = bars(pd.DatetimeIndex(["2026-11-26", "2026-11-27"]))
    assert select_last_closed_bar(source, "1d", "2026-11-27 17:59Z").empty
    assert len(select_last_closed_bar(source, "1d", "2026-11-27 18:00Z")) == 1
    summer = bars(pd.DatetimeIndex(["2026-09-01"]))
    assert select_last_closed_bar(summer, "1d", "2026-09-01 19:59Z").empty
    assert len(select_last_closed_bar(summer, "1d", "2026-09-01 20:00Z")) == 1


def test_hourly_four_hour_drop_forming_and_missing_buckets():
    source = bars(pd.date_range("2026-09-01 13:30", periods=78, freq="5min", tz="UTC"))
    assert len(resample_closed(source, "1h", "2026-09-01 15:00Z")) == 1
    assert resample_closed(source, "4h", "2026-09-01 17:29Z").empty
    assert len(resample_closed(source, "4h", "2026-09-01 17:30Z")) == 1
    assert len(resample_closed(source, "4h", "2026-09-01 20:00Z")) == 2
    assert resample_closed(source.drop(source.index[0]), "1h", "2026-09-01 14:30Z").empty


def test_weekly_monthly_never_admit_forming_period():
    weekly = bars(pd.DatetimeIndex(["2026-09-04"]))
    assert select_last_closed_bar(weekly, "1wk", "2026-09-02 20:00Z").empty
    assert len(select_last_closed_bar(weekly, "1wk", "2026-09-04 20:00Z")) == 1
    monthly = bars(pd.DatetimeIndex(["2026-09-30"]))
    assert select_last_closed_bar(monthly, "1mo", "2026-09-02 20:00Z").empty


def test_hysteresis_has_distinct_entry_exit_thresholds():
    state = False
    results = []
    for value in [24, 25, 26, 24, 21, 20, 24, 26]:
        state = adx_hysteresis(value, state)
        results.append(state)
    assert results == [False, False, True, True, True, False, False, True]
    assert not adx_hysteresis(float("nan"), True)


def trending(sign=1):
    frame = bars(pd.date_range("2025-01-01", periods=80))
    frame["Close"] = [100 + sign * i / 2 for i in range(80)]
    frame["Low"] = frame.Close - 1
    frame["High"] = frame.Close + 1
    frame["ADX14"] = 28.0
    return frame


def test_macro_has_no_micro_vote_and_requires_all_frames():
    up, down = trending(), trending(-1)
    engine = RegimeEngine()
    assert engine.evaluate(up, up, up).permission == Permission.LONG_ONLY
    assert engine.evaluate(down, down, down).permission == Permission.SHORT_ONLY
    assert engine.evaluate(up, up, down).permission == Permission.BOTH_REDUCED
    assert engine.evaluate(up, up, up.iloc[:2]).permission == Permission.NO_TRADE
    weaker = up.assign(ADX14=23.0)
    assert engine.evaluate(up, weaker, up, True).permission == Permission.LONG_ONLY
    assert engine.evaluate(up, weaker, up, False).permission == Permission.BOTH_REDUCED
    setup = SetupEngine().evaluate(up, up)
    assert setup.long_allowed and not setup.short_allowed


def test_trigger_cannot_bypass_macro_or_volume():
    frame = bars(pd.date_range("2026-09-01 13:30", periods=22, freq="5min", tz="UTC"))
    frame["StochRSI_K"], frame["StochRSI_D"] = 10.0, 15.0
    frame["MACD"], frame["MACD_signal"] = 1.0, 0.0
    frame.loc[frame.index[-1], ["StochRSI_K", "Close", "Volume"]] = [20, 102, 1500]
    setup = Setup(True, False, 95, 110, "test")
    trigger = TriggerEngine()
    assert trigger.evaluate(frame, Regime(Permission.LONG_ONLY, True, ""), setup).activated
    assert not trigger.evaluate(frame, Regime(Permission.SHORT_ONLY, True, ""), setup).activated
    frame.loc[frame.index[-1], "Volume"] = 900
    assert not trigger.evaluate(frame, Regime(Permission.LONG_ONLY, True, ""), setup).activated


def repository(tmp_path):
    repo = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repo.database.initialize()
    repo.ensure_initial_capital()
    return repo


def fill(repo, analysis, side, quantity="1"):
    repo.add_trade(TradeDraft(symbol=analysis.symbol, side=side, quantity=Decimal(quantity),
                             price_usd=Decimal("40"), commission_usd=Decimal("0"),
                             reported_total_usd=Decimal("40") * Decimal(quantity),
                             executed_at=analysis.as_of, validation_status="VERIFIED"))


def test_position_survives_restart_micro_reversal_and_rerun(tmp_path):
    repo, analysis = repository(tmp_path), _analysis()
    flat = synchronize_position(repo.database, analysis)
    assert flat.position_state == "FLAT"
    fill(repo, analysis, TradeSide.BUY)
    original_cash = repo.cash_balance_usd()
    adopted = synchronize_position(repo.database, analysis)
    assert adopted.position_state == "LONG_ACTIVE"
    flipped = replace(analysis, stoch_overbought_extreme=True, risk_veto=True,
                      probability_up=10, probability_down=90,
                      execution_levels=analysis.sell_levels)
    restarted = Database(repo.database.path)
    latest = synchronize_position(restarted, flipped)
    assert latest.execution_levels == adopted.execution_levels
    assert latest.position_state == "LONG_ACTIVE"
    assert latest.signal.value == "HOLD_LONG"
    assert executive_decision(latest).label == "GESTIONAR POSICIÓN · LONG"
    assert not latest.activation_trigger_met
    assert repo.cash_balance_usd() == original_cash
    with restarted.connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM operational_events").fetchone()[0]
    synchronize_position(restarted, flipped)
    with restarted.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM operational_events").fetchone()[0] == count
    assert macro_memory(restarted, analysis.symbol) == analysis.macro_trending
    audit_operational_history(restarted)


def test_stop_alert_does_not_sell_and_fill_releases_position(tmp_path):
    repo, analysis = repository(tmp_path), _analysis()
    fill(repo, analysis, TradeSide.BUY)
    adopted = synchronize_position(repo.database, analysis)
    newer = analysis.intraday_indicators.copy()
    stamp = analysis.as_of + timedelta(minutes=5)
    newer.loc[stamp] = newer.iloc[-1]
    newer.loc[stamp, "Low"] = adopted.execution_levels.stop_loss - 1
    moved = replace(analysis, as_of=stamp, intraday_indicators=newer)
    alert = synchronize_position(repo.database, moved)
    assert alert.position_state == "EXIT_PENDING"
    assert len(repo.list_trades()) == 1
    fill(repo, moved, TradeSide.SELL)
    assert synchronize_position(repo.database, moved).position_state == "FLAT"
    assert repo.cash_balance_usd() == Decimal("921.05")


def test_corrupted_state_fails_closed(tmp_path):
    repo, analysis = repository(tmp_path), _analysis()
    synchronize_position(repo.database, analysis)
    with repo.database.transaction() as connection:
        connection.execute("UPDATE operational_events SET payload_json='{}'")
    with pytest.raises(ValueError, match="SHA-256"):
        synchronize_position(repo.database, analysis)
