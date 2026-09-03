"""F07/F08: conservative risk and fail-closed latest-bar validation."""
from dataclasses import replace
import math
import numpy as np
import pandas as pd
import pytest

from portfolio_tracker.analytics import backtesting as bt
from portfolio_tracker.analytics import technical_probability as technical
from portfolio_tracker.analytics.technical_validity import (
    TechnicalDataVeto, current_indicator_suffix, validate_freshness,
)
from tests.test_causal_replay import dataset


def simulate(side, gap=True, costs=30):
    frame = pd.DataFrame(dict(Open=[100.,100.,90.],High=[101.,102.,99.],
        Low=[99.,99.,50.],Close=[100.,101.,98.]),index=pd.date_range('2026-09-01 13:30',periods=3,freq='5min',tz='UTC'))
    if side=='SHORT':
        original=frame.copy()
        frame['Open'],frame['Close']=200-original.Open,200-original.Close
        frame['High'],frame['Low']=200-original.Low,200-original.High
    candidate=bt.SetupCandidate(0,frame.index[0].isoformat(),side,7.,7,1.,2.,30.,
        'NEUTRAL',1.,'test','TEST',95. if side=='LONG' else 105.,110. if side=='LONG' else 90.)
    return bt._simulate_trade('TEST','OOS',frame,candidate,
        bt.BacktestConfig(holding_sessions=1,commission_bps_per_side=costs,slippage_bps_per_side=0),
        10000,.6)


@pytest.mark.parametrize('side,opening',[('LONG',90.),('SHORT',110.)])
def test_gap_stop_fills_at_open_and_does_not_mark_post_exit_extremes(side,opening):
    trade=simulate(side)
    assert trade.exit_price==opening
    assert trade.exit_reason=='Stop por gap de apertura'
    assert trade.net_pnl_usd < -100
    expected=trade.quantity*(100+opening)*.003
    assert trade.costs_usd==pytest.approx(expected)
    assert trade.maximum_adverse_excursion_usd==pytest.approx(-trade.net_pnl_usd)
    assert trade.mark_to_market[-1][1]==0  # Opening gap, not a fictitious close-time fill.
    assert min(mark[2] for mark in trade.mark_to_market)==pytest.approx(trade.net_pnl_usd)
    metrics,curve=bt.calculate_metrics([trade],setups=1,rejected=0,starting_capital=10000)
    assert metrics.maximum_drawdown_pct>=-trade.net_pnl_usd/10000*100
    assert curve[-1][1]==pytest.approx(10000+trade.net_pnl_usd)


def test_float_drawdown_is_visible_even_when_trade_finishes_profitable():
    frame=pd.DataFrame(dict(Open=[100.,100.,105.],High=[101.,110.,121.],
        Low=[99.,90.,104.],Close=[100.,105.,120.]),index=pd.date_range('2026-09-01 13:30',periods=3,freq='5min',tz='UTC'))
    candidate=bt.SetupCandidate(0,frame.index[0].isoformat(),'LONG',7.,7,1.,2.,30.,
        'NEUTRAL',1.,'test','TEST',80.,140.)
    trade=bt._simulate_trade('TEST','OOS',frame,candidate,
        bt.BacktestConfig(holding_sessions=1,commission_bps_per_side=0,slippage_bps_per_side=0),10000,.6)
    assert trade.net_pnl_usd>0
    metrics,curve=bt.calculate_metrics([trade],setups=1,rejected=0,starting_capital=10000)
    assert metrics.maximum_drawdown_pct==pytest.approx(100/10050*100)
    assert min(value for _,value in curve)<10000
    assert trade.maximum_adverse_excursion_usd==50


def test_simultaneous_positions_are_aggregated_before_drawdown():
    trade=simulate('LONG',costs=0)
    date='2026-09-01T13:35:00+00:00'
    a=replace(trade,symbol='A',net_pnl_usd=0,
              mark_to_market=((date,1,100.),(date,2,-100.),(date,4,0.)))
    b=replace(trade,symbol='B',net_pnl_usd=0,
              mark_to_market=((date,1,-100.),(date,2,100.),(date,4,0.)))
    metrics,curve=bt.calculate_metrics([a,b],setups=2,rejected=0,starting_capital=20000)
    assert metrics.maximum_drawdown_pct==0
    assert all(value==20000 for _,value in curve)


def test_legacy_trade_without_marks_is_still_supported():
    trade=replace(simulate('LONG'),mark_to_market=())
    metrics,curve=bt.calculate_metrics([trade],setups=1,rejected=0,starting_capital=10000)
    assert metrics.maximum_drawdown_pct==pytest.approx(-trade.net_pnl_usd/10000*100)
    assert len(curve)==1


def test_flat_rsi_is_undefined_but_genuine_all_gains_can_be_100():
    assert technical._rsi(pd.Series([100.]*80)).tail(40).isna().all()
    assert technical._rsi(pd.Series(range(1,81),dtype=float)).iloc[-1]==100
    recent_flat=pd.Series([100+math.sin(i) for i in range(80)]+[101.]*15)
    assert pd.isna(technical._rsi(recent_flat).iloc[-1])


@pytest.mark.parametrize('field', ['RSI14','StochRSI_K','StochRSI_D','ADX14'])
def test_invalid_latest_indicator_never_falls_back_to_old_bar(dataset,monkeypatch,field):
    original=technical.add_intraday_indicators
    def invalid(frame):
        result=original(frame)
        result.loc[result.index[-1],field]=np.nan
        return result
    monkeypatch.setattr(technical,'add_intraday_indicators',invalid)
    with pytest.raises(TechnicalDataVeto) as caught:
        technical.analyze_probability('SMCI',dataset.intraday,dataset.daily,as_of_time=dataset.as_of)
    assert caught.value.state=='UNKNOWN'
    assert caught.value.risk_veto
    assert not caught.value.activation_trigger_met
    assert caught.value.as_of==dataset.intraday.index[-1]


def test_flat_recent_prices_veto_current_cut_even_with_valid_older_history(dataset):
    flat=dataset.intraday.copy()
    flat.loc[flat.index[-15:],['Open','High','Low','Close']]=100.
    with pytest.raises(TechnicalDataVeto,match='plano'):
        technical.analyze_probability('SMCI',flat,dataset.daily,as_of_time=dataset.as_of)


def test_valid_suffix_does_not_join_discontinuous_old_indicators():
    frame=pd.DataFrame({'RSI14':[50.,np.nan,40.,45.]})
    assert current_indicator_suffix(frame,['RSI14'],'test').index.tolist()==[2,3]
    frame.loc[3,'RSI14']=None
    with pytest.raises(TechnicalDataVeto):
        current_indicator_suffix(frame,['RSI14'],'test')


def clock_frame(stamp):
    return pd.DataFrame(dict(Open=[100.],High=[101.],Low=[99.],Close=[100.],Volume=[100.]),
                        index=pd.DatetimeIndex([stamp]))


def test_exchange_freshness_handles_early_close_weekend_and_missing_last_bar():
    intraday=clock_frame('2026-11-27T17:55:00Z')
    daily=clock_frame('2026-11-27')
    validate_freshness(intraday,daily,'2026-11-28T18:00:00Z')
    with pytest.raises(TechnicalDataVeto,match='desactualizado'):
        validate_freshness(clock_frame('2026-11-27T17:50:00Z'),daily,'2026-11-28T18:00:00Z')
    with pytest.raises(TechnicalDataVeto,match='primer cierre'):
        validate_freshness(intraday,daily,'2026-11-30T14:31:00Z')
    with pytest.raises(TechnicalDataVeto,match='diario'):
        validate_freshness(intraday,clock_frame('2026-11-25'),'2026-11-27T18:00:00Z')
