"""Observed spread, causal marketable limits, and explicit missing evidence."""
import pandas as pd

from scripts.near_miss_quote_audit import QuotePath, exactly_one_failure, quote_valid


def contract(side='LONG'):
    return {'side':side,'observed_at':'2025-07-29T14:00:00Z','evaluation_starts_at':'2025-07-29T14:05:00Z',
            'timeout_at':'2025-07-29T14:06:00Z','entry_price':100.,'stop_loss':99. if side=='LONG' else 101.,
            'take_profit':101. if side=='LONG' else 99.,'execution_terms':{'entry_deadline_at':'2025-07-29T20:00:00Z'},
            'model':{'horizon_minutes':60}}


def quote(at,bid=99.98,ask=100.):
    return {'t':at,'bp':bid,'ap':ask,'bs':10,'as':10,'c':['R']}


def seeded(side='LONG'):
    p=QuotePath(contract(side))
    p.feed(quote('2025-07-29T14:00:00Z'))
    return p


def test_exact_single_operand_no_selection_by_result():
    g={'eligible':False,'side':'LONG','non_temporal':{'permission':False,'risk':True},'micro':{'volume':True},'failed_conditions':['permission']}
    assert exactly_one_failure(g)
    g['micro']['volume']=False
    assert not exactly_one_failure(g)


def test_long_uses_ask_entry_and_bid_exit_no_assumed_spread():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z',99.80,100.))
    p.feed(quote('2025-07-29T14:05:30Z',101.10,101.30))
    p.feed(quote('2025-07-29T14:06:00Z',101.20,101.40))
    r=p.finish()
    assert r['status']=='RESOLVED_QUOTE_COUNTERFACTUAL'
    assert r['exit']['outcome']=='TP_FIRST'
    assert abs(r['gross_quote_pnl_per_share']-1.10)<1e-8
    assert abs(r['possible_fill']['spread_observed']-.20)<1e-8
    assert r['net_pnl'] is None and not r['broker_fill']


def test_short_uses_bid_entry_and_ask_cover():
    p=seeded('SHORT')
    p.feed(quote('2025-07-29T14:05:00Z',100.,100.20))
    p.feed(quote('2025-07-29T14:05:30Z',98.70,98.90))
    p.feed(quote('2025-07-29T14:06:00Z',98.60,98.80))
    r=p.finish()
    assert r['exit']['outcome']=='TP_FIRST'
    assert abs(r['gross_quote_pnl_per_share']-1.10)<1e-8


def test_limit_not_filled_when_trade_price_would_touch_but_ask_does_not():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z',99.99,100.05))
    p.feed(quote('2025-07-29T14:05:30Z',99.99,100.05))
    p.feed(quote('2025-07-29T14:06:00Z',99.99,100.05))
    assert p.finish()['possible_fill'] is None


def test_stop_gap_liquidates_at_observed_bid_not_ideal_stop():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z',99.98,100.))
    p.feed(quote('2025-07-29T14:05:30Z',97.,97.10))
    p.feed(quote('2025-07-29T14:06:00Z',97.50,97.60))
    r=p.finish()
    assert r['exit']['price']==97.
    assert r['gross_quote_pnl_per_share']==-3.
    assert r['classification']=='HYPOTHETICAL_GROSS_LOSS_AVOIDED'


def test_tp_before_entry_cannot_become_blocked_gain():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z',101.10,101.20))
    p.feed(quote('2025-07-29T14:05:30Z',99.98,100.))
    p.feed(quote('2025-07-29T14:06:00Z',101.10,101.20))
    r=p.finish()
    assert r['possible_fill'] is None and r['classification']=='N_D'


def test_missing_timeout_cannot_be_resolved_with_later_quote():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z'))
    p.feed(quote('2025-07-29T14:06:01Z',101.10,101.20))
    r=p.finish()
    assert r['status']=='N_D_TIMEOUT_QUOTE'
    assert r['exit'] is None


def test_quote_gap_requires_explicit_nd():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z'))
    p.c['timeout_at']='2025-07-29T14:10:00Z'
    p.end=pd.Timestamp(p.c['timeout_at'])
    p.feed(quote('2025-07-29T14:09:59Z',101.10,101.20))
    r=p.finish()
    assert r['status']=='N_D_QUOTES_INCOMPLETE'
    assert r['gross_quote_pnl_per_share'] is None


def test_invalid_crossed_or_nonregular_quotes_excluded():
    assert not quote_valid(quote('2025-07-29T14:05:00Z',101.,100.))
    q=quote('2025-07-29T14:05:00Z');q['c']=['Y']
    assert not quote_valid(q)


def test_excursion_after_stop_not_achieved_pnl():
    p=seeded()
    p.feed(quote('2025-07-29T14:05:00Z'))
    p.feed(quote('2025-07-29T14:05:30Z',98.90,99.))
    p.feed(quote('2025-07-29T14:06:00Z',102.,102.10))
    r=p.finish()
    assert r['mfe_until_exit_per_share']==0.
    assert r['mfe_full_horizon_per_share']==2.
    assert r['exit']['outcome']=='SL_FIRST'
