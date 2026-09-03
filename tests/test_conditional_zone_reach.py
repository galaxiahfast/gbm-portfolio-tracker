import pandas as pd
import numpy as np
from portfolio_tracker.analytics.conditional_zone_reach import (
    estimate_conditional_reach, residual_budget, _daily_context,
)
from tests.test_zone_reach import history


def daily_history():
    index = pd.bdate_range('2024-01-01','2026-08-31')
    price = 80 + np.arange(len(index))*.02
    return pd.DataFrame({'Open':price, 'High':price+1, 'Low':price-1, 'Close':price},index=index)


def test_conditioned_touch_and_close_are_separate_and_no_mutation():
    data, now = history()
    daily = daily_history()
    before = data.copy()
    results = estimate_conditional_reach(data,daily,100,[(101,101,'ABOVE'),(99,99,'BELOW')],now)
    assert all(r.samples >= 20 for r in results)
    assert all(r.probability > r.close_probability == 0 for r in results)
    assert all(0 <= r.lower <= r.probability <= r.upper <= 100 for r in results)
    pd.testing.assert_frame_equal(data,before)


def test_context_is_prior_day_no_future_data_leakage():
    data, now = history()
    daily = daily_history()
    zones = [(101,101,'ABOVE')]
    baseline = estimate_conditional_reach(data,daily,100,zones,now)
    daily.loc['2026-08-31',:] = 99999
    data.loc[data.index >= now,'High'] = 99999
    assert estimate_conditional_reach(data,daily,100,zones,now) == baseline


def test_atr_budget_decreases_with_consumption_and_time():
    assert residual_budget(.02,1.5,.5) < residual_budget(.02,.5,.5)
    assert residual_budget(.02,.5,.95) < residual_budget(.02,.5,.5)
    assert residual_budget(.02,3.,1.) == 0
    assert residual_budget(.02,3.,.5) > 0


def test_insufficient_macro_evidence_never_falls_back_to_generic_frequency():
    data, now = history()
    result = estimate_conditional_reach(data,daily_history().tail(15),100,[(101,101,'ABOVE')],now)[0]
    assert result.probability is None and result.close_probability is None
    assert 'insuficiente' in result.status


def test_end_of_session_and_missing_prefix():
    data, now = history()
    daily = daily_history()
    zones = [(101,101,'ABOVE')]
    assert estimate_conditional_reach(data,daily,100,zones,'2026-08-31 20:01Z')[0].probability is None
    data = data.drop(pd.Timestamp('2026-08-31 13:30Z'))
    assert 'incompleta' in estimate_conditional_reach(data,daily,100,zones,now)[0].status


def test_completed_week_context_does_not_change_with_future_week():
    daily = daily_history()
    baseline = _daily_context(daily).loc['2026-08-20']
    daily.loc['2026-08-24':,:] *= 5
    pd.testing.assert_series_equal(_daily_context(daily).loc['2026-08-20'],baseline)


def test_different_regime_is_excluded_not_silently_pooled(monkeypatch):
    from portfolio_tracker.analytics import conditional_zone_reach as module
    data, now = history()
    context = _daily_context(daily_history())
    context.loc[:'2026-08-27', 'regime'] = 'RANGO'
    context.loc['2026-08-28', 'regime'] = 'TENDENCIA'
    monkeypatch.setattr(module, '_daily_context', lambda _: context)
    result = module.estimate_conditional_reach(data,daily_history(),100,[(101,101,'ABOVE')],now)[0]
    assert result.samples == 0 and result.probability is None


def test_farther_targets_have_no_higher_probability():
    data, now = history()
    zones = [(100.5,100.5,'ABOVE'),(101,101,'ABOVE'),(105,105,'ABOVE')]
    results = estimate_conditional_reach(data,daily_history(),100,zones,now)
    assert [r.probability for r in results] == sorted([r.probability for r in results],reverse=True)
