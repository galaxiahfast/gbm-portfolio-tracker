from dataclasses import replace
import pandas as pd
from streamlit.testing.v1 import AppTest
from portfolio_tracker.analytics import conditional_zone_reach as engine
from tests.test_zone_reach import history
from tests.test_conditional_zone_reach import daily_history


def test_weighted_uses_several_contexts_instead_of_empty_exact_filter(monkeypatch):
    data, now = history()
    context = engine._daily_context(daily_history())
    context.loc[:'2026-08-27','regime'] = 'RANGO'
    context.loc['2026-08-28','regime'] = 'TENDENCIA'
    monkeypatch.setattr(engine,'_daily_context',lambda _: context)
    result = engine.estimate_conditional_reach(data,daily_history(),100,
        [(101,101,'ABOVE')],now,matching='weighted',min_sessions=12)[0]
    assert result.probability is not None
    assert 8 <= result.effective_samples <= result.samples + 1e-8
    assert '0 coincidencias macro exactas' in result.detail
    assert 'preliminar' in result.status


def test_all_similarity_angles_affect_weight_symmetrically():
    row = engine._daily_context(daily_history()).dropna().iloc[-1]
    original = engine._similarity(row,row,.5,.5,.1,.1)
    assert original == 1
    for key in ('weekly','daily','regime','ema50','atr'):
        other = row.copy()
        other[key] = 'RANGO' if key == 'regime' else -row[key] if key != 'atr' else row[key]*2
        forward = engine._similarity(row,other,.5,.5,.1,.1)
        backward = engine._similarity(other,row,.5,.5,.1,.1)
        assert forward < original and abs(forward-backward) < 1e-12
    assert engine._similarity(row,row,.5,1.5,.1,.1) < original
    assert engine._similarity(row,row,.5,.5,.1,1.1) < original


def test_weighted_never_reads_future_or_mutates_frames():
    data, now = history()
    daily = daily_history()
    original = data.copy()
    zones = [(99,99,'BELOW'),(101,101,'ABOVE')]
    baseline = engine.estimate_conditional_reach(data,daily,100,zones,now,matching='weighted')
    pd.testing.assert_frame_equal(data,original)
    daily.loc['2026-08-31',:] = 9999
    data.loc[data.index >= now,:] = 9999
    assert engine.estimate_conditional_reach(data,daily,100,zones,now,matching='weighted') == baseline


def test_insufficient_data_still_is_not_a_made_up_percentage():
    data, now = history()
    result = engine.estimate_conditional_reach(data.tail(78*5),daily_history(),100,
        [(101,101,'ABOVE')],now,matching='weighted',min_sessions=12)[0]
    assert result.probability is None


def test_six_weighted_probabilities_are_visible_in_three_panels():
    app = AppTest.from_string('''
from dataclasses import replace
from tests.test_zone_pdf import snapshot_fixture
from portfolio_tracker.ui.price_zones import render_price_zones
a,s = snapshot_fixture()
s = replace(s, estimates=tuple(replace(e,model='conditional-v3-weighted',
    close_probability=1.,close_lower=0.,close_upper=20.,effective_samples=15.,
    close_direction='BELOW' if i<3 else 'ABOVE') for i,e in enumerate(s.estimates)))
render_price_zones(a,zone_snapshot=s)
''').run()
    assert not app.exception
    labels = [m.value for m in app.markdown if 'Probabilidad estimada de toque hoy:' in m.value]
    assert len(labels) == 6
    assert len(app.get('column')) == 3
    assert len([c for c in app.caption if 'Probabilidad de cierre' in c.value]) == 6
