import numpy as np
import pandas as pd

from portfolio_tracker.analytics.closed_bars import _calendar
from portfolio_tracker.analytics.zone_reach import estimate_zone_reach


def history():
    schedule = _calendar(2025, 2027).schedule.loc['2026-06-01':'2026-08-31']
    index = pd.DatetimeIndex([t for _, s in schedule.iterrows()
                             for t in pd.date_range(s['open'], s['close'] - pd.Timedelta(minutes=5), freq='5min')])
    data = pd.DataFrame({'Open': 100., 'High': 102., 'Low': 98., 'Close': 100.}, index=index)
    return data, pd.Timestamp('2026-08-31 16:00', tz='UTC')


def test_six_zones_have_independent_probabilities_and_no_mutation():
    data, now = history()
    before = data.copy(deep=True)
    zones = [(99., 99.), (97., 97.), (96., 96.), (101., 101.), (103., 103.), (104., 104.)]
    results = estimate_zone_reach(data, 100., zones, now)
    assert [r.probability for r in results] == [100, 0, 0, 100, 0, 0]
    assert all(r.samples >= 20 and 0 <= r.lower <= r.probability <= r.upper <= 100 for r in results)
    pd.testing.assert_frame_equal(data, before)


def test_no_future_leakage_and_no_forming_bars():
    data, now = history()
    zones = [(101., 101.)]
    baseline = estimate_zone_reach(data, 100., zones, now)
    data.loc[data.index >= now, 'High'] = 100000.
    assert estimate_zone_reach(data, 100., zones, now) == baseline


def test_closed_market_stale_sparse_and_invalid_inputs():
    data, now = history()
    zones = [(101., 101.)]
    assert estimate_zone_reach(data, 100., zones, '2026-08-30 16:00Z')[0].probability is None
    assert estimate_zone_reach(data, 100., zones, '2026-08-31 21:00Z')[0].probability is None
    assert estimate_zone_reach(data.iloc[:100], 100., zones, now)[0].probability is None
    assert estimate_zone_reach(data.tail(78), 100., zones, now)[0].probability is None
    assert estimate_zone_reach(data, 101., zones, now)[0].probability is None
    assert estimate_zone_reach(data, 100., [(None, None)], now)[0].probability is None


def test_naive_new_york_index_matches_aware_and_inside_zone():
    data, now = history()
    zones = [(99., 101.), (101., 101.)]
    expected = estimate_zone_reach(data, 100., zones, now)
    data.index = data.index.tz_convert('America/New_York').tz_localize(None)
    assert estimate_zone_reach(data, 100., zones, now) == expected
    assert expected[0].status == 'En zona al corte; no predicción'


def test_bad_or_missing_bars_are_excluded():
    data, now = history()
    data.loc[data.index < pd.Timestamp('2026-08-31', tz='UTC'), 'High'] = np.nan
    assert estimate_zone_reach(data, 100., [(101., 101.)], now)[0].samples == 0


def test_frequency_counts_sessions_not_bars_and_rebases_prices():
    data, now = history()
    dates = sorted(set(data.index.date))[:-1]
    for i, day in enumerate(dates):
        mask = data.index.date == day
        data.loc[mask, 'High'] = 104. if i % 2 == 0 else 101.
    result = estimate_zone_reach(data, 100., [(103., 103.)], now)[0]
    assert 40 < result.probability < 60
    doubled = estimate_zone_reach(data * 2, 200., [(206., 206.)], now)[0]
    assert doubled == result


def test_early_close_is_not_extended_to_normal_close():
    data, _ = history()
    # Thanksgiving Friday closes at 13:00 New York (18:00 UTC).
    result = estimate_zone_reach(data, 100., [(101., 101.)], '2026-11-27 18:01Z')[0]
    assert result.status == 'Fuera de sesión'


def test_ui_links_estimates_to_each_zone_without_context_scores():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
from portfolio_tracker.ui.price_zones import DisplayZone, _render_list
from portfolio_tracker.analytics.zone_reach import estimate_zone_reach
from tests.test_zone_reach import history
data, now = history()
levels = [99., 97., 96., 101., 103., 104.]
zones = [DisplayZone(str(i), v, v, "test", 80., "alcista") for i,v in enumerate(levels)]
estimates = estimate_zone_reach(data, 100., [(z.low,z.high) for z in zones], now)
_render_list(zones, 100., estimates)
''').run()
    assert not app.exception
    values = [c.value for c in app.caption if 'Probabilidad estimada' in c.value]
    assert len(values) == 6
    assert [v.rsplit(': ', 1)[1] for v in values] == ['100%', '0%', '0%', '100%', '0%', '0%']
    assert not any('contexto' in c.value for c in app.caption)
