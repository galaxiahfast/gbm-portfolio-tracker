"""Research-only trigger safeguards; production AND is never imported here."""
import numpy as np
import pandas as pd

from portfolio_tracker.analytics.learned_trigger_research import (
    FEATURE_NAMES, TARGET_CLASSES, completed_peer_correlation,
    embargoed_training, fit_model, predict, score_labels, strategy_counts,
)


def test_peer_correlation_excludes_current_and_future_daily_closes():
    dates = pd.bdate_range('2026-01-01', periods=40)
    first = pd.DataFrame({'Close': [100 + i + (i % 3) for i in range(40)]}, index=dates)
    peer = pd.DataFrame({'Close': [150 + 2*i + (i % 5) for i in range(40)]}, index=dates)
    cut = pd.Timestamp(str(dates[30].date()) + ' 11:00', tz='America/New_York')
    expected = completed_peer_correlation(first, peer, cut)
    changed = first.copy()
    changed.loc[dates[30]:, 'Close'] *= 50
    assert completed_peer_correlation(changed, peer, cut) == expected


def test_embargo_and_label_availability_are_both_required():
    sessions = ['2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04']
    rows = [
        {'session_ny':'2026-09-01','outcome':'TP_FIRST',
         'label_available_at':'2026-09-02T15:00:00+00:00'},
        {'session_ny':'2026-09-02','outcome':'SL_FIRST',
         'label_available_at':'2026-09-04T16:00:00+00:00'},
        {'session_ny':'2026-09-03','outcome':'TIMEOUT',
         'label_available_at':'2026-09-03T16:00:00+00:00'},
    ]
    result = embargoed_training(rows, '2026-09-04', sessions=sessions)
    assert result == rows[:1]


def test_multinomial_ridge_uses_three_classes_and_not_and_gate():
    rows = []
    for index in range(45):
        outcome = TARGET_CLASSES[index % 3]
        vector = [float((index % 3) == 0), float((index % 3) == 1)]
        vector += [float((index + j) % 7) / 10 for j in range(len(FEATURE_NAMES) - 2)]
        rows.append({'features':vector,'outcome':outcome,'net_return':
                     0.02 if outcome == 'TP_FIRST' else -0.02 if outcome == 'SL_FIRST' else 0.0,
                     'ema_rule':index % 2 == 0,'current_and':False})
    fitted = fit_model(rows, 0.1)
    predictions = predict(fitted, rows)
    assert predictions.shape == (45, 3)
    assert np.allclose(predictions.sum(axis=1), 1.0)
    assert score_labels(rows, predictions)['n'] == 45
    assert strategy_counts(rows, predictions, fitted)['CURRENT_AND']['long_signals'] == 0
