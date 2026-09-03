"""F03/F04: causal partitions and a single auditable three-class predictor."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

import pytest

from portfolio_tracker.analytics.probability_calibration import (
    CalibrationSample, calibrate_probability, calibrate_scenarios,
    chronological_split, _fit_isotonic, _predict_isotonic, _joint_prediction,
)
from portfolio_tracker.analytics.multi_timeframe import HorizonProjection
from portfolio_tracker.services.scenario_calibration import (
    make_scenario_contract, apply_scenario_calibration, outcome_class,
)
from tests.test_live_model_integrity import repo, bars, AT


def samples(count=1000):
    result = []
    for i in range(count):
        at = datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(days=i)
        vector = [(0.7,0.2,0.1), (0.2,0.6,0.2), (0.1,0.2,0.7)][i % 3]
        result.append(CalibrationSample(vector, i % 3, at,
                      at+timedelta(minutes=30), at+timedelta(hours=1)))
    return result


def horizon():
    return HorizonProjection('1 Hora', 70, 20, 10, 'Alcista', 110, 98, 102,
                             90, 2, 95, 108, engine_name='test')


def test_binary_holdout_labels_do_not_fit_curve():
    original = [(0.3,0), (0.7,1)]*250
    changed = original[:400]+[(p,1-y) for p,y in original[400:]]
    before = calibrate_probability(.7, original)
    after = calibrate_probability(.7, changed)
    assert before.calibrated_probability == after.calibrated_probability
    assert before.isotonic_curve == after.isotonic_curve
    assert after.brier_score > before.brier_score
    changed_train = [(p,1-y) for p,y in original[:300]]+original[300:]
    assert calibrate_probability(.7, changed_train) == before


def test_smoothing_cannot_break_isotonic_monotonicity():
    curve = _fit_isotonic([(.1,0)]*100 + [(.2,0)] + [(.3,0)]*200 + [(.9,1)]*20)
    predictions = [_predict_isotonic(i/100,curve) for i in range(101)]
    assert predictions == sorted(predictions)


def test_split_sorted_chronological_and_purges_late_labels():
    rows = samples(10)
    rows[5] = replace(rows[5], resolved_at=rows[6].observed_at)
    rows[7] = replace(rows[7], resolved_at=rows[8].observed_at+timedelta(seconds=1))
    split = chronological_split(reversed(rows))
    assert split.nominal_sizes == (6,2,2)
    assert tuple(map(len,(split.training,split.calibration,split.holdout))) == (5,1,2)
    assert split.excluded == 2
    assert split.training == tuple(rows[:5])


def test_split_deoverlaps_duplicates_and_rejects_naive_or_invalid_time():
    rows = samples(10)
    split = chronological_split([r for r in rows for _ in range(2)])
    assert split.excluded == 10
    for block in (split.training,split.calibration,split.holdout):
        assert all(a.available_at < b.observed_at for a,b in zip(block,block[1:]))
    with pytest.raises(ValueError):
        chronological_split([replace(rows[0], observed_at=rows[0].observed_at.replace(tzinfo=None))])
    with pytest.raises(ValueError):
        chronological_split([replace(rows[0], resolved_at=rows[0].observed_at)])


def test_multiclass_brier_is_exact_coupled_holdout_not_in_sample():
    rows = samples()
    raw = (.65,.25,.1)
    result = calibrate_scenarios(raw, rows)
    assert result.empirically_calibrated
    assert result.split.nominal_sizes == (600,200,200)
    assert sum(result.probabilities) == pytest.approx(1)
    evaluated = [_joint_prediction(r.probabilities,result.isotonic_curves) for r in rows[800:]]
    expected = sum(sum((p-int(k==r.outcome))**2 for k,p in enumerate(v))
                   for v,r in zip(evaluated,rows[800:]))/200
    assert result.brier_score == pytest.approx(expected)
    changed = rows[:800]+[replace(r,outcome=(r.outcome+1)%3) for r in rows[800:]]
    after = calibrate_scenarios(raw, changed)
    assert after.probabilities == result.probabilities
    assert after.isotonic_curves == result.isotonic_curves
    assert after.brier_score > result.brier_score
    changed_train = [replace(r,outcome=(r.outcome+1)%3) for r in rows[:600]]+rows[600:]
    assert calibrate_scenarios(raw,changed_train).probabilities == result.probabilities


@pytest.mark.parametrize('vector', [(0.7,0.3,0.2), (float('nan'),0.5,0.5), (-.1,.5,.6)])
def test_invalid_distributions_rejected_not_clipped(vector):
    with pytest.raises(ValueError):
        calibrate_scenarios(vector, [])


def test_small_or_class_deficient_calibration_remains_heuristic():
    raw = (.65,.25,.1)
    result = calibrate_scenarios(raw,samples(20))
    assert not result.empirically_calibrated
    assert result.probabilities == raw
    assert result.brier_score == result.raw_brier_score
    rows = samples()
    rows[600:800] = [replace(r,outcome=0) for r in rows[600:800]]
    assert not calibrate_scenarios(raw,rows).empirically_calibrated
    assert calibrate_scenarios(raw,[]).brier_score is None


def test_ui_keeps_exact_evaluated_vector_and_existing_levels():
    original = horizon()
    result = calibrate_scenarios((.7,.2,.1), samples())
    updated = apply_scenario_calibration(original,result)
    assert (updated.probability_up,updated.probability_range,updated.probability_down) == tuple(p*100 for p in result.probabilities)
    assert updated.bullish_target == original.bullish_target
    assert updated.range_low == original.range_low
    assert updated.range_high == original.range_high
    assert updated.bearish_target == original.bearish_target
    assert updated.calibration_holdout_samples == 200


@pytest.mark.parametrize('price,expected', [(103,0),(102,1),(98,1),(97,2)])
def test_frozen_close_classes_are_disjoint_and_exhaustive(price,expected):
    assert outcome_class(price,98,102) == expected


def add_observation(repo, at, contract):
    repo.record_live_model_observation(
        symbol='SMCI',observed_at=at,source_bar_at=at,reference_price=Decimal(100),
        raw_probability_up=Decimal('.7'),horizon_minutes=60,
        parameters_json=json.dumps({'scenario_contract':contract}),
    )
    due = at+timedelta(hours=1)
    repo.resolve_live_model_observations(symbol='SMCI',current_as_of=due+timedelta(minutes=1),
                                         historical_bars=bars((due,101)))


def test_repository_model_isolation_frozen_labels_and_asof(repo):
    repo.ensure_initial_capital()
    contract = make_scenario_contract('SMCI',horizon(),60,{'atr':2})
    other = make_scenario_contract('SMCI',horizon(),60,{'atr':3})
    # Later date inserted first; labels of the older rows must retain old bounds.
    later = AT+timedelta(days=3)
    changed_bounds = make_scenario_contract('SMCI',replace(horizon(),range_high=100.5),60,{'atr':2})
    assert changed_bounds['model_id'] == contract['model_id']
    add_observation(repo,later,changed_bounds)
    add_observation(repo,AT,contract)
    add_observation(repo,later+timedelta(days=1),other)
    cutoff = later+timedelta(days=2)
    rows = repo.live_scenario_calibration_samples('SMCI',horizon_minutes=60,model_id=contract['model_id'],as_of=cutoff)
    assert len(rows) == 2
    assert [r.observed_at for r in rows] == [AT,later]
    assert [r.outcome for r in rows] == [1,0]
    early = repo.live_scenario_calibration_samples('SMCI',horizon_minutes=60,model_id=contract['model_id'],as_of=AT+timedelta(hours=1))
    assert early == ()  # maturity passed but resolution was not known yet
    assert repo.live_scenario_calibration_samples('SMCI',horizon_minutes=360,model_id=contract['model_id'],as_of=cutoff) == ()
    assert repo.cash_balance_usd() == Decimal('921.05')
    assert repo.verify_live_model_observations() == (3,())


def test_repository_rejects_tampered_multiclass_result(repo):
    contract = make_scenario_contract('SMCI',horizon(),60,{})
    add_observation(repo,AT,contract)
    with repo.database.transaction() as connection:
        connection.execute('DROP TRIGGER live_resolution_immutable')
        connection.execute("UPDATE live_model_observations SET outcome_price='103'")
    assert repo.live_scenario_calibration_samples('SMCI',horizon_minutes=60,model_id=contract['model_id'],as_of=AT+timedelta(days=3)) == ()
