"""Signed three-class contract, shared by UI emission and historical labels."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from portfolio_tracker.analytics.probability_calibration import validate_distribution
from portfolio_tracker.services.model_observations import canonical

CONTRACT = "CLOSE_RANGE_3CLASS_V1"


@lru_cache(maxsize=1)
def engine_revision():
    root = Path(__file__).resolve().parents[1] / 'analytics'
    digest = hashlib.sha256()
    for name in ('technical_probability.py', 'multi_timeframe.py', 'decision_engines.py',
                 'fundamental_news.py', 'chart_patterns.py', 'probability_calibration.py', 'causal_core.py','technical_validity.py'):
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def make_scenario_contract(symbol, horizon, horizon_minutes, parameters):
    model = dict(symbol=symbol.upper(), engine=horizon.engine_name,
                 horizon_minutes=horizon_minutes, parameters=parameters,
                 engine_revision=engine_revision(), target=CONTRACT)
    contract = dict(
        version=CONTRACT, model=model,
        model_id=hashlib.sha256(canonical(model).encode()).hexdigest(),
        probabilities=[horizon.probability_up/100, horizon.probability_range/100,
                       horizon.probability_down/100],
        range_low=horizon.range_low, range_high=horizon.range_high,
    )
    validate_contract(contract)
    return contract


def validate_contract(contract):
    if contract['version'] != CONTRACT or contract['model']['target'] != CONTRACT:
        raise ValueError('Contrato de escenarios desconocido.')
    if contract['model_id'] != hashlib.sha256(canonical(contract['model']).encode()).hexdigest():
        raise ValueError('Identidad de modelo inconsistente.')
    vector = validate_distribution(contract['probabilities'])
    low, high = float(contract['range_low']), float(contract['range_high'])
    if not all(math.isfinite(v) for v in (low, high)) or not 0 < low <= high:
        raise ValueError('Banda de cierre inválida.')
    return vector, low, high


def outcome_class(price, low, high):
    value = float(price)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Precio de resolución inválido.')
    return 0 if value > high else 2 if value < low else 1


def apply_scenario_calibration(horizon, result):
    """Pass exactly the vector evaluated OOS to all consumers; no clipping."""
    up, ranging, down = result.probabilities
    bias = 'Alcista' if up > max(ranging, down) else 'Bajista' if down > max(up, ranging) else 'Rango'
    return replace(horizon, probability_up=100*up, probability_range=100*ranging,
                   probability_down=100*down, bias=bias,
                   probability_status=result.status, calibration_samples=result.sample_size,
                   brier_score=result.brier_score,
                   calibration_training_samples=len(result.split.training),
                   calibration_fit_samples=len(result.split.calibration),
                   calibration_holdout_samples=len(result.split.holdout),
                   calibration_excluded=result.split.excluded,
                   raw_brier_score=result.raw_brier_score,
                   baseline_brier_score=result.baseline_brier_score,
                   calibration_detail=result.reason)
