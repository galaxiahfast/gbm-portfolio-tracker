"""Technical point-in-time replay: no network, SQLite, fills or account writes.

Only the live analyzer authorizes entries; no alternative daily signal model.
Cheap necessary trigger checks and per-dataset caches bound grid-search work.
Fundamental/news revisions are NOT synthesized from today's information.
"""
from collections import Counter
from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd

from .closed_bars import select_last_closed_bar, utc, NY
from .causal_core import trigger_momentum, INTRADAY_READY

REPLAY_CONTRACT = 'closed-bars-shared-core-v1'


def _ohlcv(frame):
    columns = ['Open','High','Low','Close','Volume']
    data = frame.loc[:,columns].astype(float).copy()
    data.index = pd.DatetimeIndex(data.index)
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError('Replay requiere fechas únicas y ordenadas.')
    if not np.isfinite(data.to_numpy()).all() or (data.iloc[:,:4] <= 0).any().any() or (data.Volume < 0).any():
        raise ValueError('OHLCV no finito o fuera de dominio.')
    if ((data.High < data[['Open','Close','Low']].max(axis=1)) |
        (data.Low > data[['Open','Close','High']].min(axis=1))).any():
        raise ValueError('OHLC inconsistente para replay.')
    return data


@dataclass
class ReplayDataset:
    intraday: pd.DataFrame
    daily: pd.DataFrame
    as_of: object
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.as_of = utc(self.as_of)
        source = _ohlcv(self.intraday)
        source.index = source.index.tz_localize(NY) if source.index.tz is None else source.index.tz_convert(NY)
        source.index = source.index.tz_convert('UTC')
        if any(source.index.minute % 5) or any(source.index.second) or any(source.index.microsecond):
            raise ValueError('Las velas 5m deben estar alineadas al reloj de la sesión.')
        if len(source) < 300 or not (source.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).any():
            raise ValueError('Replay requiere al menos 300 velas reales de 5 minutos, no velas diarias.')
        self.intraday = select_last_closed_bar(source,'5m',self.as_of)
        self.daily = select_last_closed_bar(_ohlcv(self.daily),'1d',self.as_of)
        if len(self.intraday) < 300 or len(self.daily) < 200:
            raise ValueError('Histórico cerrado insuficiente: 300 velas 5m y 200 diarias mínimas.')

    def fingerprint(self, symbol):
        digest = hashlib.sha256(symbol.upper().encode())
        for frame in (self.intraday, self.daily):
            digest.update(pd.util.hash_pandas_object(frame,index=True).values.tobytes())
        digest.update(str(self.as_of).encode())
        return digest.hexdigest()

    def training_prefix(self, fraction):
        cut = int(len(self.intraday)*fraction)
        # last training candle's CLOSE, never the first holdout price
        end = self.intraday.index[cut-1]+pd.Timedelta(minutes=5)
        return ReplayDataset(self.intraday.iloc[:cut], self.daily, end)


def evaluate_replay_cut(dataset, at, *, atr_stop_multiple=2.25, symbol='REPLAY'):
    """Same technical analyzer and causal gate as live, explicit historical clock.

    Macro hysteresis is reconstructed from the closed daily prefix, identical
    to live cold-start (previous_macro_trending=None). No current news is used.
    """
    from .technical_probability import analyze_probability
    cutoff = utc(at)
    if cutoff > dataset.as_of:
        raise ValueError('El corte solicitado supera el corte disponible del replay.')
    return analyze_probability(
        symbol, dataset.intraday.loc[:cutoff], dataset.daily,
        as_of_time=cutoff, atr_stop_multiple=atr_stop_multiple,
    )


def prepare_replay(dataset, config, *, symbol='REPLAY'):
    from .technical_probability import add_intraday_indicators
    from .backtesting import SetupCandidate
    # Cache is local to these exact input frames, not shared with live sessions.
    key = (dataset.fingerprint(symbol), config.stop_atr_multiple)
    if key in dataset._cache:
        return dataset._cache[key]
    source = dataset.intraday
    indicators = add_intraday_indicators(source).dropna(subset=INTRADAY_READY)
    candidates, rejected = [], Counter()
    for i in range(21,len(indicators)):
        prefix = indicators.iloc[max(0,i-21):i+1]
        if not any(trigger_momentum(prefix)):
            continue
        label = indicators.index[i]
        cut = label+pd.Timedelta(minutes=5)
        try:
            analysis = evaluate_replay_cut(dataset,cut,atr_stop_multiple=config.stop_atr_multiple,symbol=symbol)
        except ValueError:
            rejected['Histórico insuficiente o inválido en el corte causal'] += 1
            continue
        if not analysis.activation_trigger_met or analysis.execution_plan_conditional or analysis.risk_veto or analysis.signal_rejected:
            rejected['Veto del motor técnico compartido'] += 1
            continue
        plan = analysis.execution_levels
        score = analysis.operation_probability/10.0
        candidates.append(SetupCandidate(
            signal_position=source.index.get_loc(label), signal_date=cut.isoformat(),
            side=plan.direction, score=score, strength_bucket=max(2,min(8,int(score))),
            atr=analysis.atr_5m, volume_ratio=analysis.volume_ratio, adx=analysis.adx,
            monthly_regime=analysis.monthly_trend.value, exposure_factor=analysis.exposure_factor,
            trigger=analysis.activation_trigger, market_regime=('ALCISTA' if plan.direction=='LONG' else 'BAJISTA'),
            plan_stop=plan.stop_loss, plan_target=plan.take_profit_1,
        ))
    result = (source, candidates, rejected)
    # Only one ATR entry per dataset to avoid retaining large analysis objects.
    if len(dataset._cache) >= 3:
        dataset._cache.clear()
    dataset._cache[key] = result
    return result
