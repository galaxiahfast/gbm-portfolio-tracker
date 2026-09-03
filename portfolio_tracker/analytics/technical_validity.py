"""Fail-closed technical validity. No database/state/forecast writes."""
import math
import pandas as pd
from .closed_bars import _calendar, utc, NY


class TechnicalDataVeto(ValueError):
    state = "UNKNOWN"
    risk_veto = True
    activation_trigger_met = False
    operation_probability = 0.0

    def __init__(self, reason, *, as_of=None):
        self.reason, self.as_of = reason, as_of
        super().__init__("UNKNOWN / INDEFINIDO · entrada vetada: " + reason)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_raw_tail(frame, label):
    if len(frame) < 2:
        raise TechnicalDataVeto(f"{label}: faltan velas cerradas.")
    last = frame.tail(15)
    required = ['Open','High','Low','Close','Volume']
    if not all(_finite(v) for v in last[required].to_numpy().ravel()):
        raise TechnicalDataVeto(f"{label}: OHLCV no finito en el tramo reciente.",as_of=frame.index[-1])
    if (last[['Open','High','Low','Close']] <= 0).any().any() or (last.Volume < 0).any():
        raise TechnicalDataVeto(f"{label}: precios o volumen fuera de dominio.")
    if ((last.High < last[['Open','Close','Low']].max(axis=1)) |
        (last.Low > last[['Open','Close','High']].min(axis=1))).any():
        raise TechnicalDataVeto(f"{label}: OHLC inconsistente.")
    if len(last)==15 and math.isclose(float(last.Close.max()),float(last.Close.min()),rel_tol=1e-12,abs_tol=1e-12):
        raise TechnicalDataVeto(f"{label}: precio plano durante 14 intervalos.",as_of=frame.index[-1])


def current_indicator_suffix(indicators, required, label, minimum_tail=2):
    """Only a contiguous valid suffix; never silently backtrack across bad bars."""
    finite = indicators[required].apply(lambda col: col.map(_finite)).all(axis=1)
    if len(finite)<minimum_tail or not finite.tail(minimum_tail).all():
        raise TechnicalDataVeto(f"{label}: indicadores indefinidos en la última vela o su precedente; no se reutilizan velas antiguas.",
                                as_of=indicators.index[-1] if len(indicators) else None)
    invalid_positions = [i for i,valid in enumerate(finite) if not valid]
    for name in ('RSI14','StochRSI_K','StochRSI_D','ADX14'):
        if name in required and not indicators[name].tail(minimum_tail).between(-1e-8,100+1e-8).all():
            raise TechnicalDataVeto(f'{label}: {name} fuera de dominio.')
    return indicators.iloc[invalid_positions[-1]+1 if invalid_positions else 0:].copy()


def validate_freshness(intraday, daily, cutoff):
    """Strict exchange-clock freshness for live (including holidays/early close)."""
    now = utc(cutoff)
    calendar = _calendar(now.year-1,now.year+1)
    schedule = calendar.schedule.loc[:pd.Timestamp(now.tz_convert(NY).date())]
    if not schedule.empty and schedule.iloc[-1]['open'] <= now < schedule.iloc[-1]['open']+pd.Timedelta(minutes=5):
        raise TechnicalDataVeto('Esperar el primer cierre 5m de la sesión actual.')
    expected = None
    for session in schedule.tail(30).iloc[::-1].itertuples():
        if now >= session.open + pd.Timedelta(minutes=5):
            end = min(now,session.close)
            steps = int((end-session.open)/pd.Timedelta(minutes=5))
            expected = session.open+steps*pd.Timedelta(minutes=5)
            break
    stamp = pd.Timestamp(intraday.index[-1])
    stamp = stamp.tz_localize(NY) if stamp.tzinfo is None else stamp.tz_convert(NY)
    observed = stamp.tz_convert('UTC')+pd.Timedelta(minutes=5)
    if expected is None or observed != expected:
        raise TechnicalDataVeto(f"5m desactualizado: último cierre {observed}; requerido {expected}.",as_of=observed)
    completed = schedule.loc[schedule['close'] <= now]
    if completed.empty or pd.Timestamp(daily.index[-1]).date() != completed.index[-1].date():
        raise TechnicalDataVeto("Contexto diario desactualizado: falta la última sesión cerrada.")
