"""Read-only same-clock historical first-passage estimates; not trade signals.

Historical remaining-session OHLC excursions are rebased to the latest close.
No directional scores, overnight returns, simulated sample inflation or future
observations are used. Frequencies are uncalibrated estimates, not guarantees.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar, utc


@dataclass(frozen=True)
class ReachEstimate:
    probability: float | None
    samples: int
    lower: float | None
    upper: float | None
    status: str
    close_probability: float | None = None
    close_lower: float | None = None
    close_upper: float | None = None
    close_direction: str = ""
    model: str = "historical-v1"
    detail: str = ""
    effective_samples: float | None = None


def estimate_zone_reach(frame, price, zones, now=None, min_sessions=20):
    """Estimate touching each [low, high] from last closed 5m bar to today's close.

One complete, same-clock historical suffix = one observation. Matching suffix
length handles early closes; gaps/invalid OHLC invalidate that observation.
Wilson intervals describe sampling uncertainty only, not model/regime error.
Naive bar timestamps mean New York and denote bar opens.
"""
    zones = tuple(zones)
    def unavailable(reason, n=0):
        return tuple(ReachEstimate(None, n, None, None, reason) for _ in zones)
    if min_sessions < 2:
        raise ValueError("min_sessions must be >= 2")
    now = utc(now)
    day = pd.Timestamp(now.tz_convert(NY).date())
    schedule = _calendar(day.year - 2, day.year + 1).schedule
    if day not in schedule.index:
        return unavailable("Fuera de sesión")
    session = schedule.loc[day]
    if not session['open'] <= now < session['close']:
        return unavailable("Fuera de sesión")
    if frame.empty or not {'Open', 'High', 'Low', 'Close'} <= set(frame.columns):
        return unavailable("Sin histórico intradía")
    if not price or not math.isfinite(price) or price <= 0:
        return unavailable("Precio inválido")
    source = frame[['Open', 'High', 'Low', 'Close']].copy()
    idx = pd.DatetimeIndex(source.index)
    source.index = (idx.tz_localize(NY) if idx.tz is None else idx).tz_convert('UTC')
    source = source.sort_index().loc[lambda x: ~x.index.duplicated(keep='last')]
    step = pd.Timedelta(minutes=5)
    source = source.loc[source.index + step <= now]
    today = source.loc[(source.index >= session['open']) & (source.index < session['close'])]
    if today.empty or now - (today.index[-1] + step) > 2 * step:
        return unavailable("Cotización atrasada o sin vela cerrada de hoy")
    anchor = today.index[-1] + step
    if not math.isclose(float(today.iloc[-1]['Close']), price, rel_tol=1e-8):
        return unavailable("Precio y corte no coinciden")
    elapsed = anchor - session['open']
    remaining = session['close'] - anchor
    if remaining <= pd.Timedelta(0):
        return unavailable("Sesión finalizada")
    excursions = []
    first_day = pd.Timestamp(source.index[0].tz_convert(NY).date())
    for historical_day, hours in schedule.loc[first_day:day].iloc[:-1].iterrows():
        start = hours['open'] + elapsed
        end = start + remaining
        if end > hours['close']:
            continue
        expected = pd.date_range(start - step, end - step, freq=step)
        bars = source.reindex(expected)
        values = bars.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            continue
        if ((bars.High < bars[['Open', 'Close']].max(axis=1)) |
                (bars.Low > bars[['Open', 'Close']].min(axis=1))).any():
            continue
        base = float(bars.iloc[0].Close)
        future = bars.iloc[1:]
        excursions.append((min(1., float(future.Low.min()) / base),
                           max(1., float(future.High.max()) / base)))
    n = len(excursions)
    if n < min_sessions:
        return unavailable(f"Muestra insuficiente ({n}/{min_sessions} sesiones)", n)
    lows, highs = np.array(excursions).T * price
    results = []
    for low, high in zones:
        if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)) or not 0 < low <= high:
            results.append(ReachEstimate(None, n, None, None, "Sin nivel válido"))
            continue
        if low <= price <= high:
            results.append(ReachEstimate(100., n, 100., 100., "En zona al corte; no predicción"))
            continue
        # First passage to the near edge; crossing is not an execution guarantee.
        p = float(np.mean(lows <= high) if price > high else np.mean(highs >= low))
        z = 1.959963984540054
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        results.append(ReachEstimate(100*p, n, 100*max(0, min(p, center-half)),
                                     100*min(1, max(p, center+half)), "Frecuencia histórica; no calibrada OOS"))
    return tuple(results)
