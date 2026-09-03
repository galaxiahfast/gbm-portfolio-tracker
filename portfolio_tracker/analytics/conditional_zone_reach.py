"""Point-in-time regime analogues, residual ATR scaling and session-close events.

This is an uncalibrated conditional model estimate, not established predictive
accuracy. No account access, fitted optimism, random paths or generic fallback.
"""
import math
import numpy as np
import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, _calendar, utc
from portfolio_tracker.analytics.zone_reach import ReachEstimate


def _daily_context(daily):
    d = daily[['Open', 'High', 'Low', 'Close']].copy().sort_index()
    d.index = pd.DatetimeIndex([pd.Timestamp(t.date()) for t in d.index])
    d = d.loc[~d.index.duplicated(keep='last')]
    valid = np.isfinite(d).all(axis=1) & (d > 0).all(axis=1)
    valid &= (d.High >= d[['Open', 'Close']].max(axis=1)) & (d.Low <= d[['Open', 'Close']].min(axis=1))
    d = d.loc[valid]
    tr = pd.concat([d.High-d.Low, (d.High-d.Close.shift()).abs(),
                    (d.Low-d.Close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up, down = d.High.diff(), -d.Low.diff()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dx = 100 * (plus-minus).abs() / (plus+minus).replace(0, np.nan)
    dx = dx.where((plus+minus) != 0, 0)
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ema = lambda n: d.Close.ewm(span=n, adjust=False, min_periods=n).mean()
    # Friday labels make an unfinished week unavailable before the next session.
    weekly = d.Close.resample('W-FRI').last().dropna()
    w9 = weekly.ewm(span=9, adjust=False, min_periods=9).mean()
    w21 = weekly.ewm(span=21, adjust=False, min_periods=21).mean()
    context = pd.DataFrame({'atr': atr, 'close': d.Close, 'adx': adx,
                            'daily': np.sign(ema(9)-ema(21)),
                            'ema50': np.sign(d.Close-ema(50))})
    week_bias = np.sign(w9-w21)
    if not d.empty:
        calendar = _calendar(d.index.min().year-1, d.index.max().year+1).schedule
        week_bias.index = pd.DatetimeIndex([calendar.loc[end-pd.Timedelta(days=6):end].index[-1]
                                           for end in week_bias.index])
    context['weekly'] = week_bias.reindex(context.index, method='ffill')
    context['regime'] = np.select([adx < 20, adx > 25], ['RANGO', 'TENDENCIA'], default='TRANSICION')
    return context


def residual_budget(atr_pct, consumed, elapsed_fraction):
    """Soft range budget, never a hard ATR cap. Shrinks with clock and pace.

sqrt(time) diffusion assumption + ATR-consumption/pace discount. Explicit model
assumption requiring OOS calibration, not a fitted empirical probability.
"""
    t = min(1., max(0., elapsed_fraction))
    pace = consumed / math.sqrt(max(t, 1/78))
    return atr_pct * math.sqrt(1-t) / (1 + consumed * max(1., pace))


def _interval(p, n):
    z = 1.95996398454
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return 100*max(0., min(p, center-half)), 100*min(1., max(p, center+half))


def _similarity(current, prior, consumed, used, current_move, historical_move):
    """Fixed, symmetric distance, never fitted to a ticker or future outcome.

    Four macro dimensions plus volatility, range consumption and session momentum.
    The same-clock window already controls time of day. Each mismatch reduces
    influence smoothly instead of throwing the entire historical session away.
    """
    macro = sum(float(current[key] != prior[key])
                for key in ('weekly', 'daily', 'regime', 'ema50'))
    volatility = abs(math.log((current.atr/current.close)/(prior.atr/prior.close)))
    distance = (macro + min(volatility, 2.) + min(abs(consumed-used), 2.)
                + min(abs(current_move-historical_move), 2.)) / 7
    return math.exp(-3 * distance)


def estimate_conditional_reach(frame, daily, price, zones, now=None, min_sessions=20,
                               *, matching='strict'):
    """zones: (low, high, BELOW|ABOVE), closure is beyond the FAR boundary.

Toque = future high/low crossing the near edge (or already inside at cut).
Close = final session close <= low for BELOW, >= high for ABOVE. Not a 5m close.
Historical context is the previous completed daily/weekly bar, never that day's
eventual close. Equal-length complete regular sessions only, one vote per day.
"""
    zones = tuple(zones)
    if matching not in ('strict', 'weighted'):
        raise ValueError('matching must be strict or weighted')
    model = 'conditional-v3-weighted' if matching == 'weighted' else 'conditional-v2'
    if min_sessions < 2:
        raise ValueError('min_sessions must be >= 2')
    def missing(status, n=0, detail=''):
        return tuple(ReachEstimate(None, n, None, None, status, close_direction=z[2],
                                   model=model, detail=detail) for z in zones)
    now = utc(now)
    day = pd.Timestamp(now.tz_convert(NY).date())
    schedule = _calendar(day.year-5, day.year+1).schedule
    if day not in schedule.index or not schedule.loc[day, 'open'] <= now < schedule.loc[day, 'close']:
        return missing('Fuera de sesión')
    cols = ['Open', 'High', 'Low', 'Close']
    if frame.empty or daily.empty or not set(cols) <= set(frame) or not set(cols) <= set(daily):
        return missing('Falta histórico OHLC intradía/diario')
    if not math.isfinite(price) or price <= 0:
        return missing('Precio inválido')
    source = frame[cols].copy().sort_index()
    idx = pd.DatetimeIndex(source.index)
    source.index = (idx.tz_localize(NY) if idx.tz is None else idx).tz_convert('UTC')
    source = source.loc[~source.index.duplicated(keep='last')]
    step = pd.Timedelta(minutes=5)
    source = source.loc[source.index+step <= now]
    hours = schedule.loc[day]
    today = source.loc[(source.index >= hours['open']) & (source.index < hours['close'])]
    if today.empty or now-(today.index[-1]+step) > 2*step:
        return missing('Cotización atrasada o sin vela cerrada')
    anchor = today.index[-1]+step
    if not math.isclose(float(today.iloc[-1].Close), price, rel_tol=1e-8):
        return missing('Precio y corte no coinciden')
    def complete(bars, expected):
        return (bars.index.equals(expected) and np.isfinite(bars.to_numpy()).all()
                and (bars.to_numpy() > 0).all()
                and (bars.High >= bars[['Open','Close']].max(axis=1)).all()
                and (bars.Low <= bars[['Open','Close']].min(axis=1)).all())
    if not complete(today, pd.date_range(hours['open'], anchor-step, freq=step)):
        return missing('Sesión actual incompleta: no se puede medir ATR consumido')
    # Truncate before constructing indicators, also preventing future weekly data.
    daily = daily.loc[[pd.Timestamp(t.date()) < day for t in daily.index]]
    context = _daily_context(daily)
    keys = ['weekly', 'daily', 'regime', 'ema50']
    def prior_context(session_day):
        previous = schedule.index[schedule.index < session_day][-1]
        if previous not in context.index:
            return None
        row = context.loc[previous]
        return row if row.notna().all() and row.atr > 0 else None
    current = prior_context(day)
    if current is None:
        return missing('Contexto diario/semanal insuficiente')
    elapsed, duration = anchor-hours['open'], hours['close']-hours['open']
    fraction = elapsed/duration
    consumed = float(today.High.max()-today.Low.min()) / current.atr
    budget = residual_budget(current.atr/price, consumed, fraction)
    detail = (f'S/D={current.weekly:+.0f}/{current.daily:+.0f}; ADX {current.regime}; '
              f'EMA50={current.ema50:+.0f}; ATR consumido {consumed:.0%}; '
              f'{(hours["close"]-anchor).total_seconds()/60:.0f} min restantes')
    excursions = []
    similarities = []
    exact_matches = 0
    current_move = (price-float(today.iloc[0].Open))/current.atr
    first = pd.Timestamp(source.index[0].tz_convert(NY).date())
    for historical_day, h in schedule.loc[first:day].iloc[:-1].iterrows():
        prior = prior_context(historical_day)
        if prior is None or h['close']-h['open'] != duration:
            continue
        exact = tuple(prior[keys]) == tuple(current[keys])
        if matching == 'strict' and not exact:
            continue
        expected = pd.date_range(h['open'], h['close']-step, freq=step)
        bars = source.reindex(expected)
        if not complete(bars, expected):
            continue
        cut = h['open']+elapsed
        prefix, suffix = bars.loc[bars.index < cut], bars.loc[bars.index >= cut]
        if prefix.empty or suffix.empty:
            continue
        base = float(prefix.iloc[-1].Close)
        used = float(prefix.High.max()-prefix.Low.min())/prior.atr
        historic_budget = residual_budget(prior.atr/base, used, fraction)
        if historic_budget <= 0:
            continue
        exact_matches += int(exact)
        similarities.append(_similarity(current, prior, consumed, used, current_move,
                            (base-float(prefix.iloc[0].Open))/prior.atr))
        scale = budget/historic_budget
        # Scale log excursions: positive prices and internally ordered ranges.
        excursions.append((min(0., math.log(suffix.Low.min()/base))*scale,
                           max(0., math.log(suffix.High.max()/base))*scale,
                           math.log(suffix.iloc[-1].Close/base)*scale))
    n = len(excursions)
    if n < min_sessions:
        return missing(f'Muestra condicionada insuficiente ({n}/{min_sessions})', n, detail)
    weights = np.ones(n)/n
    if matching == 'weighted':
        kernel = np.asarray(similarities)
        # Partial pooling: 25% broad same-clock sample, 75% similarity-weighted.
        # A single matching day cannot masquerade as twenty independent trials.
        weights = .25/n + .75*kernel/kernel.sum()
    effective = float(1/np.square(weights).sum())
    if matching == 'weighted' and effective < 8:
        return missing(f'Evidencia efectiva insuficiente ({effective:.1f}/8)', n, detail)
    if matching == 'weighted':
        detail += f'; comparación ponderada: {n} días, {exact_matches} coincidencias macro exactas, {effective:.1f} efectivos'
    lows, highs, closes = np.array(excursions).T
    results = []
    for low, high, direction in zones:
        if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)) or not 0 < low <= high or direction not in ('BELOW','ABOVE'):
            results.append(ReachEstimate(None,n,None,None,'Sin nivel válido', model=model, detail=detail))
            continue
        lo, hi = math.log(low/price), math.log(high/price)
        touch = 1. if low <= price <= high else float(np.dot(weights, lows <= hi) if price > high else np.dot(weights, highs >= lo))
        close = float(np.dot(weights, closes <= lo) if direction == 'BELOW' else np.dot(weights, closes >= hi))
        touch, close = min(1., max(0., touch)), min(1., max(0., close))
        lower, upper = _interval(touch,effective)
        clower, cupper = _interval(close,effective)
        status = ('Estimación multicontexto preliminar; evidencia limitada; no calibrada OOS'
                  if matching == 'weighted' else 'Estimación condicionada + ATR; no calibrada OOS')
        results.append(ReachEstimate(100*touch,n,lower,upper,
            status,100*close,clower,cupper, direction,model,detail,effective))
    return tuple(results)
