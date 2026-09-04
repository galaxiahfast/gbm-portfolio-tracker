"""Causal cross-asset evidence. No downloads, orders or calibrated probabilities.

Public metrics take an explicit histories mapping (ticker -> closed daily OHLCV).
Missing sessions are NOT filled and returns are computed before alignment.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd

from .closed_bars import NY, _calendar, select_last_closed_bar, utc

PEERS = {"SMCI": "NVDA", "NVDA": "SMCI"}
COMPONENT = "Correlación cross-asset"


def _daily(frame):
    source = frame.copy().sort_index()
    source.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in source.index])
    if source.index.has_duplicates:
        raise ValueError("Sesiones diarias duplicadas.")
    if source.empty:
        raise ValueError("Histórico diario vacío.")
    sessions = _calendar(source.index[0].year - 1, source.index[-1].year + 1).sessions_in_range(
        source.index[0], source.index[-1])
    return source.reindex(sessions)


def _finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def calculate_rolling_correlation(symbol1, symbol2, period=20, *, histories):
    """Pearson correlation of last `period` matched one-session close returns."""
    if period < 2:
        raise ValueError("Se requieren al menos dos retornos.")
    first, second = (_daily(histories[s])["Close"] for s in (symbol1, symbol2))
    returns = pd.concat([first.pct_change(fill_method=None), second.pct_change(fill_method=None)], axis=1).tail(period)
    if len(returns) < period or not np.isfinite(returns.to_numpy()).all():
        return None
    if (returns.std() <= 1e-12).any():
        return None
    return _finite(returns.iloc[:, 0].corr(returns.iloc[:, 1]))


def calculate_price_ratio(*, histories, period=50):
    """Canonical ratio SMCI/NVDA, on closed daily prices, never reversed by UI."""
    if period < 2:
        raise ValueError("Ventana de ratio inválida.")
    prices = pd.concat([_daily(histories[s])["Close"] for s in ("SMCI", "NVDA")], axis=1).tail(period)
    if len(prices) < period or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        return {"value": None, "mean50": None, "deviation_pct": None}
    ratio = prices.iloc[:, 0] / prices.iloc[:, 1]
    mean = float(ratio.mean())
    return {"value": float(ratio.iloc[-1]), "mean50": mean,
            "deviation_pct": float((ratio.iloc[-1] / mean - 1) * 100)}


def _rsi(close):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]
    if not np.isfinite([gain, loss]).all() or gain + loss <= 1e-12:
        return None
    return float(100 * gain / (gain + loss))


def get_relative_strength(symbol1="SMCI", symbol2="NVDA", *, histories):
    """Relative RSI14, not a claim that overbought means a future fall."""
    result = {}
    for symbol in (symbol1, symbol2):
        close = _daily(histories[symbol])["Close"]
        # No compressed time or RSI across missing sessions.
        result[symbol] = _rsi(close) if len(close) >= 15 and close.notna().all() else None
    return result


def _breakout(frame, lookback=20):
    if lookback < 6:
        raise ValueError("La divergencia requiere al menos seis retornos previos.")
    if len(frame) < lookback + 1:
        raise ValueError("Faltan velas para comparar extremos previos.")
    prior, last = frame.iloc[-lookback-1:-1], frame.iloc[-1]
    average = float(prior.Volume.mean())
    rvol = float(last.Volume / average) if average > 0 else 0.0
    return {"high": bool(last.High > prior.High.max()), "low": bool(last.Low < prior.Low.min()),
            "up": bool(last.Close > prior.High.max() and rvol > 1.2),
            "down": bool(last.Close < prior.Low.min() and rvol > 1.2), "rvol": rvol,
            "momentum": float(last.Close / frame.Close.iloc[-7] - 1)}


def detect_divergence(symbol1="SMCI", symbol2="NVDA", *, histories, lookback=20, timeframe="5m"):
    """New 20-bar wick extremes versus prior bars; close+RVOL confirms breakouts."""
    first, second = (histories[s] for s in (symbol1, symbol2))
    if not first.index.equals(second.index):
        raise ValueError("La divergencia exige velas alineadas sin huecos.")
    a, b = _breakout(first, lookback), _breakout(second, lookback)
    alerts = []
    for key, label in (("high", "máximo"), ("low", "mínimo")):
        if a[key] != b[key]:
            leader, follower = (symbol1, symbol2) if a[key] else (symbol2, symbol1)
            alerts.append(f"{leader} marca nuevo {label} de {lookback} velas; {follower} no confirma.")
    if a["momentum"] * b["momentum"] < 0:
        window = "30 min" if timeframe == "5m" else "6 sesiones" if timeframe == "1d" else "6 velas"
        alerts.append(f"{symbol1} {'sube' if a['momentum'] > 0 else 'baja'} pero {symbol2} "
                      f"{'sube' if b['momentum'] > 0 else 'baja'} en {window}: divergencia; posible frenazo.")
    return {"alerts": alerts, symbol1: a, symbol2: b}


def unavailable(symbol, reason):
    return {"status": "unavailable", "symbol": symbol, "peer": PEERS.get(symbol),
            "detail": f"Correlación no disponible: {reason}", "correlation": None,
            "proposed_impact": 0.0, "applied_impact": 0.0}


def build_cross_context(symbol, primary_daily, peer_daily, primary_5m, peer_5m, *, as_of):
    """Only same-cut closed candles; stale/missing evidence fails closed."""
    peer = PEERS[symbol]
    clock = utc(as_of)
    histories = {s: _daily(select_last_closed_bar(f, "1d", clock).tail(150))
                 for s, f in ((symbol, primary_daily), (peer, peer_daily))}
    calendar = _calendar(clock.year - 1, clock.year + 1)
    expected = calendar.schedule.loc[lambda x: x.close <= clock].index[-1]
    for history in histories.values():
        if history.index[-1] != expected or not np.isfinite(history.to_numpy(dtype=float)).all():
            raise ValueError("Diarios desactualizados, incompletos o no finitos.")
        if (history[["Open", "High", "Low", "Close"]] <= 0).any().any() or (history.Volume < 0).any():
            raise ValueError("Precios o volumen inválidos.")
    correlation = calculate_rolling_correlation(symbol, peer, histories=histories)
    ratio = calculate_price_ratio(histories=histories)
    rsis = get_relative_strength(symbol, peer, histories=histories)
    context = {"status": "available" if correlation is not None else "partial", "symbol": symbol, "peer": peer,
               "correlation": correlation, "return_period": 20, "return_samples": 20 if correlation is not None else 0,
               "daily_as_of": str(expected.date()), "observed_at": clock.isoformat(),
               "ratio": ratio, "rsi14": rsis, "alerts": [], "intraday_as_of": None,
               "proposed_impact": 0.0, "applied_impact": 0.0,
               "detail": "Retornos diarios cerrados; ajuste sólo del score heurístico, no de probabilidades calibradas."}
    inputs = {s: f.to_json(orient="split", date_format="iso") for s, f in histories.items()}
    aligned_daily = {s: f.tail(21) for s, f in histories.items()}
    if all(len(f) == 21 for f in aligned_daily.values()):
        daily_divergence = detect_divergence(symbol, peer, histories=aligned_daily, timeframe="1d")
        context["daily_divergence"] = daily_divergence
        context["alerts"].extend("Diario: " + message for message in daily_divergence["alerts"])
    try:
        frames = {}
        for s, f in ((symbol, primary_5m), (peer, peer_5m)):
            closed = select_last_closed_bar(f.tail(100), "5m", clock)
            closed.index = pd.DatetimeIndex(closed.index)
            closed.index = closed.index.tz_localize(NY) if closed.index.tz is None else closed.index.tz_convert(NY)
            day = pd.Timestamp(clock.tz_convert(NY).date())
            session = calendar.schedule.loc[day]
            # Latest complete regular-session bar, including session-anchored times.
            latest = min(clock.floor("5min"), session.close) - pd.Timedelta("5min")
            current_times = pd.date_range(session.open, latest, freq="5min")
            # At 11:00 there are 18 closed bars today. The 20-bar breakout/volume
            # baseline may span yesterday, but the 30-min momentum never does.
            recent = calendar.schedule.loc[day-pd.Timedelta("10d"):day]
            times = pd.DatetimeIndex([], tz="UTC")
            for prior_session in recent.itertuples():
                times = times.append(pd.date_range(prior_session.open,
                    min(prior_session.close-pd.Timedelta("5min"), latest), freq="5min"))
            times = times[-21:]
            aligned = closed.tz_convert("UTC").reindex(times)
            if clock < session.open or clock > session.close + pd.Timedelta("20min") or len(times) < 21 or len(current_times) < 7:
                raise ValueError("Faltan 21 velas cerradas o 30 min completos de momentum en la sesión actual.")
            if not np.isfinite(aligned.to_numpy(dtype=float)).all() or (aligned.Close <= 0).any():
                raise ValueError("Intradiarios desactualizados o incompletos; ajuste desactivado.")
            if ((aligned.Volume < 0).any() or (aligned.Low <= 0).any()
                    or (aligned.High < aligned[["Open", "Close", "Low"]].max(axis=1)).any()
                    or (aligned.Low > aligned[["Open", "Close"]].min(axis=1)).any()):
                raise ValueError("OHLCV intradía inconsistente; ajuste desactivado.")
            frames[s] = aligned
            inputs[s + "_5m"] = aligned.to_json(orient="split", date_format="iso")
        divergence = detect_divergence(symbol, peer, histories=frames)
        context["alerts"].extend(divergence.pop("alerts"))
        context["intraday"] = divergence
        context["intraday_as_of"] = (frames[symbol].index[-1] + pd.Timedelta("5min")).isoformat()
        a, b = divergence[symbol], divergence[peer]
        if correlation is not None and correlation > .7:
            if b["up"] and not a["up"]:
                context["alerts"].append(f"{peer} rompe resistencia con volumen >1.2x: probable seguimiento de {symbol}; requiere gatillo propio.")
                context["proposed_impact"] = 5.0
            elif b["down"] and not a["down"]:
                context["alerts"].append(f"{peer} pierde soporte con volumen >1.2x: posible seguimiento bajista de {symbol}.")
                context["proposed_impact"] = -5.0
            elif a["momentum"] * b["momentum"] < 0:
                context["proposed_impact"] = 3.0 if b["momentum"] > 0 else -3.0
    except (ValueError, KeyError, IndexError) as exc:
        context["alerts"].append(str(exc))
    context["input_sha256"] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return context


def apply_cross_context(analysis, context):
    """Bounded, idempotent contextual score; NEVER changes an execution permission.

    Keep calibrated horizons, six zone estimates, targets and position state intact.
    A prior directional score cannot be reversed solely by correlation.
    """
    from .technical_probability import ScoreComponent
    previous = getattr(analysis, "cross_asset_context", {})
    base = float(previous.get("base_score", analysis.probability_up))
    correlation = context.get("correlation")
    try:
        correlation = float(correlation)
    except (TypeError, ValueError):
        correlation = None
    eligible = (
        context.get("status") == "available"
        and correlation is not None
        and np.isfinite(correlation)
        and correlation > 0.70
    )
    impact = float(context.get("proposed_impact", 0)) if eligible else 0.0
    if not np.isfinite(impact):
        impact = 0.0
    impact = max(-5.0, min(5.0, impact))
    if getattr(analysis, "risk_veto", False) or getattr(analysis, "position_state", "FLAT") != "FLAT":
        impact = 0.0
    if impact > 0 and getattr(analysis, "long_entry_blocked", False):
        impact = 0.0
    score = round(max(15.0, min(85.0, base + impact)), 1)
    if base < 50:
        score = min(score, 49.9)
    elif base > 50:
        score = max(score, 50.1)
    metadata = dict(context, base_score=base, applied_impact=round(score-base, 1))
    breakdown = tuple(c for c in analysis.score_breakdown if c.name != COMPONENT)
    return replace(analysis, probability_up=score, probability_down=round(100-score, 1),
                   raw_probability_up=score, cross_asset_context=metadata,
                   score_breakdown=(*breakdown, ScoreComponent(COMPONENT, metadata["applied_impact"],
                       f"{metadata.get('detail', '')} Propuesto {impact:+.1f}; aplicado {metadata['applied_impact']:+.1f} puntos. "
                       "No cambia autorizaciones ni probabilidades de alcance.")))
