"""Read-only market inputs for the predictor UI, with explicit provenance.

The signed autopilot cache is a fallback, never a source of fabricated live
prices. A missing daily bar may be reconstructed only from a complete regular
NYSE session of already closed five-minute bars.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from portfolio_tracker.analytics.closed_bars import NY, _calendar, select_last_closed_bar, utc
from portfolio_tracker.config import DATA_DIR
from portfolio_tracker.services.quant_market_data import (
    QuantMarketDataError,
    download_quant_frames,
    normalize_symbol,
)


@dataclass(frozen=True)
class PredictorMarketFrames:
    intraday: pd.DataFrame
    daily: pd.DataFrame
    source: str
    reconstructed_sessions: tuple[str, ...] = ()
    source_warning: str = ""


def is_regular_nyse_session(now=None) -> bool:
    """True only inside the actual XNYS session, including early closes."""
    current = utc(now)
    schedule = _calendar(current.year - 1, current.year + 1).schedule
    session = schedule.loc[schedule.index == pd.Timestamp(current.tz_convert(NY).date())]
    return bool(not session.empty and session.iloc[0]["open"] <= current < session.iloc[0]["close"])


def complete_daily_from_intraday(intraday, daily, *, now=None):
    """Fill missing *completed* daily sessions only with all regular 5m bars."""
    cutoff = utc(now)
    closed_5m = select_last_closed_bar(intraday, "5m", cutoff)
    closed_daily = select_last_closed_bar(daily, "1d", cutoff)
    if closed_5m.empty:
        return closed_daily, ()
    bars = closed_5m.copy()
    bars.index = pd.DatetimeIndex(bars.index)
    bars.index = bars.index.tz_localize(NY) if bars.index.tz is None else bars.index.tz_convert(NY)
    bars = bars.sort_index()
    schedule = _calendar(cutoff.year - 1, cutoff.year + 1).schedule
    available = {pd.Timestamp(stamp).date() for stamp in closed_daily.index}
    rows = []
    recovered = []
    for day, session in schedule.loc[:pd.Timestamp(cutoff.tz_convert(NY).date())].tail(25).iterrows():
        if day.date() in available or session["close"] > cutoff:
            continue
        expected = pd.date_range(
            session["open"], session["close"], freq="5min", inclusive="left"
        ).tz_convert(NY)
        observed = bars.loc[bars.index.normalize() == expected[0].normalize()]
        if not observed.index.equals(expected):
            continue
        values = observed[["Open", "High", "Low", "Close", "Volume"]]
        numeric = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            continue
        if (values["Volume"] < 0).any() or (values[["Open", "High", "Low", "Close"]] <= 0).any().any():
            continue
        if ((values["High"] < values[["Open", "Close", "Low"]].max(axis=1)) |
                (values["Low"] > values[["Open", "Close", "High"]].min(axis=1))).any():
            continue
        stamp = pd.Timestamp(day.date())
        if isinstance(closed_daily.index, pd.DatetimeIndex) and closed_daily.index.tz is not None:
            stamp = stamp.tz_localize(closed_daily.index.tz)
        rows.append(pd.DataFrame([{
            "Open": float(observed["Open"].iloc[0]),
            "High": float(observed["High"].max()),
            "Low": float(observed["Low"].min()),
            "Close": float(observed["Close"].iloc[-1]),
            "Volume": float(observed["Volume"].sum()),
        }], index=pd.DatetimeIndex([stamp])))
        recovered.append(day.date().isoformat())
    if rows:
        closed_daily = pd.concat([closed_daily, *rows]).sort_index()
    return closed_daily, tuple(recovered)


def read_signed_autopilot_frames(symbol: str, *, cache_dir: Path | None = None):
    """Read both verified acquisition artifacts without downloading or writing."""
    from scripts.autopilot_market_cache import MarketCache

    root = Path(cache_dir) if cache_dir is not None else DATA_DIR / "autopilot" / "market"
    cache = MarketCache(root)
    refs = cache.artifact_refs(symbol)
    if set(refs) != {"5m", "1d"}:
        raise QuantMarketDataError(f"No hay un respaldo de velas completo para {symbol}.")
    intraday, _ = cache._read(root / symbol / "5m.json")
    daily, _ = cache._read(root / symbol / "daily.json")
    return intraday, daily


def load_predictor_market_frames(
    symbol: str, *, now: datetime | None = None,
    downloader=None, cache_loader=None,
) -> PredictorMarketFrames:
    """Prefer Yahoo; fall back to a verified local cut when it fails."""
    normalized = normalize_symbol(symbol)
    cutoff = now or datetime.now(timezone.utc)
    downloader = downloader or download_quant_frames
    cache_loader = cache_loader or read_signed_autopilot_frames
    source, warning = "Yahoo Finance", ""
    try:
        intraday, daily = downloader(normalized)
    except (QuantMarketDataError, ValueError, RuntimeError, OSError, TimeoutError) as exc:
        try:
            intraday, daily = cache_loader(normalized)
        except (QuantMarketDataError, ValueError, OSError, KeyError) as cache_exc:
            raise QuantMarketDataError(
                f"No hay velas utilizables para {normalized}: Yahoo falló ({exc}) "
                f"y el respaldo verificado no está disponible ({cache_exc})."
            ) from cache_exc
        source = "respaldo local SHA-256"
        warning = f"Yahoo no respondió; se usa el último corte local verificado ({exc})."
    intraday = select_last_closed_bar(intraday, "5m", cutoff)
    daily, reconstructed = complete_daily_from_intraday(intraday, daily, now=cutoff)
    return PredictorMarketFrames(intraday, daily, source, reconstructed, warning)


def informational_decision(decision, reason: str):
    """Display-only safety gate: stale/closed data cannot be an immediate order."""
    if decision.action == "CONFIRMAR_SALIDA":
        return replace(
            decision, position_size=0, monetary_risk=0.0,
            expected_value_total=None, trigger_met=False,
            explanation=(
                decision.explanation + " " + reason
                + " La salida pendiente debe verificarse contra el fill real en GBM+."
            ),
        )
    explanation = (
        f"Acción actual: ESPERAR. {reason} El análisis es informativo; "
        "confirma precio y posición en GBM+ antes de cualquier operación."
    )
    return replace(
        decision, action="ESPERAR", position_size=0, monetary_risk=0.0,
        expected_value_total=None, trigger_met=False, risk_veto=True,
        reasons=(reason, *decision.reasons), explanation=explanation,
    )
