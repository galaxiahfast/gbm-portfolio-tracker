"""Read-only, causal intraday plan from a verified fixed-cut forecast.

This is an observation aid, not an order or a calibrated probability. A zone
touch cannot establish a real fill; only the portfolio ledger can do that.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from portfolio_tracker.analytics.closed_bars import NY, select_last_closed_bar, utc
from portfolio_tracker.services.directional_collection import (
    COLLECTION_PROTOCOL, PREVIOUS_COLLECTION_PROTOCOL,
)
from portfolio_tracker.services.model_execution_record import execution_id
from portfolio_tracker.services.zone_forward import session_bounds


@dataclass(frozen=True, slots=True)
class IntradayPlan:
    status: str
    detail: str
    observed_at: pd.Timestamp | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    sequence: str | None = None
    sequence_at: pd.Timestamp | None = None
    eligible_at_cut: bool = False
    source: str = "Corte firmado de las 11:00 NY · hipótesis, no fill"


def _signed_daily_contract(repository, symbol: str, day: str):
    """Never substitute a current adaptive zone for a missing signed cut."""
    for protocol in (COLLECTION_PROTOCOL, PREVIOUS_COLLECTION_PROTOCOL):
        record = repository.live_model_execution_record(execution_id(symbol, day, protocol))
        if record is None or record.get("symbol") != symbol or record.get("session_date") != day:
            continue
        for prediction in record.get("predictions", ()):
            if prediction.get("horizon") == "1 Día":
                return prediction.get("operational_target")
    return None


def _closed_session_bars(frame, *, observed_at, as_of):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.iloc[:0] if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    source = select_last_closed_bar(frame, "5m", as_of=as_of)
    stamps = pd.DatetimeIndex(source.index)
    if stamps.tz is None:
        raise ValueError("Las velas 5m del plan requieren zona horaria.")
    source.index = stamps.tz_convert("UTC")
    # Do not inspect the potentially partial emission candle. This is the
    # exact prospective start used by the versioned operational target.
    start = utc(observed_at).floor("5min") + pd.Timedelta(minutes=5)
    end = utc(as_of).floor("5min")
    expected = pd.date_range(start, end, freq="5min", inclusive="left", tz="UTC")
    actual = source.loc[(source.index >= start) & (source.index < end)]
    if actual.index.has_duplicates or not actual.index.equals(expected):
        raise ValueError("Faltan velas 5m cerradas después del corte; no se infiere la secuencia.")
    return actual


def _hypothetical_sequence(bars, entry: float, stop: float, target: float):
    touched_at = None
    for at, row in bars.iterrows():
        low, high = float(row["Low"]), float(row["High"])
        if not all(math.isfinite(v) for v in (low, high)) or low <= 0 or high < low:
            return "EVIDENCIA_INCOMPLETA", None
        if touched_at is None:
            if low <= entry <= high:
                touched_at = at
            # A target reached before a possible entry is not a success.
            continue
        # Conservative intrabar precedence, matching the operational labeler.
        opening = float(row["Open"])
        if opening <= stop:
            return "SL_FIRST_HIPOTETICO", at
        if opening >= target:
            return "TP_FIRST_HIPOTETICO", at
        if low <= stop:
            return "SL_FIRST_HIPOTETICO", at
        if high >= target:
            return "TP_FIRST_HIPOTETICO", at
    return ("TOQUE_SIN_FILL" if touched_at is not None else "SIN_TOQUE"), touched_at


def build_intraday_plan(analysis, repository, *, now=None) -> IntradayPlan:
    """Freeze levels at the signed cut; update only prospective observed state.

    No call here writes forecasts, orders or ledger rows. In particular, a
    moving visual zone or a later 5m score can never re-anchor the plan.
    """
    clock = utc(now)
    bounds = session_bounds(clock)
    if bounds is None:
        return IntradayPlan("SIN_SESION", "NYSE no tiene sesión hoy; no hay plan intradía activo.")
    day, session_open, session_close = bounds
    if not session_open <= clock <= session_close:
        return IntradayPlan("MERCADO_CERRADO", "Fuera de sesión; no hay entrada intradía activa.")
    symbol = str(getattr(analysis, "symbol", "")).strip().upper()
    contract = _signed_daily_contract(repository, symbol, day)
    if contract is None:
        return IntradayPlan(
            "SIN_CORTE", "Aún no existe un corte fijo íntegro de hoy; no se crea un plan retrospectivo."
        )
    from portfolio_tracker.analytics.operational_target import validate_operational_contract
    try:
        validate_operational_contract(contract)
        observed = utc(contract["observed_at"])
        entry = float(contract["entry_price"])
        stop = float(contract["stop_loss"])
        target = float(contract["take_profit"])
        if observed > clock or contract["side"] != "LONG" or not stop < entry < target:
            raise ValueError("Contrato no corresponde a una oportunidad LONG ya emitida.")
        bars = _closed_session_bars(analysis.intraday_indicators, observed_at=observed, as_of=clock)
        sequence, at = _hypothetical_sequence(bars, entry, stop, target)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return IntradayPlan("SIN_EVIDENCIA", f"Plan firmado no evaluable: {exc}")
    if sequence == "SIN_TOQUE":
        status, detail = "ESPERAR_ENTRADA", "Entrada no tocada después del corte. No perseguir el precio."
    elif sequence == "TOQUE_SIN_FILL":
        status, detail = "VERIFICAR_GATILLO", (
            "La vela cerrada tocó la entrada, pero no consta fill. Confirmar gatillo y riesgo antes de actuar."
        )
    elif sequence == "TP_FIRST_HIPOTETICO":
        status, detail = "OBJETIVO_OBSERVADO", (
            "Tras el toque de entrada, se vio el objetivo antes que el stop; no implica ganancia sin fill real."
        )
    elif sequence == "SL_FIRST_HIPOTETICO":
        status, detail = "STOP_OBSERVADO", (
            "Tras el toque de entrada, se vio el stop primero; descartar esta oportunidad hipotética."
        )
    else:
        status, detail = "SIN_EVIDENCIA", "Velas cerradas incompletas o inválidas; esperar."
    return IntradayPlan(
        status, detail, observed, entry, stop, target, sequence, at,
        bool(contract.get("eligible_at_emission")),
    )
