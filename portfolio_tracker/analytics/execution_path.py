"""Versioned, conservative *simulated* execution path for a frozen plan.

OHLCV can establish that a limit price was reachable, not that a broker filled
an order. These results must never be described as actual GBM+ executions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pandas as pd

from .closed_bars import NY, _calendar
from .operational_target import (
    IncompleteOperationalEvidence, OperationalOutcome,
    scan_operational_outcome, validate_operational_contract,
)
from .temporal_contract import utc_timestamp
from .net_expectation import FillModel, TradingCostPolicy


EXECUTION_VERSION = "SESSION_LIMIT_POSSIBLE_FILL_V1"
CHECKPOINT_VERSION = "OHLC_EXECUTION_STATE_CHECKPOINT_V1"


def frozen_execution_terms(observed_at):
    """Freeze the entry deadline and explicit per-side cost assumptions."""
    observed = utc_timestamp(observed_at)
    day = pd.Timestamp(observed.tz_convert(NY).date())
    schedule = _calendar(day.year - 1, day.year + 1).schedule
    if day not in schedule.index:
        raise ValueError("El plan requiere una sesión XNYS.")
    return {
        "version": EXECUTION_VERSION,
        "entry_order": "LIMIT_REFERENCE_PRICE_SIMULATED",
        "entry_deadline_at": utc_timestamp(schedule.loc[day, "close"]).isoformat(),
        "commission_bps_per_side": TradingCostPolicy().commission_bps_per_side,
        "slippage_bps_per_side": TradingCostPolicy().slippage_bps_per_side,
        "spread_bps": FillModel().spread_bps,
        "same_fill_bar_policy": "STOP_FIRST_TP_DEFERRED",
        "fill_semantics": "POSSIBLE_NOT_BROKER_CONFIRMED",
    }


def validate_execution_terms(terms, observed_at):
    """Validate signed assumptions without replacing them with today's defaults."""
    if not isinstance(terms, dict):
        raise ValueError("Faltan supuestos de ejecución firmados.")
    expected = frozen_execution_terms(observed_at)
    fixed = ("version", "entry_order", "entry_deadline_at",
             "same_fill_bar_policy", "fill_semantics")
    if set(terms) != set(expected) or any(terms.get(key) != expected[key] for key in fixed):
        raise ValueError("Contrato de ejecución incompatible.")
    for key in ("commission_bps_per_side", "slippage_bps_per_side", "spread_bps"):
        value = terms[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Coste de ejecución inválido.")
        if not math.isfinite(value) or value < 0 or value > 1_000:
            raise ValueError("Coste de ejecución fuera de dominio.")
    return True


@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    status: str
    outcome: str | None = None
    fill_price: float | None = None
    fill_at: str | None = None
    exit_price: float | None = None
    exit_at: str | None = None
    net_pnl_per_share: float | None = None
    detail: str = ""
    version: str = EXECUTION_VERSION
    exit_source: str | None = None

    def as_record(self):
        return asdict(self)


def _net_pnl(side, entry, exit_price, terms):
    rate = (float(terms["commission_bps_per_side"])
            + float(terms["slippage_bps_per_side"])
            + float(terms["spread_bps"]) / 2) / 10_000
    gross = exit_price - entry if side == "LONG" else entry - exit_price
    return gross - rate * (entry + exit_price)


TERMINAL_STATUSES = {
    "LEGACY_ASSUMED_ENTRY", "LEGACY_CHECKPOINT_UNASSESSED", "NOT_AUTHORIZED",
    "NO_FILL", "NO_FILL_INVALIDATED", "NO_FILL_TARGET_PASSED",
    "AMBIGUOUS_NO_TRADE", "SIMULATED_RESOLVED",
}


def validate_execution_checkpoint(record, contract, scanned_through):
    """Reject internally inconsistent states even when the outer SHA matches."""
    if not isinstance(record, dict) or set(record) != set(ExecutionAssessment.__dataclass_fields__) | {"scanned_through", "checkpoint_version"}:
        raise ValueError("Checkpoint de ejecución mal formado.")
    if (record["version"] != EXECUTION_VERSION
            or record["checkpoint_version"] != CHECKPOINT_VERSION
            or record["scanned_through"] != utc_timestamp(scanned_through).isoformat()):
        raise ValueError("Checkpoint de ejecución desincronizado.")
    if record["status"] not in TERMINAL_STATUSES | {"PENDING_ENTRY", "FILLED_PENDING"}:
        raise ValueError("Estado de ejecución desconocido.")
    fill, exit_price, pnl = (record[key] for key in ("fill_price", "exit_price", "net_pnl_per_share"))
    for value in (fill, exit_price):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError("Precio de ejecución inválido.")
    if pnl is not None and (isinstance(pnl, bool) or not isinstance(pnl, (int, float)) or not math.isfinite(pnl)):
        raise ValueError("P&L de ejecución inválido.")
    filled = record["status"] in {"FILLED_PENDING", "SIMULATED_RESOLVED"}
    if filled != (fill is not None and record["fill_at"] is not None):
        raise ValueError("Estado de fill incoherente.")
    if record["fill_at"] is not None:
        fill_at = utc_timestamp(record["fill_at"])
        if not utc_timestamp(contract["evaluation_starts_at"]) < fill_at <= utc_timestamp(scanned_through):
            raise ValueError("Fill fuera de la evidencia escaneada.")
    resolved = record["status"] == "SIMULATED_RESOLVED"
    if resolved != all(record[key] is not None for key in (
        "outcome", "exit_price", "exit_at", "net_pnl_per_share", "exit_source"
    )):
        raise ValueError("Resultado de ejecución incompleto.")
    if resolved:
        if record["outcome"] not in {item.value for item in OperationalOutcome}:
            raise ValueError("Clase de ejecución inválida.")
        exit_at = utc_timestamp(record["exit_at"])
        if not utc_timestamp(record["fill_at"]) <= exit_at <= utc_timestamp(scanned_through):
            raise ValueError("Salida anterior al fill o posterior a la evidencia.")
        if record["outcome"] == "TIMEOUT" and exit_at != utc_timestamp(contract["timeout_at"]):
            raise ValueError("Timeout de ejecución fuera de vencimiento.")
    elif any(record[key] is not None for key in (
        "outcome", "exit_price", "exit_at", "net_pnl_per_share", "exit_source"
    )):
        raise ValueError("Estado no resuelto con resultado presente.")
    return True


def _resolved(contract, fill_price, fill_at, outcome, exit_price, exit_at, source):
    pnl = _net_pnl(contract["side"], fill_price, exit_price, contract["execution_terms"])
    if not math.isfinite(pnl):
        raise ValueError("P&L de ejecución no finito.")
    return ExecutionAssessment(
        "SIMULATED_RESOLVED", outcome, fill_price, fill_at,
        exit_price, exit_at, pnl,
        "Fill posible por OHLC; no confirmado por corredor.",
        exit_source=source,
    )


def advance_execution_checkpoint(
    contract, observed_bars, scanned_through, *, prior=None,
    previous_scanned_through=None,
):
    """Advance only on the barrier scanner's verified *new* rows.

    The returned state carries a matching watermark and is signed inside the
    existing SQLite checkpoint. No old 5m bars are required after restart.
    """
    validate_operational_contract(contract)
    watermark = utc_timestamp(scanned_through)
    prior_mark = utc_timestamp(previous_scanned_through) if previous_scanned_through else None
    if prior is not None:
        if prior_mark is None:
            raise ValueError("Estado previo sin cursor firmado.")
        validate_execution_checkpoint(prior, contract, prior_mark)
        state = ExecutionAssessment(**{key: prior[key] for key in ExecutionAssessment.__dataclass_fields__})
    elif prior_mark is not None:
        # An older signed checkpoint has no fill history. Never reconstruct a
        # profitable entry from later candles or promote it to model evidence.
        state = ExecutionAssessment("LEGACY_CHECKPOINT_UNASSESSED", detail="Fill previo desconocido.")
    elif "execution_terms" not in contract:
        state = ExecutionAssessment("LEGACY_ASSUMED_ENTRY", detail="Sin contrato de fill firmado.")
    elif not contract.get("eligible_at_emission"):
        state = ExecutionAssessment("NOT_AUTHORIZED", detail="No había entrada autorizada al corte.")
    else:
        state = ExecutionAssessment("PENDING_ENTRY")

    stop, target, reference = (float(contract[key]) for key in
                               ("stop_loss", "take_profit", "entry_price"))
    timeout = utc_timestamp(contract["timeout_at"])
    deadline = min(utc_timestamp(contract["execution_terms"]["entry_deadline_at"]), timeout) if "execution_terms" in contract else timeout
    last_end = prior_mark or utc_timestamp(contract["evaluation_starts_at"])
    for start, end, opening, high, low, close, _, source in observed_bars:
        start, end = utc_timestamp(start), utc_timestamp(end)
        if (start < last_end or end <= start or end > watermark
                or end > timeout or source not in {"5m", "1d"}):
            raise ValueError("Vela fuera del tramo de ejecución firmado.")
        last_end = end
        if state.status in TERMINAL_STATUSES:
            continue
        if state.status == "PENDING_ENTRY":
            if start >= deadline:
                state = ExecutionAssessment("NO_FILL", detail="Venció la ventana de entrada.")
                continue
            if source != "5m":
                raise ValueError("Una vela diaria no puede crear un fill de 5 minutos.")
            if contract["side"] == "LONG":
                if opening <= stop:
                    state = ExecutionAssessment("NO_FILL_INVALIDATED", detail="Apertura bajo stop antes del fill.")
                    continue
                if opening >= target or (high >= target and low > reference):
                    state = ExecutionAssessment("NO_FILL_TARGET_PASSED", detail="Objetivo antes de entrada.")
                    continue
                possible, fill = low <= reference, min(opening, reference)
                same_stop, same_target = low <= stop, high >= target
            else:
                if opening >= stop:
                    state = ExecutionAssessment("NO_FILL_INVALIDATED", detail="Apertura sobre stop antes del fill.")
                    continue
                if opening <= target or (low <= target and high < reference):
                    state = ExecutionAssessment("NO_FILL_TARGET_PASSED", detail="Objetivo antes de entrada.")
                    continue
                possible, fill = high >= reference, max(opening, reference)
                same_stop, same_target = high >= stop, low <= target
            if not possible:
                if end >= deadline:
                    state = ExecutionAssessment("NO_FILL", detail="Entrada no tocada antes del vencimiento.")
                continue
            if same_target and not same_stop:
                state = ExecutionAssessment(
                    "AMBIGUOUS_NO_TRADE", detail="TP y entrada en la misma vela: orden desconocido."
                )
                continue
            fill_at = end.isoformat()
            if same_stop:
                state = _resolved(contract, fill, fill_at, "SL_FIRST", stop, fill_at,
                                  "5m:possible-fill-bar-stop-conservative")
            elif end == timeout:
                state = _resolved(contract, fill, fill_at, "TIMEOUT", close, fill_at,
                                  "5m:timeout-close")
            else:
                state = ExecutionAssessment("FILLED_PENDING", fill_price=fill, fill_at=fill_at)
            continue

        # From here the possible fill is already signed in an older checkpoint.
        if contract["side"] == "LONG":
            outcome, exit_price, source_tag = (
                ("SL_FIRST", opening, "gap-open") if opening <= stop else
                ("TP_FIRST", target, "gap-target") if opening >= target else
                ("SL_FIRST", stop, "barrier") if low <= stop else
                ("TP_FIRST", target, "barrier") if high >= target else
                (None, None, None)
            )
        else:
            outcome, exit_price, source_tag = (
                ("SL_FIRST", opening, "gap-open") if opening >= stop else
                ("TP_FIRST", target, "gap-target") if opening <= target else
                ("SL_FIRST", stop, "barrier") if high >= stop else
                ("TP_FIRST", target, "barrier") if low <= target else
                (None, None, None)
            )
        if outcome is None and end == timeout:
            outcome, exit_price, source_tag = "TIMEOUT", close, "timeout-close"
        if outcome is not None:
            state = _resolved(
                contract, state.fill_price, state.fill_at, outcome, exit_price,
                end.isoformat(), f"{source}:{source_tag}",
            )
    record = {
        **state.as_record(), "scanned_through": watermark.isoformat(),
        "checkpoint_version": CHECKPOINT_VERSION,
    }
    validate_execution_checkpoint(record, contract, watermark)
    return record


def assess_execution_path(contract, bars_5m, as_of, *, daily_bars=None):
    """One-shot replay of the same incremental, signed execution state."""
    validate_operational_contract(contract)
    if "execution_terms" not in contract:
        return ExecutionAssessment("LEGACY_ASSUMED_ENTRY", detail="Sin contrato de fill firmado.")
    if not contract.get("eligible_at_emission"):
        return ExecutionAssessment("NOT_AUTHORIZED", detail="No había entrada autorizada al corte.")
    try:
        scan = scan_operational_outcome(contract, bars_5m, as_of, daily_bars=daily_bars)
    except (IncompleteOperationalEvidence, ValueError) as exc:
        return ExecutionAssessment("EVIDENCE_INCOMPLETE", detail=str(exc))
    if not scan.observed_bars:
        return ExecutionAssessment("PENDING_ENTRY")
    record = advance_execution_checkpoint(contract, scan.observed_bars, scan.scanned_through)
    return ExecutionAssessment(**{key: record[key] for key in ExecutionAssessment.__dataclass_fields__})
