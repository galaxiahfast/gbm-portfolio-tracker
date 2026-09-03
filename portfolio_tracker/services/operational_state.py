"""Operational memory separate from accounting. Signals NEVER create fills.

Existing ledger positions are adopted with a frozen management plan on first
observation. Stop/target alerts produce EXIT_PENDING, not a fictitious sale.
SHA-256 chaining detects modified payloads; it is not cryptographic authenticity
against an attacker able to rewrite the entire database and its hashes.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import pandas as pd

from portfolio_tracker.analytics.multi_timeframe import ExecutionLevels


def _digest(symbol, previous, payload):
    return hashlib.sha256((symbol + "\n" + previous + "\n" + payload).encode()).hexdigest()


def read_state(connection, symbol):
    previous, state = "", None
    for row in connection.execute("SELECT * FROM operational_events WHERE symbol=? ORDER BY id", (symbol,)):
        if row["previous_sha256"] != previous or row["sha256"] != _digest(symbol, previous, row["payload_json"]):
            raise ValueError("Integridad SHA-256 del estado operativo inválida; ejecución bloqueada.")
        previous = row["sha256"]
        state = json.loads(row["payload_json"])
    return state, previous


def audit_operational_history(database):
    with database.connect() as connection:
        for row in connection.execute("SELECT DISTINCT symbol FROM operational_events"):
            read_state(connection, row["symbol"])


def macro_memory(database, symbol):
    with database.connect() as connection:
        state, _ = read_state(connection, symbol)
    return state["macro_trending"] if state else None


def synchronize_position(database, analysis):
    """Atomic reconciliation with actual fills; safe under reruns/concurrency.

    State transitions: FLAT → *_ACTIVE only on ledger inventory; *_ACTIVE →
    EXIT_PENDING on fixed risk levels; → FLAT only when ledger inventory is zero.
    """
    symbol = analysis.symbol
    with database.transaction() as connection:
        old, previous = read_state(connection, symbol)
        quantity, episode = Decimal(0), None
        for trade in connection.execute("SELECT * FROM trades WHERE symbol=? ORDER BY executed_at,id", (symbol,)):
            delta = Decimal(trade["quantity"]) * (1 if trade["side"] == "BUY" else -1)
            before = quantity
            quantity += delta
            if before == 0 or before * quantity < 0:
                episode = trade["id"]
            if quantity == 0:
                episode = None
        current_bar = analysis.as_of.isoformat()
        state = dict(old or {})
        # An older concurrent response may reconcile fills, but not reverse the
        # latest market observation or regime memory.
        fresh = not old or datetime.fromisoformat(current_bar) >= datetime.fromisoformat(old["bar"])
        if fresh:
            state.update(bar=current_bar, macro_trending=analysis.macro_trending,
                         macro_permission=analysis.macro_permission)
        state["quantity"] = str(quantity)
        state["episode"] = episode
        if not quantity:
            state.update(status="FLAT", levels=None, management="Sin posición contable; evaluar entradas autorizadas.")
        elif not old or old.get("episode") != episode or old.get("status") == "FLAT":
            direction = "LONG" if quantity > 0 else "SHORT"
            levels = analysis.buy_levels if direction == "LONG" else analysis.sell_levels
            levels = levels or analysis.execution_levels
            state.update(status=f"{direction}_ACTIVE", levels=asdict(levels), adopted_at=current_bar,
                         higher_bar=(analysis.four_hour_indicators.index[-1].isoformat() if not analysis.four_hour_indicators.empty else None),
                         structural_support=analysis.structural_support,
                         structural_resistance=analysis.structural_resistance,
                         management="Posición contable adoptada: plan fijado en este corte, no reconstruye el plan original.")
        elif fresh and state["status"] != "EXIT_PENDING":
            levels = ExecutionLevels(**state["levels"])
            bars = analysis.intraday_indicators
            newer = bars.loc[bars.index > datetime.fromisoformat(old["bar"])]
            long = levels.direction == "LONG"
            hit_stop = not newer.empty and (newer.Low.min() <= levels.stop_loss if long else newer.High.max() >= levels.stop_loss)
            hit_target = not newer.empty and (newer.High.max() >= levels.take_profit_1 if long else newer.Low.min() <= levels.take_profit_1)
            higher = analysis.four_hour_indicators
            # Fixed structure from adoption, checked ONLY on subsequent 4h close.
            structural = False
            if not higher.empty and (state.get("higher_bar") is None or pd.Timestamp(higher.index[-1]) > pd.Timestamp(state["higher_bar"])):
                close = float(higher.Close.iloc[-1])
                structural = (state["structural_support"] > 0 and close < state["structural_support"]) if long else (state["structural_resistance"] > 0 and close > state["structural_resistance"])
                state["higher_bar"] = higher.index[-1].isoformat()
            if hit_stop or hit_target or structural:
                reason = "Stop alcanzado" if hit_stop else "TP1 alcanzado: revisar salida parcial" if hit_target else "Invalidación estructural 4h"
                state.update(status="EXIT_PENDING", management=reason + "; confirmar operación real. No se ha registrado ninguna venta automática.")
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if old != state:
            connection.execute("INSERT INTO operational_events(symbol,payload_json,previous_sha256,sha256,created_at) VALUES(?,?,?,?,?)",
                               (symbol, payload, previous, _digest(symbol, previous, payload), datetime.now(timezone.utc).isoformat()))
    if state["status"] == "FLAT":
        return replace(analysis, position_state="FLAT",
                       operation_probability=analysis.operation_probability if analysis.activation_trigger_met and not analysis.risk_veto else 0.0)
    fixed = ExecutionLevels(**state["levels"])
    return replace(analysis, position_state=state["status"], position_management=state["management"],
                   signal=type(analysis.signal)("HOLD_LONG" if fixed.direction == "LONG" else "HOLD_SHORT"),
                   verdict=state["management"], suggested_level=fixed.entry_low,
                   execution_levels=fixed,
                   buy_levels=fixed if fixed.direction == "LONG" else analysis.buy_levels,
                   sell_levels=fixed if fixed.direction == "SHORT" else analysis.sell_levels,
                   execution_plan_conditional=True, activation_trigger_met=False, operation_probability=0.0,
                   execution_plan_label="GESTIÓN DE POSICIÓN · entradas congeladas",
                   activation_trigger="No abrir ni invertir por osciladores 5m. Gestionar únicamente el plan fijado y la estructura 4h.",
                   scenario=state["management"])
