"""Acceso a datos. Mantiene SQL fuera de la interfaz y de los calculos."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
from typing import Any

from .config import INITIAL_CAPITAL_USD
from .db import Database
from .models import CashMovementKind, FxQuote, PriceQuote, TradeDraft, money


DECIMAL_TRADE_FIELDS = {
    "quantity",
    "price_usd",
    "commission_usd",
    "commission_rate_pct",
    "gross_usd",
    "reported_total_usd",
    "cash_delta_usd",
    "fx_rate",
    "ocr_confidence",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _decimal_or_none(value: Any) -> Decimal | None:
    return None if value is None or value == "" else Decimal(str(value))


class PortfolioRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_initial_capital(self) -> None:
        """Registra $921.05 USD una sola vez, aun despues de reiniciar."""

        now = _utc_now().isoformat()
        with self.database.transaction() as connection:
            exists = connection.execute(
                "SELECT value FROM settings WHERE key = 'initial_capital_seeded'"
            ).fetchone()
            if exists:
                return
            connection.execute(
                """
                INSERT INTO cash_movements (
                    kind, original_amount, original_currency, usd_amount,
                    fx_rate, occurred_at, notes, created_at
                ) VALUES (?, ?, 'USD', ?, NULL, ?, ?, ?)
                """,
                (
                    CashMovementKind.INITIAL.value,
                    str(INITIAL_CAPITAL_USD),
                    str(INITIAL_CAPITAL_USD),
                    now,
                    "Capital inicial configurado al crear la aplicacion",
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO settings(key, value) VALUES('initial_capital_seeded', ?)",
                (now,),
            )

    def add_cash_movement(
        self,
        *,
        kind: CashMovementKind,
        original_amount: Decimal,
        original_currency: str,
        usd_amount: Decimal,
        fx_rate: Decimal | None,
        occurred_at: datetime,
        notes: str = "",
    ) -> int:
        if kind is CashMovementKind.INITIAL:
            raise ValueError("El capital inicial solo puede crearlo el sistema.")
        if original_amount <= 0 or usd_amount <= 0:
            raise ValueError("El importe debe ser mayor que cero.")
        now = _utc_now().isoformat()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO cash_movements (
                    kind, original_amount, original_currency, usd_amount,
                    fx_rate, occurred_at, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind.value,
                    str(original_amount),
                    original_currency,
                    str(money(usd_amount)),
                    str(fx_rate) if fx_rate is not None else None,
                    occurred_at.isoformat(),
                    notes.strip(),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_cash_movements(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cash_movements ORDER BY occurred_at DESC, id DESC"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in ("original_amount", "usd_amount", "fx_rate"):
                item[field] = _decimal_or_none(item[field])
            result.append(item)
        return result

    def cash_balance_usd(self) -> Decimal:
        balance = Decimal("0")
        for movement in self.list_cash_movements():
            amount = movement["usd_amount"] or Decimal("0")
            if movement["kind"] == CashMovementKind.WITHDRAWAL.value:
                balance -= amount
            else:
                balance += amount
        for trade in self.list_trades(ascending=True):
            balance += trade["cash_delta_usd"] or Decimal("0")
        return money(balance)

    def add_trade(self, trade: TradeDraft, receipt_id: int | None = None) -> int:
        now = _utc_now().isoformat()
        with self.database.transaction() as connection:
            if receipt_id is not None:
                duplicate = connection.execute(
                    "SELECT id FROM trades WHERE receipt_id = ? LIMIT 1",
                    (receipt_id,),
                ).fetchone()
                if duplicate:
                    raise ValueError(
                        "Este comprobante ya está vinculado a la operación "
                        f"#{int(duplicate['id'])}; no se guardó un duplicado."
                    )
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO trades (
                        symbol, product, side, order_type, quantity, price_usd,
                        commission_usd, commission_rate_pct, gross_usd,
                        reported_total_usd, reported_total_type, cash_delta_usd,
                        fx_rate, executed_at,
                        notes, receipt_id, ocr_text, ocr_confidence,
                        validation_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.symbol.strip().upper(),
                        trade.product.strip().upper(),
                        trade.side.value,
                        trade.order_type.strip(),
                        str(trade.quantity),
                        str(trade.price_usd),
                        str(money(trade.commission_usd)),
                        str(trade.commission_rate_pct)
                        if trade.commission_rate_pct is not None
                        else None,
                        str(trade.gross_usd),
                        str(trade.reported_total_usd)
                        if trade.reported_total_usd is not None
                        else None,
                        trade.reported_total_type.value,
                        str(trade.cash_delta_usd),
                        str(trade.fx_rate) if trade.fx_rate is not None else None,
                        trade.executed_at.isoformat(),
                        trade.notes.strip(),
                        receipt_id,
                        trade.ocr_text,
                        str(trade.ocr_confidence)
                        if trade.ocr_confidence is not None
                        else None,
                        trade.validation_status,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if "receipt_already_linked" in str(exc):
                    raise ValueError(
                        "Este comprobante ya está vinculado a otra operación; "
                        "no se guardó un duplicado."
                    ) from exc
                raise
            return int(cursor.lastrowid)

    def list_receipts(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM receipts ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_trades(self, *, ascending: bool = False) -> list[dict[str, Any]]:
        direction = "ASC" if ascending else "DESC"
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT t.*, r.thumbnail_path, r.original_path, r.original_filename
                FROM trades t
                LEFT JOIN receipts r ON r.id = t.receipt_id
                ORDER BY t.executed_at {direction}, t.id {direction}
                """
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in DECIMAL_TRADE_FIELDS:
                item[field] = _decimal_or_none(item.get(field))
            result.append(item)
        return result

    def delete_trade(self, trade_id: int) -> bool:
        with self.database.transaction() as connection:
            cursor = connection.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            return cursor.rowcount == 1

    def upsert_receipt(
        self,
        *,
        sha256: str,
        original_filename: str,
        mime_type: str,
        original_path: str,
        thumbnail_path: str,
        width: int,
        height: int,
        byte_size: int,
    ) -> int:
        now = _utc_now().isoformat()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM receipts WHERE sha256 = ?", (sha256,)
            ).fetchone()
            if row:
                return int(row["id"])
            cursor = connection.execute(
                """
                INSERT INTO receipts (
                    sha256, original_filename, mime_type, original_path,
                    thumbnail_path, width, height, byte_size, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sha256,
                    original_filename,
                    mime_type,
                    original_path,
                    thumbnail_path,
                    width,
                    height,
                    byte_size,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def add_fx_quote(self, quote: FxQuote) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fx_rates (
                    base_currency, quote_currency, rate, observed_at,
                    provider, is_reference
                ) VALUES ('USD', 'MXN', ?, ?, ?, ?)
                """,
                (
                    str(quote.rate),
                    quote.observed_at.isoformat(),
                    quote.provider,
                    int(quote.is_reference),
                ),
            )

    def fx_quote_exists(self, quote: FxQuote) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM fx_rates
                WHERE base_currency = 'USD' AND quote_currency = 'MXN'
                  AND observed_at = ? AND provider = ?
                LIMIT 1
                """,
                (quote.observed_at.isoformat(), quote.provider),
            ).fetchone()
        return row is not None

    def latest_fx_quote(self) -> FxQuote | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM fx_rates
                WHERE base_currency = 'USD' AND quote_currency = 'MXN'
                ORDER BY observed_at DESC, id DESC LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        return FxQuote(
            rate=Decimal(row["rate"]),
            observed_at=datetime.fromisoformat(row["observed_at"]),
            provider=row["provider"],
            is_reference=bool(row["is_reference"]),
        )

    def add_price_quote(self, quote: PriceQuote, *, is_manual: bool = False) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO price_snapshots (
                    symbol, price_usd, observed_at, provider, is_manual
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    quote.symbol.upper(),
                    str(quote.price_usd),
                    quote.observed_at.isoformat(),
                    quote.provider,
                    int(is_manual),
                ),
            )

    def price_quote_exists(self, quote: PriceQuote) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM price_snapshots
                WHERE symbol = ? AND observed_at = ? AND provider = ?
                LIMIT 1
                """,
                (quote.symbol.upper(), quote.observed_at.isoformat(), quote.provider),
            ).fetchone()
        return row is not None

    def latest_price_quotes(self, symbols: list[str]) -> dict[str, PriceQuote]:
        quotes: dict[str, PriceQuote] = {}
        with self.database.connect() as connection:
            for symbol in sorted({item.upper() for item in symbols}):
                row = connection.execute(
                    """
                    SELECT * FROM price_snapshots
                    WHERE symbol = ?
                    ORDER BY observed_at DESC, id DESC LIMIT 1
                    """,
                    (symbol,),
                ).fetchone()
                if row:
                    quotes[symbol] = PriceQuote(
                        symbol=symbol,
                        price_usd=Decimal(row["price_usd"]),
                        observed_at=datetime.fromisoformat(row["observed_at"]),
                        provider=row["provider"],
                    )
        return quotes

    def add_portfolio_snapshot(
        self,
        *,
        cash_usd: Decimal,
        holdings_value_usd: Decimal,
        equity_usd: Decimal,
        fx_rate: Decimal,
        observed_at: datetime | None = None,
    ) -> bool:
        observed_at = observed_at or _utc_now()
        with self.database.transaction() as connection:
            last = connection.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY observed_at DESC LIMIT 1"
            ).fetchone()
            if last:
                last_time = datetime.fromisoformat(last["observed_at"])
                recent = observed_at - last_time < timedelta(minutes=10)
                unchanged = money(Decimal(last["equity_usd"])) == money(equity_usd)
                if recent and unchanged:
                    return False
            connection.execute(
                """
                INSERT INTO portfolio_snapshots (
                    cash_usd, holdings_value_usd, equity_usd, fx_rate, observed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(money(cash_usd)),
                    str(money(holdings_value_usd)),
                    str(money(equity_usd)),
                    str(fx_rate),
                    observed_at.isoformat(),
                ),
            )
            return True

    def list_portfolio_snapshots(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY observed_at ASC"
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for field in ("cash_usd", "holdings_value_usd", "equity_usd", "fx_rate"):
                item[field] = Decimal(item[field])
            result.append(item)
        return result

    def record_audit_run(
        self,
        *,
        status: str,
        passed: int,
        warnings: int,
        errors: int,
        details_json: str,
    ) -> int:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_runs(
                    status, passed, warnings, errors, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    status,
                    passed,
                    warnings,
                    errors,
                    details_json,
                    _utc_now().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def list_audit_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_runs
                ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_backtest_run(
        self,
        *,
        engine_version: str,
        symbols_json: str,
        parameters_json: str,
        dataset_sha256: str,
        payload_json: str,
        status: str,
    ) -> int:
        normalized_status = status.upper()
        if normalized_status not in {"APPROVED", "REJECTED"}:
            raise ValueError("Estado de backtest no reconocido.")
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        with self.database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs(
                    engine_version, symbols_json, parameters_json,
                    dataset_sha256, payload_json, payload_sha256,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    engine_version,
                    symbols_json,
                    parameters_json,
                    dataset_sha256,
                    payload_json,
                    payload_hash,
                    normalized_status,
                    _utc_now().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def list_backtest_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, engine_version, symbols_json, parameters_json,
                       dataset_sha256, payload_sha256, status, created_at
                FROM backtest_runs
                ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def verify_backtest_runs(self) -> tuple[int, tuple[int, ...]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, payload_json, payload_sha256 FROM backtest_runs"
            ).fetchall()
        invalid: list[int] = []
        for row in rows:
            calculated = hashlib.sha256(
                str(row["payload_json"]).encode("utf-8")
            ).hexdigest()
            if calculated != str(row["payload_sha256"]):
                invalid.append(int(row["id"]))
        return len(rows) - len(invalid), tuple(invalid)
