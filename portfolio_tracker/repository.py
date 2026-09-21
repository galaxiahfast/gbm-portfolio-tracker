"""Acceso a datos. Mantiene SQL fuera de la interfaz y de los calculos."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

from .config import INITIAL_CAPITAL_USD
from .db import Database
from .services.zone_forward import ZonePrediction
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

    def ensure_zone_forward_schema(self) -> None:
        from .services.zone_forward import ensure_schema
        ensure_schema(self.database)

    def save_prediction(self, prediction: ZonePrediction, *, now=None) -> bool:
        from .services.zone_forward import save_prediction
        return save_prediction(self.database, prediction, now=now)

    def resolve_predictions(self, provider, *, now=None):
        from .services.zone_forward import resolve_predictions
        return resolve_predictions(self.database, provider, now=now)

    def zone_predictions(self):
        from .services.zone_forward import read_predictions
        return read_predictions(self.database)

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

    def latest_backtest_run(self) -> dict[str, Any] | None:
        """Devuelve el último resultado completo para el reporte maestro local."""

        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, engine_version, symbols_json, parameters_json,
                       dataset_sha256, payload_json, payload_sha256,
                       status, created_at
                FROM backtest_runs
                ORDER BY created_at DESC, id DESC LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

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

    def latest_backtest_parameters(self, *, symbol: str, engine_version: str) -> dict[str, Any] | None:
        """Only approved, hash-verified replay evidence for this asset/version.

        Sidecar metadata must agree with the hashed payload. Legacy/rejected
        runs stay in the audit history but can never configure live analysis.
        """
        import math
        from dataclasses import fields
        from .analytics.backtesting import BacktestConfig, PerformanceMetrics, evaluate_capital_preservation, ENGINE_VERSION
        from .analytics.causal_core import causal_revision
        from .analytics.replay import REPLAY_CONTRACT
        symbol = symbol.strip().upper()
        if not symbol or not engine_version:
            raise ValueError('Activo y versión validada son obligatorios.')
        if engine_version != ENGINE_VERSION:
            return None
        revision = causal_revision()
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM backtest_runs WHERE status='APPROVED' AND engine_version=?
                   ORDER BY created_at DESC, id DESC LIMIT 100""", (engine_version,)
            ).fetchall()
        for row in rows:
            try:
                if hashlib.sha256(row['payload_json'].encode()).hexdigest() != row['payload_sha256']:
                    continue
                payload = json.loads(row['payload_json'])
                parameters = json.loads(row['parameters_json'])
                symbols = json.loads(row['symbols_json'])
                if (payload['engine_version'] != engine_version or payload['core_sha256'] != revision
                    or payload['replay_contract'] != REPLAY_CONTRACT
                    or payload['dataset_sha256'] != row['dataset_sha256']
                    or not payload['aggregate_decision'].startswith('APROBADO')
                    or set(symbols) != {r['symbol'] for r in payload['results']} or symbol not in symbols):
                    continue
                config_values = payload['config']
                if set(config_values) != {f.name for f in fields(BacktestConfig)}:
                    continue
                if any(parameters.get(k) != v for k,v in config_values.items()):
                    continue
                if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in config_values.values()):
                    continue
                config = BacktestConfig(**config_values)
                matching = [r for r in payload['results'] if r['symbol']==symbol]
                if len(matching)!=1:
                    continue
                result = matching[0]
                if result.get('replay_contract') != REPLAY_CONTRACT or not result['decision'].startswith('APROBADO'):
                    continue
                metrics = PerformanceMetrics(**result['validation'])
                if not all(v is None or (isinstance(v,(int,float)) and math.isfinite(v))
                           for v in result['validation'].values()):
                    continue
                if metrics.trades != metrics.wins+metrics.losses or metrics.setups < metrics.trades:
                    continue
                if not evaluate_capital_preservation(metrics,config)[0].startswith('APROBADO'):
                    continue
                return dict(config_values)
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                continue
        return None

    @staticmethod
    def _live_observation_payload(
        *,
        symbol: str,
        observed_at: str,
        horizon_minutes: int,
        reference_price: str,
        raw_probability_up: str,
        predicted_direction: str,
        parameters_json: str,
    ) -> str:
        """Historical v1 serialization only; never certifies v2 live outcomes.

        Kept for identifying old audit evidence. New writes and all consumers
        use model_observations.forecast_digest/resolution_digest instead.
        """
        return json.dumps(
            {
                "symbol": symbol,
                "observed_at": observed_at,
                "horizon_minutes": horizon_minutes,
                "reference_price": reference_price,
                "raw_probability_up": raw_probability_up,
                "predicted_direction": predicted_direction,
                "parameters_json": parameters_json,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def record_live_model_observation(
        self,
        *,
        symbol: str,
        observed_at: datetime,
        source_bar_at: datetime,
        reference_price: Decimal,
        raw_probability_up: Decimal,
        parameters_json: str,
        horizon_minutes: int = 390,
    ) -> bool:
        """Record a forecast at emission, referencing a known closed 5m bar.

        source_bar_at denotes the source bar CLOSE, not its open label.
        available_at is the immutable target close, never a later spot quote.
        """
        from .services.model_observations import VERSION
        row = self._live_observation_row(
            symbol=symbol, observed_at=observed_at, source_bar_at=source_bar_at,
            reference_price=reference_price, raw_probability_up=raw_probability_up,
            parameters_json=parameters_json, horizon_minutes=horizon_minutes,
        )
        with self.database.transaction() as connection:
            # Reruns must not manufacture independent samples from the same bar.
            duplicate = connection.execute(
                """SELECT 1 FROM live_model_observations
                   WHERE symbol=? AND source_bar_at=? AND horizon_minutes=?
                     AND integrity_version=? LIMIT 1""",
                (row["symbol"], row["source_bar_at"], horizon_minutes, VERSION),
            ).fetchone()
            if duplicate:
                return False
            names = tuple(row)
            connection.execute(
                f"INSERT INTO live_model_observations({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                tuple(row[name] for name in names),
            )
        return True

    @staticmethod
    def _live_observation_row(
        *, symbol: str, observed_at: datetime, source_bar_at: datetime,
        reference_price: Decimal, raw_probability_up: Decimal,
        parameters_json: str, horizon_minutes: int,
    ) -> dict[str, Any]:
        from .services.model_observations import (
            VERSION, POLICY, maturity, utc_timestamp, is_regular_close,
            forecast_digest, canonical,
        )
        observed = utc_timestamp(observed_at)
        source = utc_timestamp(source_bar_at)
        if not source <= observed < source + timedelta(minutes=5) or not is_regular_close(source):
            raise ValueError("El precio de referencia debe provenir de una vela cerrada reciente (menos de 5 min).")
        if not reference_price.is_finite() or reference_price <= 0:
            raise ValueError("Precio de referencia inválido.")
        if not raw_probability_up.is_finite() or not Decimal(0) <= raw_probability_up <= Decimal(1):
            raise ValueError("Probabilidad fuera de [0, 1].")
        parameters = json.loads(parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("Los parámetros del modelo deben ser un objeto JSON.")
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("Falta el símbolo del pronóstico.")
        row = dict(
            symbol=symbol, observed_at=observed.isoformat(),
            available_at=maturity(observed, horizon_minutes).isoformat(),
            horizon_minutes=horizon_minutes, source_bar_at=source.isoformat(),
            reference_price=str(reference_price), raw_probability_up=str(raw_probability_up),
            predicted_direction="UP" if raw_probability_up >= Decimal("0.5") else "DOWN",
            parameters_json=canonical(parameters), integrity_version=VERSION,
            horizon_policy=POLICY, created_at=_utc_now().isoformat(),
        )
        row["observation_sha256"] = forecast_digest(row)
        row["resolution_status"] = "PENDING"
        return row

    def record_fixed_live_observations(
        self, *, symbol: str, observed_at: datetime, source_bar_at: datetime,
        reference_price: Decimal, forecasts: list[dict[str, Any]],
        session_date: str, protocol: str,
    ) -> int:
        """Atomic fixed-session cohort. A retry never adds a second 11 NY sample."""
        from .services.model_observations import VERSION, canonical, valid_observation
        from .services.directional_collection import HORIZON_MINUTES
        from .services.model_execution_record import execution_id
        from .analytics.operational_target import (
            TARGET_VERSION, validate_operational_contract, valid_operational_outcome,
        )
        if len(forecasts) != len(HORIZON_MINUTES) or {
            int(item["horizon_minutes"]) for item in forecasts
        } != set(HORIZON_MINUTES.values()):
            raise ValueError("El corte direccional necesita seis horizontes únicos.")
        rows = [self._live_observation_row(
            symbol=symbol, observed_at=observed_at, source_bar_at=source_bar_at,
            reference_price=reference_price,
            raw_probability_up=item["raw_probability_up"],
            parameters_json=item["parameters_json"],
            horizon_minutes=item["horizon_minutes"],
        ) for item in forecasts]
        metadata = [json.loads(row["parameters_json"]) for row in rows]
        if any(item.get("collection_protocol") != protocol
               or item.get("session_date") != session_date for item in metadata):
            raise ValueError("El protocolo y la sesión del lote no coinciden.")
        replay = metadata[0].get("replay")
        if (not isinstance(replay, dict)
                or replay.get("run_id") != execution_id(symbol, session_date, protocol)
                or replay.get("observed_at") != rows[0]["observed_at"]
                or replay.get("source_bar_closed_at") != rows[0]["source_bar_at"]
                or any(canonical(item.get("replay")) != canonical(replay) for item in metadata)):
            raise ValueError("El snapshot reproducible del lote no coincide.")
        for row, item in zip(rows, metadata):
            operational = item.get("operational_contract")
            validate_operational_contract(operational)
            if (item.get("primary_validation_target") != TARGET_VERSION
                    or operational["model"]["symbol"] != row["symbol"]
                    or operational["model"]["horizon_minutes"] != row["horizon_minutes"]
                    or operational["observed_at"] != row["observed_at"]
                    or operational["entry_at"] != row["source_bar_at"]
                    or abs(float(operational["entry_price"]) - float(row["reference_price"])) > 1e-9):
                raise ValueError("El objetivo operativo no coincide con la observación firmada.")
        with self.database.transaction() as connection:
            prior = connection.execute(
                """SELECT * FROM live_model_observations
                   WHERE symbol=? AND integrity_version=?
                     AND CASE WHEN json_valid(parameters_json)
                         THEN json_extract(parameters_json, '$.collection_protocol') END=?
                     AND CASE WHEN json_valid(parameters_json)
                         THEN json_extract(parameters_json, '$.session_date') END=?""",
                (symbol.strip().upper(), VERSION, protocol, session_date),
            ).fetchall()
            if prior:
                ids = [int(item["id"]) for item in prior]
                placeholders = ",".join("?" for _ in ids)
                children = connection.execute(
                    f"SELECT * FROM operational_model_outcomes WHERE observation_id IN ({placeholders})",
                    ids,
                ).fetchall()
                parents = {int(item["id"]): item for item in prior}
                if len(prior) == len(rows) and {
                    int(item["horizon_minutes"]) for item in prior
                } == set(HORIZON_MINUTES.values()) and all(valid_observation(item) for item in prior) \
                        and len(children) == len(rows) and all(
                            valid_operational_outcome(child, parents[int(child["observation_id"])])
                            for child in children
                        ):
                    return 0
                raise ValueError("Cohorte direccional incompleta o con integridad inválida; revisión manual.")
            # Existing UI-era V3 observations at the same bar must not be
            # overwritten or silently substituted for this fixed protocol.
            conflict = connection.execute(
                """SELECT 1 FROM live_model_observations
                   WHERE symbol=? AND source_bar_at=? AND integrity_version=? LIMIT 1""",
                (rows[0]["symbol"], rows[0]["source_bar_at"], VERSION),
            ).fetchone()
            if conflict:
                raise ValueError("Ya existe una emisión V3 de este cierre fuera del corte fijo.")
            for row in rows:
                names = tuple(row)
                cursor = connection.execute(
                    f"INSERT INTO live_model_observations({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                    tuple(row[name] for name in names),
                )
                contract = json.loads(row["parameters_json"])["operational_contract"]
                connection.execute(
                    """INSERT INTO operational_model_outcomes(
                        observation_id,target_version,contract_sha256,resolution_status,created_at
                    ) VALUES (?,?,?,'PENDING',?)""",
                    (cursor.lastrowid, TARGET_VERSION, contract["contract_sha256"], row["created_at"]),
                )
        return len(rows)

    def live_model_execution_record(self, run_id: str) -> dict[str, Any] | None:
        """Read one complete, signed six-horizon cut with its eventual outcomes.

        A missing, partial, inconsistent or tampered cut is never presented as
        reproducible evidence. Historical V3 rows without replay metadata keep
        their old calibration behavior, but are not falsely upgraded here.
        """
        from .services.directional_collection import HORIZON_MINUTES, SUPPORTED_COLLECTION_PROTOCOLS
        from .services.model_execution_record import execution_id
        from .services.model_observations import VERSION, canonical, valid_observation
        from .services.scenario_calibration import validate_contract

        if not isinstance(run_id, str) or len(run_id) != 64 or any(
            character not in "0123456789abcdef" for character in run_id
        ):
            return None
        with self.database.connect() as connection:
            stored = connection.execute(
                """SELECT * FROM live_model_observations
                   WHERE integrity_version=? AND CASE WHEN json_valid(parameters_json)
                       THEN json_extract(parameters_json, '$.replay.run_id') END=?
                   ORDER BY horizon_minutes""",
                (VERSION, run_id),
            ).fetchall()
            child_rows = connection.execute(
                """SELECT operational_model_outcomes.* FROM operational_model_outcomes
                   JOIN live_model_observations
                     ON live_model_observations.id=operational_model_outcomes.observation_id
                   WHERE live_model_observations.integrity_version=?
                     AND CASE WHEN json_valid(live_model_observations.parameters_json)
                         THEN json_extract(live_model_observations.parameters_json, '$.replay.run_id') END=?""",
                (VERSION, run_id),
            ).fetchall()
        if len(stored) != len(HORIZON_MINUTES) or not all(valid_observation(row) for row in stored):
            return None
        rows = [dict(row) for row in stored]
        operational_by_id = {int(row["observation_id"]): row for row in child_rows}
        if {row["horizon_minutes"] for row in rows} != set(HORIZON_MINUTES.values()):
            return None
        try:
            metadata = [json.loads(row["parameters_json"]) for row in rows]
            from .analytics.operational_target import (
                validate_operational_contract, valid_operational_outcome,
            )
            from .services.directional_collection import (
                COLLECTION_PROTOCOL, LEGACY_COLLECTION_PROTOCOL,
            )
            replay = metadata[0]["replay"]
            if not isinstance(replay, dict) or any(
                canonical(item["replay"]) != canonical(replay) for item in metadata
            ):
                return None
            symbol, session = replay["symbol"], replay["session_date"]
            stored_protocol = replay["collection_protocol"]
            if (stored_protocol not in SUPPORTED_COLLECTION_PROTOCOLS
                    or replay["run_id"] != execution_id(symbol, session, stored_protocol)):
                return None
            requires_operational_target = stored_protocol == COLLECTION_PROTOCOL
            if stored_protocol == LEGACY_COLLECTION_PROTOCOL:
                # V1 predates the first-passage target.  It remains valid
                # directional evidence, but must never be silently upgraded
                # with an unsigned or synthetic operational child.
                if child_rows or any(
                    "operational_contract" in item or "primary_validation_target" in item
                    for item in metadata
                ):
                    return None
            predictions = []
            by_minutes = {minutes: label for label, minutes in HORIZON_MINUTES.items()}
            for row, item in zip(rows, metadata):
                contract = item["scenario_contract"]
                vector, low, high = validate_contract(contract)
                snapshot = item["prediction_snapshot"]
                operational_contract = None
                operational_result = None
                if requires_operational_target:
                    operational_contract = item["operational_contract"]
                    validate_operational_contract(operational_contract)
                    operational_result = operational_by_id.get(int(row["id"]))
                    if (operational_result is None
                            or not valid_operational_outcome(operational_result, row)):
                        return None
                if (row["symbol"] != symbol or row["observed_at"] != replay["observed_at"]
                    or row["source_bar_at"] != replay["source_bar_closed_at"]
                    or item["session_date"] != session
                    or item["collection_protocol"] != stored_protocol
                    or contract["model"]["symbol"] != symbol
                    or contract["model"]["horizon_minutes"] != row["horizon_minutes"]
                    or contract["model"]["engine"] != snapshot["engine_name"]
                    or snapshot["label"] != by_minutes[row["horizon_minutes"]]
                    or abs(vector[0] - float(row["raw_probability_up"])) > 1e-9
                    or abs(vector[0] - float(snapshot["probability_up"]) / 100) > 1e-9
                    or abs(vector[1] - float(snapshot["probability_range"]) / 100) > 1e-9
                    or abs(vector[2] - float(snapshot["probability_down"]) / 100) > 1e-9
                    or abs(low - float(snapshot["range_low"])) > 1e-9
                    or abs(high - float(snapshot["range_high"])) > 1e-9):
                    return None
                prediction = {
                    "horizon": snapshot["label"],
                    "horizon_minutes": row["horizon_minutes"],
                    "model_id": contract["model_id"],
                    "engine_revision": contract["model"]["engine_revision"],
                    "prediction": snapshot,
                    "observed_at": row["observed_at"],
                    "available_at": row["available_at"],
                    "reference_price": row["reference_price"],
                    "predicted_direction": row["predicted_direction"],
                    "resolution_status": row["resolution_status"],
                    "outcome_price": row["outcome_price"],
                    "outcome_up": row["outcome_up"],
                    "successful": row["successful"],
                    "resolved_at": row["resolved_at"],
                    "outcome_bar_at": row["outcome_bar_at"],
                    "outcome_source": row["outcome_source"],
                    "observation_sha256": row["observation_sha256"],
                    "resolution_sha256": row["resolution_sha256"],
                }
                if requires_operational_target:
                    prediction["operational_target"] = operational_contract
                    prediction["operational_result"] = {
                        "resolution_status": operational_result["resolution_status"],
                        "outcome": operational_result["outcome"],
                        "exit_price": operational_result["exit_price"],
                        "exit_at": operational_result["exit_at"],
                        "exit_source": operational_result["exit_source"],
                        "evidence_sha256": operational_result["evidence_sha256"],
                        "evidence": json.loads(operational_result["evidence_json"])
                            if operational_result["evidence_json"] else None,
                        "resolved_at": operational_result["resolved_at"],
                        "outcome_sha256": operational_result["outcome_sha256"],
                        "checkpoint": {
                            "scanned_through": operational_result["scanned_through"],
                            "evidence_sha256": operational_result["scan_evidence_sha256"],
                            "evidence_count": operational_result["scan_evidence_count"],
                            "evidence": json.loads(operational_result["scan_evidence_json"])
                                if operational_result["scan_evidence_json"] else None,
                            "updated_at": operational_result["checkpoint_updated_at"],
                            "checkpoint_sha256": operational_result["checkpoint_sha256"],
                        },
                    }
                predictions.append(prediction)
            return {**replay, "predictions": predictions}
        except (KeyError, TypeError, ValueError, OverflowError, IndexError):
            return None

    def pending_operational_model_symbols(self, as_of: datetime) -> tuple[str, ...]:
        """All active first-passage paths must be scanned, even before timeout."""
        from .services.model_observations import VERSION, utc_timestamp
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT live.symbol FROM operational_model_outcomes AS target
                   JOIN live_model_observations AS live ON live.id=target.observation_id
                   WHERE live.integrity_version=? AND target.resolution_status='PENDING'
                     AND live.observed_at<=? ORDER BY live.symbol""",
                (VERSION, utc_timestamp(as_of).isoformat()),
            ).fetchall()
        return tuple(row["symbol"] for row in rows)

    def resolve_operational_model_outcomes(
        self, *, symbol: str, current_as_of: datetime, historical_bars=None,
        historical_daily_bars=None, source: str = "yfinance:5m:raw-ohlcv",
        daily_source: str = "yfinance:1d:raw-ohlcv",
        evidence_artifacts=None,
    ) -> int:
        """Advance signed evidence and attach TP/SL/timeout exactly once.

        All pandas/calendar work happens before the short compare-and-swap
        transaction so analytical catch-up never holds the ledger writer lock.
        """
        from .analytics.operational_target import (
            IncompleteOperationalEvidence, OperationalResult,
            operational_checkpoint_digest, operational_outcome_digest,
            scan_operational_outcome, valid_operational_outcome,
        )
        from .services.model_observations import VERSION, canonical, utc_timestamp, valid_observation
        now = utc_timestamp(current_as_of)
        with self.database.connect() as connection:
            joined = connection.execute(
                """SELECT live.*, target.observation_id AS target_observation_id,
                          target.target_version AS target_version,
                          target.contract_sha256 AS contract_sha256,
                          target.resolution_status AS target_resolution_status,
                          target.outcome AS target_outcome,
                          target.exit_price AS target_exit_price,
                          target.exit_at AS target_exit_at,
                          target.exit_source AS target_exit_source,
                          target.evidence_sha256 AS target_evidence_sha256,
                          target.evidence_json AS target_evidence_json,
                          target.resolved_at AS target_resolved_at,
                          target.outcome_sha256 AS target_outcome_sha256,
                          target.scanned_through AS target_scanned_through,
                          target.scan_evidence_sha256 AS target_scan_evidence_sha256,
                          target.scan_evidence_count AS target_scan_evidence_count,
                          target.scan_evidence_json AS target_scan_evidence_json,
                          target.checkpoint_updated_at AS target_checkpoint_updated_at,
                          target.checkpoint_sha256 AS target_checkpoint_sha256,
                          target.created_at AS target_created_at
                   FROM live_model_observations AS live
                   JOIN operational_model_outcomes AS target ON target.observation_id=live.id
                   WHERE live.symbol=? AND live.integrity_version=?
                     AND target.resolution_status='PENDING' AND live.observed_at<=?
                   ORDER BY live.observed_at,live.horizon_minutes""",
                (symbol.strip().upper(), VERSION, now.isoformat()),
            ).fetchall()
        proposals = []
        for joined_row in joined:
            combined = dict(joined_row)
            parent = {key: combined[key] for key in joined_row.keys()
                      if not key.startswith("target_") and key != "target_observation_id"}
            child = {
                "observation_id": combined["target_observation_id"],
                "target_version": combined["target_version"],
                "contract_sha256": combined["contract_sha256"],
                "resolution_status": combined["target_resolution_status"],
                "outcome": combined["target_outcome"],
                "exit_price": combined["target_exit_price"],
                "exit_at": combined["target_exit_at"],
                "exit_source": combined["target_exit_source"],
                "evidence_sha256": combined["target_evidence_sha256"],
                "evidence_json": combined["target_evidence_json"],
                "resolved_at": combined["target_resolved_at"],
                "outcome_sha256": combined["target_outcome_sha256"],
                "scanned_through": combined["target_scanned_through"],
                "scan_evidence_sha256": combined["target_scan_evidence_sha256"],
                "scan_evidence_count": combined["target_scan_evidence_count"],
                "scan_evidence_json": combined["target_scan_evidence_json"],
                "checkpoint_updated_at": combined["target_checkpoint_updated_at"],
                "checkpoint_sha256": combined["target_checkpoint_sha256"],
                "created_at": combined["target_created_at"],
            }
            if not valid_observation(parent):
                raise ValueError(
                    f"Observación operativa {child['observation_id']} con integridad inválida."
                )
            if not valid_operational_outcome(child, parent):
                raise ValueError(
                    f"Resultado operativo {child['observation_id']} con integridad inválida."
                )
            contract = json.loads(parent["parameters_json"])["operational_contract"]
            old_count = int(child["scan_evidence_count"] or 0)
            try:
                scan = scan_operational_outcome(
                    contract, historical_bars, now, daily_bars=historical_daily_bars,
                    resume_at=child["scanned_through"],
                    previous_evidence_sha256=child["scan_evidence_sha256"],
                    previous_evidence_count=old_count,
                )
            except IncompleteOperationalEvidence:
                continue
            if scan.evidence_count == old_count:
                continue
            prior_evidence = (
                json.loads(child["scan_evidence_json"])
                if child["scan_evidence_json"] else {"chunks": []}
            )
            chunks = list(prior_evidence.get("chunks", []))
            chunks.append({
                "scanned_at": now.isoformat(),
                "scanned_through": scan.scanned_through,
                "evidence_count": scan.evidence_count,
                "artifacts": dict(evidence_artifacts or {}),
            })
            evidence_json = canonical({"chunks": chunks})
            checkpoint_updated_at = now.isoformat()
            checkpoint_sha = operational_checkpoint_digest(
                parent["observation_sha256"], child["contract_sha256"],
                scan.scanned_through, scan.evidence_sha256, scan.evidence_count,
                evidence_json, checkpoint_updated_at,
            )
            result = scan.result
            if result is not None:
                provider = daily_source if result.exit_source.startswith("1d:") else source
                result = OperationalResult(
                    result.outcome, result.exit_price, result.exit_at,
                    f"{provider}:{result.exit_source}", result.evidence_sha256,
                )
                outcome_sha = operational_outcome_digest(
                    parent["observation_sha256"], child["contract_sha256"], result,
                    checkpoint_updated_at, evidence_json,
                )
            else:
                outcome_sha = None
            proposals.append((parent, child, scan, result, evidence_json,
                              checkpoint_updated_at, checkpoint_sha, outcome_sha))

        resolved = 0
        with self.database.transaction() as connection:
            for parent, child, scan, result, evidence_json, updated_at, checkpoint_sha, outcome_sha in proposals:
                common = (
                    scan.scanned_through, scan.evidence_sha256, scan.evidence_count,
                    evidence_json, updated_at, checkpoint_sha,
                )
                cas = (child["observation_id"], int(child["scan_evidence_count"] or 0),
                       child["checkpoint_sha256"] or "")
                if result is None:
                    cursor = connection.execute(
                        """UPDATE operational_model_outcomes SET
                             scanned_through=?,scan_evidence_sha256=?,scan_evidence_count=?,
                             scan_evidence_json=?,checkpoint_updated_at=?,checkpoint_sha256=?
                           WHERE observation_id=? AND resolution_status='PENDING'
                             AND scan_evidence_count=? AND COALESCE(checkpoint_sha256,'')=?""",
                        (*common, *cas),
                    )
                else:
                    cursor = connection.execute(
                        """UPDATE operational_model_outcomes SET
                             resolution_status='RESOLVED',outcome=?,exit_price=?,exit_at=?,
                             exit_source=?,evidence_sha256=?,evidence_json=?,resolved_at=?,outcome_sha256=?,
                             scanned_through=?,scan_evidence_sha256=?,scan_evidence_count=?,
                             scan_evidence_json=?,checkpoint_updated_at=?,checkpoint_sha256=?
                           WHERE observation_id=? AND resolution_status='PENDING'
                             AND scan_evidence_count=? AND COALESCE(checkpoint_sha256,'')=?""",
                        (result.outcome.value, str(result.exit_price), result.exit_at,
                         result.exit_source, result.evidence_sha256, evidence_json,
                         updated_at, outcome_sha, *common, *cas),
                    )
                if result is not None and cursor.rowcount:
                    resolved += 1
        return resolved

    def verify_operational_model_outcomes(self) -> tuple[int, tuple[int, ...]]:
        from .analytics.operational_target import valid_operational_outcome
        from .services.model_observations import valid_observation
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT live.*, target.observation_id AS child_id,
                          target.target_version AS child_target_version,
                          target.contract_sha256 AS child_contract_sha256,
                          target.resolution_status AS child_resolution_status,
                          target.outcome AS child_outcome,target.exit_price AS child_exit_price,
                          target.exit_at AS child_exit_at,target.exit_source AS child_exit_source,
                          target.evidence_sha256 AS child_evidence_sha256,
                          target.evidence_json AS child_evidence_json,
                          target.resolved_at AS child_resolved_at,
                          target.outcome_sha256 AS child_outcome_sha256,
                          target.scanned_through AS child_scanned_through,
                          target.scan_evidence_sha256 AS child_scan_evidence_sha256,
                          target.scan_evidence_count AS child_scan_evidence_count,
                          target.scan_evidence_json AS child_scan_evidence_json,
                          target.checkpoint_updated_at AS child_checkpoint_updated_at,
                          target.checkpoint_sha256 AS child_checkpoint_sha256,
                          target.created_at AS child_created_at
                   FROM operational_model_outcomes AS target
                   JOIN live_model_observations AS live ON live.id=target.observation_id"""
            ).fetchall()
        invalid = []
        for stored in rows:
            data = dict(stored)
            child = {"observation_id": data["child_id"], "target_version": data["child_target_version"],
                     "contract_sha256": data["child_contract_sha256"],
                     "resolution_status": data["child_resolution_status"],
                     "outcome": data["child_outcome"], "exit_price": data["child_exit_price"],
                     "exit_at": data["child_exit_at"], "exit_source": data["child_exit_source"],
                     "evidence_sha256": data["child_evidence_sha256"],
                     "evidence_json": data["child_evidence_json"],
                     "resolved_at": data["child_resolved_at"],
                     "outcome_sha256": data["child_outcome_sha256"],
                     "scanned_through": data["child_scanned_through"],
                     "scan_evidence_sha256": data["child_scan_evidence_sha256"],
                     "scan_evidence_count": data["child_scan_evidence_count"],
                     "scan_evidence_json": data["child_scan_evidence_json"],
                     "checkpoint_updated_at": data["child_checkpoint_updated_at"],
                     "checkpoint_sha256": data["child_checkpoint_sha256"],
                     "created_at": data["child_created_at"]}
            parent = {key: data[key] for key in stored.keys() if not key.startswith("child_")}
            if not valid_observation(parent) or not valid_operational_outcome(child, parent):
                invalid.append(int(child["observation_id"]))
        return len(rows) - len(invalid), tuple(invalid)

    def resolve_live_model_observations(
        self,
        *,
        symbol: str,
        current_as_of: datetime,
        historical_bars=None,
        historical_daily_bars=None,
        source: str = "yfinance:5m:raw-close",
        daily_source: str = "yfinance:1d:raw-close",
        current_price: Decimal | None = None,
    ) -> int:
        """Resolve only exact historical closes. Missing bars remain pending.

        current_as_of is processing time, not the last quote time.
        Legacy current_price calls are explicitly prohibited.
        """
        from .services.model_observations import (
            utc_timestamp, is_regular_close, exact_closed_prices,
            exact_daily_closed_prices, valid_observation, resolution_digest,
            POLICY, SESSION_HORIZONS,
        )
        if current_price is not None:
            raise ValueError("No se permite resolver con precio actual; proporciona velas históricas de 5m.")
        if not source or not isinstance(source, str) or not daily_source or not isinstance(daily_source, str):
            raise ValueError("Falta la procedencia del precio histórico.")
        now = utc_timestamp(current_as_of)
        intraday_prices = exact_closed_prices(historical_bars, now)
        daily_prices = exact_daily_closed_prices(historical_daily_bars, now)
        resolved = 0
        with self.database.transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM live_model_observations
                   WHERE symbol=? AND integrity_version>=2 AND resolution_status='PENDING'
                     AND available_at<=? ORDER BY available_at, id""",
                (symbol.strip().upper(), now.isoformat()),
            ).fetchall()
            for stored in rows:
                row = dict(stored)
                # Never sign an already altered pending observation.
                if not valid_observation(row):
                    continue
                due = row["available_at"]
                if not is_regular_close(due):
                    row.update(resolution_status="INVALID_MARKET_CLOSED", resolved_at=now.isoformat())
                else:
                    session_based = (
                        row["horizon_policy"] == POLICY
                        and int(row["horizon_minutes"]) in SESSION_HORIZONS
                    )
                    prices = daily_prices if session_based else intraday_prices
                    outcome_source = daily_source if session_based else source
                if row["resolution_status"] == "PENDING" and due not in prices:
                    continue
                if row["resolution_status"] == "PENDING":
                    outcome = Decimal(prices[due])
                    up = int(outcome > Decimal(row["reference_price"]))
                    row.update(
                        outcome_price=str(outcome), outcome_up=up,
                        successful=int((row["predicted_direction"] == "UP") == bool(up)),
                        resolved_at=now.isoformat(), outcome_bar_at=due,
                        outcome_source=outcome_source, resolution_status="RESOLVED",
                    )
                    resolved += 1
                row["resolution_sha256"] = resolution_digest(row)
                connection.execute(
                    """UPDATE live_model_observations
                       SET outcome_price=:outcome_price, outcome_up=:outcome_up,
                           successful=:successful, resolved_at=:resolved_at,
                           outcome_bar_at=:outcome_bar_at, outcome_source=:outcome_source,
                           resolution_status=:resolution_status, resolution_sha256=:resolution_sha256
                       WHERE id=:id AND resolution_status='PENDING'""", row,
                )
        return resolved

    def pending_live_model_symbols(self, as_of: datetime) -> tuple[str, ...]:
        """Symbols with due signed V3 forecasts, independent of zone predictions."""
        from .services.model_observations import VERSION, utc_timestamp
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT symbol FROM live_model_observations
                   WHERE integrity_version=? AND resolution_status='PENDING'
                     AND available_at<=? ORDER BY symbol""",
                (VERSION, utc_timestamp(as_of).isoformat()),
            ).fetchall()
        return tuple(row["symbol"] for row in rows)

    def _verified_live_results(self, symbol: str, *, horizon_minutes=None, limit=2000):
        from .services.model_observations import VERSION, valid_observation
        query = """SELECT * FROM live_model_observations WHERE symbol=?
                   AND integrity_version=? AND resolution_status='RESOLVED'"""
        values = [symbol.strip().upper(), VERSION]
        if horizon_minutes is not None:
            query += " AND horizon_minutes=?"
            values.append(int(horizon_minutes))
        query += " ORDER BY resolved_at DESC, id DESC LIMIT ?"
        values.append(max(1, min(int(limit), 10000)))
        with self.database.connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [row for row in rows if valid_observation(row)]

    def live_model_stats(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Return aggregate monitoring only; never an execution threshold.

        Rows can belong to different horizons, so accuracy and Brier are useful
        for operational health/audit but are not a statistically coherent input
        to any individual horizon decision.
        """
        rows = self._verified_live_results(symbol, limit=max(1, min(limit, 500)))
        total = len(rows)
        wins = sum(int(row["successful"]) for row in rows)
        accuracy = wins / total if total else 0.0
        from .analytics.probability_calibration import CalibrationSample, chronological_split
        timed = [CalibrationSample((float(r['raw_probability_up']),), int(r['outcome_up']),
                 *(datetime.fromisoformat(r[k]) for k in ('observed_at','available_at','resolved_at')))
                 for r in rows]
        holdout = chronological_split(timed).holdout
        brier = (sum((r.probabilities[0]-r.outcome)**2 for r in holdout)/len(holdout)
                 if holdout else None)
        return dict(resolved=total, wins=wins, accuracy=accuracy,
                    brier_score=brier, brier_holdout_samples=len(holdout),
                    decision_eligible=False,
                    scope="AGGREGATE_DIAGNOSTIC_ONLY")

    def live_model_calibration_samples(
        self, symbol: str, *, horizon_minutes: int = 390, limit: int = 2000,
    ) -> tuple[tuple[float, int], ...]:
        """Unsigned legacy or tampered resolutions never enter calibration."""
        rows = self._verified_live_results(symbol, horizon_minutes=horizon_minutes, limit=limit)
        rows.sort(key=lambda row: datetime.fromisoformat(row["observed_at"]))
        return tuple((float(row["raw_probability_up"]), int(row["outcome_up"])) for row in rows)

    def live_scenario_calibration_samples(
        self, symbol: str, *, horizon_minutes: int, model_id: str,
        as_of: datetime, limit: int = 10000,
    ):
        """One signed model/target; emission order and point-in-time labels.

        Legacy V2 binary records and other model versions are never
        reinterpreted as samples of the current V3 session contract.
        The calibrator performs 60/20/20 splitting and temporal purging.
        """
        from .analytics.probability_calibration import CalibrationSample
        from .services.model_observations import VERSION, valid_observation, utc_timestamp
        from .services.scenario_calibration import validate_contract, outcome_class
        cutoff = utc_timestamp(as_of)
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM live_model_observations
                   WHERE symbol=? AND horizon_minutes=? AND integrity_version=?
                     AND resolution_status='RESOLVED' AND resolved_at<=?
                     AND CASE WHEN json_valid(parameters_json)
                         THEN json_extract(parameters_json, '$.scenario_contract.model_id') END = ?
                   ORDER BY observed_at DESC, id DESC LIMIT ?""",
                (symbol.strip().upper(), horizon_minutes, VERSION, cutoff.isoformat(), model_id,
                 max(1, min(limit, 10000))),
            ).fetchall()
        samples = []
        for row in rows:
            if not valid_observation(row):
                continue
            try:
                contract = json.loads(row['parameters_json'])['scenario_contract']
                vector, low, high = validate_contract(contract)
                if contract['model']['symbol'] != row['symbol'] or contract['model']['horizon_minutes'] != row['horizon_minutes']:
                    continue
                if abs(vector[0]-float(row['raw_probability_up'])) > 1e-9:
                    continue
                observed, available, resolved = [utc_timestamp(row[k]).to_pydatetime()
                    for k in ('observed_at', 'available_at', 'resolved_at')]
                if not observed < available <= resolved <= cutoff:
                    continue
                samples.append(CalibrationSample(vector, outcome_class(row['outcome_price'],low,high),
                                                  observed, available, resolved))
            except (KeyError, TypeError, ValueError):
                continue
        # One independent cohort per NY trading session and horizon. Repeated
        # five-minute emissions are useful for audit/replay, but must not inflate
        # the calibration sample size. Keep the first knowable forecast of each
        # session for this immutable model contract.
        ny = ZoneInfo("America/New_York")
        daily_cohort = {}
        for sample in sorted(samples, key=lambda item: item.observed_at):
            session_date = sample.observed_at.astimezone(ny).date()
            daily_cohort.setdefault(session_date, sample)
        return tuple(daily_cohort.values())

    def verify_live_model_observations(self) -> tuple[int, tuple[int, ...]]:
        """Invalid IDs include legacy rows whose outcomes cannot be certified."""
        from .services.model_observations import valid_observation
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM live_model_observations").fetchall()
        invalid = tuple(int(row["id"]) for row in rows if not valid_observation(row))
        return len(rows) - len(invalid), invalid

    def record_fundamental_news_snapshot(
        self,
        *,
        symbol: str,
        observed_at: datetime,
        provider: str,
        engine_version: str,
        payload_json: str,
    ) -> tuple[int, str]:
        """Versiona un corte inmutable; el JSON exacto es la unidad auditada."""

        digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        normalized_symbol = symbol.strip().upper()
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO fundamental_news_snapshots(
                    symbol, observed_at, provider, engine_version,
                    payload_json, payload_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_symbol,
                    observed_at.astimezone(timezone.utc).isoformat(),
                    provider,
                    engine_version,
                    payload_json,
                    digest,
                    _utc_now().isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT id FROM fundamental_news_snapshots WHERE payload_sha256 = ?",
                (digest,),
            ).fetchone()
        if not row:
            raise RuntimeError("No fue posible versionar el corte fundamental.")
        return int(row["id"]), digest

    def latest_fundamental_news_snapshot(
        self, symbol: str
    ) -> dict[str, Any] | None:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM fundamental_news_snapshots
                WHERE symbol = ?
                ORDER BY observed_at DESC, id DESC LIMIT 1
                """,
                (symbol.strip().upper(),),
            ).fetchone()
        if not row:
            return None
        payload = dict(row)
        calculated = hashlib.sha256(
            str(payload["payload_json"]).encode("utf-8")
        ).hexdigest()
        return payload if calculated == str(payload["payload_sha256"]) else None

    def verify_fundamental_news_snapshots(self) -> tuple[int, tuple[int, ...]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, payload_json, payload_sha256 FROM fundamental_news_snapshots"
            ).fetchall()
        invalid = tuple(
            int(row["id"])
            for row in rows
            if hashlib.sha256(str(row["payload_json"]).encode("utf-8")).hexdigest()
            != str(row["payload_sha256"])
        )
        return len(rows) - len(invalid), invalid
