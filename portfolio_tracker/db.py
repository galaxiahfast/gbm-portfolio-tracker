"""Conexion SQLite, esquema y migraciones idempotentes."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import DB_PATH, ensure_data_directories


BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cash_movements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('INITIAL', 'DEPOSIT', 'WITHDRAWAL')),
    original_amount TEXT NOT NULL,
    original_currency TEXT NOT NULL CHECK (original_currency IN ('USD', 'MXN')),
    usd_amount TEXT NOT NULL,
    fx_rate TEXT,
    occurred_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    original_path TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    byte_size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    product TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price_usd TEXT NOT NULL,
    commission_usd TEXT NOT NULL,
    commission_rate_pct TEXT,
    gross_usd TEXT NOT NULL,
    reported_total_usd TEXT,
    reported_total_type TEXT NOT NULL DEFAULT 'GROSS'
        CHECK (reported_total_type IN ('GROSS', 'SETTLEMENT')),
    cash_delta_usd TEXT NOT NULL,
    fx_rate TEXT,
    executed_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    receipt_id INTEGER REFERENCES receipts(id),
    ocr_text TEXT NOT NULL DEFAULT '',
    ocr_confidence TEXT,
    validation_status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
    ON trades(symbol, executed_at, id);
CREATE INDEX IF NOT EXISTS idx_cash_time
    ON cash_movements(occurred_at, id);

CREATE TABLE IF NOT EXISTS fx_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    rate TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_reference INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_fx_pair_time
    ON fx_rates(base_currency, quote_currency, observed_at DESC);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    price_usd TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_manual INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol_time
    ON price_snapshots(symbol, observed_at DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cash_usd TEXT NOT NULL,
    holdings_value_usd TEXT NOT NULL,
    equity_usd TEXT NOT NULL,
    fx_rate TEXT NOT NULL,
    observed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_time
    ON portfolio_snapshots(observed_at DESC);

-- Tablas vacias preparadas para indicadores y modelos futuros. No se generan
-- predicciones ficticias: cada resultado debera indicar modelo, version y fecha.
CREATE TABLE IF NOT EXISTS analytics_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(symbol, as_of, feature_name, source)
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    training_cutoff TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES prediction_runs(id),
    symbol TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    probability_up TEXT NOT NULL,
    expected_return TEXT,
    confidence TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_version TEXT NOT NULL,
    symbols_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('APPROVED', 'REJECTED')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_time
    ON backtest_runs(created_at DESC, id DESC);
"""

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, "esquema_inicial"),
    (2, "tipo_total_comprobante"),
    (3, "historial_auditorias"),
    (4, "comprobante_unico_por_operacion"),
    (5, "calibracion_estadistica_backtesting"),
)


class Database:
    """Fabrica conexiones cortas; SQLite WAL permite lecturas fluidas."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        ensure_data_directories()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        """Aplica migraciones pendientes sin reconstruir tablas existentes."""

        existed_before = self.path.exists()
        with self.connect() as connection:
            migrations_table_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'schema_migrations'
                """
            ).fetchone()
            applied_before = (
                {
                    int(row["version"])
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                if migrations_table_exists
                else set()
            )
            pending = [version for version, _ in MIGRATIONS if version not in applied_before]
            if existed_before and pending:
                self._backup_before_migrations(connection, max(pending))

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            # Verificación de arranque: repone cualquier tabla base faltante
            # sin tocar las que ya existen ni sus registros.
            connection.executescript(BASE_SCHEMA)
            connection.commit()
            applied = {
                int(row["version"])
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                ).fetchall()
            }

            for version, name in MIGRATIONS:
                if version in applied:
                    continue
                try:
                    if version == 1:
                        # BASE_SCHEMA ya fue verificado arriba. Registrar esta
                        # versión también adopta instalaciones anteriores.
                        pass
                    else:
                        connection.execute("BEGIN IMMEDIATE")
                        if version == 2:
                            self._add_reported_total_type(connection)
                        elif version == 3:
                            self._create_audit_history(connection)
                        elif version == 4:
                            self._prevent_duplicate_receipt_links(connection)
                        elif version == 5:
                            self._create_backtest_history(connection)
                    connection.execute(
                        """
                        INSERT INTO schema_migrations(version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (version, name, datetime.now(timezone.utc).isoformat()),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

            # También repone la tabla de auditorías si fue retirada
            # accidentalmente después de aplicar la migración 3.
            self._create_audit_history(connection)
            self._prevent_duplicate_receipt_links(connection)
            self._create_backtest_history(connection)
            connection.commit()

    def _backup_before_migrations(
        self, connection: sqlite3.Connection, target_version: int
    ) -> Path:
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        backup_path = backup_dir / (
            f"{self.path.stem}-before-v{target_version}-{timestamp}{self.path.suffix}"
        )
        destination = sqlite3.connect(backup_path)
        try:
            connection.backup(destination)
        finally:
            destination.close()
        return backup_path

    @staticmethod
    def _column_names(
        connection: sqlite3.Connection, table: str
    ) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _add_reported_total_type(self, connection: sqlite3.Connection) -> None:
        if "reported_total_type" not in self._column_names(connection, "trades"):
            connection.execute(
                """
                ALTER TABLE trades ADD COLUMN reported_total_type TEXT
                NOT NULL DEFAULT 'GROSS'
                CHECK (reported_total_type IN ('GROSS', 'SETTLEMENT'))
                """
            )

    @staticmethod
    def _create_audit_history(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL CHECK (status IN ('PASS', 'WARNING', 'ERROR')),
                passed INTEGER NOT NULL,
                warnings INTEGER NOT NULL,
                errors INTEGER NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_runs_time
            ON audit_runs(created_at DESC)
            """
        )

    @staticmethod
    def _prevent_duplicate_receipt_links(connection: sqlite3.Connection) -> None:
        """Impide nuevas operaciones ligadas al mismo comprobante.

        Se usan triggers en vez de reconstruir la tabla o borrar posibles datos
        históricos duplicados. Así la migración es segura para instalaciones ya
        existentes y también protege inserciones concurrentes.
        """

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(trades)").fetchall()
        }
        if "receipt_id" not in columns:
            return

        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS prevent_duplicate_trade_receipt_insert
            BEFORE INSERT ON trades
            WHEN NEW.receipt_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM trades WHERE receipt_id = NEW.receipt_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'receipt_already_linked');
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS prevent_duplicate_trade_receipt_update
            BEFORE UPDATE OF receipt_id ON trades
            WHEN NEW.receipt_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM trades
                WHERE receipt_id = NEW.receipt_id AND id <> OLD.id
            )
            BEGIN
                SELECT RAISE(ABORT, 'receipt_already_linked');
            END
            """
        )

    @staticmethod
    def _create_backtest_history(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine_version TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('APPROVED', 'REJECTED')),
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backtest_runs_time
            ON backtest_runs(created_at DESC, id DESC)
            """
        )

    def schema_version(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
        return int(row["version"])

    def integrity_check(self) -> str:
        with self.connect() as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "sin resultado"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
