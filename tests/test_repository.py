import sqlite3
from datetime import datetime, timezone
from decimal import Decimal

from portfolio_tracker.db import Database
from portfolio_tracker.models import CashMovementKind, TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository


def test_initial_capital_is_seeded_only_once(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    repository.ensure_initial_capital()
    repository.ensure_initial_capital()

    assert repository.cash_balance_usd() == Decimal("921.05")
    assert len(repository.list_cash_movements()) == 1


def test_mxn_deposit_and_usd_withdrawal(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    repository.ensure_initial_capital()
    now = datetime.now(timezone.utc)

    repository.add_cash_movement(
        kind=CashMovementKind.DEPOSIT,
        original_amount=Decimal("1800"),
        original_currency="MXN",
        usd_amount=Decimal("100"),
        fx_rate=Decimal("18"),
        occurred_at=now,
    )
    repository.add_cash_movement(
        kind=CashMovementKind.WITHDRAWAL,
        original_amount=Decimal("360"),
        original_currency="MXN",
        usd_amount=Decimal("20"),
        fx_rate=Decimal("18"),
        occurred_at=now,
    )

    assert repository.cash_balance_usd() == Decimal("1001.05")


def test_migrations_adopt_existing_database_and_recreate_missing_table(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            executed_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    assert database.schema_version() == 5
    with database.connect() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(trades)")
        }
        assert "reported_total_type" in columns
        connection.execute("DROP TABLE audit_runs")
        connection.commit()

    database.initialize()
    with database.connect() as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'audit_runs'"
        ).fetchone()
    assert table is not None


def test_same_receipt_cannot_create_two_trades(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    receipt_id = repository.upsert_receipt(
        sha256="a" * 64,
        original_filename="orden.png",
        mime_type="image/png",
        original_path="data/receipts/a.png",
        thumbnail_path="data/receipts/a-thumb.jpg",
        width=100,
        height=200,
        byte_size=123,
    )
    trade = TradeDraft(
        symbol="SMCI",
        side=TradeSide.BUY,
        quantity=Decimal("1"),
        price_usd=Decimal("40"),
        commission_usd=Decimal("0.10"),
        executed_at=datetime.now(timezone.utc),
    )
    repository.add_trade(trade, receipt_id)

    try:
        repository.add_trade(trade, receipt_id)
    except ValueError as exc:
        assert "ya está vinculado" in str(exc)
    else:
        raise AssertionError("Se permitió importar dos veces el mismo comprobante")


def test_backtest_payload_is_persisted_with_auditable_hash(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    run_id = repository.record_backtest_run(
        engine_version="test-v1",
        symbols_json='["SMCI"]',
        parameters_json='{"training_fraction":0.7}',
        dataset_sha256="d" * 64,
        payload_json='{"result":"strict-oos"}',
        status="REJECTED",
    )

    history = repository.list_backtest_runs()
    valid, invalid = repository.verify_backtest_runs()
    assert history[0]["id"] == run_id
    assert history[0]["status"] == "REJECTED"
    assert valid == 1
    assert invalid == ()
