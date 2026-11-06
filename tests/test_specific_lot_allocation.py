"""Una venta de las cinco acciones recientes deja intacto el lote anterior."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from portfolio_tracker.db import Database
from portfolio_tracker.models import PriceQuote, TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.lot_sale_simulator import open_buy_lots
from portfolio_tracker.services.portfolio import PortfolioCalculator
from portfolio_tracker.services.realized_performance import realized_sales


def _recent_trades() -> list[dict]:
    return [
        {"id": 17, "executed_at": "2026-10-08T08:48:00-06:00", "symbol": "SMCI",
         "side": "BUY", "quantity": Decimal("22"), "price_usd": Decimal("43.28"),
         "gross_usd": Decimal("952.10"), "commission_usd": Decimal("2.38"),
         "cash_delta_usd": Decimal("-954.48")},
        {"id": 18, "executed_at": "2026-10-16T09:39:00-06:00", "symbol": "SMCI",
         "side": "BUY", "quantity": Decimal("5"), "price_usd": Decimal("32.56"),
         "gross_usd": Decimal("162.80"), "commission_usd": Decimal("0.40"),
         "cash_delta_usd": Decimal("-163.20")},
        {"id": 19, "executed_at": "2026-10-19T09:46:00-06:00", "symbol": "SMCI",
         "side": "SELL", "quantity": Decimal("5"), "price_usd": Decimal("33.00"),
         "gross_usd": Decimal("165.00"), "commission_usd": Decimal("0.41"),
         "cash_delta_usd": Decimal("164.59"), "allocated_buy_trade_id": 18},
    ]


def test_specific_allocation_gives_139_and_preserves_older_22() -> None:
    trades = _recent_trades()
    sale = realized_sales(trades, timezone=timezone.utc)[0]
    lots = open_buy_lots(trades)
    quote = PriceQuote("SMCI", Decimal("33.00"), datetime.now(timezone.utc), "test")
    summary = PortfolioCalculator().summarize(trades=trades, cash_movements=[], prices={"SMCI": quote})

    assert sale.trade_id == 19
    assert sale.net_proceeds_usd == Decimal("164.59")
    assert sale.allocated_cost_usd == Decimal("163.20")
    assert sale.gain_usd == Decimal("1.39")
    assert [(lot.trade_id, lot.remaining_quantity) for lot in lots] == [(17, Decimal("22"))]
    assert summary.positions[0].quantity == Decimal("22")
    assert summary.positions[0].cost_basis_usd == Decimal("954.48")
    assert summary.realized_pnl_usd == Decimal("1.39")
    assert summary.cash_usd == Decimal("-953.09")


def test_without_explicit_allocation_remains_fifo() -> None:
    trades = _recent_trades()
    trades[-1].pop("allocated_buy_trade_id")
    sale = realized_sales(trades, timezone=timezone.utc)[0]
    assert sale.gain_usd.quantize(Decimal("0.01")) == Decimal("-52.34")
    assert [(lot.trade_id, lot.remaining_quantity) for lot in open_buy_lots(trades)] == [
        (17, Decimal("17")), (18, Decimal("5")),
    ]


def test_repository_assignment_is_validated_durable_and_idempotent(tmp_path) -> None:
    repo = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repo.database.initialize()
    before = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("22"),
        price_usd=Decimal("43.28"), commission_usd=Decimal("2.38"),
        executed_at=datetime(2026, 10, 8, 14, 48, tzinfo=timezone.utc),
    ))
    recent = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("5"),
        price_usd=Decimal("32.56"), commission_usd=Decimal("0.40"),
        executed_at=datetime(2026, 10, 16, 15, 39, tzinfo=timezone.utc),
    ))
    sale = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.SELL, quantity=Decimal("5"),
        price_usd=Decimal("33.00"), commission_usd=Decimal("0.41"),
        executed_at=datetime(2026, 10, 19, 15, 46, tzinfo=timezone.utc),
    ))

    assert repo.assign_sale_to_buy_lot(
        sale_trade_id=sale, buy_trade_id=recent, reason="Comprobante de la venta del lote reciente",
    ) is True
    assert repo.assign_sale_to_buy_lot(
        sale_trade_id=sale, buy_trade_id=recent, reason="Repetición idempotente",
    ) is False
    trades = repo.list_trades(ascending=True)
    assert trades[-1]["allocated_buy_trade_id"] == recent
    assert realized_sales(trades, timezone=timezone.utc)[0].gain_usd == Decimal("1.39")
    assert [(lot.trade_id, lot.remaining_quantity) for lot in open_buy_lots(trades)] == [
        (before, Decimal("22")),
    ]
    with pytest.raises(ValueError, match="ya tiene otro lote"):
        repo.assign_sale_to_buy_lot(sale_trade_id=sale, buy_trade_id=before, reason="No")
    with repo.database.connect() as connection:
        row = connection.execute("SELECT reason FROM sale_lot_allocations WHERE sale_trade_id = ?", (sale,)).fetchone()
    assert row["reason"] == "Comprobante de la venta del lote reciente"


def test_assignment_rejects_lot_with_insufficient_shares(tmp_path) -> None:
    repo = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repo.database.initialize()
    first = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("5"),
        price_usd=Decimal("10"), commission_usd=Decimal("0"),
        executed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ))
    second = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.BUY, quantity=Decimal("1"),
        price_usd=Decimal("11"), commission_usd=Decimal("0"),
        executed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    ))
    sale = repo.add_trade(TradeDraft(
        symbol="SMCI", side=TradeSide.SELL, quantity=Decimal("5"),
        price_usd=Decimal("12"), commission_usd=Decimal("0"),
        executed_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    ))
    with pytest.raises(ValueError, match="no tiene títulos suficientes"):
        repo.assign_sale_to_buy_lot(sale_trade_id=sale, buy_trade_id=second, reason="Demasiadas acciones")
    assert repo.list_trades()[-1]["allocated_buy_trade_id"] is None
    assert first != second
