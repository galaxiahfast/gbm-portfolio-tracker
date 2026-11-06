"""Desglose de ganancias de la interfaz, reconciliado con el FIFO contable."""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from portfolio_tracker.services.portfolio import PortfolioCalculator
from portfolio_tracker.services.realized_performance import (
    realized_gain_total,
    realized_sales,
)


LOCAL = ZoneInfo("America/Mexico_City")


def _trade(
    trade_id: int, when: str, side: str, quantity: str, gross: str, commission: str,
) -> dict:
    gross_usd = Decimal(gross)
    commission_usd = Decimal(commission)
    return {
        "id": trade_id,
        "executed_at": when,
        "symbol": "SMCI",
        "side": side,
        "quantity": Decimal(quantity),
        "price_usd": gross_usd / Decimal(quantity),
        "gross_usd": gross_usd,
        "commission_usd": commission_usd,
        "cash_delta_usd": (
            -gross_usd - commission_usd if side == "BUY"
            else gross_usd - commission_usd
        ),
    }


def test_three_buys_then_full_sale_reports_one_net_fifo_gain() -> None:
    trades = [
        _trade(1, "2026-01-15T16:00:00+00:00", "BUY", "3", "30", "0.30"),
        _trade(2, "2026-01-20T16:00:00+00:00", "BUY", "2", "24", "0.24"),
        _trade(3, "2026-02-01T16:00:00+00:00", "BUY", "1", "8", "0.08"),
        _trade(4, "2026-02-10T16:00:00+00:00", "SELL", "6", "90", "0.90"),
    ]

    sales = realized_sales(trades, timezone=LOCAL)

    assert len(sales) == 1
    assert sales[0].allocated_cost_usd == Decimal("62.62")
    assert sales[0].net_proceeds_usd == Decimal("89.10")
    assert sales[0].gain_usd == Decimal("26.48")
    assert realized_gain_total(sales, year=2026, month=2) == Decimal("26.48")
    assert realized_gain_total(sales, year=2026, month=3) == Decimal("0.00")
    summary = PortfolioCalculator().summarize(
        trades=trades, cash_movements=[], prices={},
    )
    assert realized_gain_total(sales) == summary.realized_pnl_usd


def test_partial_sales_crossing_month_boundary_reset_only_month_gain() -> None:
    trades = [
        _trade(1, "2026-01-02T16:00:00+00:00", "BUY", "3", "30", "0.30"),
        _trade(2, "2026-01-30T16:00:00+00:00", "SELL", "1", "12", "0.12"),
        _trade(3, "2026-02-02T16:00:00+00:00", "SELL", "2", "28", "0.28"),
    ]

    sales = realized_sales(trades, timezone=LOCAL)

    assert [sale.gain_usd for sale in sales] == [Decimal("1.78"), Decimal("7.52")]
    assert realized_gain_total(sales, year=2026, month=1) == Decimal("1.78")
    assert realized_gain_total(sales, year=2026, month=2) == Decimal("7.52")
    assert realized_gain_total(sales) == Decimal("9.30")
    summary = PortfolioCalculator().summarize(
        trades=trades, cash_movements=[], prices={},
    )
    assert realized_gain_total(sales) == summary.realized_pnl_usd


def test_month_uses_local_execution_date_and_unknown_cost_is_not_gain() -> None:
    trades = [
        _trade(1, "2026-03-01T02:00:00+00:00", "BUY", "1", "10", "0.10"),
        _trade(2, "2026-03-01T03:00:00+00:00", "SELL", "1", "12", "0.12"),
        _trade(3, "2026-03-02T16:00:00+00:00", "SELL", "1", "14", "0.14"),
    ]

    sales = realized_sales(trades, timezone=LOCAL)

    assert sales[0].executed_at.date() == datetime(2026, 2, 28).date()
    assert realized_gain_total(sales, year=2026, month=2) == Decimal("1.78")
    assert sales[1].gain_usd is None
    assert realized_gain_total(sales, year=2026, month=3) is None
    assert realized_gain_total(sales) is None
