from datetime import datetime, timezone
from decimal import Decimal

from portfolio_tracker.models import PriceQuote
from portfolio_tracker.services.portfolio import PortfolioCalculator


def test_fifo_cash_and_performance() -> None:
    movements = [
        {
            "kind": "INITIAL",
            "usd_amount": Decimal("1000.00"),
        }
    ]
    trades = [
        {
            "id": 1,
            "symbol": "ABC",
            "side": "BUY",
            "quantity": Decimal("10"),
            "price_usd": Decimal("10"),
            "gross_usd": Decimal("100"),
            "commission_usd": Decimal("1"),
            "cash_delta_usd": Decimal("-101"),
            "executed_at": "2026-01-01T12:00:00+00:00",
        },
        {
            "id": 2,
            "symbol": "ABC",
            "side": "SELL",
            "quantity": Decimal("4"),
            "price_usd": Decimal("15"),
            "gross_usd": Decimal("60"),
            "commission_usd": Decimal("1"),
            "cash_delta_usd": Decimal("59"),
            "executed_at": "2026-02-01T12:00:00+00:00",
        },
    ]
    quote = PriceQuote(
        symbol="ABC",
        price_usd=Decimal("12"),
        observed_at=datetime.now(timezone.utc),
        provider="test",
    )

    summary = PortfolioCalculator().summarize(
        trades=trades, cash_movements=movements, prices={"ABC": quote}
    )

    assert summary.cash_usd == Decimal("958.00")
    assert summary.holdings_value_usd == Decimal("72.00")
    assert summary.equity_usd == Decimal("1030.00")
    assert summary.realized_pnl_usd == Decimal("18.60")
    assert summary.unrealized_pnl_usd == Decimal("11.40")
    assert summary.total_pnl_usd == Decimal("30.00")
    assert summary.commissions_usd == Decimal("2.00")
    assert summary.total_return_pct == Decimal("3.00")
    assert summary.positions[0].quantity == Decimal("6")


def test_fifo_orders_by_instant_not_by_iso_string_with_mixed_offsets():
    """10:00-06:00 (16:00Z) happens after 15:00+00:00 even though it sorts first as text."""
    from decimal import Decimal
    from portfolio_tracker.services.portfolio import PortfolioCalculator

    def trade(id_, side, qty, price, at):
        gross = Decimal(qty) * Decimal(price)
        delta = -gross if side == "BUY" else gross
        return {"id": id_, "symbol": "ABC", "side": side, "quantity": qty,
                "price_usd": price, "gross_usd": str(gross), "commission_usd": "0",
                "cash_delta_usd": str(delta), "executed_at": at}

    trades = [
        trade(1, "BUY", "1", "10", "2026-10-01T15:00:00+00:00"),   # 15:00Z
        trade(2, "BUY", "1", "20", "2026-10-01T10:00:00-06:00"),   # 16:00Z, sorts first as text
        trade(3, "SELL", "1", "30", "2026-10-01T17:00:00+00:00"),  # 17:00Z
    ]
    summary = PortfolioCalculator().summarize(
        trades=trades, cash_movements=[{"kind": "DEPOSIT", "usd_amount": "100"}], prices={})
    # FIFO sells the $10 lot: realized 30-10 = 20 and the $20 lot stays open.
    assert summary.realized_pnl_usd == Decimal("20.00")
    assert summary.positions[0].average_cost_usd == Decimal("20.00")
    assert not summary.warnings or all("excede" not in w for w in summary.warnings)
