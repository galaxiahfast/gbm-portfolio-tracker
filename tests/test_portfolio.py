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
