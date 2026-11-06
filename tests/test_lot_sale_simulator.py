"""La simulación por lote es informativa y no altera el FIFO contable."""

from copy import deepcopy
from decimal import Decimal

import pytest

from portfolio_tracker.services.lot_sale_simulator import open_buy_lots, simulate_lot_sale


def _trade(
    trade_id: int, when: str, side: str, quantity: str,
    gross: str, commission: str,
) -> dict:
    return {
        "id": trade_id, "executed_at": when, "symbol": "SMCI", "side": side,
        "quantity": Decimal(quantity),
        "price_usd": Decimal(gross) / Decimal(quantity),
        "gross_usd": Decimal(gross), "commission_usd": Decimal(commission),
    }


def test_recent_five_shares_profit_differs_from_real_fifo_loss() -> None:
    trades = [
        _trade(17, "2026-10-08T14:00:00+00:00", "BUY", "22", "952.10", "2.38"),
        _trade(18, "2026-10-16T15:39:00+00:00", "BUY", "5", "162.80", "0.40"),
    ]
    original = deepcopy(trades)

    result = simulate_lot_sale(
        trades, buy_trade_id=18, quantity=Decimal("5"),
        sale_price_usd=Decimal("32.88"),
    )

    assert result.gross_proceeds_usd == Decimal("164.40")
    assert result.sale_commission_usd == Decimal("0.41")
    assert result.net_proceeds_usd == Decimal("163.99")
    assert result.selected_lot_cost_usd == Decimal("163.20")
    assert result.selected_lot_gain_usd == Decimal("0.79")
    assert result.fifo_cost_usd == Decimal("216.93")
    assert result.fifo_gain_usd == Decimal("-52.94")
    assert trades == original


def test_open_lots_reflect_prior_fifo_sales_but_do_not_mutate_trades() -> None:
    trades = [
        _trade(1, "2026-01-01T16:00:00+00:00", "BUY", "3", "30", "0.30"),
        _trade(2, "2026-01-02T16:00:00+00:00", "BUY", "5", "50", "0.50"),
        _trade(3, "2026-01-03T16:00:00+00:00", "SELL", "4", "48", "0.12"),
    ]
    original = deepcopy(trades)

    lots = open_buy_lots(trades)

    assert [(lot.trade_id, lot.remaining_quantity) for lot in lots] == [
        (2, Decimal("4")),
    ]
    assert trades == original
    with pytest.raises(ValueError, match="no tiene acciones disponibles"):
        simulate_lot_sale(
            trades, buy_trade_id=1, quantity=Decimal("1"),
            sale_price_usd=Decimal("12"),
        )
    with pytest.raises(ValueError, match="no superar el lote"):
        simulate_lot_sale(
            trades, buy_trade_id=2, quantity=Decimal("5"),
            sale_price_usd=Decimal("12"),
        )


def test_simulator_rejects_incomplete_cost_history() -> None:
    trades = [
        _trade(1, "2026-01-01T16:00:00+00:00", "BUY", "1", "10", "0.10"),
        _trade(2, "2026-01-02T16:00:00+00:00", "SELL", "2", "20", "0.20"),
    ]
    with pytest.raises(ValueError, match="sin compras suficientes"):
        open_buy_lots(trades)
