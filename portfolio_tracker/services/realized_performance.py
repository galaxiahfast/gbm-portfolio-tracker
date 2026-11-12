"""Vista de ganancias realizadas por venta; no modifica el libro contable.

Usa el mismo criterio FIFO que PortfolioCalculator: cada lote compra incorpora
su comisión al costo unitario y cada venta descuenta su propia comisión.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, tzinfo
from decimal import Decimal
from typing import Any

from ..models import TradeSide, chronological_key, money


@dataclass(frozen=True, slots=True)
class RealizedSale:
    trade_id: int
    executed_at: datetime
    symbol: str
    quantity: Decimal
    allocated_cost_usd: Decimal | None
    net_proceeds_usd: Decimal
    gain_usd: Decimal | None


def _trade_time(value: str | datetime, timezone: tzinfo) -> datetime:
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone)
    return stamp.astimezone(timezone)


def realized_sales(
    trades: list[dict[str, Any]], *, timezone: tzinfo,
) -> tuple[RealizedSale, ...]:
    """Proyecta resultado de cada venta completa sin reescribir transacciones.

    Si falta el costo de alguna acción vendida, su ganancia se marca N/D en vez
    de atribuirle costo cero. Las compras no generan ganancia realizada.
    """
    lots: dict[str, deque[list[Decimal]]] = defaultdict(deque)
    sales: list[RealizedSale] = []
    for trade in sorted(trades, key=chronological_key):
        symbol = str(trade["symbol"]).upper()
        quantity = Decimal(trade["quantity"])
        gross = Decimal(trade["gross_usd"])
        commission = Decimal(trade["commission_usd"])
        if quantity <= 0:
            raise ValueError(f"Operación {trade['id']}: títulos no positivos.")

        if trade["side"] == TradeSide.BUY.value:
            lots[symbol].append([quantity, (gross + commission) / quantity, int(trade["id"])])
            continue
        if trade["side"] != TradeSide.SELL.value:
            raise ValueError(f"Operación {trade['id']}: lado desconocido.")

        remaining = quantity
        allocated_cost = Decimal("0")
        selected_buy_id = trade.get("allocated_buy_trade_id")
        if selected_buy_id is not None:
            chosen = next(
                (lot for lot in lots[symbol] if int(lot[2]) == int(selected_buy_id)),
                None,
            )
            if chosen is None or chosen[0] < remaining:
                raise ValueError(
                    f"Operación {trade['id']}: el lote elegido no tiene títulos suficientes."
                )
            allocated_cost = remaining * chosen[1]
            chosen[0] -= remaining
            if chosen[0] == 0:
                lots[symbol].remove(chosen)
            remaining = Decimal("0")
        while remaining > 0 and lots[symbol]:
            lot = lots[symbol][0]
            used = min(remaining, lot[0])
            allocated_cost += used * lot[1]
            lot[0] -= used
            remaining -= used
            if lot[0] == 0:
                lots[symbol].popleft()
        complete = remaining == 0
        net_proceeds = gross - commission
        sales.append(RealizedSale(
            trade_id=int(trade["id"]),
            executed_at=_trade_time(trade["executed_at"], timezone),
            symbol=symbol,
            quantity=quantity,
            allocated_cost_usd=allocated_cost if complete else None,
            net_proceeds_usd=net_proceeds,
            gain_usd=net_proceeds - allocated_cost if complete else None,
        ))
    return tuple(sales)


def realized_gain_total(
    sales: tuple[RealizedSale, ...], *, year: int | None = None,
    month: int | None = None,
) -> Decimal | None:
    """Total neto de ventas; el filtro mensual usa la fecha de ejecución local."""
    selected = [
        sale for sale in sales
        if (year is None or sale.executed_at.year == year)
        and (month is None or sale.executed_at.month == month)
    ]
    if any(sale.gain_usd is None for sale in selected):
        return None
    return money(sum((sale.gain_usd for sale in selected), Decimal("0")))
