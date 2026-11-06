"""Simulación de venta por lote, separada de la contabilidad FIFO real.

No persiste operaciones ni altera los lotes recibidos. La comparación FIFO
reproduce el orden de consumo del cálculo contable del portafolio.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..models import TradeSide, money


@dataclass(frozen=True, slots=True)
class OpenBuyLot:
    trade_id: int
    symbol: str
    executed_at: str
    remaining_quantity: Decimal
    unit_cost_usd: Decimal
    purchase_price_usd: Decimal


@dataclass(frozen=True, slots=True)
class LotSaleSimulation:
    symbol: str
    quantity: Decimal
    sale_price_usd: Decimal
    gross_proceeds_usd: Decimal
    sale_commission_usd: Decimal
    net_proceeds_usd: Decimal
    selected_lot_cost_usd: Decimal
    selected_lot_gain_usd: Decimal
    fifo_cost_usd: Decimal
    fifo_gain_usd: Decimal


def open_buy_lots(trades: list[dict[str, Any]]) -> tuple[OpenBuyLot, ...]:
    """Reconstruye los lotes abiertos después de las ventas FIFO registradas."""
    lots: dict[str, deque[OpenBuyLot]] = defaultdict(deque)
    for trade in sorted(trades, key=lambda row: (row["executed_at"], row["id"])):
        symbol = str(trade["symbol"]).upper()
        quantity = Decimal(str(trade["quantity"]))
        if quantity <= 0:
            raise ValueError(f"Operación {trade['id']}: títulos no positivos.")
        if trade["side"] == TradeSide.BUY.value:
            gross = Decimal(str(trade["gross_usd"]))
            commission = Decimal(str(trade["commission_usd"]))
            lots[symbol].append(OpenBuyLot(
                trade_id=int(trade["id"]), symbol=symbol,
                executed_at=str(trade["executed_at"]),
                remaining_quantity=quantity,
                unit_cost_usd=(gross + commission) / quantity,
                purchase_price_usd=Decimal(str(trade["price_usd"])),
            ))
            continue
        if trade["side"] != TradeSide.SELL.value:
            raise ValueError(f"Operación {trade['id']}: lado desconocido.")
        remaining = quantity
        selected_buy_id = trade.get("allocated_buy_trade_id")
        if selected_buy_id is not None:
            chosen = next(
                (lot for lot in lots[symbol] if lot.trade_id == int(selected_buy_id)),
                None,
            )
            if chosen is None or chosen.remaining_quantity < remaining:
                raise ValueError(
                    f"Operación {trade['id']}: el lote elegido no tiene títulos suficientes."
                )
            lots[symbol].remove(chosen)
            left = chosen.remaining_quantity - remaining
            if left > 0:
                replacement = OpenBuyLot(
                    trade_id=chosen.trade_id, symbol=chosen.symbol,
                    executed_at=chosen.executed_at, remaining_quantity=left,
                    unit_cost_usd=chosen.unit_cost_usd,
                    purchase_price_usd=chosen.purchase_price_usd,
                )
                # Conservar el orden original para futuras ventas FIFO.
                position = next(
                    (index for index, lot in enumerate(lots[symbol])
                     if (lot.executed_at, lot.trade_id) > (chosen.executed_at, chosen.trade_id)),
                    len(lots[symbol]),
                )
                lots[symbol].insert(position, replacement)
            continue
        while remaining > 0 and lots[symbol]:
            lot = lots[symbol][0]
            used = min(remaining, lot.remaining_quantity)
            new_quantity = lot.remaining_quantity - used
            remaining -= used
            lots[symbol].popleft()
            if new_quantity > 0:
                lots[symbol].appendleft(OpenBuyLot(
                    trade_id=lot.trade_id, symbol=lot.symbol,
                    executed_at=lot.executed_at,
                    remaining_quantity=new_quantity,
                    unit_cost_usd=lot.unit_cost_usd,
                    purchase_price_usd=lot.purchase_price_usd,
                ))
        if remaining > 0:
            raise ValueError(
                f"{symbol}: el historial contiene una venta sin compras suficientes; "
                "no se puede simular un costo confiable."
            )
    return tuple(lot for symbol_lots in lots.values() for lot in symbol_lots)


def simulate_lot_sale(
    trades: list[dict[str, Any]], *, buy_trade_id: int,
    quantity: Decimal, sale_price_usd: Decimal,
    commission_rate_pct: Decimal = Decimal("0.25"),
) -> LotSaleSimulation:
    """Compara el lote elegido con FIFO sin registrar una venta."""
    available = open_buy_lots(trades)
    selected = next((lot for lot in available if lot.trade_id == buy_trade_id), None)
    if selected is None:
        raise ValueError("La compra seleccionada ya no tiene acciones disponibles.")
    quantity = Decimal(str(quantity))
    price = Decimal(str(sale_price_usd))
    rate = Decimal(str(commission_rate_pct))
    if not quantity.is_finite() or quantity <= 0 or quantity > selected.remaining_quantity:
        raise ValueError("La cantidad debe ser positiva y no superar el lote abierto.")
    if not price.is_finite() or price <= 0:
        raise ValueError("El precio de venta debe ser positivo.")
    if not rate.is_finite() or rate < 0:
        raise ValueError("La comisión no puede ser negativa.")

    gross = money(quantity * price)
    commission = money(gross * rate / Decimal("100"))
    net = money(gross - commission)
    selected_cost = money(quantity * selected.unit_cost_usd)
    remaining = quantity
    fifo_cost_raw = Decimal("0")
    for lot in available:
        if lot.symbol != selected.symbol or remaining <= 0:
            continue
        used = min(remaining, lot.remaining_quantity)
        fifo_cost_raw += used * lot.unit_cost_usd
        remaining -= used
    if remaining > 0:
        raise ValueError("No hay acciones suficientes para simular FIFO.")
    fifo_cost = money(fifo_cost_raw)
    return LotSaleSimulation(
        symbol=selected.symbol, quantity=quantity, sale_price_usd=price,
        gross_proceeds_usd=gross, sale_commission_usd=commission,
        net_proceeds_usd=net, selected_lot_cost_usd=selected_cost,
        selected_lot_gain_usd=money(net - selected_cost),
        fifo_cost_usd=fifo_cost, fifo_gain_usd=money(net - fifo_cost),
    )
