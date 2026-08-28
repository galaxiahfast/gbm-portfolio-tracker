"""Contabilidad de efectivo, lotes FIFO y rendimiento del portafolio."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..models import (
    CashMovementKind,
    PortfolioSummary,
    Position,
    PriceQuote,
    TradeSide,
    money,
)


@dataclass(slots=True)
class _Lot:
    quantity: Decimal
    unit_cost_usd: Decimal


class PortfolioCalculator:
    """Motor puro y testeable; no conoce Streamlit, red ni SQLite."""

    def summarize(
        self,
        *,
        trades: list[dict[str, Any]],
        cash_movements: list[dict[str, Any]],
        prices: dict[str, PriceQuote],
    ) -> PortfolioSummary:
        cash = Decimal("0")
        net_contributions = Decimal("0")
        for movement in cash_movements:
            amount = Decimal(movement["usd_amount"])
            if movement["kind"] == CashMovementKind.WITHDRAWAL.value:
                cash -= amount
                net_contributions -= amount
            else:
                cash += amount
                net_contributions += amount

        lots: dict[str, deque[_Lot]] = defaultdict(deque)
        realized_by_symbol: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        last_trade_price: dict[str, Decimal] = {}
        warnings: list[str] = []
        commissions = Decimal("0")

        for trade in sorted(trades, key=lambda item: (item["executed_at"], item["id"])):
            symbol = str(trade["symbol"]).upper()
            quantity = Decimal(trade["quantity"])
            gross = Decimal(trade["gross_usd"])
            commission = Decimal(trade["commission_usd"])
            commissions += commission
            cash += Decimal(trade["cash_delta_usd"])
            last_trade_price[symbol] = Decimal(trade["price_usd"])

            if trade["side"] == TradeSide.BUY.value:
                unit_cost = (gross + commission) / quantity
                lots[symbol].append(_Lot(quantity, unit_cost))
                continue

            remaining = quantity
            removed_cost = Decimal("0")
            while remaining > 0 and lots[symbol]:
                lot = lots[symbol][0]
                used = min(remaining, lot.quantity)
                removed_cost += used * lot.unit_cost_usd
                lot.quantity -= used
                remaining -= used
                if lot.quantity == 0:
                    lots[symbol].popleft()
            if remaining > 0:
                warnings.append(
                    f"{symbol}: una venta excede los titulos comprados por {remaining}."
                )
                # No se inventa costo para la parte sin historial.
            realized_by_symbol[symbol] += gross - commission - removed_cost

        positions: list[Position] = []
        unrealized_total = Decimal("0")
        holdings_value = Decimal("0")
        for symbol in sorted(lots):
            quantity = sum((lot.quantity for lot in lots[symbol]), Decimal("0"))
            if quantity <= 0:
                continue
            cost_basis = sum(
                (lot.quantity * lot.unit_cost_usd for lot in lots[symbol]),
                Decimal("0"),
            )
            quote = prices.get(symbol)
            if quote:
                current_price = quote.price_usd
            else:
                current_price = last_trade_price.get(symbol, Decimal("0"))
                warnings.append(
                    f"{symbol}: valuado al ultimo precio registrado; cotizacion no disponible."
                )
            market_value = quantity * current_price
            unrealized = market_value - cost_basis
            holdings_value += market_value
            unrealized_total += unrealized
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=quantity,
                    cost_basis_usd=money(cost_basis),
                    average_cost_usd=money(cost_basis / quantity),
                    current_price_usd=money(current_price),
                    market_value_usd=money(market_value),
                    realized_pnl_usd=money(realized_by_symbol[symbol]),
                    unrealized_pnl_usd=money(unrealized),
                )
            )

        equity = cash + holdings_value
        realized_total = sum(realized_by_symbol.values(), Decimal("0"))
        total_return_pct = Decimal("0")
        if net_contributions != 0:
            total_return_pct = (
                (equity - net_contributions) / abs(net_contributions) * Decimal("100")
            ).quantize(Decimal("0.01"))

        return PortfolioSummary(
            cash_usd=money(cash),
            holdings_value_usd=money(holdings_value),
            equity_usd=money(equity),
            net_contributions_usd=money(net_contributions),
            realized_pnl_usd=money(realized_total),
            unrealized_pnl_usd=money(unrealized_total),
            commissions_usd=money(commissions),
            total_return_pct=total_return_pct,
            positions=positions,
            warnings=warnings,
        )

    def quantities(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]:
        result: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        for trade in trades:
            sign = Decimal("1") if trade["side"] == TradeSide.BUY.value else Decimal("-1")
            result[str(trade["symbol"]).upper()] += sign * Decimal(trade["quantity"])
        return dict(result)

