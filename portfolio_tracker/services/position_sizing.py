"""Read-only portfolio context and conservative 2% risk sizing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import math

from portfolio_tracker.models import PriceQuote
from portfolio_tracker.services.portfolio import PortfolioCalculator


@dataclass(frozen=True, slots=True)
class PortfolioRiskContext:
    symbol: str
    average_price: float | None
    current_shares: float
    cash_available: float
    total_capital: float
    current_market_value: float
    concentration: float
    receipt_backed_trades: int


@dataclass(frozen=True, slots=True)
class PositionSize:
    shares: int
    risk_budget: float
    risk_per_share: float
    monetary_risk: float
    cash_required: float
    concentration_after: float
    limiting_factor: str


def portfolio_risk_context(repository, symbol, current_price):
    symbol = str(symbol).strip().upper()
    trades = repository.list_trades(ascending=True)
    symbols = sorted({str(row["symbol"]).upper() for row in trades})
    prices = repository.latest_price_quotes(symbols)
    prices[symbol] = PriceQuote(
        symbol=symbol,
        price_usd=Decimal(str(current_price)),
        observed_at=datetime.now(timezone.utc),
        provider="Motor cuantitativo · último cierre válido",
    )
    summary = PortfolioCalculator().summarize(
        trades=trades,
        cash_movements=repository.list_cash_movements(),
        prices=prices,
    )
    position = next((item for item in summary.positions if item.symbol == symbol), None)
    equity = max(0.0, float(summary.equity_usd))
    market_value = 0.0 if position is None else float(position.market_value_usd)
    return PortfolioRiskContext(
        symbol=symbol,
        average_price=None if position is None else float(position.average_cost_usd),
        current_shares=0.0 if position is None else float(position.quantity),
        cash_available=max(0.0, float(summary.cash_usd)),
        total_capital=equity,
        current_market_value=market_value,
        concentration=market_value / equity if equity > 0 else 0.0,
        receipt_backed_trades=sum(
            row.get("receipt_id") is not None and str(row["symbol"]).upper() == symbol
            for row in trades
        ),
    )


def calculate_position_size(context, entry_price, stop_loss, *, risk_fraction=0.02,
                            maximum_concentration=0.30):
    entry, stop = float(entry_price), float(stop_loss)
    if not all(math.isfinite(value) and value > 0 for value in (entry, stop)) or entry <= stop:
        raise ValueError("Entrada/stop LONG inválidos.")
    capital = max(0.0, float(context.total_capital))
    risk_budget = capital * max(0.0, min(float(risk_fraction), 0.02))
    per_share = entry - stop
    risk_cap = math.floor(risk_budget / per_share) if per_share > 0 else 0
    cash_cap = math.floor(max(0.0, context.cash_available) / entry)
    headroom = max(0.0, maximum_concentration * capital - context.current_market_value)
    concentration_cap = math.floor(headroom / entry)
    caps = {"riesgo 2%": risk_cap, "efectivo": cash_cap, "concentración 30%": concentration_cap}
    limiting, shares = min(caps.items(), key=lambda item: item[1])
    shares = max(0, int(shares))
    market_value_after = context.current_market_value + shares * entry
    return PositionSize(
        shares=shares,
        risk_budget=risk_budget,
        risk_per_share=per_share,
        monetary_risk=shares * per_share,
        cash_required=shares * entry,
        concentration_after=market_value_after / capital if capital > 0 else 0.0,
        limiting_factor=limiting,
    )
