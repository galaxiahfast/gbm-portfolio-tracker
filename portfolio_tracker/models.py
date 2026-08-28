"""Modelos de dominio sin dependencias de Streamlit o SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum


CENT = Decimal("0.01")
SHARE_PRECISION = Decimal("0.000001")


def money(value: Decimal | str | int | float) -> Decimal:
    """Normaliza un importe monetario a centavos con redondeo contable."""

    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def shares(value: Decimal | str | int | float) -> Decimal:
    """Conserva hasta seis decimales para soportar titulos fraccionarios."""

    return Decimal(str(value)).quantize(SHARE_PRECISION, rounding=ROUND_HALF_UP)


class TradeSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def spanish(self) -> str:
        return "Compra" if self is TradeSide.BUY else "Venta"


class ReportedTotalType(StrEnum):
    """Indica que representa el total visible en el comprobante.

    GBM normalmente muestra un total bruto en el encabezado, mientras que
    otros comprobantes pueden mostrar el cargo/abono final con comision.
    Conservar esta distincion evita sumar la comision dos veces.
    """

    GROSS = "GROSS"
    SETTLEMENT = "SETTLEMENT"


class CashMovementKind(StrEnum):
    INITIAL = "INITIAL"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"


@dataclass(slots=True)
class FxQuote:
    rate: Decimal
    observed_at: datetime
    provider: str
    is_reference: bool = True


@dataclass(slots=True)
class PriceQuote:
    symbol: str
    price_usd: Decimal
    observed_at: datetime
    provider: str
    currency: str = "USD"


@dataclass(slots=True)
class TradeDraft:
    symbol: str
    side: TradeSide
    quantity: Decimal
    price_usd: Decimal
    commission_usd: Decimal
    executed_at: datetime
    product: str = "USA"
    order_type: str = "Limitada"
    commission_rate_pct: Decimal | None = None
    reported_total_usd: Decimal | None = None
    reported_total_type: ReportedTotalType = ReportedTotalType.GROSS
    fx_rate: Decimal | None = None
    notes: str = ""
    ocr_text: str = ""
    ocr_confidence: Decimal | None = None
    validation_status: str = "PENDING"

    @property
    def calculated_gross_usd(self) -> Decimal:
        return money(self.quantity * self.price_usd)

    @property
    def gross_usd(self) -> Decimal:
        # GBM puede mostrar el precio unitario redondeado. Si el comprobante
        # contiene el total ejecutado, ese total es la fuente contable primaria.
        if (
            self.reported_total_usd is not None
            and self.reported_total_type is ReportedTotalType.GROSS
        ):
            return money(self.reported_total_usd)
        return self.calculated_gross_usd

    @property
    def calculated_settlement_usd(self) -> Decimal:
        """Cargo final en compra o abono neto en venta."""

        if self.side is TradeSide.BUY:
            return money(self.gross_usd + self.commission_usd)
        return money(self.gross_usd - self.commission_usd)

    @property
    def settlement_usd(self) -> Decimal:
        if (
            self.reported_total_usd is not None
            and self.reported_total_type is ReportedTotalType.SETTLEMENT
        ):
            return money(self.reported_total_usd)
        return self.calculated_settlement_usd

    @property
    def cash_delta_usd(self) -> Decimal:
        sign = Decimal("-1") if self.side is TradeSide.BUY else Decimal("1")
        return money(sign * self.settlement_usd)


@dataclass(slots=True)
class ValidationReport:
    is_valid: bool
    status: str
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: Decimal
    cost_basis_usd: Decimal
    average_cost_usd: Decimal
    current_price_usd: Decimal
    market_value_usd: Decimal
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal


@dataclass(slots=True)
class PortfolioSummary:
    cash_usd: Decimal
    holdings_value_usd: Decimal
    equity_usd: Decimal
    net_contributions_usd: Decimal
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal
    commissions_usd: Decimal
    total_return_pct: Decimal
    positions: list[Position]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_pnl_usd(self) -> Decimal:
        """Resultado económico total: operaciones cerradas más posiciones abiertas."""

        return money(self.realized_pnl_usd + self.unrealized_pnl_usd)
