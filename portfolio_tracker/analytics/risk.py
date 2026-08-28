"""Primer modulo util: alertas de concentracion, no señales de compra/venta."""

from __future__ import annotations

from decimal import Decimal

from ..models import PortfolioSummary


def concentration_warnings(
    summary: PortfolioSummary, threshold_pct: Decimal = Decimal("30")
) -> list[str]:
    """Aplica disciplina de riesgo y diversificacion de forma transparente."""

    if summary.holdings_value_usd <= 0:
        return []
    warnings: list[str] = []
    for position in summary.positions:
        weight = position.market_value_usd / summary.holdings_value_usd * Decimal("100")
        if weight > threshold_pct:
            warnings.append(
                f"{position.symbol} representa {weight.quantize(Decimal('0.1'))}% "
                f"de las posiciones (umbral {threshold_pct}%)."
            )
    return warnings

