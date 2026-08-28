"""Doble chequeo aritmetico de comprobantes y operaciones manuales."""

from __future__ import annotations

import re
from decimal import Decimal

from ..models import ReportedTotalType, TradeDraft, TradeSide, ValidationReport, money


SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")


def validate_trade(trade: TradeDraft) -> ValidationReport:
    checks: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    symbol = trade.symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        errors.append("La emisora/ticker tiene un formato invalido.")
    if trade.quantity <= 0:
        errors.append("Los titulos deben ser mayores que cero.")
    if trade.price_usd <= 0:
        errors.append("El precio debe ser mayor que cero.")
    if trade.commission_usd < 0:
        errors.append("La comision no puede ser negativa.")

    calculated = trade.calculated_gross_usd
    if trade.reported_total_usd is not None:
        if trade.reported_total_type is ReportedTotalType.SETTLEMENT:
            expected = money(
                calculated + trade.commission_usd
                if trade.side is TradeSide.BUY
                else calculated - trade.commission_usd
            )
            total_label = "cargo final" if trade.side is TradeSide.BUY else "abono neto"
        else:
            expected = calculated
            total_label = "total bruto"
        difference = abs(trade.reported_total_usd - expected)
        # El precio mostrado por GBM suele estar redondeado a centavos; se
        # permite medio centavo por titulo, mas un centavo de holgura.
        rounding_tolerance = max(
            Decimal("0.02"), trade.quantity * Decimal("0.005") + Decimal("0.01")
        )
        if difference <= rounding_tolerance:
            checks.append(
                f"{total_label.capitalize()} conciliado: comprobante "
                f"${money(trade.reported_total_usd)}; esperado ${expected}."
            )
        else:
            errors.append(
                f"El {total_label} difiere ${money(difference)} del cálculo "
                "títulos × precio y comisión."
            )
    else:
        warnings.append("No se detecto total ejecutado; se usara titulos x precio.")

    if trade.commission_rate_pct is not None:
        commission_base = trade.gross_usd
        expected_commission = money(
            commission_base * trade.commission_rate_pct / Decimal("100")
        )
        difference = abs(expected_commission - trade.commission_usd)
        if difference <= Decimal("0.02"):
            checks.append(
                f"Comision conciliada: {trade.commission_rate_pct}% de "
                f"${commission_base} = ${expected_commission}."
            )
        else:
            warnings.append(
                f"La comision capturada (${money(trade.commission_usd)}) no coincide "
                f"con la tasa indicada (${expected_commission})."
            )
    elif trade.commission_usd == 0:
        warnings.append("La operacion no incluye comision ni tasa de comision.")

    is_valid = not errors
    if errors:
        status = "REJECTED"
    elif warnings:
        status = "REVIEW"
    else:
        status = "VERIFIED"
    return ValidationReport(is_valid, status, checks, warnings, errors)
