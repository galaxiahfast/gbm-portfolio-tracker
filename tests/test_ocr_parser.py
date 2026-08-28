from decimal import Decimal

from portfolio_tracker.models import ReportedTotalType, TradeDraft, TradeSide
from portfolio_tracker.services.ocr import parse_gbm_text
from portfolio_tracker.services.validation import validate_trade


SAMPLE = """
Orden completada
$953.82 USD
25 ago 2026 – 08:41 h
GBM
Producto USA
Emisora SMCI
Tipo de Operación Venta
Tipo de Orden Limitada
Títulos 25
Precio por título $38.15 USD
Vigencia 25 ago 2026
Comisión 0.25% • $2.38 USD
"""


def test_parse_attached_gbm_format() -> None:
    extraction = parse_gbm_text(SAMPLE)
    assert extraction.fields["symbol"] == "SMCI"
    assert extraction.fields["product"] == "USA"
    assert extraction.fields["side"] == "Venta"
    assert extraction.fields["order_type"] == "Limitada"
    assert extraction.fields["quantity"] == Decimal("25")
    assert extraction.fields["price_usd"] == Decimal("38.15")
    assert extraction.fields["reported_total_usd"] == Decimal("953.82")
    assert extraction.fields["reported_total_type"] == "GROSS"
    assert extraction.fields["commission_rate_pct"] == Decimal("0.25")
    assert extraction.fields["commission_usd"] == Decimal("2.38")
    assert extraction.fields["executed_at"].hour == 8
    assert extraction.fields["executed_at"].minute == 41


def test_parse_repairs_corrupted_accents_and_missing_i_in_titles() -> None:
    extraction = parse_gbm_text(
        """Orden completada
        $556.50 USD
        28 ago 2026 � 10:16 h
        Producto USA
        Emisora SMCI
        Tipo de Operaci�n Compra
        Tipo de Orden Limitada
        Ttulos 15
        Precio por titulo $37.10 USD
        Comisi�n 0.25% � $1.39 USD
        """
    )

    assert extraction.fields["side"] == "Compra"
    assert extraction.fields["quantity"] == Decimal("15")
    assert extraction.fields["commission_rate_pct"] == Decimal("0.25")
    assert extraction.fields["commission_usd"] == Decimal("1.39")
    assert not extraction.warnings


def test_missing_field_warnings_use_user_facing_spanish_labels() -> None:
    extraction = parse_gbm_text("Emisora SMCI")

    assert "títulos" in extraction.warnings[0]
    assert "quantity" not in extraction.warnings[0]


def test_sample_passes_rounding_and_commission_checks() -> None:
    fields = parse_gbm_text(SAMPLE).fields
    trade = TradeDraft(
        symbol=fields["symbol"],
        side=TradeSide.SELL,
        quantity=fields["quantity"],
        price_usd=fields["price_usd"],
        commission_usd=fields["commission_usd"],
        commission_rate_pct=fields["commission_rate_pct"],
        reported_total_usd=fields["reported_total_usd"],
        executed_at=fields["executed_at"],
    )
    report = validate_trade(trade)
    assert report.is_valid
    assert report.status == "VERIFIED"
    assert trade.gross_usd == Decimal("953.82")
    assert trade.cash_delta_usd == Decimal("951.44")


def test_buy_settlement_total_includes_commission() -> None:
    trade = TradeDraft(
        symbol="TSLA",
        side=TradeSide.BUY,
        quantity=Decimal("2"),
        price_usd=Decimal("100"),
        commission_usd=Decimal("0.50"),
        reported_total_usd=Decimal("200.50"),
        reported_total_type=ReportedTotalType.SETTLEMENT,
        executed_at=fields_datetime(),
    )
    report = validate_trade(trade)
    assert report.is_valid
    assert trade.gross_usd == Decimal("200.00")
    assert trade.settlement_usd == Decimal("200.50")
    assert trade.cash_delta_usd == Decimal("-200.50")


def fields_datetime():
    return parse_gbm_text(SAMPLE).fields["executed_at"]
