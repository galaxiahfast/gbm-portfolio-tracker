import hashlib
from datetime import datetime, timezone
from decimal import Decimal

from portfolio_tracker.db import Database
from portfolio_tracker.models import TradeDraft, TradeSide
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.audit import AuditLevel, PortfolioAuditor


def test_audit_reconciles_cash_positions_and_receipt_hash(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    repository.ensure_initial_capital()

    content = b"immutable receipt test"
    receipt_path = tmp_path / "receipt.jpg"
    receipt_path.write_bytes(content)
    receipt_id = repository.upsert_receipt(
        sha256=hashlib.sha256(content).hexdigest(),
        original_filename="receipt.jpg",
        mime_type="image/jpeg",
        original_path="receipt.jpg",
        thumbnail_path="receipt.jpg",
        width=1,
        height=1,
        byte_size=len(content),
    )
    trade = TradeDraft(
        symbol="ABC",
        side=TradeSide.BUY,
        quantity=Decimal("1"),
        price_usd=Decimal("10"),
        commission_usd=Decimal("0.01"),
        reported_total_usd=Decimal("10"),
        executed_at=datetime.now(timezone.utc),
        validation_status="VERIFIED",
    )
    repository.add_trade(trade, receipt_id)

    report = PortfolioAuditor(repository, tmp_path).run()
    assert report.status is AuditLevel.PASS
    assert report.errors == 0
    assert report.warnings == 0
