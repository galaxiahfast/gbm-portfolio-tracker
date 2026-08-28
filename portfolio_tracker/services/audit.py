"""Auditoria independiente de SQLite, efectivo, posiciones y comprobantes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..models import ReportedTotalType, TradeDraft, TradeSide, money
from ..repository import PortfolioRepository
from .validation import validate_trade


class AuditLevel(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class AuditFinding:
    area: str
    level: AuditLevel
    message: str
    reference: str = ""


@dataclass(frozen=True, slots=True)
class AuditReport:
    generated_at: datetime
    findings: tuple[AuditFinding, ...]

    @property
    def passed(self) -> int:
        return sum(item.level is AuditLevel.PASS for item in self.findings)

    @property
    def warnings(self) -> int:
        return sum(item.level is AuditLevel.WARNING for item in self.findings)

    @property
    def errors(self) -> int:
        return sum(item.level is AuditLevel.ERROR for item in self.findings)

    @property
    def status(self) -> AuditLevel:
        if self.errors:
            return AuditLevel.ERROR
        if self.warnings:
            return AuditLevel.WARNING
        return AuditLevel.PASS

    def to_json(self) -> str:
        payload = {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "findings": [
                {
                    **asdict(item),
                    "level": item.level.value,
                }
                for item in self.findings
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class PortfolioAuditor:
    """Ejecuta controles reproducibles sin depender de la interfaz."""

    def __init__(
        self,
        repository: PortfolioRepository,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.repository = repository
        self.project_root = project_root

    def run(self) -> AuditReport:
        findings: list[AuditFinding] = []
        self._check_database(findings)
        trades = self.repository.list_trades(ascending=True)
        movements = self.repository.list_cash_movements()
        self._check_cash(findings, trades, movements)
        self._check_positions(findings, trades)
        self._check_trade_math(findings, trades)
        self._check_receipts(findings, trades)
        self._check_backtests(findings)
        return AuditReport(datetime.now(timezone.utc), tuple(findings))

    def _check_database(self, findings: list[AuditFinding]) -> None:
        integrity = self.repository.database.integrity_check()
        level = AuditLevel.PASS if integrity.lower() == "ok" else AuditLevel.ERROR
        findings.append(
            AuditFinding(
                "Base de datos",
                level,
                "SQLite confirmó la integridad interna."
                if level is AuditLevel.PASS
                else f"SQLite reportó: {integrity}",
            )
        )
        findings.append(
            AuditFinding(
                "Base de datos",
                AuditLevel.PASS,
                f"Esquema actualizado a la migración {self.repository.database.schema_version()}.",
            )
        )

    @staticmethod
    def _expected_trade_delta(trade: dict[str, Any]) -> Decimal:
        side = TradeSide(str(trade["side"]))
        gross = Decimal(trade["gross_usd"])
        commission = Decimal(trade["commission_usd"])
        total_type = ReportedTotalType(
            str(trade.get("reported_total_type") or ReportedTotalType.GROSS.value)
        )
        reported = trade.get("reported_total_usd")
        if total_type is ReportedTotalType.SETTLEMENT and reported is not None:
            settlement = Decimal(reported)
        elif side is TradeSide.BUY:
            settlement = gross + commission
        else:
            settlement = gross - commission
        return money(-settlement if side is TradeSide.BUY else settlement)

    def _check_cash(
        self,
        findings: list[AuditFinding],
        trades: list[dict[str, Any]],
        movements: list[dict[str, Any]],
    ) -> None:
        expected_cash = Decimal("0")
        movement_errors = 0
        for movement in movements:
            usd_amount = Decimal(movement["usd_amount"])
            if usd_amount <= 0:
                movement_errors += 1
            sign = Decimal("-1") if movement["kind"] == "WITHDRAWAL" else Decimal("1")
            expected_cash += sign * usd_amount

            rate = movement.get("fx_rate")
            if movement["original_currency"] == "MXN" and rate:
                original = Decimal(movement["original_amount"])
                expected_original = (
                    usd_amount * Decimal(rate)
                    if movement["kind"] == "WITHDRAWAL"
                    else usd_amount * Decimal(rate)
                )
                conversion_tolerance = Decimal(rate) * Decimal("0.005") + Decimal("0.01")
                if abs(original - expected_original) > conversion_tolerance:
                    movement_errors += 1
                    findings.append(
                        AuditFinding(
                            "Efectivo",
                            AuditLevel.ERROR,
                            "La conversión MXN/USD no coincide con la tasa guardada.",
                            f"movimiento #{movement['id']}",
                        )
                    )

        trade_delta_errors = 0
        for trade in trades:
            expected_delta = self._expected_trade_delta(trade)
            stored_delta = money(Decimal(trade["cash_delta_usd"]))
            if stored_delta != expected_delta:
                trade_delta_errors += 1
                findings.append(
                    AuditFinding(
                        "Efectivo",
                        AuditLevel.ERROR,
                        f"Delta guardado {stored_delta} USD; esperado {expected_delta} USD.",
                        f"operación #{trade['id']}",
                    )
                )
            expected_cash += expected_delta

        stored_ledger_cash = self.repository.cash_balance_usd()
        expected_cash = money(expected_cash)
        if stored_ledger_cash == expected_cash and not movement_errors and not trade_delta_errors:
            findings.append(
                AuditFinding(
                    "Efectivo",
                    AuditLevel.PASS,
                    f"El libro de efectivo concilia en {expected_cash:,.2f} USD.",
                )
            )
        elif stored_ledger_cash != expected_cash:
            findings.append(
                AuditFinding(
                    "Efectivo",
                    AuditLevel.ERROR,
                    f"Saldo registrado {stored_ledger_cash:,.2f}; saldo reconstruido {expected_cash:,.2f} USD.",
                )
            )

    @staticmethod
    def _check_positions(
        findings: list[AuditFinding], trades: list[dict[str, Any]]
    ) -> None:
        quantities: dict[str, Decimal] = {}
        negative: list[str] = []
        for trade in trades:
            symbol = str(trade["symbol"]).upper()
            previous = quantities.get(symbol, Decimal("0"))
            quantity = Decimal(trade["quantity"])
            quantities[symbol] = previous + (
                quantity if trade["side"] == TradeSide.BUY.value else -quantity
            )
            if quantities[symbol] < 0:
                negative.append(f"{symbol} en operación #{trade['id']}")
        if negative:
            for reference in negative:
                findings.append(
                    AuditFinding(
                        "Posiciones",
                        AuditLevel.ERROR,
                        "La secuencia histórica genera títulos negativos.",
                        reference,
                    )
                )
        else:
            findings.append(
                AuditFinding(
                    "Posiciones",
                    AuditLevel.PASS,
                    "Ninguna emisora queda negativa en la secuencia histórica.",
                )
            )

    @staticmethod
    def _check_trade_math(
        findings: list[AuditFinding], trades: list[dict[str, Any]]
    ) -> None:
        invalid = 0
        review = 0
        for row in trades:
            trade = TradeDraft(
                symbol=str(row["symbol"]),
                product=str(row["product"]),
                side=TradeSide(str(row["side"])),
                order_type=str(row["order_type"]),
                quantity=Decimal(row["quantity"]),
                price_usd=Decimal(row["price_usd"]),
                commission_usd=Decimal(row["commission_usd"]),
                commission_rate_pct=(
                    Decimal(row["commission_rate_pct"])
                    if row.get("commission_rate_pct") is not None
                    else None
                ),
                reported_total_usd=(
                    Decimal(row["reported_total_usd"])
                    if row.get("reported_total_usd") is not None
                    else None
                ),
                reported_total_type=ReportedTotalType(
                    str(row.get("reported_total_type") or "GROSS")
                ),
                executed_at=datetime.fromisoformat(str(row["executed_at"])),
            )
            report = validate_trade(trade)
            if not report.is_valid:
                invalid += 1
                findings.append(
                    AuditFinding(
                        "Operaciones",
                        AuditLevel.ERROR,
                        " ".join(report.errors),
                        f"operación #{row['id']}",
                    )
                )
            elif report.warnings:
                review += 1
        if not invalid:
            level = AuditLevel.WARNING if review else AuditLevel.PASS
            message = (
                f"{review} operación(es) requieren revisión no bloqueante."
                if review
                else f"Las {len(trades)} operación(es) pasan la conciliación aritmética."
            )
            findings.append(AuditFinding("Operaciones", level, message))

    def _absolute_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _check_receipts(
        self,
        findings: list[AuditFinding],
        trades: list[dict[str, Any]],
    ) -> None:
        receipts = self.repository.list_receipts()
        linked_ids = {
            int(trade["receipt_id"])
            for trade in trades
            if trade.get("receipt_id") is not None
        }
        receipt_ids = {int(receipt["id"]) for receipt in receipts}
        missing_links = linked_ids - receipt_ids
        for receipt_id in sorted(missing_links):
            findings.append(
                AuditFinding(
                    "Comprobantes",
                    AuditLevel.ERROR,
                    "La operación apunta a un registro de comprobante inexistente.",
                    f"comprobante #{receipt_id}",
                )
            )

        valid_hashes = 0
        for receipt in receipts:
            path = self._absolute_path(str(receipt["original_path"]))
            reference = f"comprobante #{receipt['id']}"
            if not path.is_file():
                findings.append(
                    AuditFinding(
                        "Comprobantes",
                        AuditLevel.ERROR,
                        "No se encontró el archivo original.",
                        reference,
                    )
                )
                continue
            if self._sha256(path) != str(receipt["sha256"]):
                findings.append(
                    AuditFinding(
                        "Comprobantes",
                        AuditLevel.ERROR,
                        "La huella SHA-256 ya no coincide con el archivo original.",
                        reference,
                    )
                )
                continue
            valid_hashes += 1

        if valid_hashes == len(receipts) and not missing_links:
            findings.append(
                AuditFinding(
                    "Comprobantes",
                    AuditLevel.PASS,
                    f"SHA-256 verificado en {valid_hashes} comprobante(s).",
                )
            )

        unlinked = receipt_ids - linked_ids
        if unlinked:
            findings.append(
                AuditFinding(
                    "Comprobantes",
                    AuditLevel.WARNING,
                    f"Hay {len(unlinked)} comprobante(s) conservado(s) sin operación; puede ocurrir al borrar una transacción.",
                )
            )
        linked_receipt_sequence = [
            int(trade["receipt_id"])
            for trade in trades
            if trade.get("receipt_id") is not None
        ]
        duplicated_links = {
            receipt_id
            for receipt_id, count in Counter(linked_receipt_sequence).items()
            if count > 1
        }
        for receipt_id in sorted(duplicated_links):
            findings.append(
                AuditFinding(
                    "Comprobantes",
                    AuditLevel.ERROR,
                    "El mismo comprobante está vinculado a más de una operación.",
                    f"comprobante #{receipt_id}",
                )
            )
        without_receipt = sum(trade.get("receipt_id") is None for trade in trades)
        if without_receipt:
            findings.append(
                AuditFinding(
                    "Comprobantes",
                    AuditLevel.WARNING,
                    f"{without_receipt} operación(es) fueron capturadas sin imagen.",
                )
            )

    def _check_backtests(self, findings: list[AuditFinding]) -> None:
        valid, invalid_ids = self.repository.verify_backtest_runs()
        if invalid_ids:
            findings.append(
                AuditFinding(
                    "Backtesting",
                    AuditLevel.ERROR,
                    "La huella del resultado estadístico no coincide con su contenido.",
                    "ejecuciones " + ", ".join(f"#{item}" for item in invalid_ids),
                )
            )
            return
        findings.append(
            AuditFinding(
                "Backtesting",
                AuditLevel.PASS,
                f"Integridad SHA-256 confirmada en {valid} ejecución(es) estadística(s).",
            )
        )
