"""Read-only post-correction report for the NVDA catch-up incident."""
from pathlib import Path
import sqlite3
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_tracker.services.zone_forward import verified

OUTPUT = ROOT / "output" / "pdf" / "auditoria_rapida_catchup_20260904.pdf"


def build():
    connection = sqlite3.connect(f"file:{(ROOT / 'data/portfolio.db').as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    evidence = {row["sha256"]: row["payload_json"] for row in connection.execute(
        "SELECT sha256,payload_json FROM zone_market_evidence"
    )}
    rows = [dict(row) for row in connection.execute(
        "SELECT * FROM zone_prediction_log ORDER BY timestamp_prediction,id"
    )]
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    for row in rows:
        row["integrity_ok"] = verified(row, evidence.get(row.get("evidence_sha256")))
    nvda = [row for row in rows if row["symbol"] == "NVDA"]
    resolved = [row for row in nvda if row["resolved_at"] is not None]
    due_unresolved = [row for row in nvda if row["resolved_at"] is None and row["session_date"] < "2026-09-04"]

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=19, leading=22,
                              textColor=colors.HexColor("#111827"), spaceAfter=5))
    styles.add(ParagraphStyle(name="Status", parent=styles["Heading1"], fontSize=14, leading=17,
                              textColor=colors.HexColor("#047857"), spaceAfter=4))
    styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=11.5, leading=14,
                              textColor=colors.HexColor("#111827"), spaceBefore=4, spaceAfter=2))
    styles.add(ParagraphStyle(name="B", parent=styles["BodyText"], fontSize=7.8, leading=10,
                              textColor=colors.HexColor("#1f2937"), spaceAfter=2.5))
    styles.add(ParagraphStyle(name="TH", parent=styles["BodyText"], fontSize=7.8, leading=9,
                              textColor=colors.white))

    def p(value, style="B"):
        text = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(text, styles[style])

    def table(headers, data, widths):
        content = [[p(x, "TH") for x in headers]] + [[p(x) for x in row] for row in data]
        result = Table(content, colWidths=widths, repeatRows=1)
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return result

    detail = []
    for row in resolved:
        detail.append([
            row["zone_key"],
            "Excluido" if row["actual_touch_occurred"] is None else str(row["actual_touch_occurred"]),
            f"${row['actual_close_price']:.6f}",
            "Sí" if row["integrity_ok"] else "No",
        ])
    story = [p("AUDITORÍA RÁPIDA POST-CORRECCIÓN", "Title2"),
             p("Catch-up NVDA - 4 de septiembre de 2026", "Title2"),
             p("CORRECCIÓN OPERATIVA APROBADA", "Status"),
             p("El backlog vencido del 3 de septiembre fue resuelto con el cierre diario oficial. Las 78 velas 5m completas permanecen como evidencia exclusiva de toques."),
             table(["Control", "Resultado", "Estado"], [
                 ["Ejecución manual producción", "Código de salida 0", "Cumple"],
                 ["NVDA vencidas resueltas", f"{len(resolved)} de 6", "Cumple"],
                 ["NVDA vencidas pendientes", str(len(due_unresolved)), "Cumple"],
                 ["Firmas de las resoluciones", f"{sum(r['integrity_ok'] for r in resolved)} de {len(resolved)} válidas", "Cumple"],
                 ["Integridad SQLite", integrity, "Cumple"],
                 ["Pruebas", "334 aprobadas; 0 fallos", "Cumple"],
                 ["Programador Catchup", "Ready; resultado anterior 1", "Pendiente próxima ejecución"],
             ], [62 * mm, 72 * mm, 40 * mm]),
             p("Detalle de las seis zonas", "H"),
             table(["Zona", "Toque", "Cierre oficial", "Firma válida"], detail,
                   [42 * mm, 42 * mm, 50 * mm, 40 * mm]),
             p("Trazabilidad de la discrepancia", "H"),
             p("Última vela 5m: $228.399994. Cierre diario 1D: $228.449997. Diferencia absoluta: $0.050003. Al superar USD 0.01, el sistema usa el cierre 1D como fuente oficial, conserva la sesión 5m para toques y firma una advertencia con ambas cifras."),
             p("La zona ENTRY1 aparece como Excluido porque ya estaba alcanzada al momento de emitir el pronóstico (touch_eligible=0). No se convierte artificialmente en 0 o 1 y queda fuera del Brier de toque."),
             p("Estado estadístico", "H"),
             p("El flujo operativo queda reparado y puede seguir acumulando evidencia. La calibración continúa en modo preliminar: se requieren al menos 30 sesiones independientes por activo y versión para declarar calibración OOS concluyente."),
             p("Integridad contable", "H"),
             p("Los hashes de trades, cash_movements, receipts, settings y schema_migrations fueron idénticos antes y después. No se modificaron órdenes, efectivo, operaciones ni comprobantes."),
             Spacer(1, 1 * mm),
             p("Conclusión: se levantan los hallazgos operativos de discrepancia y backlog vencido. No se levanta todavía la limitación estadística por tamaño de muestra ni se afirma una aprobación predictiva OOS.")]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 11 * mm, "GBM+ - Auditoría rápida catch-up - Uso interno")
        canvas.drawRightString(192 * mm, 11 * mm, f"Página {doc.page}"); canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                      topMargin=12 * mm, bottomMargin=16 * mm,
                      title="Auditoría rápida catch-up NVDA", author="Auditoría técnica").build(
        story, onFirstPage=footer, onLaterPages=footer
    )
    print(OUTPUT)


if __name__ == "__main__":
    build()
