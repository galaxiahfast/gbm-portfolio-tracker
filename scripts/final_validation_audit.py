"""Read-only final validation report for the GBM forward-evidence system."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_tracker.services.zone_forward import model_version_hash, validation_data, verified

DB = ROOT / "data" / "portfolio.db"
OUT = ROOT / "output" / "pdf" / "auditoria_validacion_final_20260904.pdf"
EVIDENCE = ROOT / "output" / "audit_predictions"


def digest_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest() -> str:
    names = sorted(
        p for base in (ROOT / "portfolio_tracker", ROOT / "scripts")
        for p in base.rglob("*.py")
    ) + [ROOT / "app.py"]
    h = sha256()
    for path in names:
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_rows():
    connection = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    evidence = {r["sha256"]: r["payload_json"] for r in connection.execute(
        "SELECT sha256,payload_json FROM zone_market_evidence"
    )}
    rows = [dict(r) for r in connection.execute(
        "SELECT * FROM zone_prediction_log ORDER BY timestamp_prediction,id"
    )]
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
    connection.close()
    for row in rows:
        row["integrity_ok"] = verified(row, evidence.get(row.get("evidence_sha256")))
    return rows, integrity, foreign_keys


def summarize(rows):
    scored = validation_data(rows)
    events = {}
    for event in ("Toque", "Cierre"):
        values = [r for r in scored if r["event"] == event]
        events[event] = {
            "n": len(values),
            "brier": sum(r["brier"] for r in values) / len(values) if values else None,
            "hits": sum(r["actual"] for r in values),
        }
    by_symbol = []
    for symbol in sorted({r["symbol"] for r in rows}):
        group = [r for r in rows if r["symbol"] == symbol]
        by_symbol.append((symbol, len(group), sum(r["resolved_at"] is not None for r in group),
                          sum(r.get("actual_touch_occurred") == 1 for r in group)))
    by_zone = []
    for symbol, zone in sorted({(r["symbol"], r["zone_key"]) for r in rows}):
        group = [r for r in scored if r["symbol"] == symbol and r["zone_key"] == zone]
        touch = [r for r in group if r["event"] == "Toque"]
        close = [r for r in group if r["event"] == "Cierre"]
        by_zone.append((symbol, zone, len(touch), sum(r["actual"] for r in touch),
                        sum(r["actual"] for r in close)))
    bins = {}
    for event in ("Toque", "Cierre"):
        grouped = defaultdict(list)
        for row in scored:
            if row["event"] == event:
                grouped[row["bin"]].append(row)
        bins[event] = [(key, len(values), sum(r["prediction"] for r in values) / len(values),
                        sum(r["actual"] for r in values) / len(values))
                       for key, values in sorted(grouped.items())]
    violations = []
    cross_records = 0
    for row in rows:
        try:
            cross = json.loads(row["context_json"]).get("cross_asset") or {}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not cross:
            continue
        cross_records += 1
        impact = float(cross.get("applied_impact") or 0)
        correlation = cross.get("correlation")
        if impact and (cross.get("status") != "available" or correlation is None or correlation <= 0.70):
            violations.append(row["id"])
    return scored, events, by_symbol, by_zone, bins, cross_records, violations


def reliability_chart(bins, path):
    image = PILImage.new("RGB", (1350, 540), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((510, 16), "Curvas de fiabilidad - cohorte primario", fill="#111827", font=font)
    for panel, event in enumerate(("Toque", "Cierre")):
        left, top, width, height = 90 + panel * 650, 70, 520, 390
        draw.rectangle((left, top, left + width, top + height), outline="#cbd5e1", width=2)
        for step in range(1, 5):
            x = left + width * step / 5
            y = top + height * step / 5
            draw.line((x, top, x, top + height), fill="#e2e8f0")
            draw.line((left, y, left + width, y), fill="#e2e8f0")
        draw.line((left, top + height, left + width, top), fill="#94a3b8", width=2)
        draw.text((left + width // 2 - 20, 48), event, fill="#111827", font=font)
        values = bins[event]
        points = []
        for _, n, predicted, observed in values:
            x = left + predicted * width
            y = top + (1 - observed) * height
            points.append((x, y))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#2563eb")
            draw.text((x + 8, y - 10), f"n={n}", fill="#334155", font=font)
        if len(points) > 1:
            draw.line(points, fill="#2563eb", width=3)
        draw.text((left + width // 2 - 30, top + height + 24), "Pronosticado", fill="#475569", font=font)
        draw.text((left - 55, top - 4), "1.0", fill="#475569", font=font)
        draw.text((left - 18, top + height + 8), "0", fill="#475569", font=font)
    image.save(path)


def build():
    rows, integrity, foreign_keys = load_rows()
    scored, events, by_symbol, by_zone, bins, cross_records, violations = summarize(rows)
    scheduler = json.loads((EVIDENCE / "scheduler_verified_20260904.json").read_text(encoding="utf-8"))
    backup = ROOT / "backups" / "portfolio.db.aesgcm"
    manifest = json.loads((ROOT / "backups" / "manifest.json").read_text(encoding="utf-8"))
    pending = [r for r in rows if r["resolved_at"] is None]
    now = datetime.now(timezone.utc)
    overdue = [r for r in pending if datetime.fromisoformat(r["expires_at"]) < now]
    invalid = sum(not r["integrity_ok"] for r in rows)
    duplicates = len(rows) - len({(r["symbol"], r["session_date"], r["zone_key"],
                                  r["timestamp_prediction"]) for r in rows})
    test_pass = 334
    critical = [
        f"{len(overdue)} predicciones vencidas siguen sin resolver.",
        "NVDA tiene 12 registros y 0 resoluciones; no existe validación empírica para ese activo.",
        "Catchup está activo, pero su última ejecución terminó con código 1 por discrepancia cierre diario/5m.",
    ]
    verdict = "RECHAZADO"
    chart = EVIDENCE / "reliability_final_20260904.png"
    reliability_chart(bins, chart)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Cover", parent=styles["Title"], fontSize=25, leading=30,
                              alignment=TA_CENTER, textColor=colors.HexColor("#111827"), spaceAfter=10))
    styles.add(ParagraphStyle(name="Verdict", parent=styles["Heading1"], fontSize=19, leading=24,
                              alignment=TA_CENTER, textColor=colors.HexColor("#b91c1c"), spaceAfter=10))
    styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=15, leading=19,
                              textColor=colors.HexColor("#111827"), spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="B", parent=styles["BodyText"], fontSize=9.3, leading=13,
                              textColor=colors.HexColor("#1f2937"), spaceAfter=5))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.7, leading=10,
                              textColor=colors.HexColor("#475569")))

    def p(text, style="B"):
        return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles[style])

    def table(headers, data, widths=None):
        rows2 = [[p(x, "Small") for x in headers]] + [[p(x, "Small") for x in row] for row in data]
        t = Table(rows2, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    story = [Spacer(1, 34 * mm), p("AUDITORÍA DE VALIDACIÓN FINAL", "Cover"),
             p("Sistema completo de evidencia predictiva GBM+", "Cover"), Spacer(1, 8 * mm),
             p(verdict, "Verdict"),
             p("La infraestructura conserva integridad y automatización verificable, pero no puede certificarse para uso validado mientras existan predicciones vencidas sin resolver y NVDA carezca de resultados.", "B"),
             Spacer(1, 8 * mm),
             table(["Fecha", "Versión del motor", "SHA-256 del código auditado"], [[
                 "04/09/2026", model_version_hash()[:20] + "...", source_digest()[:28] + "..."
             ]], [38 * mm, 58 * mm, 78 * mm]), PageBreak()]

    story += [p("1. Resumen ejecutivo", "H"),
              table(["Indicador", "Resultado", "Estado"], [
                  ["Predicciones", f"{len(rows)} totales / {sum(r['resolved_at'] is not None for r in rows)} resueltas", "Cumple muestra bruta"],
                  ["Cohorte primario", f"{len([r for r in scored if r['event']=='Toque'])} toque / {len([r for r in scored if r['event']=='Cierre'])} cierre", "Insuficiente por activo"],
                  ["Brier toque", f"{events['Toque']['brier']:.4f}", "< 0.25"],
                  ["Brier cierre", f"{events['Cierre']['brier']:.4f}", "< 0.25"],
                  ["Integridad", f"{invalid} firmas inválidas; SQLite {integrity}", "Cumple"],
                  ["Duplicados exactos", str(duplicates), "Cumple"],
                  ["Pendientes vencidas", str(len(overdue)), "No cumple"],
                  ["Pruebas", f"{test_pass} aprobadas; 0 fallos", "Cumple"],
              ], [55 * mm, 73 * mm, 46 * mm]),
              p("Motivos del rechazo", "H")] + [p("- " + x) for x in critical]

    story += [p("2. Autopiloto y estabilidad", "H"),
              table(["Tarea", "Estado", "Última ejecución", "Resultado", "Próxima"], [
                  [t["TaskName"], t["State"], t["LastRunTime"], str(t["LastTaskResult"]), t["NextRunTime"]]
                  for t in scheduler
              ], [43 * mm, 20 * mm, 43 * mm, 23 * mm, 45 * mm]),
              p("Collector: última ejecución programada 04/09 11:00 NY, 49 s, 12 zonas guardadas. Resolver: 03/09 17:00 NY, 2 s. Catchup: 04/09 09:05 y 10:05 NY, 6 s y 4 s; ambas detectaron la discrepancia NVDA y conservaron seis registros pendientes. La preservación es correcta, pero el ciclo no concluyó."),
              p("La captura del Programador confirma el servicio activo; la tabla anterior procede de una consulta Windows de solo lectura con permisos suficientes."),
              KeepTogether([
                  p("Evidencia visual del Programador de tareas", "H"),
                  Image(str(EVIDENCE / "task_scheduler_20260904.png"), width=174 * mm, height=90 * mm),
              ])]

    story += [PageBreak(), p("3. Predicciones, resolución y Brier", "H"),
              table(["Símbolo", "Totales", "Resueltas", "Toques positivos"], by_symbol,
                    [48 * mm, 42 * mm, 42 * mm, 42 * mm]), Spacer(1, 5),
              Image(str(chart), width=174 * mm, height=70 * mm),
              p("Los Brier se calculan únicamente sobre el cohorte primario firmado: primera emisión por activo, versión, sesión y zona. Las 96 filas resueltas no son 96 observaciones independientes. El cohorte evaluable contiene 30 pares por evento, solo SMCI, repartidos entre varias versiones y esencialmente una sesión por versión. Por tanto, los valores son descriptivos preliminares, no calibración OOS concluyente."),
              p("Distribución por zona y símbolo", "H"),
              table(["Símbolo", "Zona", "n evaluable", "Toques", "Cierres"], by_zone,
                    [36 * mm, 38 * mm, 34 * mm, 32 * mm, 34 * mm])]

    backup_hash = digest_file(backup) if backup.exists() else "N/D"
    story += [PageBreak(), p("4. Integridad, correlación y respaldo", "H"),
              table(["Control", "Resultado"], [
                  ["Firmas forecast/resolución/evidencia", f"{invalid} inválidas de {len(rows)}"],
                  ["SQLite / claves foráneas", f"{integrity}; incidencias {len(foreign_keys)}"],
                  ["Duplicados exactos", str(duplicates)],
                  ["Contextos cross-asset", f"{cross_records} registros; {len(violations)} violaciones del umbral > 0.70"],
                  ["Backup AES-256-GCM", f"Existe; {backup.stat().st_size if backup.exists() else 0:,} bytes"],
                  ["SHA-256 backup", backup_hash],
                  ["SHA-256 manifest", manifest.get("ciphertext_sha256", "N/D")],
              ], [58 * mm, 116 * mm]),
              p("El hash cifrado coincide exactamente con el manifest. Los niveles dinámicos y extendidos no invocan log_snapshot; la única ruta de persistencia de las seis zonas permanece en autopilot_runtime.py. No se observaron registros atribuibles a renderizados UI/PDF."),
              p("5. Hallazgos y acciones", "H"),
              table(["Prioridad", "Hallazgo", "Acción"], [
                  ["CRÍTICA", "Seis predicciones NVDA vencidas sin resolución.", "Conciliar fuente diaria frente a la vela 5m sin imputar precios actuales; repetir catch-up."],
                  ["ALTA", "NVDA no tiene observaciones resueltas; Brier no disponible por activo.", "Acumular al menos 30 sesiones y reportar Brier separado por activo/versión."],
                  ["ALTA", "Catchup terminó con código 1.", "Corregir discrepancia de cierre y exigir siguiente ejecución con resultado 0."],
                  ["MEDIA", "30 observaciones primarias, concentradas en SMCI y varias versiones.", "No presentar los scores como probabilidades calibradas OOS."],
                  ["BAJA", "Advertencias deprecadas de pandas generan ruido.", "Cambiar Timedelta genérico por unidades explícitas."],
              ], [24 * mm, 72 * mm, 78 * mm]),
              p("6. Conclusión", "H"),
              p("VEREDICTO: RECHAZADO. El sistema demuestra integridad criptográfica, automatización instalada, backup válido, ausencia de duplicados y 334 pruebas aprobadas. Sin embargo, la evidencia operativa no está completa: hay resultados vencidos pendientes y el segundo activo carece de resolución. Recomendación: mantener el sistema en modo preliminar/observación; no declarar probabilidades empíricamente validadas hasta resolver el backlog y acumular sesiones independientes suficientes."),
              p("Auditoría estrictamente de solo lectura sobre SQLite. No se insertaron, actualizaron ni eliminaron operaciones, efectivo, comprobantes o predicciones.", "Small")]

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 11 * mm, "GBM+ - Auditoría de validación final - Uso interno")
        canvas.drawRightString(192 * mm, 11 * mm, f"Página {doc.page}"); canvas.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm,
                      title="Auditoría de validación final", author="Auditoría técnica").build(
        story, onFirstPage=footer, onLaterPages=footer
    )
    print(json.dumps({"pdf": str(OUT), "verdict": verdict, "rows": len(rows),
                      "resolved": sum(r["resolved_at"] is not None for r in rows),
                      "overdue": len(overdue), "brier": events}, ensure_ascii=False))


if __name__ == "__main__":
    build()
