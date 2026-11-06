"""Render a read-only operational audit from signed research archive evidence."""
from __future__ import annotations

from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_massive_history import sha
from scripts.prospective_coverage import build_report
from scripts.audit_score_map_resolution import audit as audit_score_map


OUTPUT = ROOT / "output" / "pdf" / "auditoria_operativa_fase5_20261005.pdf"


def render():
    report = build_report(save=True)
    score_audit = audit_score_map()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AuditTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#141414"),
        alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="AuditBody", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#222222"),
        spaceAfter=7))
    styles.add(ParagraphStyle(name="AuditHead", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#111111"),
        spaceBefore=10, spaceAfter=5))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(210*mm, 297*mm),
        leftMargin=17*mm, rightMargin=17*mm, topMargin=16*mm, bottomMargin=16*mm,
        title="Auditoria operativa Fase 5 - 05-10-2026", author="STOCK research")
    story = [Paragraph("Auditoría operativa · Motor cuantitativo Fase 5", styles["AuditTitle"]),
             Paragraph(f"Corte de reporte UTC: {report['generated_at_utc']}<br/>"
                       f"SHA-256 de cobertura: {sha(report)}", styles["AuditBody"]),
             Paragraph("1. Cortes perdidos y contención", styles["AuditHead"]),
             Paragraph("Los cortes de las 10:00 NY del 01 y 05 de octubre no se ejecutaron porque "
                       "Windows estaba suspendido. Eventos Power-Troubleshooter ID 1: despertar "
                       "14:46:27 UTC y 14:51:41 UTC, respectivamente; fuera de la ventana prospectiva "
                       "de cinco minutos. No existe evidencia de error API en esos cortes y el historial "
                       "detallado de Task Scheduler estaba deshabilitado.", styles["AuditBody"]),
             Paragraph("Se registró STOCK-Prospective-Cut-Wake con WakeToRun, tres minutos antes "
                       "de cada posible corte local. En corriente alterna los temporizadores de "
                       "reactivación están habilitados; un equipo apagado o en batería con temporizadores "
                       "deshabilitados no puede garantizar el corte. Los ocho backfills firmados "
                       "son DEGRADADOS y permanecen fuera de snapshots, entrenamiento y denominadores.",
                       styles["AuditBody"]),
             Paragraph("2. Cobertura prospectiva - fuente Alpaca IEX", styles["AuditHead"])]
    header = ["Activo", "Horizonte", "Snapshots", "Resueltos", "Pendientes", "Entradas autorizadas"]
    cells = [header] + [[r["symbol"], r["horizon"], str(r["snapshots"]), str(r["resolved"]),
                         str(r["pending"]), str(r["production_anchor_authorized"])]
                        for r in report["rows"]]
    table = Table(cells, colWidths=[25*mm, 25*mm, 25*mm, 24*mm, 26*mm, 45*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#1d1d1d")),
        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0),(-1,-1), 8),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("TOPPADDING", (0,0),(-1,-1), 6),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("LINEBELOW", (0,-1),(-1,-1), 0.5, colors.HexColor("#777777")),
    ]))
    story += [table, Spacer(1, 5*mm),
              Paragraph(f"Cortes prospectivos ausentes: {len(report['missing_prospective_cuts'])} "
                        f"(01 y 05 oct, 10:00 NY). Backfills degradados excluidos: "
                        f"{report['backfill_count_excluded_from_denominators']}.", styles["AuditBody"]),
              Paragraph("Cinco cortes del mismo día y horizontes solapados no son observaciones "
                        "independientes. Los 300 son umbral de evaluación, no precisión alcanzada.",
                        styles["AuditBody"]),
              Paragraph("3. Procedencia y mapa de scores", styles["AuditHead"]),
              Paragraph("Archivo prospectivo: Alpaca IEX (velas split-adjusted). App interactiva: "
                        "Yahoo Finance; ante fallo, respaldo local firmado. Replay de 400 sesiones: "
                        "Massive Stocks Basic, no Alpaca SIP. No se promedian feeds.", styles["AuditBody"]),
              Paragraph("El mapa CONTINUOUS_CONTEXT_LOGIT_V2 preserva distancias EMA/MACD "
                        "normalizadas por ATR y elimina el recorte final fijo. Reproducido sobre "
                        f"{score_audit['rows']} filas archivadas: "
                        f"{score_audit['distinct_score_triples']} ternas de scores distintas y "
                        f"{score_audit['plateau_70_26_4']} filas exactamente 70/26/4. "
                        "Son scores PRELIMINARES /100, no probabilidades calibradas. "
                        "El toque de niveles es otro suceso: desde cada corte hasta el cierre.", styles["AuditBody"]),
              Paragraph("No se entrenó, no se aprobaron modelos ni se autorizaron entradas. "
                        "Esta rutina no abre portfolio.db; una app abierta puede actualizar "
                        "tablas analíticas concurrentemente.", styles["AuditBody"])]
    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(render())
