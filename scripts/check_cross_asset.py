"""Read-only market integration check. Never opens the portfolio database."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pandas as pd

from portfolio_tracker.analytics.cross_correlation import PEERS, build_cross_context
from portfolio_tracker.services.cross_asset import _download_peer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", choices=sorted(PEERS), default="SMCI")
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    peer = PEERS[args.symbol]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_download_peer, args.symbol)
        second = pool.submit(_download_peer, peer)
        daily, intraday = first.result()
        peer_daily, peer_intraday = second.result()
    context = build_cross_context(args.symbol, daily, peer_daily, intraday, peer_intraday,
                                  as_of=pd.Timestamp.now(tz="UTC"))
    print(json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False))
    if args.pdf:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from portfolio_tracker.services.pdf_report import _cross_asset_story, _report_styles
        styles = _report_styles()
        story = [Paragraph("Verificación cross-asset · datos de mercado", styles["ReportTitle"]),
                 Paragraph(context["observed_at"], styles["ReportSubtitle"]),
                 Paragraph("Consulta real de Yahoo Finance. No contiene operaciones, simulaciones de acierto ni "
                           "modificaciones al portafolio. Sólo valida la integración del contexto.", styles["BodySmall"]),
                 Spacer(1, 10)]
        story.extend(_cross_asset_story(SimpleNamespace(cross_asset_context=context), styles))
        story.append(Paragraph("El ajuste aplicado figura en cero porque esta comprobación no ejecuta el "
                               "motor principal ni evalúa sus vetos. Los reportes de la app muestran su impacto aplicado real.",
                               styles["BodySmall"]))
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(str(args.pdf), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                          topMargin=18*mm, bottomMargin=18*mm).build(story)
        evidence = {"context": context, "provider": "yfinance / auto_adjust=False",
                    "frames": {"primary_daily": daily.to_json(orient="table", date_format="iso"),
                               "peer_daily": peer_daily.to_json(orient="table", date_format="iso"),
                               "primary_5m": intraday.to_json(orient="table", date_format="iso"),
                               "peer_5m": peer_intraday.to_json(orient="table", date_format="iso")}}
        args.pdf.with_suffix(".json").write_text(json.dumps(evidence, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"PDF: {args.pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
