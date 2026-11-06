"""Append observed quote evidence to closed diagnostics; retain earlier snapshots."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT, digest_file
from scripts.audit_trigger_quote_windows import inspect_quotes
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.measure_trigger_selectivity import DEFAULT_OUTPUT
from scripts.report_trigger_selectivity import artifact, table, number


def local(at):
    return pd.Timestamp(at).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")


def run():
    out, qdir = DEFAULT_OUTPUT, DEFAULT_OUTPUT / "quote_access"
    report = read_signed(out / "measurement_report.json")
    diagnosis = read_signed(OUTPUT / "diagnosis.json")
    snapshot = read_signed(qdir / "provided_screenshot_pair.json")
    original_window = read_signed(qdir / "quote_window_audit.json")
    protocol = read_signed(OUTPUT / "preregistration.json")
    for path, expected in protocol["engine_code_sha256"].items():
        if digest_file(ROOT / path) != expected:
            raise ValueError("Changed production/replay engine.")
    ledger = digest_file(ROOT / "data/portfolio.db")
    if ledger != report["ledger_after_sha256"]:
        raise ValueError("Ledger changed.")
    events = {e["event_id"]: e for e in diagnosis["symbols"]["SMCI"]["events"]}
    rows, costs, summaries = [], [], []
    for e in report["fill_evidence"]:
        eid = e["event_id"]
        w = next(x for x in original_window["events"] if x["event_id"] == eid)
        raw = read_signed(out / w["raw_file"])
        if sha(raw) != w["raw_sha256"]:
            raise ValueError("Altered raw quote path.")
        # execution_path.py records fill_at=end (bar close), NOT bar open.
        # Retain raw download unchanged; exclude the extra 5m from conclusions.
        end = pd.Timestamp(e["resolutions"][0]["fill_at_possible_OHLC"])
        selected = [q for q in raw["quotes"] if pd.Timestamp(q["t"]) <= end]
        evidence = inspect_quotes(selected, e["side"], e["limit_price"], events[eid]["evaluation_starts_at"], end)
        cut = next(x for x in snapshot["quotes"] if x["event_id"] == eid and x["reference"] == "cut")
        first = evidence["first_compatible_quote"]
        if first is None:
            raise ValueError("No compatible quote in true fill window; do not certify possible fill.")
        first_spread = float(first["ap"]) - float(first["bp"])
        summary = {"event_id": eid, "cut_at": e["cut_at"], "side": e["side"],
                   "start": events[eid]["evaluation_starts_at"], "end": end.isoformat(),
                   "raw_quote_file": w["raw_file"], "raw_quote_file_sha256": digest_file(out / w["raw_file"]),
                   "raw_quotes_downloaded": len(raw["quotes"]), "excluded_after_fill_bar_close": len(raw["quotes"])-len(selected),
                   "cut_quote": cut, "first_compatible_quote_spread": first_spread, **evidence}
        summaries.append(summary)
        rows.append([local(e["cut_at"])[:16], e["side"], number(e["limit_price"], 4),
                     f"{cut['quote']['bp']:.2f} / {cut['quote']['ap']:.2f}", number(cut["spread_observed"], 2),
                     local(first["t"])[11:], f"{first['bp']:.2f} / {first['ap']:.2f}", number(first_spread, 2),
                     evidence["compatible_with_limit_count"], "No, SHORT" if e["side"] == "SHORT" else "LONG; fill sin confirmar"])
        for h in e["resolutions"]:
            exit_quote = next(x for x in snapshot["quotes"] if x["event_id"] == eid and x["reference"] == f"simulated_exit_{h['horizon_minutes']}")
            if not exit_quote["valid_positive_uncrossed_market"]:
                raise ValueError("Invalid exit reference quote.")
            spread_midpoint_proxy = (first_spread + exit_quote["spread_observed"]) / 2
            costs.append([local(e["cut_at"])[:16], "1h" if h["horizon_minutes"] == 60 else "1D",
                          number(h["costs_per_share_assumed"]["commission_roundtrip_assumed"], 4),
                          number(first_spread, 2), number(exit_quote["spread_observed"], 2), number(spread_midpoint_proxy, 4), "N/D"])
            summary.setdefault("cost_reference_by_horizon", []).append({"horizon_minutes": h["horizon_minutes"],
                "commission_roundtrip_frozen_tariff_estimate": h["costs_per_share_assumed"]["commission_roundtrip_assumed"],
                "entry_first_compatible_quote_spread_observed": first_spread,
                "exit_bar_close_quote_spread_observed_NOT_exact_exit_tick": exit_quote["spread_observed"],
                "half_spread_crossing_proxy_NOT_paid_cost": spread_midpoint_proxy,
                "paid_commission": None, "paid_spread_or_slippage": None})
    resolved = {"version": "OBSERVED_SIP_QUOTE_DIAGNOSTIC_V2", "quote_capture_at": snapshot["generated_at"],
                "measurement_report_sha256": sha(report), "source_script_sha256": digest_file(Path(__file__)),
                "semantics_erratum": "Earlier quote-window metadata called fill_at/exit_at bar opens. Core actually uses BAR CLOSE; immutable raw evidence preserved; exact-fill conclusions now stop at fill_at, removing 5m trailing quotes.",
                "events": summaries, "ledger_before_and_after_sha256": ledger,
                "no_filters_parameters_or_training_changed": True, "broker_fills_observed": False}
    write_signed(qdir / "quote_window_exact_fill_audit.json", resolved)
    section = ["## 7. Las seis entradas SMCI: evidencia bid/ask observada",
               f"Complemento obtenido en {snapshot['generated_at']}. Se consultó únicamente Market Data de Alpaca SIP: {sum(s['raw_quotes_downloaded'] for s in summaries):,} cotizaciones en seis ventanas, sin compras de planes ni llamadas a órdenes/cuentas. Los OHLCV del replay siguen siendo los mismos de Massive; no se reejecuta ni modifica el fill engine.",
               "La primera pareja de credenciales proporcionada respondió HTTP401; la pareja de la captura permitió el acceso. Las claves no se guardaron en scripts, archivos de evidencia ni logs; permanecieron solo en memoria durante la consulta.",
               "Según la [documentación de Alpaca](https://docs.alpaca.markets/us/docs/market-data-faq), las consultas SIP históricas con vencimiento de más de 15 minutos pueden hacerse sin suscripción en ese acceso. Se verificó respuesta HTTP200; coste incremental pagado en esta tarea: USD0, sin contratar nada.",
               "### Ventanas y criterio de fill posible",
               "Se descargó desde evaluation_starts_at del contrato original hasta fill_at+5m. Al verificar execution_path.py se comprobó que fill_at es CIERRE de vela de 5m, no apertura: para las conclusiones se excluyó todo quote posterior a fill_at. Los archivos originales y su metadata se conservaron; la corrección semántica está explícita en quote_window_exact_fill_audit.json.",
               "Prueba de compatibilidad: LONG ask ≤ límite; SHORT bid ≥ límite; bid/ask positivos, no cruzados y tamaños cotizados positivos. Ningún quote futuro al cierre de la vela de fill se usa para afirmar posibilidad. Los quotes al corte tienen edades 0.05–0.89 segundos; fecha, timestamp nanosegundo y exchanges están archivados.",
               "Las seis ventanas tienen cotizaciones compatibles con su límite. Eso demuestra oportunidad cotizada bajo el contrato hipotético, NO envío de orden, ejecución del broker, permanencia de liquidez, cola, latencia ni volumen disponible para un tamaño real.",
               table(["Corte NY", "Lado", "Límite USD", "Bid/ask al corte", "Spread al corte", "Primer quote compatible NY", "Bid/ask compatible", "Spread compatible", "Quotes compatibles hasta fill close", "Trading USA"], rows),
               "La columna de quotes compatibles es cantidad de actualizaciones de cotización, NO operaciones, fills, oportunidades independientes ni muestras para entrenamiento. Cada ventana sigue perteneciendo a UN evento.",
               "Según [Trading USA de GBM](https://gbm.com/trading/), no se abren posiciones cortas propias en ese producto. Cinco SHORT permanecen separados como no ejecutables allí. La única LONG tiene quote compatible, pero fill GBM no confirmado. Política consultada hoy, no certificación de permisos históricos de otra cuenta/producto.",
               "### Comisión estimada y spread observado (USD por una acción)",
               table(["Corte NY", "H", "Comisión ida/vuelta estimada", "Spread primer quote compatible", "Spread al cierre de vela de salida", "Proxy mitad spreads (NO pagado)", "Coste realizado broker"], costs),
               "Comisión estimada conserva exactamente fill/salida y tarifa de la simulación: (fill+salida)×0.0025. La [tarifa pública de Trading USA](https://gbm.com/faqs/existe-alguna-comision-por-invertir-en-trading-usa) es 0.25% por lado; no prueba cargos históricos pagados. No hay tamaño real de orden ni comisión realizada para estos eventos hipotéticos.",
               "Spread observado = ask−bid. No es automáticamente un cargo: depende del tipo de orden y precio realizado. Proxy de cruce respecto al midpoint = mitad del spread de entrada más mitad del spread al cierre de la vela de salida; NO es pérdida realizada, coste exacto de stop/TP ni slippage observado. El quote al cierre de vela puede ser posterior al toque intravela. No se modifica el spread supuesto original de 4bps del replay.",
               "La trayectoria y salida real todavía requerirían envío de orden, routing, tamaños, prints y confirmación del broker. Los cinco SHORT carecen de permiso en Trading USA, aunque existan quotes compatibles. No se sustituyó esta falta de ejecución por una afirmación de operabilidad."]
    original = (out / "reporte_cerrado.md").read_text(encoding="utf-8")
    prefix = original.split("## 7. Las seis entradas SMCI: fills y costes")[0]
    suffix = "## 8." + original.split("## 8.", 1)[1]
    final = prefix + "\n\n".join(section) + "\n\n" + suffix
    final = final.replace("# Auditoría cerrada de selectividad del gatillo", "# Auditoría final de selectividad del gatillo y spread observado", 1)
    final += "\n\n## 11. Cierre con cotizaciones observadas\n\nEl reporte_cerrado.md conserva el estado previo a obtener quotes (N/D). Este reporte_final.md incorpora la evidencia SIP posterior sin borrar el registro anterior. quote_access contiene respuestas sanitizadas, metodología, cotizaciones originales, ventanas verificadas y su errata temporal; manifest_final.json inventaría la evidencia final. Conteos, filtros y eventos del replay NO cambiaron.\n"
    artifact(out / "reporte_final.md", final)
    base_manifest = read_signed(out / "closure_manifest.json")
    files = [out / n for n in base_manifest["files_sha256"]]
    files += list(qdir.rglob("*.json")) + [out / "reporte_final.md"]
    source_files = [ROOT / f"scripts/{n}.py" for n in ("collect_trigger_quotes", "audit_trigger_quote_windows", "close_trigger_quote_report")]
    for file in source_files:
        target = out / "source_archive" / digest_file(file) / file.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    write_signed(out / "manifest_final.json", {"version": "TRIGGER_AUDIT_FINAL_WITH_QUOTES_V2", "base_manifest_sha256": sha(base_manifest),
        "files_sha256": {str(f.relative_to(out)): digest_file(f) for f in files},
        "source_sha256": {str(f.relative_to(ROOT)): digest_file(f) for f in source_files},
        "ledger_before_and_after_sha256": ledger, "engine_code_sha256": protocol["engine_code_sha256"]})
    print("Final report:", out / "reporte_final.md")
    print("Quoted compatible events:", len(summaries), "original engine and ledger unchanged.")


if __name__ == "__main__":
    run()
