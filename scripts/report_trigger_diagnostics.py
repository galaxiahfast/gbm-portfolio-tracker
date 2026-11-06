"""Readable post-run gate diagnostics. Counts only; no fitting or selection."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_trigger_replay import ROOT, OUTPUT, TIMES, joint_counts
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.build_historical_replay import load_dataset
from scripts.audit_trigger_replay import INPUT
import pandas as pd


def same_bar_ambiguity(row, dataset, *, hypothetical):
    """Separate unobservable intrabar order from declared conservative labels."""
    c = row.get("operational_contract")
    result = row["operational_result" if hypothetical else "execution_result"]
    if (not c or result.get("outcome") not in ("TP_FIRST", "SL_FIRST")
            or not str(result.get("exit_source", "")).startswith("5m")
            or "gap" in result.get("exit_source", "")):
        return False
    bar_at = pd.Timestamp(result["exit_at"]) - pd.Timedelta(minutes=5)
    if bar_at not in dataset.intraday.index:
        return False
    b = dataset.intraday.loc[bar_at]
    lo, hi = sorted((c["stop_loss"], c["take_profit"]))
    return bool(lo < b.Open < hi and b.Low <= lo and b.High >= hi)


def legacy_rebound(features, candle):
    """Exact reconstruction of the legacy WATCH logic, not the final gate."""
    price, support, atr = features["last_price"], features["nearest_support"], features["atr_5m"]
    interaction = (abs(price - support) <= max(.5 * atr, .002 * price)
                   or candle["Low"] <= support <= candle["High"])
    reversal = features["candle_pattern"] in ("HAMMER", "BULLISH_ENGULFING")
    chart = any(p["timeframe"] == "5m" and p["confirmed"] and p["confidence"] > 75
                and p["pattern_type"] in ("DOUBLE_BOTTOM", "TRIPLE_BOTTOM")
                for p in features["chart_patterns"])
    return bool(min(features["stochastic_k"], features["stochastic_d"]) < 20
                and (interaction or reversal or chart)), bool(reversal)


def distinct_events(records):
    """Group identical barriers/cut first, then embargo full max timeout.

    Outcome and actual exit time never participate in event selection.
    """
    grouped = {}
    for r in records:
        for h in r["horizons"]:
            c = h["operational_contract"]
            if not c or not c["eligible_at_emission"]:
                continue
            identity = {k: c[k] for k in ("side", "entry_price", "stop_loss", "take_profit", "evaluation_starts_at")}
            key = sha(identity)
            if key not in grouped:
                grouped[key] = {"event_id": key, "symbol": r["symbol"], "cut_at": r["cut_at"],
                                "time_ny": r["time_ny"], **identity,
                                "max_timeout": c["timeout_at"], "resolutions": []}
            event = grouped[key]
            event["max_timeout"] = max(event["max_timeout"], c["timeout_at"])
            event["resolutions"].append({"horizon_minutes": h["horizon_minutes"],
                                          "execution_status": h["execution_result"]["status"],
                                          "outcome": h["execution_result"].get("outcome"),
                                          "fill_price": h["execution_result"].get("fill_price"),
                                          "timeout_at": c["timeout_at"]})
    events = sorted(grouped.values(), key=lambda e: e["evaluation_starts_at"])
    boundary = None
    for event in events:
        event["nonoverlapping_selected"] = boundary is None or event["evaluation_starts_at"] >= boundary
        if event["nonoverlapping_selected"]:
            boundary = event["max_timeout"]
    return events


def main(output=OUTPUT):
    protocol = read_signed(output / "preregistration.json")
    report = read_signed(output / "report.json")
    audit = read_signed(output / "baseline_gate_audit.json")
    diagnosis = {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "preregistration_sha256": sha(protocol), "counts_report_sha256": sha(report), "symbols": {}}
    lines = ["# Diagnóstico operativo del gatillo", "", f"Fecha UTC: {diagnosis['generated_at']}",
             "", "Periodo: 21/07/2025–25/09/2026. SMCI y NVDA; mismo histórico firmado y mismas reglas técnicas.",
             "No se entrenó, no se ajustaron parámetros ni se seleccionaron horarios por resultados.",
             "", "## 1. Fallos independientes: 299 cortes de las 11:00 NY por activo", "",
             "| Condición | SMCI fallos | NVDA fallos | % SMCI / NVDA | % conjunto (598) |",
             "|---|---:|---:|---|---:|"]
    labels = [
        ("Precio dentro de zona LONG **(descriptivo; no gate final)**", "inside_buy_zone_descriptive"),
        ("Vela alcista cerrada **(diagnóstico; no equivale a rebote autorizado)**", "bullish_closed_candle_descriptive"),
        ("Volumen > media de 20 velas previas", "volume_above_previous20"),
        ("MACD 5m ≥ señal", "macd_long"),
        ("Régimen autoriza LONG: LONG_ONLY", "regime_long_only"),
        ("Veto técnico inactivo", "risk_veto_inactive"),
        ("Gatillo final completo: cualquier dirección", "final_trigger_any_direction"),
        ("Cruce StochRSI K>D en la última vela", "stoch_cross_up"),
        ("StochRSI K y D ≤80", "stoch_not_overbought"),
        ("Cierre ≥ máximo de vela previa", "previous_high_break"),
        ("Setup 4h/1h permite LONG", "setup_long"),
        ("Impulso LONG completo (sin régimen/setup/vetos)", "long_momentum_all"),
        ("Gatillo LONG completo antes de vetos", "long_core_complete"),
    ]
    for label, key in labels:
        a, b = [audit["symbols"][s]["counts"][key]["fail"] for s in ("SMCI", "NVDA")]
        lines.append(f"| {label} | {a} | {b} | {a/299:.1%} / {b/299:.1%} | {(a+b)/598:.1%} |")
    lines += ["", "**No existe una puerta final llamada ‘vela de rebote’.** El texto heredado la menciona, pero",
              "`apply_hierarchy` sustituye el gatillo por cruce StochRSI + ruptura de vela previa + MACD + volumen + régimen/setup.",
              "Se muestran diagnósticos descriptivos aparte, sin atribuirles falsamente el bloqueo final.",
              "El veto de noticias/fundamentales no se puede contar: el replay no dispone de noticias punto-en-tiempo.",
              "Los fallos se superponen: no sumarlos ni interpretarlos como causas mutuamente excluyentes.",
              "", "## 2. Cuello de botella de las 11:00", ""]
    for symbol in ("SMCI", "NVDA"):
        baseline = json.loads(next((ROOT / "output/historical_replay/pilot_1h_1d").glob(f"{symbol}*.json")).read_text(encoding="utf-8"))
        old_rows = audit["symbols"][symbol]["rows"]
        joints = joint_counts(old_rows)
        ratios = [r["gates"]["inputs"]["volume"] / r["gates"]["inputs"]["previous20_volume_mean"] for r in old_rows]
        legacy = [legacy_rebound(c["feature_snapshot"]["features"],
                                  c["feature_snapshot"]["indicator_tail"]["intraday_indicators"]["values"])
                  for c in baseline["observations"]]
        diagnosis["symbols"][symbol] = {
            "baseline_joint_counts": joints,
            "volume_ratio_p10_p50_p90": np.percentile(ratios, [10, 50, 90]).tolist(),
            "legacy_rebound_watch_failures": sum(not a for a, b in legacy),
            "hammer_or_bullish_engulfing_failures": sum(not b for a, b in legacy),
        }
        lines += [f"### {symbol}", "",
                  f"- Volumen y cruce alcista coinciden en **{joints['pair_pass_counts']['volume_above_previous20 AND stoch_cross_up']}/299** cortes.",
                  f"- Ninguno cumple todo el impulso LONG. Volumen es el único requisito de impulso ausente en {joints['sole_missing_momentum_gate']['volume_above_previous20']} cortes; eso todavía NO autoriza entrar.",
                  f"- Ratio mediano de volumen: **{np.median(ratios):.3f}×**, frente al requisito >1.000×.",
                  f"- Permisos macro: {audit['symbols'][symbol]['macro_permissions']}.",
                  f"- Vigilancia de rebote heredada ausente: {sum(not a for a,b in legacy)}/299; martillo/envolvente alcista ausentes: {sum(not b for a,b in legacy)}/299. Son diagnósticos, no puertas finales.", ""]
    lines += ["La condición individual más bloqueante es el volumen, en ambos activos. También pesan el régimen y el cruce exacto.",
              "No basta con quitar una condición: se requiere coincidencia simultánea de varias en una única vela del corte.",
              "La media de volumen usa 20 velas anteriores, no 20 sesiones a la misma hora. A las 11:00 normales mezcla",
              "17 velas de esa mañana y 3 del cierre de la sesión anterior. Esto puede explicar parte del sesgo horario;",
              "es una hipótesis sobre el filtro, NO prueba de que deba relajarse.",
              "", "## 3. Preregistro y reloj", "",
              f"Registrado antes de ejecutar nuevos cortes: {protocol['registered_at']}. SHA-256: `{sha(protocol)}`.",
              "- Intradía: 10:00, 11:00, 12:30, 14:00, 15:30 NY, cada uno con 1h y 1D.",
              "- Diario 1D: 15:30 NY. Es la misma observación de intradía15:30/1D; no se cuenta dos veces.",
              "- Semanal 1S: 15:30 NY en la última sesión XNYS de cada semana completa (viernes normalmente; jueves si viernes festivo).",
              "- Si ese horario cae después de un cierre anticipado: OMITIR y registrar; no desplazarlo.",
              "- Contrato 1h: comienza en la próxima vela completa y vence después de 60 minutos negociados. A las 15:30 puede vencer en la siguiente sesión.",
              "- Contrato 1D: cierre de la siguiente sesión; 1S: cierre de la quinta sesión siguiente. No se redefine ‘diario’ como cierre del mismo día.",
              "- El núcleo de señales/indicadores y parámetros es el mismo del piloto. El reloj comparativo es un adaptador aislado de investigación, versionado y probado.",
              "- Producción no acepta sus contratos fuera de 11:00: NO se amplió el permiso en vivo ni se emitieron órdenes.",
              "", "## 4. Conteo operativo por horario y horizonte", "",
              "LONG/SHORT = permiso técnico bajo el reloj de investigación; no aprobación de modelo, permiso SHORT de broker ni ejecución real.",
              "TP/SL/timeout se cuentan solamente en la población autorizada con fill posible, nunca desde toques hipotéticos.",
              "", "| Activo | Corte NY / cohorte | Horizonte | Cortes | LONG | SHORT | Fills | TP | SL | Timeout | Ambiguos | Pendientes | Sin fill |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels_h = {"60": "1h", "1440": "1D", "10080": "1S"}
    for symbol, data in report["symbols"].items():
        for cohort, horizons in data["cohorts"].items():
            for horizon, row in horizons.items():
                lines.append(f"| {symbol} | {cohort} | {labels_h[horizon]} | " + " | ".join(str(row[k]) for k in
                    ("cuts", "long", "short", "possible_fills", "TP_FIRST", "SL_FIRST", "TIMEOUT", "ambiguous", "pending", "no_fill")) + " |")
        lines += ["", f"{symbol}: {data['unique_cuts']} cortes únicos; {len(data['skipped'])} horarios omitidos por cierre anticipado.", ""]
        completion = read_signed(output / f"{symbol}_completion.json")
        records = [read_signed(output / item["file"]) for item in completion["records"]]
        events = distinct_events(records)
        diagnosis["symbols"][symbol]["events"] = events
        diagnosis["symbols"][symbol]["distinct_authorized_events"] = len(events)
        diagnosis["symbols"][symbol]["nonoverlapping_authorized_events"] = sum(e["nonoverlapping_selected"] for e in events)
        long_gates = {clock: data["cohorts"][f"intraday_{clock.replace(':', '')}"]["60"]["gate_counts"] for clock in TIMES}
        diagnosis["symbols"][symbol]["counts_by_time"] = long_gates
        dataset = load_dataset(INPUT, symbol)
        ambiguity = {}
        for cohort in data["cohorts"]:
            ambiguity[cohort] = {}
            for horizon in data["cohorts"][cohort]:
                rows = [h for r in records if cohort in r["cohorts"] for h in r["horizons"]
                        if h["horizon_minutes"] == int(horizon)]
                ambiguity[cohort][horizon] = {
                    "hypothetical_tp_sl_same_bar_subset_of_conservative_sl": sum(same_bar_ambiguity(h, dataset, hypothetical=True) for h in rows),
                    "filled_tp_sl_same_bar_subset_of_conservative_sl": sum(same_bar_ambiguity(h, dataset, hypothetical=False) for h in rows
                        if (h["operational_contract"] or {}).get("eligible_at_emission"))}
        diagnosis["symbols"][symbol]["same_bar_ambiguity"] = ambiguity
    lines += ["", "## 5. Eventos compartidos y solapamiento", "",
              "| Activo | Entradas distintas (sin duplicar horizontes) | Ventanas no solapadas (embargo al timeout máximo) |",
              "|---|---:|---:|"]
    for s, d in diagnosis["symbols"].items():
        lines.append(f"| {s} | {d['distinct_authorized_events']} | {d['nonoverlapping_authorized_events']} |")
    lines += ["", "Un mismo fill puede resolver distintos horizontes: eso es UN evento con varias resoluciones, no varios aciertos.",
              "No se suman las tablas por horario como muestras independientes. El embargo usa el vencimiento completo, nunca la salida favorable observada.",
              "No solapado no significa estadísticamente IID.", "", "## 6. Barreras hipotéticas: conteo separado", "",
              "| Activo | Corte/cohorte | Horizonte | TP primero | SL primero | Timeout | Pendientes |",
              "|---|---|---|---:|---:|---:|---:|"]
    for symbol, data in report["symbols"].items():
        for cohort, horizons in data["cohorts"].items():
            for h, row in horizons.items():
                hy = row["hypothetical"]
                lines.append(f"| {symbol} | {cohort} | {labels_h[h]} | {hy['TP_FIRST']} | {hy['SL_FIRST']} | {hy['TIMEOUT']} | {row['hypothetical_pending']} |")
    lines += ["", "Estos no son fills ni operaciones. Se conserva SL conservador cuando ambas barreras están en una vela;",
              "la intravela de 5m no permite conocer el orden exacto de los ticks.",
              "La columna ‘Ambiguos’ operativa cuenta entrada/TP sin orden conocido (sin trade). El JSON diagnosis.json",
              "también separa velas TP+SL de orden desconocido que el etiquetador original incluye conservadoramente en SL; no son aciertos verificables.",
              "", "## 7. Qué bloquea después de los cinco horarios", "",
              "| Activo | Hora | Volumen falla | Cruce K>D falla | Ruptura previa falla | MACD falla | LONG_ONLY falla | Impulso completo falla |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, d in diagnosis["symbols"].items():
        for t, counts in d["counts_by_time"].items():
            lines.append(f"| {s} | {t} | " + " | ".join(str(counts[k]["fail"]) for k in
                ("volume_above_previous20", "stoch_cross_up", "previous_high_break", "macd_long", "regime_long_only", "long_momentum_all")) + " |")
        aggregate = {k: sum(c[k]["fail"] for c in d["counts_by_time"].values()) for k in
                     ("volume_above_previous20", "stoch_cross_up", "previous_high_break", "macd_long", "regime_long_only", "setup_long", "risk_veto_inactive")}
        d["aggregate_individual_gate_failures"] = aggregate
        d["most_blocking_individual_condition"] = max(aggregate, key=aggregate.get)
        lines += ["", f"**{s}:** condición individual con más fallos en los cinco cortes: `{d['most_blocking_individual_condition']}` ({max(aggregate.values())}).", ""]
    lines += ["## 8. Evidencia necesaria antes de tocar filtros", "",
              "1. Cobertura por activo, hora y régimen, incluidos candidatos cercanos al gatillo: registrar TODOS los candidatos, no solo ganadores.",
              "2. Perfil de volumen por hora y sesión comparable, calculado únicamente con sesiones anteriores: comprobar si la SMA20 discrimina participación o principalmente horario.",
              "3. Persistencia del cruce: medir cuándo se cumplen ruptura/MACD/volumen frente al instante del cruce; comprobar si exigir todo en una misma vela elimina casi toda cobertura.",
              "4. Justificación de LONG_ONLY versus BOTH_REDUCED: medir cobertura de ambos sin autorizar operaciones ni bajar filtros.",
              "5. Spread, liquidez, permiso SHORT, órdenes límite y fills observados; los OHLC son insuficientes para afirmar ejecución.",
              "6. Si se propone un cambio: hipótesis y parámetros preregistrados, una sola modificación por experimento en desarrollo; evaluación posterior en periodo no usado, con purga/embargo.",
              "   Esta auditoría mide selectividad, no demuestra que un filtro sea rentable ni que sea ‘demasiado estricto’ en términos de riesgo.",
              "", "## 9. Discrepancias conservadas, no corregidas aquí", "",
              "- UI/texto heredado de rebote no coincide con el gatillo final del núcleo causal.",
              "- Precio dentro de la zona no es puerta del gatillo; el fill simulado usa una orden límite al último cierre, no el rango dibujado de entrada.",
              "- Correlación cross-asset no disponible en los 299 cortes originales de cada activo: error de offsets de timezone en slicing. No se corrigió para mantener la versión comparada.",
              "- Fundamental/noticias y estado de cuenta histórico no disponibles. No se sustituye ausencia por seguridad.",
              "- Datos diarios de warmup limitados a la adquisición original del piloto; no se amplió el histórico ni se cambió la inicialización de indicadores.",
              "- El único cambio previo de fuente frente al piloto es la compatibilidad del validador de historical_replay; el núcleo de señales, TP/SL, fills y parámetros tiene los mismos hashes.",
              "", "## 10. Integridad", "",
              f"portfolio.db antes y después: `{report['ledger_after_sha256']}` (sin cambios).",
              "Artefactos aislados de las tablas de observaciones reales y de la contabilidad. SHA-256 verifica integridad, no firma externa del proveedor.",
              "Código del núcleo, scripts de investigación, datos y preregistro están inventariados/archivados. Cada corte tiene checksum y puede reanudarse."]
    diagnosis["diagnostic_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_signed(output / "diagnosis.json", diagnosis)
    path = output / "diagnostico.md"
    if path.exists():
        raise ValueError("Diagnóstico ya generado; no sobrescribir.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in d.items() if k in
          ("distinct_authorized_events", "nonoverlapping_authorized_events", "most_blocking_individual_condition", "aggregate_individual_gate_failures")}
          for s, d in diagnosis["symbols"].items()}, indent=2))


if __name__ == "__main__":
    main()
