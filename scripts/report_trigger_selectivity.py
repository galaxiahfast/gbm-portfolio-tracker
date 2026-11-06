"""Close signed diagnostic evidence into human-readable reports; never fit or trade."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_trigger_replay import OUTPUT, INPUT, TIMES, digest_file
from scripts.build_historical_replay import load_dataset
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.measure_trigger_selectivity import DEFAULT_OUTPUT, volume_summary

GATE_LABELS = {
    "inside_buy_zone_descriptive": "Precio dentro de zona LONG (solo descriptivo)",
    "bullish_closed_candle_descriptive": "Vela alcista cerrada (no es gate de rebote)",
    "volume_above_previous20": "Volumen > media de 20 velas anteriores",
    "macd_long": "MACD 5m ≥ señal",
    "regime_long_only": "Permiso LONG_ONLY",
    "risk_veto_inactive": "Veto técnico inactivo",
    "final_trigger_any_direction": "Gatillo final completo, cualquier dirección",
    "stoch_cross_up": "Cruce StochRSI K>D en la vela del corte",
    "stoch_not_overbought": "K y D ≤80",
    "previous_high_break": "Cierre ≥ máximo de vela previa",
    "setup_long": "Setup 4h/1h LONG",
    "long_momentum_all": "Impulso LONG completo antes de macro/setup",
    "long_core_complete": "Gatillo LONG completo antes de vetos",
}
HORIZONS = {"60": "1h", "1440": "1D", "10080": "1S"}


def table(headers, rows):
    clean = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "|" + "|".join("---" for _ in headers) + "|",
                       *("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)])


def number(value, places=2):
    return "N/D" if value is None else f"{value:.{places}f}"


def artifact(path, content):
    """Published evidence is append-only, including human-readable artifacts."""
    encoded = (content.rstrip() + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Artefacto cerrado diferente: {path}; usar nueva versión.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)


def same_clock_volume(symbol, cuts):
    """Supplementary RVOL, past 20 sessions at the SAME source-bar clock.

    This is a distribution only. It is never substituted into the trigger.
    No same-day total, future session or forward split factor is used.
    """
    data = load_dataset(INPUT, symbol).intraday
    clocks = data.index.tz_convert("America/New_York").strftime("%H:%M")
    baseline = data.Volume.groupby(clocks).transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    rows = []
    for cut in cuts:
        source = pd.Timestamp(cut["cut_at"]) - pd.Timedelta(minutes=5)
        mean = float(baseline.loc[source])
        if not mean > 0:
            raise ValueError("Faltan 20 sesiones previas comparables; no inventar denominador.")
        rows.append({"cut_at": cut["cut_at"], "time_ny": cut["time_ny"],
                     "source_bar_at": source.isoformat(), "volume": float(data.Volume.loc[source]),
                     "past20_same_clock_mean": mean,
                     "same_clock_ratio": float(data.Volume.loc[source]) / mean})
    return {"cuts": rows, "by_clock": {t: volume_summary([x["same_clock_ratio"] for x in rows if x["time_ny"] == t]) for t in TIMES}}


def proposal(report, protocol):
    return {
        "status": "PROPOSED_NOT_APPLIED_NOT_PREREGISTERED_FOR_EXECUTION",
        "created_at": report["created_at"],
        "reference_engine_sha256": protocol["engine_code_sha256"],
        "inspected_period_NOT_holdout": {"start": protocol["start"], "end": protocol["end"]},
        "unseen_internal_holdout_proposed": {"start": "2026-09-28", "end": "2026-12-31", "calendar": "XNYS",
            "open_once_after_last_contract_matures": True,
            "alternative": "Un bloque histórico no adquirido/evaluado antes, delimitado y sellado antes de acceso; demostrar ausencia de uso previo."},
        "symbols": ["SMCI", "NVDA"], "fixed_times_ny": list(TIMES), "horizons": [60, 1440],
        "not_training": True, "no_live_orders": True,
        "variants_one_at_a_time": [
            {"id": "V_VOLUME", "only_changed_condition": "volume_above_previous20",
             "proposal": "Volumen actual > media de 20 sesiones PREVIAS de la misma vela horaria; umbral 1.0 fijo. No modificar cruce, MACD, ruptura, permisos, vetos ni sizing."},
            {"id": "V_STOCH", "only_changed_condition": "stoch_cross_up",
             "proposal": "Cruce alcista en vela del corte O en la vela cerrada anterior, y K>D vigente. Cap ≤80 actual intacto. Tolerancia fija 1 vela; no explorar 2/3 tras abrir holdout."},
            {"id": "V_LONG_PERMISSION", "only_changed_condition": "regime_long_only",
             "proposal": "Puerta de unanimidad: ADX de tendencia intacto; S/D/4h sin voto negativo, al menos dos positivos y uno neutral. Resto intacto; no modificar clasificador de régimen. Exposición reducida existente si permiso original es BOTH_REDUCED."},
            {"id": "V_BOTH_PERMISSION", "only_changed_condition": "regime_long_only",
             "proposal": "Puerta de permiso: aceptar LONG también con permiso original BOTH_REDUCED y setup LONG existente. Resto intacto; mantener factor 0.25 ya previsto. SHORT no habilitado en Trading USA."},
        ],
        "controls": [
            "C0: gatillo original intacto, con mismos datos, reloj, órdenes límite, costes y barreras.",
            "C1 simple: cierre rompe máximo previo, bajo mismos controles de riesgo, permiso macro original, setup, brackets y fill. Sin Stoch/MACD/volumen. Es comparador, no variante de una sola condición ni propuesta de despliegue.",
            "C2: no entrada; cobertura y exposición cero como control negativo.",
        ],
        "independence": "Agrupar por activo+corte+lado+entrada+TP+SL; 1h/1D son resoluciones del mismo evento. Purga y embargo hasta timeout máximo, nunca hasta salida favorable. Selección cronológica fija; solapados reportados aparte.",
        "quotes_and_pit": "Registrar NBBO al envío y trayectoria de la orden; tamaños, prints, latencia y fills del broker si disponibles. Sin ellos ejecución N/D. Noticias históricas ausentes => no certificar autorización de riesgo real.",
        "endpoints": ["Cobertura: cortes y eventos elegibles no solapados", "Fills posibles / no fill / ambiguos / pendientes",
                      "Costes por acción, spread/slippage y pérdida por gaps", "Excursión adversa antes de salida y exposición; distribuciones, no precisión"],
        "minimum_evidence_proposed": "30 eventos LONG no solapados por activo/variante como mínimo DESCRIPTIVO, no validación comercial. Menos => inconcluso; no ampliar fechas ni modificar filtros después de ver resultados.",
        "analysis_lock": "Congelar fechas, endpoints y variantes antes de generar salidas; abrir holdout una vez. Bootstrap por bloques de sesión/semana para IC cuando haya soporte; ajuste Holm para cuatro comparaciones si se realizan pruebas. No elegir mejor hora ni parámetros con este periodo.",
        "decision_boundary": "Más entradas no prueba mejor protección. Sin costes/spread observados y cobertura de riesgo PIT no concluir que una variante es operable. No aplicar hasta nueva autorización del usuario.",
    }


def per_cut_report(measurements):
    out = ["# Desfase por corte fallido — medición descriptiva",
           "Cada edad se mide en velas cerradas de 5m hasta el corte. Se usa la última confirmación verdadera de cada condición, dentro de la misma sesión y una ventana censurada de 60 velas.",
           "N/D significa ausencia de alguna confirmación dentro de la ventana, no desfase cero. No se arrastra macro, veto ni sobrecompra de velas anteriores. Tol=1/2/3 informa si pasaría el shadow COMPLETO; no fill ni operación autorizada."]
    for symbol, payload in measurements.items():
        rows = []
        for r in payload["cuts"]:
            if not r["failed_original_trigger"]:
                continue
            lag = r["selected_side_lag"]
            a = lag["ages_in_bars"]
            local = pd.Timestamp(r["cut_at"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M %Z")
            rows.append([local, r["selected_side"], *(number(a[k], 0) for k in ("volume", "stoch_cross", "macd", "previous_bar_break")),
                         number(lag["span_first_to_last_bars"], 0), number(lag["oldest_confirmation_age_bars"], 0),
                         "/".join("Sí" if r["full_shadow_by_tolerance"][str(k)] else "No" for k in (1, 2, 3)),
                         ", ".join(lag["missing_in_lookback"]) or "—"])
        out.extend([f"## {symbol}: {len(rows)} cortes fallidos", table(
            ["Corte NY", "Lado del plan", "Edad vol", "Edad Stoch", "Edad MACD", "Edad ruptura", "Desfase", "Edad máxima", "Tol 1/2/3", "Ausentes"], rows)])
    return "\n\n".join(out)


def render(report, diagnosis, protocol, method, comparable, experiment):
    closed = report["closed_counts"]["symbols"]
    lines = ["# Auditoría cerrada de selectividad del gatillo",
             f"Fecha de medición UTC: {report['created_at']}. Periodo: {protocol['start']} a {protocol['end']}.",
             "SMCI y NVDA. Mismo histórico firmado, mismos filtros, parámetros, niveles y motor del replay. No entrenamiento, órdenes, cambios de producción ni métricas de precisión. Artefactos fuera de contabilidad.",
             "## 1. Diagnóstico directo",
             "Hay selectividad extrema y dependencia del horario, pero todavía no evidencia suficiente para afirmar que el filtro protege correctamente o es excesivo en riesgo económico. La sincronización importa; no explica por sí sola toda la falta de entradas. La población operable en Trading USA es menor que la población técnicamente SHORT."]
    summary = []
    for s, v in report["symbols"].items():
        a = v["lag_overall"]
        summary.append([s, a["cuts"], a["full_shadow_all_cuts"]["0"], *(a["full_shadow_all_cuts"][str(k)] for k in (1, 2, 3)),
                        " / ".join(str(a["full_shadow_sides"][str(k)].get("LONG", 0)) for k in (0, 1, 2, 3))])
    lines.append(table(["Activo", "Cortes únicos", "Actual", "Shadow tol1", "Shadow tol2", "Shadow tol3", "LONG actual/1/2/3"], summary))
    lines += ["Shadow = conteo descriptivo de cortes, no ejecución de una variante ni replay de sus resultados. Con 3 velas, SMCI pasa de 6 a 15 cortes (LONG de 1 a 3); NVDA de 0 a 4 LONG. No se evalúan fills ni TP/SL de esos nuevos shadow.",
              "## 2. Conteo por horario y horizonte (cerrado)",
              "Permiso técnico no implica modelo aprobado, noticias PIT disponibles ni autorización del broker. TP/SL/timeout se cuentan aquí SOLO con elegibilidad técnica y fill OHLC posible."]
    for s in closed:
        rows = []
        for cohort, horizons in closed[s]["cohorts"].items():
            for h, d in horizons.items():
                rows.append([cohort, HORIZONS[h], d["cuts"], d["long"], d["short"], d["possible_fills"],
                             d["TP_FIRST"], d["SL_FIRST"], d["TIMEOUT"], d["ambiguous"], d["pending"], d["no_fill"], d["missing_evidence"]])
        lines += [f"### {s}", table(["Cohorte NY", "Horizonte", "Cortes", "LONG", "SHORT", "Fill posible", "TP", "SL", "Timeout", "Ambiguos", "Pendientes", "Sin fill", "Evidencia incompleta"], rows)]
    lines += ["Por activo: 1,491 cortes únicos; 299 por 10:00/11:00/12:30 y 297 por 14:00/15:30. Los cuatro cortes omitidos corresponden a los cierres anticipados del 28/11/2025 y 24/12/2025, sin desplazarlos.",
              "La cohorte diaria 15:30/1D es la MISMA observación intradía 15:30/1D, no otra entrada. Semanal: 61 cortes de última sesión XNYS de semana completa, 15:30.",
              "Contrato conservado: 1h comienza en la siguiente vela completa (11:05 si corte 11:00) y suma 60 minutos negociados. 1D vence al cierre de la SIGUIENTE sesión; no es cierre de hoy. 1S vence a quinta sesión siguiente. Se conserva esta definición, no se corrige ni reoptimiza aquí.",
              "Reloj adicional = adaptador aislado de investigación. Producción mantiene su corte validado 11:00 y no acepta automáticamente contratos de este experimento.",
              "## 3. Eventos agrupados y toques hipotéticos separados"]
    lines.append(table(["Activo", "Entradas/eventos distintos", "No solapados hasta timeout máximo", "LONG", "SHORT"],
        [[s, diagnosis["symbols"][s]["distinct_authorized_events"], diagnosis["symbols"][s]["nonoverlapping_authorized_events"],
          sum(e["side"] == "LONG" for e in diagnosis["symbols"][s]["events"]), sum(e["side"] == "SHORT" for e in diagnosis["symbols"][s]["events"])] for s in closed]))
    lines += ["SMCI: seis eventos con dos resoluciones por evento, no doce operaciones. El evento de 31/12/2025 tiene timeout en 1h y TP primero en 1D: no convertirlo en dos eventos independientes. El embargo se basa en el timeout máximo, nunca en una salida favorable temprana. No solapamiento no prueba independencia IID."]
    rows = []
    for s in closed:
        for cohort, horizons in closed[s]["cohorts"].items():
            for h, d in horizons.items():
                hyp = d["hypothetical"]
                ambiguous = diagnosis["symbols"][s]["same_bar_ambiguity"][cohort][h]["hypothetical_tp_sl_same_bar_subset_of_conservative_sl"]
                rows.append([s, cohort, HORIZONS[h], hyp.get("TP_FIRST", 0), hyp.get("SL_FIRST", 0), hyp.get("TIMEOUT", 0), d["hypothetical_pending"], ambiguous])
    lines += [table(["Activo", "Cohorte", "H", "TP hipotético", "SL hipotético", "Timeout", "Pendientes", "TP+SL misma vela (subconjunto SL)"], rows),
              "Toques de barreras NO son entradas/fills. TP+SL en la misma vela no permite conocer orden: el núcleo conserva SL conservador; la última columna lo identifica sin sumarlo dos veces. No hay ambigüedad TP+SL en los seis eventos elegibles.",
              "## 4. Fallos por condición: independientes, no sumar porcentajes",
              "Estas puertas son LONG salvo la fila explícita de gatillo final de cualquier dirección. No atribuir a los cinco SHORT fallos de un cruce alcista.",
              "### Corte original 11:00 — 299 por activo"]
    rows = []
    for key, label in GATE_LABELS.items():
        a, b = [closed[s]["cohorts"]["intraday_1100"]["60"]["gate_counts"][key]["fail"] for s in closed]
        rows.append([label, a, b, f"{100*a/299:.1f}% / {100*b/299:.1f}%", f"{100*(a+b)/598:.1f}%"])
    lines.append(table(["Condición", "SMCI fallos", "NVDA fallos", "% por activo", "% total 598"], rows))
    lines += ["No existe una puerta final ‘vela de rebote’. Zona dibujada y cuerpo alcista son descriptivos; la jerarquía efectiva usa cruce, MACD, ruptura, volumen, macro/setup y vetos. No inventar ese gate para explicar bloqueos.",
              "Veto contabilizado = técnico disponible. Fundamental/noticias históricos punto-en-tiempo ausentes: no confundir ‘no disponible’ con veto de riesgo real inactivo.",
              "### Todos los horarios — 1,491 por activo (sin duplicar horizontes)"]
    rows = []
    for key in GATE_LABELS:
        vals = [sum(closed[s]["cohorts"][f"intraday_{t.replace(':', '')}"]["60"]["gate_counts"][key]["fail"] for t in TIMES) for s in closed]
        rows.append([GATE_LABELS[key], *vals, f"{100*vals[0]/1491:.1f}% / {100*vals[1]/1491:.1f}%"])
    lines.append(table(["Condición", "SMCI fallos", "NVDA fallos", "% por activo"], rows))
    lines += ["### Fallos LONG por hora (misma población, condiciones superpuestas)"]
    rows = []
    for s in closed:
        for t in TIMES:
            d = closed[s]["cohorts"][f"intraday_{t.replace(':', '')}"]["60"]
            rows.append([s, t, d["cuts"], *(d["gate_counts"][k]["fail"] for k in ("volume_above_previous20", "stoch_cross_up", "macd_long", "previous_high_break", "regime_long_only", "setup_long", "risk_veto_inactive"))])
    lines += [table(["Activo", "NY", "n", "Volumen", "Stoch", "MACD", "Ruptura", "LONG_ONLY", "Setup", "Veto"], rows),
              "A las 11:00 el volumen es la condición individual más bloqueante; en el conjunto de horarios son LONG_ONLY y el cruce exacto, casi empatados. Frecuencia de fallo marginal no mide daño evitado ni demuestra cuál debería quitarse.",
              "## 5. Desfase entre confirmaciones (solo lectura)",
              "Cuatro confirmaciones temporales: volumen, cruce Stoch, MACD y ruptura de vela previa, en el LADO DEL PLAN ORIGINAL. Para cada una se localiza la última vela cerrada que la cumplió antes del corte. Desfase = edad máxima menos mínima; se informa también edad máxima para no reciclar una secuencia compacta pero antigua.",
              "Ventana descriptiva: últimas 60 velas, censurada al inicio de la misma sesión. A las 10:00 solo existen seis velas de hoy: si falta una confirmación es N/D, no se presta del cierre anterior. No implica que todas las confirmaciones ‘primarias’ fueran simultáneamente vigentes durante ese lapso.",
              "Tol k exige que TODAS estén entre corte−k velas y el corte, inclusive. Macro, setup, veto/rechazo, lado del plan, bracket, score mínimo y cap Stoch actual permanecen como estaban en el corte. Valores anteriores de MACD pueden haberse revertido: este shadow NO autoriza una compra.",
              "Control: tol0 reproduce exactamente elegibilidad original (6 SMCI, 0 NVDA), y cada corte reconstruido reproduce indicadores/volumen del snapshot firmado."]
    rows = []
    for s, payload in report["symbols"].items():
        for t, d in payload["by_clock"].items():
            l = d["lag"]
            rows.append([s, t, l["failed_trigger"], l["span_observable"], l["span_unavailable_missing_confirmation"],
                         "/".join(number(x, 0) for x in l["span_p25_median_p75"]),
                         "/".join(str(l["micro_shadow_all_cuts"][str(k)]) for k in (1, 2, 3)),
                         "/".join(str(l["full_shadow_all_cuts"][str(k)]) for k in (1, 2, 3)),
                         "/".join(str(l["additional_failed_cuts_full_shadow"][str(k)]) for k in (1, 2, 3))])
    lines += [table(["Activo", "NY", "Gatillo falló", "Desfase medible", "N/D", "P25/mediana/P75 velas", "Solo micro tol1/2/3", "Completo tol1/2/3", "Nuevos entre fallidos 1/2/3"], rows),
              "La tabla completa de cada corte fallido está en desfase_por_corte.md; edades, tiempos exactos UTC, dirección, todas las puertas congeladas y ambos lados de diagnóstico están en SMCI_measurements.json y NVDA_measurements.json. No son eventos resueltos nuevos.",
              "## 6. Volumen por horario: distribución y comparabilidad",
              "Ratio operativo = volumen de última vela cerrada / media de las 20 velas anteriores (excluye la actual). P(≤1) ubica empíricamente el umbral: no es una probabilidad financiera. El > es estricto."]
    rows = []
    for s, payload in report["symbols"].items():
        for t, d in payload["by_clock"].items():
            v = d["volume"]
            rows.append([s, t, v["n"], number(v["p25"], 3), number(v["median"], 3), number(v["p75"], 3),
                         f"P{v['threshold1_empirical_percentile']:.1f}", v["above1"], f"{100*v['above1']/v['n']:.1f}%"])
    lines += [table(["Activo", "NY", "n", "P25", "Mediana", "P75", "Umbral 1.0 cae en", "Pasan >1", "% pasan"], rows),
              "A las 11:00, >1 pide aproximadamente el percentil 89.6 en SMCI y 95.7 en NVDA. A las 15:30 cae cerca del percentil 34–35. Es un filtro fuertemente condicionado por horario, no un criterio uniforme de participación.",
              "En sesión normal, a las 11:00 el denominador mezcla 17 velas de la mañana con 3 del cierre anterior; a las 10:00 mezcla 5 de la mañana y 15 del cierre anterior. Esto compara segmentos con perfiles de actividad distintos. Su selectividad sí está medida; su utilidad protectora aún no.",
              "### Comparación adicional: 20 sesiones PREVIAS de la misma vela horaria",
              "Solo distribución descriptiva: NO se sustituyó el denominador del gatillo ni se generaron entradas con este ratio. Cada baseline usa 20 sesiones anteriores, sin incluir la actual."]
    rows = []
    for s, payload in comparable.items():
        for t, v in payload["by_clock"].items():
            rows.append([s, t, number(v["p25"], 3), number(v["median"], 3), number(v["p75"], 3),
                         f"P{v['threshold1_empirical_percentile']:.1f}", v["above1"], v["n"]])
    lines += [table(["Activo", "NY", "P25 comparable", "Mediana", "P75", "Posición de 1.0", "Sobre media comparable", "n"], rows),
              "No elegir un umbral de esta tabla por cobertura deseada. La diferencia de perfiles justifica medir una alternativa predefinida en otro periodo, no aprobarla con estos datos.",
              "## 7. Las seis entradas SMCI: fills y costes",
              "El histórico adquirido solo tiene OHLCV; no contiene bid/ask, NBBO, tamaños de cola, latencia de orden ni comprobantes de ejecución de estos eventos hipotéticos. Fill OHLC posible ≠ fill observado. Spread real y fill con spread real = N/D en las seis entradas.",
              "Según la descripción pública actual de [Trading USA de GBM](https://gbm.com/trading/), no se abren cortos propios en ese producto. Se separan cinco SHORT técnicamente elegibles pero NO ejecutables allí. No equivale a una venta de acciones ya poseídas ni certifica permiso histórico de 2025.",
              "La [tarifa pública actual de GBM](https://gbm.com/faqs/existe-alguna-comision-por-invertir-en-trading-usa) es 0.25% por cada lado. Coincide con el supuesto congelado del replay, no prueba cargos pagados en esas fechas. Impuestos, aranceles, préstamo y FX no se inventan ni se incluyen como observados.",
              "[Massive Basic](https://massive.com/pricing) no incluye Quotes; el piloto descargó agregados, no se adquirieron cotizaciones ni plan adicional."]
    rows = []
    cost_rows = []
    for e in report["fill_evidence"]:
        cut = pd.Timestamp(e["cut_at"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")
        first = e["resolutions"][0]
        fill_at = pd.Timestamp(first["fill_at_possible_OHLC"]).tz_convert("America/New_York").strftime("%H:%M")
        rows.append([cut, e["side"], number(e["limit_price"], 4), number(first["fill_price_possible_OHLC"], 4), fill_at,
                     "No (SHORT)" if e["side"] == "SHORT" else "LONG; fill no confirmado", "N/D", "N/D"])
        for h in e["resolutions"]:
            c = h["costs_per_share_assumed"]
            cost_rows.append([cut, HORIZONS[str(h["horizon_minutes"])] , number(h["exit_price_simulated"], 4),
                              number(c["commission_entry_assumed"], 4), number(c["commission_exit_assumed"], 4),
                              number(c["commission_roundtrip_assumed"], 4), number(c["spread_roundtrip_assumed_NOT_observed"], 4),
                              number(c["slippage_roundtrip_assumed_NOT_observed"], 4)])
    lines += [table(["Corte NY", "Lado", "Límite simulado USD", "Fill posible OHLC", "Hora fill NY", "Trading USA", "Spread real", "Fill broker"], rows),
              "### Coste hipotético por UNA acción, USD (no pagado)",
              table(["Corte NY", "H", "Salida simulada", "Com. entrada", "Com. salida", "Com. ida/vuelta", "Spread supuesto", "Slippage supuesto"], cost_rows),
              "Comisión = (fill + salida) ×0.0025. Spread supuesto = (fill + salida) ×0.0002 (4 bps completos, mitad por lado); slippage = (fill + salida) ×0.0005. Se mantienen estos supuestos originales; no se atribuyen al mercado observado. Multiplicar por tamaño hipotético solo si se declara; el replay no conoce tamaño real de orden.",
              "Para cerrar ejecución real hacen falta cotizaciones NBBO y prints de cada ventana de orden, tamaños/liquidez, retraso de envío, condiciones de la orden y confirmación del broker. Una única cotización al corte tampoco demuestra el fill posterior ni el coste realizado.",
              "## 8. Propuesta de experimento controlado — NO aplicada",
              "Estado: propuesta diseñada, sin variantes ejecutadas, sin parámetros modificados ni entrenamiento. El shadow de desfase de esta auditoría es solo descriptivo y no selecciona la variante que ‘ganó’.",
              "### Variantes aisladas (comparar cada una con C0)"]
    lines.append(table(["Variante", "Única condición", "Cambio propuesto, no implementado"], [[v["id"], v["only_changed_condition"], v["proposal"]] for v in experiment["variants_one_at_a_time"]]))
    lines += ["V_LONG_PERMISSION y V_BOTH_PERMISSION son alternativas de la MISMA puerta macro, no se combinan. Todos los demás filtros, capital, riesgo y costes siguen congelados. Relajar Stoch solo en el cruce no modifica la sobrecompra de seguridad.",
              "### Periodo no usado y controles",
              "Todo el periodo 21/07/2025–25/09/2026 ya fue inspeccionado. Separar ahora su último 20% NO crea un holdout virgen. Propuesta: bloque interno sellado 28/09/2026–31/12/2026, abrirlo una vez solo tras madurar el último contrato. Esto no es una instrucción de esperar para esta auditoría: la auditoría actual está cerrada.",
              "Alternativa retrospectiva para acelerar: fijar y sellar otro bloque de 5m nunca adquirido/evaluado, demostrarlo con inventario de accesos, antes de descargarlo y consultar sus resultados. No se afirma aquí que un periodo anterior conocido sea automáticamente virgen. No hay compra/descarga adicional autorizada ni realizada en esta tarea.",
              "Horarios inalterados: 10:00,11:00,12:30,14:00,15:30 NY; 1h/1D; mismas reglas de omisión y vencimiento. No seleccionar hora después de conocer salidas.",
              "\n\n".join(experiment["controls"]), experiment["independence"],
              "Noticias PIT ausentes se reportan como límite de elegibilidad operativa, no se imputan con noticias actuales. Sin NBBO/fills se comparan candidatos y fills OHLC posibles, sin proclamar ejecución real.",
              "### Métricas y criterio de conclusión",
              "Endpoints fijados: cobertura/eventos LONG no solapados por activo, fills/no-fill/ambiguos/pendientes, costes observados cuando disponibles, pérdidas por gaps y excursión adversa/exposición. Reportar los cuatro brazos también si generan cero candidatos.",
              experiment["minimum_evidence_proposed"], experiment["analysis_lock"], experiment["decision_boundary"],
              "## 9. Límites que siguen abiertos",
              "No es válido concluir ‘el filtro protege’ por falta de operaciones ni ‘es excesivo’ por pocas entradas. Faltan una comparación temporal independiente y ejecución/costes medidos. Solo seis eventos técnicos y un LONG no permiten esa conclusión.",
              "No hay históricos PIT de fundamentales/noticias/capital/estado de cuenta; permiso SHORT Trading USA no disponible. Cross-asset quedó no disponible por la discrepancia de timezone de la versión auditada. No se corrige aquí, porque cambiaría la comparación. ‘Vela de rebote’ heredada no describe el gatillo final.",
              "## 10. Integridad, artefactos y reproducción",
              f"portfolio.db antes/después: `{method['ledger_before_sha256']}` / `{report['ledger_after_sha256']}` (idénticos).",
              f"Preregistro de horarios: `{report['closed_counts']['protocol_sha256']}`. Método de medición: `{report['method_sha256']}`.",
              "Todos los cortes y medidas usan envolturas con SHA-256 verificable. Es checksum de integridad, no firma digital del proveedor ni certificación de precios. Código del motor intacto; archivos de diagnóstico nuevos y outputs separados.",
              "Artefactos: reporte_cerrado.md (este), desfase_por_corte.md, measurement_report.json, measurement_method.json, SMCI_measurements.json, NVDA_measurements.json, same_clock_volume.json, experimento_propuesto.json y closure_manifest.json.",
              "Reproducir verificación y reporte: `.venv\\Scripts\\python.exe scripts/report_trigger_selectivity.py`. La medición se obtuvo con `scripts/measure_trigger_selectivity.py`; el resultado cerrado es append-only y no debe sobrescribirse con nueva fecha."]
    return "\n\n".join(lines)


def run(output=DEFAULT_OUTPUT):
    report = read_signed(output / "measurement_report.json")
    method = read_signed(output / "measurement_method.json")
    protocol = read_signed(OUTPUT / "preregistration.json")
    diagnosis = read_signed(OUTPUT / "diagnosis.json")
    if report["method_sha256"] != sha(method):
        raise ValueError("Reporte no corresponde al método.")
    for path, expected in protocol["engine_code_sha256"].items():
        if digest_file(ROOT / path) != expected:
            raise ValueError(f"Motor congelado modificado: {path}")
    if digest_file(ROOT / "data/portfolio.db") != method["ledger_before_sha256"]:
        raise ValueError("Ledger actual difiere del previo; detener sin abrir SQL.")
    measurements = {s: read_signed(output / f"{s}_measurements.json") for s in protocol["symbols"]}
    for s, data in measurements.items():
        if data["method_sha256"] != report["method_sha256"] or len(data["cuts"]) != report["symbols"][s]["lag_overall"]["cuts"]:
            raise ValueError("Desfase individual no corresponde al resumen.")
    comparison_path = output / "same_clock_volume.json"
    if comparison_path.exists():
        supplementary = read_signed(comparison_path)
        if supplementary["measurement_method_sha256"] != report["method_sha256"]:
            raise ValueError("Comparación horaria de otro método.")
    else:
        supplementary = {"measurement_method_sha256": report["method_sha256"],
                         "report_script_sha256": digest_file(Path(__file__)),
                         "method": "20 PREVIOUS SESSIONS SAME 5m SOURCE BAR CLOCK; DESCRIPTIVE ONLY NO GATE REPLACEMENT",
                         "symbols": {s: same_clock_volume(s, p["cuts"]) for s, p in measurements.items()}}
        write_signed(comparison_path, supplementary)
    experiment = proposal(report, protocol)
    write_signed(output / "experimento_propuesto.json", experiment)
    artifact(output / "desfase_por_corte.md", per_cut_report(measurements))
    artifact(output / "reporte_cerrado.md", render(report, diagnosis, protocol, method, supplementary["symbols"], experiment))
    archive = output / "source_archive"
    files = [Path(__file__), ROOT / "scripts/measure_trigger_selectivity.py", ROOT / "tests/test_trigger_selectivity_measurement.py"]
    for file in files:
        target = archive / digest_file(file) / file.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    names = ["reporte_cerrado.md", "desfase_por_corte.md", "measurement_report.json", "measurement_method.json", "SMCI_measurements.json", "NVDA_measurements.json", "same_clock_volume.json", "experimento_propuesto.json"]
    closure = {"version": "TRIGGER_DIAGNOSTIC_CLOSURE_V1", "created_at": report["created_at"],
               "files_sha256": {name: digest_file(output / name) for name in names},
               "source_sha256": {str(f.relative_to(ROOT)): digest_file(f) for f in files},
               "engine_unchanged": True, "no_training_or_live_orders": True,
               "portfolio_before_and_after_sha256": digest_file(ROOT / "data/portfolio.db")}
    write_signed(output / "closure_manifest.json", closure)
    print("Reporte cerrado:", output / "reporte_cerrado.md")
    print("Desfase por corte:", output / "desfase_por_corte.md")
    print("Ledger y motor: hashes idénticos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
