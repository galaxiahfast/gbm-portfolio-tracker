"""Report all locked replay arms after final holdout is consumed; no promotion."""
from collections import Counter
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import TIMES, digest_file
from scripts.seal_trigger_experiment_holdout import OUTPUT
from scripts.run_trigger_experiment import validate_environment
from scripts.trigger_experiment_rules import VARIANTS
from scripts.report_trigger_selectivity import table, artifact, number


def run():
    p = read_signed(OUTPUT / "preregistration.json")
    validate_environment(p)
    seal = read_signed(OUTPUT / "holdout_seal.json")
    opening = read_signed(OUTPUT / "holdout_opening.json")
    stages = {s:read_signed(OUTPUT / f"{s}_complete.json") for s in ("development","holdout")}
    if opening["development_sha256"] != sha(stages["development"]):
        raise ValueError("Holdout opened before this development closure.")
    for s,d in stages.items():
        if d["errors"] or d["protocol_sha256"] != sha(p):
            raise ValueError("Incomplete/different experimental stage.")
        for symbol,obj in d["symbols"].items():
            for f in obj["checkpoint_files"]:
                if digest_file(OUTPUT / f["path"]) != f["sha256"]:
                    raise ValueError("Altered execution checkpoint.")
    lines = ["# Experimento controlado del gatillo — replay aislado",
             f"Cerrado UTC: {stages['holdout']['completed_at']}. Protocolo: `{sha(p)}`.",
             "No entrenamiento, calibración, precisión, órdenes ni modificación del gatillo de producción. portfolio.db no abierto; solo checksum antes/después.",
             "## 1. Diseño fijado antes de evaluar",
             f"Preregistro: {p['registered_at']}. Holdout interno 01/07/2024–30/08/2024, nuevo bloque 5m fuera de inputs/replays inventariados. Se adquirió con Alpaca SIP, coste incremental USD0, sin contratación.",
             "Es un holdout retrospectivo anterior al desarrollo, elegido por fechas antes de descargar resultados. No se presenta como validación walk-forward cronológica ni como prueba de precisión. El inventario demuestra ausencia de replay 5m local en ese bloque, no que nadie hubiera visto su gráfico diario.",
             f"Sello AES256GCM + DPAPI: {seal['sealed_at']}; SHA `{sha(seal)}`. Apertura registrada: {opening['opened_at']}, DESPUÉS de cerrar el desarrollo {stages['development']['completed_at']}. Todas las variantes bloqueadas y evaluadas juntas al abrir; sin seleccionar ganadoras intermedias.",
             "El periodo previamente auditado 21/07/2025–25/09/2026 es desarrollo exploratorio, nunca holdout. Este experimento NO recicla su último 20% como prueba independiente.",
             "Horarios: 10:00, 11:00,12:30,14:00,15:30 NY; 1h y 1D. Cierre anticipado se omite. Se conserva siguiente vela completa; 1h son 60 minutos negociados y 1D vence al cierre de la siguiente sesión.",
             table(["Brazo", "Única modificación / control"], list(p["variants"].items())),
             "P50 significa volumen por encima de la mediana de las 20 sesiones PREVIAS de ESA misma vela horaria. Cambia el umbral de volumen según activo/hora/historia; no usa un ratio fijo >1 ni datos del día siguiente. La regla y el percentil no se eligieron después de ver entradas.",
             "Se conserva siempre el lado del plan original, su entrada/TP/SL, setup, vetos/rechazo, score mínimo 65, sobrecompra LONG y costes. Sin plan no se inventa operación. BOTH_REDUCED admite ambos lados solo en investigación, con exposición máxima 0.25; SHORT sigue no ejecutable en Trading USA.",
             "## 2. Datos y cierre del holdout"]
    lines.append(table(["Activo", "5m regulares adquiridas", "Sesiones adquiridas", "Diarias calentamiento", "Huecos", "Cortes desarrollo", "Cortes holdout"],
                       [[s,seal["audits"][s]["regular_bars"],seal["audits"][s]["sessions"],seal["audits"][s]["daily_sessions"],seal["audits"][s]["missing_regular_bars"],
                         stages["development"]["symbols"][s]["processed_cuts"],stages["holdout"]["symbols"][s]["processed_cuts"]] for s in p["symbols"]]))
    lines += ["XNYS y America/New_York verificados. 5m adquiridas mayo–04/09/2024; indicadores diarios desde mayo2023; resto solo calentamiento/vencimientos, no cortes de prueba fuera del bloque.",
              "Se pidió RAW. NVDA: split 10:1 del 10/06/2024, anterior a TODOS los cortes del holdout; se ajustó calentamiento anterior para mantener la misma base. SMCI se conserva en precios reales pre-split: NO se aplica el split futuro de septiembre2024. Los precios absolutos de SMCI2024 y2025 no se comparan como si fueran la misma unidad.",
              "## 3. Tabla comparativa principal — eventos NO solapados, horizonte 1D",
              "Se agrupan 1h/1D de la misma entrada. Se elige cronológicamente el primer evento por activo y brazo; embargo hasta el timeout máximo de ambos horizontes, no hasta una salida favorable observada. LONG/SHORT aquí son eventos técnicamente autorizados por el brazo, no operaciones reales."]
    for stage, content in stages.items():
        rows=[]
        for symbol,d in content["symbols"].items():
            base=d["variants"]["CURRENT"]["independent"]["1440"]
            simple=d["variants"]["SIMPLE_EMA_CROSS"]["independent"]["1440"]
            for v,obj in d["variants"].items():
                r=obj["independent"]["1440"]
                rows.append([symbol,v,r["LONG"],r["SHORT"],r["possible_fills"],f"{r['TP_FIRST']}/{r['SL_FIRST']}/{r['TIMEOUT']}",
                             r["ambiguous"],r["no_fill"],r["pending"],f"{r['LONG']-base['LONG']:+d} LONG",f"{r['LONG']-simple['LONG']:+d} LONG"])
        lines += [f"### {stage}: {p[stage]['start']} a {p[stage]['end']}",
                  table(["Activo","Variante","LONG","SHORT","Fills OHLC","TP/SL/timeout","Ambiguos","Sin fill","Pendientes","vs actual","vs EMA simple"],rows)]
    lines += ["## 4. Horizonte 1h — resoluciones de los MISMOS eventos, no más muestras"]
    rows=[]
    for stage,content in stages.items():
        for symbol,d in content["symbols"].items():
            for v,obj in d["variants"].items():
                r=obj["independent"]["60"]
                rows.append([stage,symbol,v,r["LONG"],r["SHORT"],r["possible_fills"],r["TP_FIRST"],r["SL_FIRST"],r["TIMEOUT"],r["ambiguous"],r["no_fill"],r["pending"])
    lines.append(table(["Bloque","Activo","Variante","LONG","SHORT","Fills OHLC","TP","SL","Timeout","Ambiguos","Sin fill","Pendientes"],rows))
    lines += ["## 5. Cobertura bruta antes de embargo (NO independiente)"]
    rows=[]
    for stage,content in stages.items():
        for symbol,d in content["symbols"].items():
            for v,obj in d["variants"].items():
                r=obj["raw"]["1440"]
                rows.append([stage,symbol,v,r["LONG"],r["SHORT"],obj["independent_events"],obj["embargo_excluded"],r["possible_fills"],f"{r['TP_FIRST']}/{r['SL_FIRST']}/{r['TIMEOUT']}"])
    lines.append(table(["Bloque","Activo","Variante","LONG brutos","SHORT brutos","Eventos no solapados","Excluidos embargo","Fills brutos","TP/SL/timeout brutos"],rows))
    lines += ["### Desglose bruto por horario — ambos horizontes en JSON"]
    rows=[]
    blockers={}
    for stage,content in stages.items():
        blockers[stage]={}
        for symbol,d in content["symbols"].items():
            cuts=[]
            for item in d["checkpoint_files"]:
                cuts.extend(read_signed(OUTPUT/item["path"])["cuts"])
            native_long=[r["variants"]["CURRENT"] for r in cuts if r["variants"]["CURRENT"]["side"]=="LONG"]
            blockers[stage][symbol]={"LONG_plan_cuts":len(native_long),"failures_overlapping":dict(Counter(k for g in native_long for k in g.get("failed_conditions",[])))}
            for clock in TIMES:
                subset=[r for r in cuts if r["time_ny"]==clock]
                for v in VARIANTS:
                    gs=[r["variants"][v] for r in subset if r["variants"][v]["eligible"]]
                    rows.append([stage,symbol,clock,v,len(subset),sum(g["side"]=="LONG" for g in gs),sum(g["side"]=="SHORT" for g in gs)])
    lines.append(table(["Bloque","Activo","NY","Variante","Cortes","LONG brutos","SHORT brutos"],rows))
    lines += ["## 6. Costes supuestos y diagnóstico económico, NO precisión",
              "Todos los fills son posibles OHLC, no reales. Comisión 25bps por lado; slippage 5bps por lado; spread completo 4bps (mitad por lado). Son los mismos supuestos originales. No se importa bid/ask de seis eventos2025 como spread real de los nuevos eventos2024.",
              "PNL neto por acción = resultado al fill/salida simulado menos (entrada+salida)×0.0032. Conserva gaps al precio real OHLC de apertura. Para comparar bases de split distintas se informa bps = PNL/fill×10000×exposición; es una unidad hipotética de capital, NO rentabilidad realizada de cuenta ni suma de retornos de un portafolio con reinversión.",
              "BOTH_REDUCED usa como máximo factor0.25. Esto reduce efectivamente la exposición monetaria normalizada; no se cambió cantidad real de acciones ni capital contable. En las tablas económicas se excluyen TODOS los SHORT no ejecutables de la conclusión LONG."]
    rows=[]
    for stage,content in stages.items():
        for symbol,d in content["symbols"].items():
            for v,obj in d["variants"].items():
                r=obj["independent"]["1440"]["LONG_only"]
                rows.append([stage,symbol,v,r["LONG"],r["resolved_pnl_observations"],number(r["net_bps_sum_exposure_scaled"]),number(r["net_bps_median_exposure_scaled"]),number(r["worst_net_bps_exposure_scaled"])])
    lines.append(table(["Bloque","Activo","Variante","LONG","PNL resueltos","Suma bps netos supuestos","Mediana bps","Peor bps"],rows))
    lines += ["## 7. Conclusión descriptiva, sin promoción"]
    conclusions={}
    for stage,content in stages.items():
        combined={v:sum(d["variants"][v]["independent"]["1440"]["LONG"] for d in content["symbols"].values()) for v in VARIANTS}
        arms=[v for v in VARIANTS if v not in ("CURRENT","SIMPLE_EMA_CROSS")]
        top=max(combined[v] for v in arms)
        best=[v for v in arms if combined[v]==top]
        conclusions[stage]={"LONG_independent_combined_by_arm":combined,"most_LONG_among_single_gate_variants":best,"most_LONG_count":top}
        lines.append(f"{stage}: más LONG no solapados entre las cuatro variantes = {', '.join(best)} ({top}); actual={combined['CURRENT']}; EMA simple={combined['SIMPLE_EMA_CROSS']}. Se reportan todos los brazos, no se eliminan los que generaron cero.")
    lines += ["Un aumento de entradas no demuestra mejora económica. Comparar la tabla LONG de bps netos supuestos, sin sumar SHORT ni duplicar horizontes. Con pocas operaciones o sin cobertura de ejecución/noticias PIT, ‘no destruir el rendimiento’ no puede certificarse. La evaluación de un bloque elegido una vez no permite escoger parámetros y volver a validarlos aquí.",
              "### Condiciones restantes: fallos independientes entre cortes con plan LONG"]
    rows=[]
    for stage, symbols in blockers.items():
        for symbol,d in symbols.items():
            for condition,n in sorted(d["failures_overlapping"].items(),key=lambda x:-x[1]):
                rows.append([stage,symbol,d["LONG_plan_cuts"],condition,n])
    lines += [table(["Bloque","Activo","Planes LONG","Condición","Fallos (superpuestos)"],rows),
              "Si ninguna variante mejora: el candidato a cuello de botella es la interacción entre las condiciones que permanecen, no necesariamente la más frecuente por sí sola. Hace falta cubrir candidatos cercanos, comparar daño adverso evitado y costes/fills medidos en OTRO bloque no consumido, con una sola hipótesis fijada. No relajar varios filtros por prueba-error con este holdout.",
              "## 8. Integridad y reproducibilidad",
              f"portfolio.db SHA256 antes/después: `{p['ledger_sha256']}`. Motor producción conservado: inventario en preregistration.json.",
              "Desarrollo cerrado antes de apertura; registro SQLite independiente de aperturas en output/historical_replay, nunca portfolio.db. Reabrir después de holdout_complete.json falla; ejecución interrumpida solo reanuda el MISMO protocolo con checkpoints, no crea una nueva evaluación.",
              "Artefactos: preregistration.json, holdout_seal.json, holdout_opening.json, development_complete.json, holdout_complete.json, report.json, reporte.md y manifest.json. AES/DPAPI conserva adquisición original cifrada; datos de resultados históricos son investigación, no observaciones live.",
              "Fuentes: [Alpaca SIP histórico](https://docs.alpaca.markets/us/docs/market-data-faq), [NVDA split Jun2024 SEC](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm), [SMCI split posterior Sep2024 Nasdaq](https://classic.nasdaqtrader.com/TraderNews.aspx?id=ECA2024-451)."]
    payload={"protocol_sha256":sha(p),"stages":stages,"conclusions":conclusions,"remaining_LONG_blockers":blockers,
             "no_training_or_production_change":True,"ledger_sha256":digest_file(ROOT/"data/portfolio.db")}
    write_signed(OUTPUT/"report.json",payload)
    artifact(OUTPUT/"reporte.md","\n\n".join(lines))
    files=["preregistration.json","holdout_seal.json","holdout_opening.json","development_complete.json","holdout_complete.json","report.json","reporte.md","tests_passed.json"]
    for name,h in p["experiment_source_sha256"].items():
        source=ROOT/name; dest=OUTPUT/"source_archive"/h/source.name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    write_signed(OUTPUT/"manifest.json",{"files_sha256":{n:digest_file(OUTPUT/n) for n in files},"protocol_sha256":sha(p),
                 "ledger_sha256":digest_file(ROOT/"data/portfolio.db"),"experiment_source_sha256":p["experiment_source_sha256"]})
    print("REPORT:",OUTPUT/"reporte.md")
    print(conclusions)


if __name__=="__main__":
    run()
