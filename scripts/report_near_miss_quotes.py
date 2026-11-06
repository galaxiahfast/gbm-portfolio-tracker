"""Close near-miss diagnosis; quote counterfactuals are NOT actual trades."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.near_miss_quote_audit import OUTPUT, verify
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import digest_file
from scripts.report_trigger_selectivity import table, number, artifact

LABELS={"permission":"Permiso macro", "cross":"Cruce StochRSI", "volume":"Volumen", "break":"Ruptura vela previa"}


def run():
    method=read_signed(OUTPUT/"method.json")
    verify(method)
    events=[]
    files={"method.json":digest_file(OUTPUT/"method.json")}
    for e in method["events"]:
        path=OUTPUT/"results"/f"{e['event_id']}.json"
        result=read_signed(path) if path.exists() else None
        if result and result["method_sha256"]!=sha(method):
            raise ValueError("Different result protocol.")
        if result:
            files[str(path.relative_to(OUTPUT))]=digest_file(path)
        events.append({**e,"quote_result":result})
    counts={}
    for stage in ("development","holdout"):
        for side in ("LONG","SHORT"):
            for selection in ("raw","nonoverlapping"):
                subset=[e for e in events if e["stage"]==stage and e["side"]==side and (selection=="raw" or e["nonoverlapping"])]
                for h in ("60","1440"):
                    groups=defaultdict(list)
                    for e in subset:
                        groups[e["missing_condition"]].append(e)
                    for condition,rows in groups.items():
                        classes=Counter(); outcomes=Counter(); fills=0; spread=[]; mf=[]; ma=[]; pnl=[]
                        for e in rows:
                            r=e["quote_result"]["horizons"][h] if e["quote_result"] else None
                            classes.update([r["classification"] if r else "N_D"])
                            if r and not r["quality_flags"]:
                                if r["possible_fill"]:
                                    fills+=1; spread.append(r["possible_fill"]["spread_observed"])
                                if r["exit"]:
                                    outcomes.update([r["exit"]["outcome"]])
                                if r["gross_quote_pnl_per_share"] is not None:
                                    pnl.append(r["gross_quote_pnl_per_share"])
                                if r["mfe_until_exit_per_share"] is not None:
                                    mf.append(r["mfe_until_exit_per_share"])
                                    ma.append(r["mae_until_exit_per_share"])
                        counts['|'.join((stage,side,selection,h,condition))]={
                            "cuts":len(rows),"possible_quote_fills":fills,"classification":dict(classes),"outcomes":dict(outcomes),
                            "median_entry_spread_observed":float(np.median(spread)) if spread else None,
                            "median_mfe_until_exit":float(np.median(mf)) if mf else None,
                            "median_mae_until_exit":float(np.median(ma)) if ma else None,
                            "gross_quote_pnl_sum_per_share_unweighted_not_portfolio_return":sum(pnl) if pnl else None}
    lines=["# Cortes que casi pasan el gatillo — bid/ask SIP observado",
           "Diagnóstico retrospectivo de bloques YA consumidos. No nuevo holdout, entrenamiento, aprobación ni relajación de producción. Método fijado antes de descargar las cotizaciones de estos candidatos.",
           "## Población y método",
           f"Método SHA256 `{sha(method)}`; registrado {method['registered_at']}. {len(events)} cortes, 13 LONG y 18 SHORT. Se incluyen todos los cortes del control con EXACTAMENTE un operando AND falso; no se eligen por su resultado. Precio dentro de zona no es un filtro del gatillo actual. Los operandos veto/rechazo redundantes no se fusionan artificialmente.",
           "Desarrollo previo 21/07/2025–25/09/2026: 28 candidatos. Bloque previamente abierto 01/07/2024–30/08/2024: 3 candidatos SHORT, ninguno LONG. Solo se leyeron sus snapshots cerrados: no se descifró ni reabrió el holdout. No se usa para aprobar una nueva variante.",
           "Una misma entrada se observa a 1h y 1D, sin contar dos muestras. La tabla principal conserva eventos no solapados por activo, embargo hasta el mayor timeout firmado; cobertura bruta separada. Otros brazos del experimento no alimentan esta selección.",
           "Las cotizaciones son históricas reales de Alpaca, feed SIP, USD, sesiones XNYS, America/New_York. Se guardan TODAS las páginas comprimidas con SHA256, timestamps y condiciones; no solo una instantánea favorable. Solo condiciones R regulares, bid/ask finitos positivos no cruzados y ambos tamaños positivos. No rellenar cotizaciones ausentes ni sustituir otro día. Un hueco >60s o falta de quote final <=10s exige N/D.",
           "Contrafactual de una acción: mantener el lado, entrada limitada, stop y TP originales; empezar en la siguiente vela completa y respetar ventana de entrada/vencimiento. LONG: primer ask compatible con límite, salida en bid observado. SHORT: primer bid compatible, cobertura en ask observado. Si stop/objetivo se sobrepasa antes de la entrada, cancelar; no inventar ganancias. Stop referido a precio liquidable de quote, no a una ejecución/trigger real de GBM.",
           "Spread observado queda incorporado al comprar en ask/vender en bid (inverso SHORT). NO se resta spread supuesto. Comisión, slippage, impuestos, préstamo, cola y ejecución real son N/D; por eso se reporta PNL BRUTO bid/ask, no ganancia neta realizada. Tamaño de quote no prueba capacidad del broker. SHORT sigue no ejecutable en Trading USA.",
           "## Principal: 1D, eventos no solapados, por filtro y lado"]
    rows=[]
    for key,v in counts.items():
        stage,side,selection,h,condition=key.split('|')
        if selection!='nonoverlapping' or h!='1440':continue
        c=v['classification']; o=v['outcomes']
        rows.append([stage,side,LABELS.get(condition,condition),v['cuts'],v['possible_quote_fills'],
                     c.get('HYPOTHETICAL_GROSS_GAIN_BLOCKED',0),c.get('HYPOTHETICAL_GROSS_LOSS_AVOIDED',0),
                     f"{o.get('TP_FIRST',0)}/{o.get('SL_FIRST',0)}/{o.get('TIMEOUT',0)}",c.get('N_D',0)])
    lines.append(table(['Bloque','Lado','Único filtro fallido','Cortes','Entrada posible quote','Ganancia bruta hipotética bloqueada','Pérdida bruta hipotética evitada','TP/SL/timeout','N/D o sin entrada'],rows))
    lines += ["Las etiquetas ‘bloqueada/evitada’ significan solo el resultado de ignorar ese único operando en este contrafactual. No son ganancias/pérdidas reales ni prueban causalmente que un filtro esté bien o mal calibrado. Un timeout puede dar beneficio/pérdida bruto sin tocar TP/SL.",
              "## Detalle de cada corte: ambos horizontes (NO muestras independientes)"]
    rows=[]; quote_total=0; unavailable=[]
    for e in events:
        result=e['quote_result']
        if result:
            quote_total+=sum(w['quotes'] for w in result['evidence'])
        else:
            unavailable.append(e['event_id'])
        for h in ('60','1440'):
            r=result['horizons'][h] if result else None
            fill=r['possible_fill'] if r else None; exit=r['exit'] if r else None
            rows.append([e['stage'],e['symbol'],pd.Timestamp(e['cut_at']).tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M'),
                         e['side'],LABELS.get(e['missing_condition'],e['missing_condition']),h,
                         'sí' if e['nonoverlapping'] else 'excluido embargo',number(fill['price']) if fill else 'N/D',
                         number(fill['spread_observed'],4) if fill else 'N/D',exit['outcome'] if exit else (r['status'] if r else 'N/D descarga'),
                         number(r['gross_quote_pnl_per_share'],4) if r else 'N/D',
                         number(r['mfe_until_exit_per_share']) if r else 'N/D',number(r['mae_until_exit_per_share']) if r else 'N/D',
                         '; '.join(r['quality_flags']) if r else 'sin cotizaciones completas'])
    lines.append(table(['Bloque','Activo','Corte NY','Lado','Filtro','Min','Evento independiente','Entrada quote USD','Spread entrada USD','Resultado','PNL bruto/acción USD','MFE hasta salida USD','MAE hasta salida USD','Limitación'],rows))
    lines += ["MFE favorable y MAE adversa se miden con el precio liquidable, desde entrada posible hasta primera salida. El JSON conserva además las excursiones hasta vencimiento completo: son EX POST y no un beneficio obtenido después de un stop. Valores de trayectorias con flags no se utilizan para clasificar beneficio/pérdida.",
              f"## Cobertura e integridad\n\nCotizaciones adquiridas: {quote_total:,}. Eventos sin adquisición completa: {len(unavailable)}. Datos originales de ambos proveedores permanecen sin cambios; no se fabrican observaciones live.",
              f"portfolio.db SHA256 antes/después: `{method['original_ledger_sha256']}`. Solo se calcula checksum; sin abrir tabla, orden ni registro contable.",
              "## Qué permite concluir\n\nEsta tabla localiza filtros que descartaron candidatos positivos o negativos en un historial ya inspeccionado. No permite escoger ahora el filtro con PNL más alto y tratarlo como evidencia independiente. La muestra LONG es pequeña, no contiene fundamentales/noticias históricos PIT y la ejecución GBM no está observada. Cualquier cambio requiere otra hipótesis fijada y datos no consumidos. No hay entrenamiento, Brier ni precisión reportada.",
              "Fuente: [Alpaca histórico SIP y condiciones de acceso](https://docs.alpaca.markets/us/docs/market-data-faq), [endpoint de cotizaciones históricas](https://docs.alpaca.markets/us/reference/stockquotes-1)."]
    payload={'method_sha256':sha(method),'events':events,'counts':counts,'raw_quotes_downloaded':quote_total,
             'events_without_complete_acquisition':unavailable,'ledger_sha256':digest_file(ROOT/'data/portfolio.db'),
             'production_unchanged':True,'no_approval_or_training':True}
    write_signed(OUTPUT/'report.json',payload)
    artifact(OUTPUT/'reporte.md','\n\n'.join(lines))
    files.update({n:digest_file(OUTPUT/n) for n in ('report.json','reporte.md')})
    write_signed(OUTPUT/'manifest.json',{'method_sha256':sha(method),'files_sha256':files,'ledger_sha256':digest_file(ROOT/'data/portfolio.db')})
    print('REPORT',OUTPUT/'reporte.md','quotes',quote_total,'unavailable',len(unavailable))


if __name__=='__main__':
    run()
