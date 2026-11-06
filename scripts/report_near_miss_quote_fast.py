"""Reuse only the numeric reporter in a private research namespace."""
from pathlib import Path
from types import FunctionType
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import report_near_miss_quotes as original
from scripts.near_miss_quote_fast import OUTPUT, verify
from scripts.download_massive_history import read_signed,write_signed,sha
from scripts.audit_trigger_replay import digest_file
from scripts.report_trigger_selectivity import artifact,table,number


def publish(path,text):
    old='El JSON conserva además las excursiones hasta vencimiento completo: son EX POST y no un beneficio obtenido después de un stop.'
    if old not in text:raise ValueError('Unexpected original report prose.')
    text=text.replace(old,'No se mide la excursión después de la primera salida: los campos de horizonte completo quedan N/D. No se cuenta un rebote posterior al stop como beneficio conseguido.')
    report=read_signed(OUTPUT/'report.json')
    rows=[]
    for key,v in report['counts'].items():
        stage,side,selection,h,condition=key.split('|')
        if side=='LONG' and selection=='nonoverlapping' and h=='1440':
            rows.append([stage,original.LABELS.get(condition,condition),v['cuts'],v['possible_quote_fills'],
                         number(v['median_entry_spread_observed'],4),number(v['median_mfe_until_exit']),number(v['median_mae_until_exit'])])
    text+='\n\n## Spread y excursión LONG por filtro — medianas hasta primera salida\n\n'+table(
        ['Bloque','Filtro','Cortes','Fills posibles quote','Spread entrada USD','MFE USD/acción','MAE USD/acción'],rows)
    text+='\n\n## Enmienda de adquisición, no de reglas\n\n'+read_signed(OUTPUT/'method.json')['acquisition_amendment']
    text+='\n\nLa condición de parada es el cierre del plan predefinido para ambos horizontes, cualquiera sea su resultado; no se eligen casos favorables. Las páginas originales de v1 se preservan. Cada página usada está identificada por ruta y SHA256 en los resultados v2. Cotizaciones posteriores a esa salida no son necesarias para esta pregunta ni se convierten en PNL.'
    artifact(path,text)


def run():
    namespace=dict(original.run.__globals__)
    namespace.update(OUTPUT=OUTPUT,verify=verify,artifact=publish)
    reporter=FunctionType(original.run.__code__,namespace,original.run.__name__,original.run.__defaults__,original.run.__closure__)
    reporter()
    report=read_signed(OUTPUT/'report.json')
    raw={}
    for e in report['events']:
        if not e['quote_result']:continue
        for window in e['quote_result']['evidence']:
            for page in window['pages']:
                p=ROOT/page['path_from_root']
                if digest_file(p)!=page['sha256']:raise ValueError('Raw page altered.')
                raw[page['path_from_root']]=page['sha256']
    write_signed(OUTPUT/'quote_inventory.json',{'report_sha256':sha(report),'used_raw_pages_sha256':raw,
                 'method_sha256':sha(read_signed(OUTPUT/'method.json')),'ledger_sha256':digest_file(ROOT/'data/portfolio.db')})
    print('RAW_QUOTE_MANIFEST_VALID',len(raw),'pages')


if __name__=='__main__':
    run()
