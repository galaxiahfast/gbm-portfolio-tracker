"""Auditoría reproducible, sin importar app.py ni abrir la base de producción.

Uso: .venv/Scripts/python.exe scripts/audit_phase5.py --run-tests
Solo escribe informes en output/ y datos de prueba en un directorio temporal.
Los hallazgos son revisión humana versionada, no diagnósticos inventados por AST.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from collections import Counter
from decimal import Decimal
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    return sorted({ROOT/'app.py', ROOT/'requirements.txt', ROOT/'iniciar_app.bat',
                   *ROOT.joinpath('portfolio_tracker').rglob('*.py'),
                   *ROOT.joinpath('scripts').glob('*.py'), *ROOT.joinpath('tests').glob('*.py'),
                   *ROOT.joinpath('docs').glob('*.md'), *ROOT.joinpath('docs').glob('audit_phase5*.json')})


def protected_manifest():
    files = set(sources())
    # Hash, never open SQLite connections or report private row/image contents.
    private = Path(os.getenv('GBM_PORTFOLIO_DATA_DIR', str(ROOT/'data'))).expanduser()
    for folder in (private, ROOT/'backups'):
        if folder.exists():
            files.update(p for p in folder.rglob('*') if p.is_file() and
                         ('yfinance_cache' not in p.parts) and ('__pycache__' not in p.parts))
    return {str(p.resolve()): digest(p) for p in sorted(files) if p.is_file()}


def inventory():
    result = []
    for p in sources():
        raw = p.read_text(encoding='utf-8-sig')
        entry = dict(path=p.relative_to(ROOT).as_posix(), sha256=digest(p),
                     lines=len(raw.splitlines()), nonblank=sum(bool(s.strip()) for s in raw.splitlines()))
        if p.suffix == '.py':
            try:
                tree = ast.parse(raw)
                funcs = [n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                entry['functions'] = len(funcs)
                entry['test_functions'] = sum(n.name.startswith('test_') for n in funcs)
                entry['max_function_lines'] = max((n.end_lineno-n.lineno+1 for n in funcs),default=0)
                entry['branches_approx'] = sum(isinstance(n,(ast.If,ast.For,ast.While,ast.ExceptHandler,ast.IfExp)) for n in ast.walk(tree))
                entry['broad_except'] = sum(isinstance(n,ast.ExceptHandler) and
                    (n.type is None or isinstance(n.type,ast.Name) and n.type.id=='Exception') for n in ast.walk(tree))
                entry['parse'] = 'OK'
            except SyntaxError as exc:
                entry['parse'] = str(exc)
        result.append(entry)
    return result


def probes():
    """Synthetic controls in memory only; no live quotes or network calls."""
    import pandas as pd
    from portfolio_tracker.analytics import backtesting as bt
    from portfolio_tracker.analytics.probability_calibration import calibrate_probability
    from portfolio_tracker.analytics.technical_probability import _rsi, add_intraday_indicators
    from portfolio_tracker.analytics.fundamental_news import build_fundamental_snapshot
    from portfolio_tracker.analytics.multi_timeframe import calculate_15_day_projection
    from portfolio_tracker.analytics.closed_bars import _calendar
    results = []
    def record(name, values):
        results.append(dict(name=name, values=values))
    samples = [(0.1,1)]*250 + [(0.9,0)]*250
    cal = calibrate_probability(.9,samples)
    record('calibration_brier', dict(status=cal.status, published_brier_oos=cal.brier_score,
           raw_brier_oos=cal.raw_brier_score, holdout=cal.holdout_samples,
           calibrated_prediction=cal.calibrated_probability, sample_size=cal.sample_size,
           note='Caso invertido binario sin diversidad en calibración: debe permanecer preliminar.'))
    def signals(short):
        n=30
        k,d=(70,80) if short else (30,20)
        f=pd.DataFrame(dict(Close=[100.]*n, Open=[100.]*n, High=[101.]*n, Low=[99.]*n,
            Volume=[1500.]*n, Volume_MA20=[1000.]*n, StochRSI_K=[k]*n, StochRSI_D=[d]*n,
            ADX14=[30.]*n, MACD=[-1. if short else 1.]*n, MACD_signal=[0.]*n,
            EMA9=[99. if short else 101.]*n, EMA21=[101. if short else 99.]*n,
            EMA50=[101. if short else 99.]*n, PriorHigh20=[110.]*n, PriorLow20=[90.]*n,
            MonthlyBias=[0.]*n, WeeklyBias=[0.]*n, ATR14=[1.]*n,
            ATRPercent=[.02]*n, ATRPercentMedian=[.02]*n),index=pd.bdate_range('2026-01-01',periods=n))
        f.iloc[0,f.columns.get_loc('StochRSI_K')] = 90 if short else 10
        return f
    config=bt.BacktestConfig()
    longs,lr=bt._generate_candidates(signals(False),config)
    shorts,sr=bt._generate_candidates(signals(True),config)
    record('short_asymmetry',dict(long_count=len(longs),short_count=len(shorts),
           long_scores=[x.score for x in longs],short_scores=[x.score for x in shorts],short_rejections=dict(sr)))
    if longs:
        f=signals(False)
        # Entry row 2 stays above stop; following session gaps below it.
        f.iloc[3,f.columns.get_indexer(['Open','High','Low','Close'])]=[90,91,89,90]
        trade=bt._simulate_trade('TEST','AUDIT',f,longs[0],config,10000,.6)
        record('gap_fill',dict(entry=trade.entry_price,stop=trade.stop_price,
                              gap_open=90,simulated_exit=trade.exit_price))
    flat=pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':1000.},
                      index=pd.date_range('2026-01-05 14:30Z',periods=120,freq='5min'))
    indicators=add_intraday_indicators(flat)
    from portfolio_tracker.analytics.technical_validity import validate_raw_tail, TechnicalDataVeto
    try:
        validate_raw_tail(flat,'5m')
        flat_state='SIN VETO'
    except TechnicalDataVeto as exc:
        flat_state=exc.state
    rsi=float(_rsi(flat.Close).iloc[-1])
    record('flat_oscillator',dict(last_rsi=rsi if math.isfinite(rsi) else None, state=flat_state,
           finite_stoch=int(indicators.StochRSI_K.notna().sum()),
           finite_adx=int(indicators.ADX14.notna().sum())))
    f=signals(False)
    projection=calculate_15_day_projection(100,f,50,False,pd.Timestamp('2026-09-04'))
    cal_schedule=_calendar(2026,2026).schedule
    invalid=[str(p.session_date) for p in projection if pd.Timestamp(p.session_date) not in cal_schedule.index]
    record('projection_holidays',dict(non_exchange_sessions=invalid))
    snap=build_fundamental_snapshot('TEST',info={'operatingCashflow':400,'freeCashflow':200},
           income_statement=pd.DataFrame({pd.Timestamp('2026-06-30'):[1000.,100.]},index=['Total Revenue','Net Income']),
           cashflow=pd.DataFrame({pd.Timestamp('2026-06-30'):[100.,50.]},index=['Operating Cash Flow','Free Cash Flow']))
    record('period_mismatch',dict(cash_conversion=snap.metrics['cash_conversion'],
           fcf_margin=snap.metrics['fcf_margin'],basis=snap.metrics['cash_conversion_basis'],
           expected='N/D sin metadatos; pruebas F09 adicionales cubren pares compatibles.'))
    # Use an isolated SQLite connection implementing the repository's DB interface.
    import sqlite3
    from portfolio_tracker.repository import PortfolioRepository
    class MemoryDB:
        def __init__(self):
            self.c=sqlite3.connect(':memory:'); self.c.row_factory=sqlite3.Row
            self.c.execute('''CREATE TABLE live_model_observations(id INTEGER PRIMARY KEY,
              symbol,observed_at,horizon_minutes,reference_price,raw_probability_up,predicted_direction,
              parameters_json,observation_sha256,created_at,outcome_price,outcome_up,successful,resolved_at,
              integrity_version,available_at,source_bar_at,horizon_policy,resolution_status,
              outcome_bar_at,outcome_source,resolution_sha256)''')
        @contextmanager
        def connect(self):
            yield self.c
        @contextmanager
        def transaction(self):
            with self.c:
                yield self.c
    db=MemoryDB(); repo=PortfolioRepository(db)
    at=datetime(2026,8,28,19,tzinfo=timezone.utc)
    repo.record_live_model_observation(symbol='TEST',observed_at=at,source_bar_at=at,reference_price=Decimal(100),
              raw_probability_up=Decimal('.8'),parameters_json='{}',horizon_minutes=60)
    late=at+timedelta(days=3)
    historical=pd.DataFrame(dict(Open=[101.],High=[102.],Low=[100.],Close=[101.],Volume=[100.]),
                            index=pd.DatetimeIndex([at+timedelta(minutes=55)]))
    repo.resolve_live_model_observations(symbol='TEST',historical_bars=historical,current_as_of=late)
    resolved_bar=db.c.execute('SELECT outcome_bar_at FROM live_model_observations').fetchone()[0]
    before=repo.verify_live_model_observations()
    db.c.execute('UPDATE live_model_observations SET outcome_up=0,outcome_price=90,successful=0')
    after=repo.verify_live_model_observations()
    record('outcome_integrity',dict(valid_before=before[0],valid_after_tamper=after[0],
           invalid_after=list(after[1]),samples_after_tamper=repo.live_model_calibration_samples('TEST',horizon_minutes=60)))
    record('late_horizon',dict(requested_minutes=60,
           resolved_minutes=int((datetime.fromisoformat(resolved_bar)-at).total_seconds()/60),
           processing_delay_minutes=int((late-at).total_seconds()/60)))
    db.c.close()
    # Timings are synthetic local wall times, not production load guarantees.
    from tests.test_pdf_report import _analysis
    from portfolio_tracker.services.pdf_report import build_executive_report,build_technical_report,build_probability_report,build_master_report
    from portfolio_tracker.services.price_zones import build_zone_snapshot
    start=time.perf_counter(); a=_analysis(); engine_seconds=time.perf_counter()-start
    snapshot=build_zone_snapshot(a,now='2026-08-30 15:00Z')
    sizes={}; timings={}
    for name,fn in [('executive',build_executive_report),('technical',build_technical_report),('combined',build_probability_report),('master',build_master_report)]:
        start=time.perf_counter()
        b=fn(a,None,zone_snapshot=snapshot) if name=='master' else fn(a,zone_snapshot=snapshot)
        timings[name]=round(time.perf_counter()-start,4); sizes[name]=len(b)
    record('synthetic_performance',dict(engine_seconds=round(engine_seconds,4),pdf_seconds=timings,pdf_bytes=sizes))
    return results


def build_pdf(report, destination):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyAudit',fontName='Helvetica',fontSize=9,leading=13,spaceAfter=6,textColor=colors.HexColor('#222222')))
    styles.add(ParagraphStyle(name='SmallAudit',parent=styles['BodyAudit'],fontSize=7.5,leading=10))
    styles['Title'].alignment=0
    def p(s,style='BodyAudit'):
        # Built-in PDF fonts do not reliably render Unicode arrows everywhere.
        text = str(s).replace('→', ' -> ').replace('–', '-').replace('—', '-')
        return Paragraph(escape(text),styles[style])
    def table(rows,widths):
        cells=[[p(c,'SmallAudit') for c in row] for row in rows]
        t=Table(cells,colWidths=[x*mm for x in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5E7EB')),
           ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F7F7')]),
           ('LINEBELOW',(0,0),(-1,0),.7,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),6),
           ('TOPPADDING',(0,0),(-1,-1),6)]))
        return t
    story=[p('AUDITORÍA TÉCNICA','Title'),p('Motor cuantitativo · Fase 5','Heading1'),
           p('Fecha UTC: '+datetime.fromisoformat(report['generated_at']).strftime('%d/%m/%Y %H:%M:%S')),p('Revisión local de código y evidencia sintética. Sin órdenes, cotizaciones externas ni conexión a la base de producción.'),Spacer(1,8*mm),
           p('Dictamen','Heading1'),p('La suite funcional aprobada no acredita precisión predictiva ni equivalencia entre simulación y operación en vivo. Se detectaron defectos concretos en calibración, simulación y trazabilidad. No se recomienda interpretar el estado “100% implementado” como certificación cuantitativa.'),
           p('ESTADO POST-CORRECCIÓN','Heading2'),
           p(' | '.join(f'{key}: {value}' for key,value in report['status_counts'].items())),
           p(f"Hallazgos revisados: {len(report['findings'])}; prioridades corresponden al catálogo original. Archivos inventariados: {len(report['inventory'])}"),
           p(f"Verificación de archivos protegidos: {report['preservation']['status']}"),
           p(f"Pruebas: {report['tests']['summary']}"),
           p('Prioridad propuesta','Heading2'),p('1. Completar replay persistente, vigencia 1h/4h y costes del benchmark (F06-F08). 2. Riesgo del remanente, plan único y capital por fold (F10/F11/F18). 3. Calendario de proyección y validación OOS de zonas (F12/F13). 4. Fuentes de eventos, rendimiento, visibilidad y ciclo de patrones (F14-F17).'),
           p('Alcance y límites','Heading2'),p('Inspección estática integral por inventario/AST, lectura detallada del flujo cuantitativo y sus fronteras, pruebas existentes y reproducciones sintéticas seleccionadas. No se ejecutó trading real, no se auditó una cuenta de GBM, no se certificó seguridad criptográfica completa ni rentabilidad histórica real. Los módulos contables se revisaron solo como frontera de integración.'),
           p('Cambios de esta auditoría','Heading2'),p('Únicamente el generador, catálogo de hallazgos y artefactos del informe. No se implementaron correcciones de producción. El árbol ya contenía cambios locales anteriores; no se atribuyen a esta auditoría.'),PageBreak()]
    story += [p('1. Arquitectura y discrepancias','Heading1'),
       table([['Capa','Flujo observado','Riesgo principal'],
          ['Datos','Velas cerradas; veto central 5m/diario en UI; 1h/4h derivados','Falta validar vigencia del último bucket superior tras huecos.'],
          ['Reglas','analyze_probability → apply_hierarchy → fundamentales → calibración → synchronize_position','Múltiples representaciones de señal, veto, plan y proyección.'],
          ['Zonas','Snapshot compartido UI/PDF → comparación ponderada diaria','Heurística ajustada sin calibración OOS de toque/cierre.'],
          ['Validación','Núcleo técnico compartido + promoción por activo/versión/hash; OOS 60/20/20','Replay no reproduce todo el estado, noticias ni filtros vivos.'],
          ['Persistencia','Pronóstico y resolución con SHA-256 enlazados; legacy excluido','Integridad verificada no equivale a autenticidad criptográfica.']], [28,80,66]),
       p('Cómo interpretar la evidencia','Heading2'),p('REPRODUCIDO: el script ejecutó un caso sintético y guardó el resultado. ESTÁTICO: la discrepancia se desprende de rutas/expresiones del código; su frecuencia real no fue medida. RIESGO METODOLÓGICO: se requiere validación estadística adicional, no significa que cada estimación sea incorrecta.'),
       p('Protecciones que sí existen','Heading2'),p('Calendario XNYS para velas cerradas; jerarquía sin voto 5m en el régimen; histéresis ADX; posiciones derivadas del libro sin fills ficticios; cadena SHA-256 de estado; instantáneas fundamentales; AES-256-GCM para respaldos; entrada de simulación en la siguiente vela; supuesto stop-first si stop y objetivo coinciden. Se preservaron estas implementaciones.'),
       p('Escala','Heading2'),p('ALTA: corregir antes de confiar en validación o gestión de riesgo. MEDIA: afecta disponibilidad, interpretación o estabilidad. No se declara una vulneración del libro contable ni una pérdida de fondos.'),PageBreak(),
       p('2. Resumen de hallazgos','Heading1'),
       p('RESUELTO: criterio específico corregido y probado; no certifica rentabilidad. PARCIAL: hay corrección comprobada y un componente pendiente. PENDIENTE: el defecto o necesidad metodológica continúa. Las pruebas citadas son evidencia relacionada; una prueba aprobada no cierra por sí sola el hallazgo.'),
       table([['ID','Prioridad','Estado','Hallazgo']]+[[f['id'],f['priority'],f['status'],f['title']] for f in report['findings']],[12,20,28,114]),PageBreak()]
    for i,f in enumerate(report['findings']):
        block=[p(f"{f['id']} | {f['status']} | {f['priority']} | {f['title']}",'Heading2'),p(f"Evidencia: {f['evidence_type']}"),
               p('Qué ocurre: '+f['observation']),p('Consecuencia: '+f['impact']),p('Acción propuesta: '+f['recommendation']),
               p('Criterio de aceptación original: '+f['acceptance']),p('Localización: '+'; '.join(f['locations']),'SmallAudit'),
               p('Pruebas relacionadas: '+', '.join(f.get('test_files',[])),'SmallAudit')]
        if f.get('probe'):
            evidence=next((x['values'] for x in report['probes'] if x['name']==f['probe']),{'error':'No se ejecutó'})
            block.append(p('Reproducción: '+json.dumps(evidence,ensure_ascii=False),'SmallAudit'))
        story.extend(block)
        if i%2==1 and i<len(report['findings'])-1:
            story.append(PageBreak())
    story += [PageBreak(),p('3. Métricas, pruebas y reproducibilidad','Heading1'),
              p('Los recuentos AST son métricas descriptivas. “Ramas aproximadas” cuenta if/for/while/except/if-expressions; no equivale a complejidad ciclomática certificada.'),
              table([['Archivo','Líneas','Funciones','Mayor función','Ramas aprox.']]+
                    [[x['path'],x['lines'],x.get('functions','-'),x.get('max_function_lines','-'),x.get('branches_approx','-')]
                     for x in sorted(report['inventory'],key=lambda x:x.get('max_function_lines',0),reverse=True)[:12]], [94,18,20,20,22]),
              p('Resultado de pruebas','Heading2'),p(report['tests']['summary']),
              p('Advertencias y compatibilidad del entorno','Heading2'),
              p(f"Advertencias emitidas: {report['tests'].get('warning_count',0)}. No se suprimen mediante filtros en este comando. Pasar todas las pruebas no significa una ejecución libre de advertencias."),
              p('Se observa DeprecationWarning de unidades genéricas de timedelta en el entorno NumPy/pandas/exchange-calendars y sus llamadas desde el proyecto. No prueba resultados incorrectos hoy, pero anuncia incompatibilidad futura. Recomendación: reproducir con versiones compatibles en un entorno aislado y añadir una ejecución focalizada con warnings-as-errors antes de actualizar dependencias. No se cambiaron paquetes ni código productivo durante esta auditoría.'),
              p('Clases detectadas: '+', '.join(report['tests'].get('warning_categories',[])),'SmallAudit'),
              p('Comparación con auditoría anterior','Heading2'),p(json.dumps(report.get('comparison',{}),ensure_ascii=False),'SmallAudit'),
              p('Las pruebas se ejecutan con GBM_PORTFOLIO_DATA_DIR temporal y sin modificar el contenido de tests/. No se midió cobertura de líneas; pasar pruebas no demuestra ausencia de fallos.'),
              PageBreak(),
              p('Reproducciones adicionales','Heading2')]
    for probe in report['probes']:
        story += [p(probe['name'],'Heading3'),p(json.dumps(probe['values'],ensure_ascii=False),'SmallAudit')]
    story += [PageBreak(),p('4. Hoja de ruta segura','Heading1'),
              p('Etapa A - Conservar correcciones','Heading2'),p('Mantener SHA enlazado, resolución histórica exacta, OOS independiente, vector multiclase y ratios compatibles. No recertificar registros legacy ni confundir Brier medido con habilidad predictiva demostrada.'),
              p('Etapa B - Cerrar brechas de simulación','Heading2'),p('Completar paridad secuencial de estado, gestión del remanente, vigencia superior y liquidación del benchmark. Conciliar capital por fold. Exigir regresiones específicas antes de aprobar cambios.'),
              p('Etapa C - Modelos diarios de alcance','Heading2'),p('Guardar pronósticos por nivel y sus resultados reales de toque/cierre. Comparar el modelo ponderado contra frecuencia simple y un baseline de volatilidad con Brier OOS, curvas de fiabilidad, tamaño efectivo y pruebas por régimen. No optimizar pesos sobre la misma muestra que se reporta.'),
              p('Etapa D - Entrega','Heading2'),p('Implementar cada corrección en una rama aislada, con copia de prueba y regresiones específicas. Medir rendimiento antes/después, revisión visual y hashes del libro. Solicitar aprobación antes de cualquier migración o cambio de política operativa.'),
              p('Reejecución','Heading2'),p('.venv/Scripts/python.exe scripts/audit_phase5.py --run-tests','SmallAudit'),
              p('El generador no es un auditor autónomo: recompila hallazgos humanos versionados, métricas y reproducciones. Si cambia el SHA-256 del archivo que sustenta un hallazgo, exige revisión y lo indica; no convierte este informe en una aprobación de versiones futuras.'),
              p('Dependencias utilizadas','Heading2'),p(json.dumps(report['versions'],ensure_ascii=False),'SmallAudit'),PageBreak(),
              p('Anexo. Inventario de fuentes','Heading1'),p('Inventario estático completo. Hashes íntegros en audit.json; abajo se muestran prefijos para lectura. No se incluyen contenidos ni hashes individuales del libro o comprobantes en el PDF.')]
    story.append(table([['Archivo','Líneas','SHA-256 (prefijo)']]+[[x['path'],x['lines'],x['sha256'][:16]] for x in report['inventory']],[119,15,40]))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
        canvas.drawString(18*mm,10*mm,'AUDITORÍA FASE 5 | '+report['generated_at'][:10])
        canvas.drawRightString(192*mm,10*mm,str(doc.page))
    SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
                      topMargin=18*mm,bottomMargin=18*mm,title='Auditoría técnica - Motor cuantitativo Fase 5',
                      author='Auditoría de código local').build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests',action='store_true')
    parser.add_argument('--findings',default='docs/audit_phase5_post_review.json')
    parser.add_argument('--baseline',default=None,help='audit.json anterior para comparación explícita')
    args=parser.parse_args()
    started=datetime.now(timezone.utc); slug=started.strftime('%Y%m%d_%H%M%S')
    out=ROOT/'output'/'audit_phase5'/slug; out.mkdir(parents=True)
    pdf=ROOT/'output'/'pdf'/f'auditoria_fase5_{slug}.pdf';pdf.parent.mkdir(parents=True,exist_ok=True)
    before=protected_manifest()
    inv=inventory()
    tests={'summary':'No ejecutadas en esta corrida','returncode':None}
    with tempfile.TemporaryDirectory(prefix='gbm-audit-') as temporary:
        original=os.environ.get('GBM_PORTFOLIO_DATA_DIR')
        os.environ['GBM_PORTFOLIO_DATA_DIR']=str(Path(temporary)/'data')
        try:
            checks=probes()
            if args.run_tests:
                command=[sys.executable,'-m','pytest','-q','--tb=short',
                         '-p','no:cacheprovider','--basetemp',str(Path(temporary)/'pytest'),
                         '--junitxml',str(out/'tests.xml')]
                start=time.perf_counter()
                run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
                (out/'tests.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
                tests=dict(returncode=run.returncode,seconds=round(time.perf_counter()-start,3),
                           summary=next((line.strip() for line in reversed(run.stdout.splitlines()) if 'passed' in line or 'failed' in line), 'Ver tests.log'))
                warning_match=re.search(r'(\d+) warnings?',tests['summary'])
                tests['warning_count']=int(warning_match.group(1)) if warning_match else 0
                tests['warning_categories']=sorted(set(re.findall(r'\b(\w+Warning):',run.stdout)))
                if (out/'tests.xml').exists():
                    suites=ET.parse(out/'tests.xml').getroot()
                    tests['junit']={key:sum(int(s.get(key,0)) for s in suites.findall('testsuite'))
                                    for key in ('tests','failures','errors','skipped')}
        finally:
            if original is None: os.environ.pop('GBM_PORTFOLIO_DATA_DIR',None)
            else: os.environ['GBM_PORTFOLIO_DATA_DIR']=original
    after=protected_manifest()
    changed=[p for p,h in before.items() if after.get(p)!=h]
    added=[p for p in after if p not in before]
    preservation=dict(status='SIN CAMBIOS' if not changed and not added else 'CAMBIOS DETECTADOS: REVISAR',
                      compared=len(before),changed_count=len(changed),added_count=len(added))
    catalog=ROOT/args.findings
    findings=json.loads(catalog.read_text(encoding='utf-8'))
    for f in findings:
        locations=[]
        for ref in f['references']:
            path=ROOT/ref['path'];lines=path.read_text(encoding='utf-8').splitlines()
            matches=[i for i,s in enumerate(lines,1) if ref['anchor'] in s]
            unchanged=digest(path)==ref['reviewed_sha256']
            locations.append(f"{ref['path']}:{matches[0] if matches else '?'}"+('' if unchanged else ' [fuente cambiada: revalidar hallazgo]'))
            if not unchanged or not matches:
                f['status']='REVALIDAR'
        f['locations']=locations
    comparison={}
    if args.baseline:
        baseline_path=ROOT/args.baseline
        old=json.loads(baseline_path.read_text(encoding='utf-8'))
        old_files={f['path']:f for f in old['inventory']}
        comparison=dict(baseline=args.baseline, baseline_sha256=digest(baseline_path),
            previous_tests=old['tests']['summary'],current_tests=tests['summary'],
            previous_files=len(old_files),current_files=len(inv),
            changed_sources=sum(f['path'] in old_files and f['sha256']!=old_files[f['path']]['sha256'] for f in inv),
            added_sources=sum(f['path'] not in old_files for f in inv),
            previous_lines=sum(f['lines'] for f in old['inventory']),current_lines=sum(f['lines'] for f in inv),
            note='Diferencias de todo el árbol desde ese corte, no atribución causal ni medición de precisión.')
    versions={}
    for package in ('pandas','numpy','streamlit','reportlab','yfinance','exchange-calendars','pytest','cryptography'):
        try: versions[package]=version(package)
        except PackageNotFoundError: versions[package]='no instalado'
    report=dict(generated_at=started.isoformat(),inventory=inv,tests=tests,probes=checks,
                findings_catalog_sha256=digest(catalog), comparison=comparison,
                status_counts=dict(Counter(f['status'] for f in findings)),
                findings=findings,preservation=preservation,versions=versions)
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    build_pdf(report,pdf)
    (out/'pdf.sha256').write_text(digest(pdf)+'\n',encoding='ascii')
    print(json.dumps(dict(pdf=str(pdf),evidence=str(out),tests=tests,preservation=preservation),ensure_ascii=False))
    if tests.get('returncode') not in (0,None) or changed or added or 'REVALIDAR' in report['status_counts']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
