"""Read-only production audit; all destructive probes use disposable databases.

Run with project Python: scripts/audit_predictions.py --run-tests --benchmark-network
Outputs PDF + JSON evidence, never orders, model corrections or scheduled tasks.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, default=json_safe, allow_nan=False).encode()).hexdigest()


def source_inventory():
    paths = [ROOT/'app.py', ROOT/'requirements.txt', *ROOT.joinpath('portfolio_tracker').rglob('*.py'),
             *ROOT.joinpath('scripts').glob('*.py'), *ROOT.joinpath('scripts').glob('*.ps1'),
             *ROOT.joinpath('tests').glob('*.py')]
    return {p.relative_to(ROOT).as_posix(): {'sha256': sha256(p.read_bytes()).hexdigest(),
            'lines': len(p.read_text(encoding='utf-8-sig').splitlines())} for p in sorted(set(paths))}


def snapshot_production(path):
    """SQLite backup from mode=ro into RAM: respects WAL, never changes pragmas."""
    if not path.is_file():
        raise FileNotFoundError(path)
    origin = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=10)
    memory = sqlite3.connect(':memory:')
    memory.row_factory = sqlite3.Row
    try:
        origin.backup(memory)
    finally:
        origin.close()
    memory.execute('PRAGMA query_only=ON')
    return memory


def ledger_manifest(conn):
    result = {}
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ('trades', 'cash_movements', 'receipts', 'settings', 'schema_migrations'):
        if name in names:
            values = [tuple(row) for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]
            result[name] = {'count': len(values), 'sha256': fingerprint(values)}
    return result


def inspect_database(conn, now):
    from portfolio_tracker.services.zone_forward import verified, validation_data, FORECAST_COLUMNS, digest, model_version_hash
    rows = [dict(r) for r in conn.execute('SELECT * FROM zone_prediction_log ORDER BY timestamp_prediction,id')]
    evidence = {r[0]: r[1] for r in conn.execute('SELECT sha256,payload_json FROM zone_market_evidence')}
    for row in rows:
        row['integrity_ok'] = verified(row, evidence.get(row['evidence_sha256']))
    current = model_version_hash()
    issues, near, contexts = [], [], []
    mandatory = [r[1] for r in conn.execute('PRAGMA table_info(zone_prediction_log)') if r[3]]
    last = {}
    for row in rows:
        errors = []
        if any(row.get(k) is None for k in mandatory):
            errors.append('null obligatorio')
        for name in ('predicted_touch_probability', 'predicted_close_probability'):
            p = row[name]
            if p is not None and (not np.isfinite(p) or not 0 <= p <= 1):
                errors.append(name+' fuera de rango')
        if not 0 < row['zone_low'] <= row['zone_high'] or row['reference_price'] <= 0:
            errors.append('precios incoherentes')
        if not re.fullmatch('[a-f0-9]{64}', row['model_version_hash'] or ''):
            errors.append('version inválida')
        try:
            emitted, closed, expires = (pd.Timestamp(row[n]) for n in ('timestamp_prediction','source_bar_closed_at','expires_at'))
            if any(t.tzinfo is None for t in (emitted, closed, expires)) or not closed <= emitted < expires:
                errors.append('cronología de emisión inválida')
            if row['resolved_at']:
                resolved = pd.Timestamp(row['resolved_at'])
                if resolved <= emitted or resolved < expires+pd.Timedelta('15min'):
                    errors.append('resolución anticipada')
                if row['actual_close_price'] is None:
                    errors.append('resuelto sin cierre')
                if row['actual_touch_occurred'] is None and not any(s in (row['resolution_note'] or '') for s in ('AMBIGUA','YA ALCANZADA')):
                    errors.append('toque nulo sin exclusión documentada')
            key = tuple(row[k] for k in ('symbol','model_version_hash','session_date','zone_key'))
            if key in last:
                minutes = (emitted-pd.Timestamp(last[key]['timestamp_prediction'])).total_seconds()/60
                if minutes <= 5:
                    near.append({'id':row['id'], 'previous':last[key]['id'], 'minutes':minutes,
                                 'same_closed_bar':row['source_bar_closed_at']==last[key]['source_bar_closed_at']})
            last[key] = row
        except (ValueError, TypeError):
            errors.append('timestamp no interpretable')
        try:
            cross = json.loads(row['context_json']).get('cross_asset')
            if cross:
                contexts.append({'id':row['id'], **cross})
        except (ValueError, TypeError):
            errors.append('contexto no JSON')
        if errors:
            issues.append({'id':row['id'], 'errors':errors})
    technical = Counter(tuple(r[k] for k in ('symbol','model_version_hash','source_bar_closed_at','zone_key')) for r in rows)
    resolved = [r for r in rows if r['resolved_at']]
    pending = [r for r in rows if not r['resolved_at']]
    overdue = [r for r in pending if pd.Timestamp(r['expires_at'])+pd.Timedelta('15min') <= now]
    scored = validation_data(rows)
    groups = defaultdict(list)
    for r in scored:
        for zone in (r['zone_key'], 'GLOBAL'):
            groups[(r['model_version_hash'],r['event'],zone)].append(r)
    metrics = []
    for (version,event,zone), items in sorted(groups.items()):
        sessions = defaultdict(list)
        for r in items:
            sessions[r['session_date']].append(r['brier'])
        metrics.append({'version':version,'event':event,'zone':zone,'n':len(items),'sessions':len(sessions),
                        'brier':float(np.mean([r['brier'] for r in items])),
                        'brier_equal_session':float(np.mean([np.mean(v) for v in sessions.values()])),
                        'hits':sum(r['actual'] for r in items)})
    bins = []
    for key in sorted({(r['model_version_hash'],r['event'],r['bin']) for r in scored}):
        items = [r for r in scored if (r['model_version_hash'],r['event'],r['bin']) == key]
        bins.append({'version':key[0],'event':key[1],'bin':key[2],'n':len(items),
                     'predicted':float(np.mean([r['prediction'] for r in items])),
                     'observed':float(np.mean([r['actual'] for r in items]))})
    mutation = []
    for row in rows:
        changed = dict(row, context_json='{"audit_mutation":true}')
        mutation.append(digest({k: changed[k] for k in FORECAST_COLUMNS}) != row['forecast_sha256'])
    return dict(total=len(rows), resolved=len(resolved), complete_results=sum(r['actual_touch_occurred'] is not None and r['actual_close_price'] is not None for r in resolved),
                pending=len(pending), overdue=len(overdue), overdue_ids=[r['id'] for r in overdue],
                resolved_percent=100*len(resolved)/len(rows) if rows else None,
                invalid_hashes=sum(not r['integrity_ok'] for r in rows), field_issues=issues,
                duplicates=sum(n-1 for n in technical.values() if n>1), near=near,
                repeated_hashes=sum(n-1 for n in Counter(r['forecast_sha256'] for r in rows).values() if n>1),
                coverage=[dict(symbol=s,day=d,zone=z,n=n) for (s,d,z),n in sorted(Counter((r['symbol'],r['session_date'],r['zone_key']) for r in rows).items())],
                models=dict(Counter(r['model_version_hash'] for r in rows)), current_model=current,
                current_model_records=sum(r['model_version_hash']==current for r in rows),
                correlation_records=len(contexts), numeric_correlation_records=sum(c.get('correlation') is not None for c in contexts),
                mutation_sensitive=sum(mutation), metrics=metrics, bins=bins,
                integrity=[r[0] for r in conn.execute('PRAGMA integrity_check')],
                foreign_keys=[tuple(r) for r in conn.execute('PRAGMA foreign_key_check')]), rows


def inspect_logs():
    result = {}
    for name in ('collector.log','resolver.log','catchup.log'):
        path = ROOT/'logs'/name
        if not path.exists():
            result[name] = {'exists':False}
            continue
        text = path.read_text(encoding='utf-8',errors='replace')
        lines = text.splitlines()
        starts, runs = {}, []
        for line in lines:
            match = re.match(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (EST|EDT) - \w+ - (Inicio|Fin) (\w+)', line)
            if match:
                stamp, tz, action, job = match.groups()
                value = datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S')
                if action == 'Inicio':
                    starts[job] = value
                elif job in starts:
                    runs.append({'job':job,'start':str(starts.pop(job)),'end':str(value),'seconds':(value-starts.get(job,value)).total_seconds()})
                    runs[-1]['seconds'] = (value-datetime.fromisoformat(runs[-1]['start'])).total_seconds()
        result[name] = dict(exists=True,sha256=sha256(path.read_bytes()).hexdigest(),lines=len(lines),runs=runs,
                            errors=[s for s in lines if ' - ERROR - ' in s or ' - WARNING - ' in s],tail=lines[-35:])
    return result


def scheduler_inventory():
    command = "Get-ScheduledTask -ErrorAction Stop | Where-Object { $_.TaskName -like 'GBM_Forward_*' } | Select-Object TaskName,State,TaskPath | ConvertTo-Json -Depth 3"
    run = subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,timeout=30)
    return {'checked_at':datetime.now(timezone.utc).isoformat(),'returncode':run.returncode,
            'tasks':json.loads(run.stdout) if run.returncode==0 and run.stdout.strip() else [],
            'error':run.stderr[-1200:] if run.returncode else ''}


def isolated_repo(path):
    from portfolio_tracker.db import Database
    from portfolio_tracker.repository import PortfolioRepository
    repo = PortfolioRepository(Database(path))
    repo.database.initialize()
    repo.ensure_zone_forward_schema()
    return repo


def probes(folder):
    from scripts import autopilot_runtime as runtime
    from scripts.autopilot_market_cache import MarketCache
    from portfolio_tracker.services import forward_market, zone_forward
    from portfolio_tracker.analytics import cross_correlation as cc
    from tests.test_zone_forward import prediction, market
    from tests.test_pdf_report import _analysis
    results = []
    def check(name, action):
        try:
            result = action()
            results.append(dict(name=name,**result))
        except Exception as exc:
            results.append(dict(name=name,passed=False,error=repr(exc)))
    def offline():
        repo = isolated_repo(folder/'offline.db')
        log = runtime.logger(folder/'offline_logs')
        with patch.object(runtime,'fundamental_context',return_value=(None,'')), patch.object(MarketCache,'frames',side_effect=ConnectionError('AUDIT simulated network outage')):
            code = runtime.collect(repo,['SMCI','NVDA'],folder/'offline_state',log,now_fn=lambda:pd.Timestamp('2026-09-03T15:00:10Z').to_pydatetime())
        text = (folder/'offline_logs/collector.log').read_text(encoding='utf-8')
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)
        return dict(passed=code==1 and not repo.zone_predictions() and 'SMCI' in text and 'NVDA' in text and 'ERROR' in text,
                    exit_code=code, detail='Fallo inyectado en proveedor; continúa ambos símbolos; salida controlada 1 para reintento.',log=text[-3500:])
    check('Red desconectada (inyección controlada)',offline)
    def early():
        repo = isolated_repo(folder/'early.db')
        item = prediction()
        repo.save_prediction(item,now=item.timestamp_prediction)
        with patch.object(forward_market,'resolution_frames',side_effect=AssertionError('No debe descargar antes del cierre')):
            code = runtime.resolve(repo,['SMCI'],logging.getLogger('audit'),now=pd.Timestamp('2026-08-31T19:00:00Z').to_pydatetime())
        return dict(passed=code==0 and repo.zone_predictions()[0]['resolved_at'] is None, detail='15:00 NY: no consulta cierre futuro y conserva pendiente.')
    check('Resolvedor antes del cierre',early)
    def catchup():
        repo = isolated_repo(folder/'catchup.db')
        for symbol in ('SMCI','NVDA'):
            for day in ('2026-08-27','2026-08-28'):
                item = prediction(symbol=symbol,timestamp_prediction=day+'T15:00:00Z',source_bar_closed_at=day+'T15:00:00Z')
                repo.save_prediction(item,now=item.timestamp_prediction)
        calls=[]
        def provider(symbol,day):
            calls.append((symbol,day))
            return market(day)
        with patch.object(forward_market,'resolution_frames',side_effect=provider):
            first=runtime.resolve(repo,['SMCI','NVDA'],logging.getLogger('audit'),catchup=True,now=pd.Timestamp('2026-08-31T13:05:00Z').to_pydatetime())
            second=runtime.resolve(repo,['SMCI','NVDA'],logging.getLogger('audit'),catchup=True,now=pd.Timestamp('2026-08-31T13:06:00Z').to_pydatetime())
        rows=repo.zone_predictions()
        return dict(passed=first==second==0 and len(calls)==4 and all(r['resolved_at'] and r['integrity_ok'] for r in rows),
                    detail='Dos activos, dos sesiones atrasadas; resuelve 4 registros. Segundo arranque: cero descargas adicionales.',calls=calls)
    check('Catch-up multiactivo, atrasos e idempotencia',catchup)
    def crash():
        path=folder/'crash.db'
        isolated_repo(path)
        code="""import sys,os
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from tests.test_zone_forward import prediction
r=PortfolioRepository(Database(sys.argv[1]))
item=prediction()
r.save_prediction(item,now=item.timestamp_prediction)
with r.database.transaction() as conn:
    conn.execute("INSERT INTO zone_market_evidence VALUES (?,?)",('audit-uncommitted','{}'))
    os._exit(79)
"""
        run=subprocess.run([sys.executable,'-c',code,str(path)],cwd=ROOT,capture_output=True,text=True,timeout=20)
        with sqlite3.connect(path) as conn:
            integrity=conn.execute('PRAGMA integrity_check').fetchone()[0]
            committed=conn.execute('SELECT count(*) FROM zone_prediction_log').fetchone()[0]
            uncommitted=conn.execute('SELECT count(*) FROM zone_market_evidence').fetchone()[0]
        return dict(passed=run.returncode==79 and integrity=='ok' and committed==1 and uncommitted==0,
                    detail='Proceso hijo termina sin finally/rollback: commit previo persiste; transacción abierta desaparece. No equivale a corte eléctrico físico.',
                    child_exit=run.returncode,integrity=integrity,committed=committed,uncommitted=uncommitted)
    check('Terminación abrupta durante transacción SQLite',crash)
    def corr_threshold():
        base=replace(_analysis(),probability_up=60,probability_down=40,risk_veto=False,long_entry_blocked=False,position_state='FLAT')
        values=[]
        for corr in (.2,.70,.71):
            supplied={'status':'available','symbol':'SMCI','peer':'NVDA','correlation':corr,'proposed_impact':5.,'detail':'audit controlled context'}
            value=cc.apply_cross_context(base,supplied).probability_up-base.probability_up
            values.append({'correlation':corr,'applied':value})
        return dict(passed=values[0]['applied']==values[1]['applied']==0 and values[2]['applied']==5,
                    detail='Prueba de defensa de la función de aplicación; el constructor normal sí filtra >0.70, pero se comprueba también la frontera pública.',values=values)
    check('Umbral 0.70 revalidado en apply_cross_context',corr_threshold)
    def hash_identity():
        r1,r2=isolated_repo(folder/'id1.db'),isolated_repo(folder/'id2.db')
        item=prediction()
        for r in (r1,r2):
            r.save_prediction(item,now=item.timestamp_prediction)
        a,b=r1.zone_predictions()[0],r2.zone_predictions()[0]
        content=[k for k in zone_forward.FORECAST_COLUMNS if k!='id']
        return dict(passed=all(a[k]==b[k] for k in content),
                    detail='Mismo contenido da UUID y forecast_sha256 diferentes por diseño. Comparar contenido canónico sin id, no firmas de registros distintos.',
                    same_forecast_sha256=a['forecast_sha256']==b['forecast_sha256'],same_content=all(a[k]==b[k] for k in content))
    check('Equivalencia semántica frente a identidad SHA-256',hash_identity)
    return results


def benchmark_and_parity(folder, network):
    """Replay exact UI call AST, same frames, clock, memory and peer snapshot."""
    from scripts import autopilot_runtime as runtime
    from scripts.autopilot_market_cache import MarketCache
    from portfolio_tracker.services import cross_asset
    from portfolio_tracker.services.price_zones import build_zone_snapshot
    from portfolio_tracker.services.operational_state import macro_memory, synchronize_position
    from portfolio_tracker.analytics.technical_probability import analyze_probability
    from portfolio_tracker.analytics.fundamental_news import apply_fundamental_filter
    from portfolio_tracker.services.zone_forward import log_snapshot, FORECAST_COLUMNS
    app_tree=ast.parse((ROOT/'app.py').read_text(encoding='utf-8'))
    func=next(n for n in app_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_probability_predictor_content')
    call_names=('analyze_probability','enrich_cross_asset','synchronize_position')
    expressions={name:next(n for n in ast.walk(func) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name) for name in call_names}
    results=[]
    if not network:
        return [{'status':'NO MEDIDO','reason':'Usar --benchmark-network para captura real y medición fría/caliente.'}]
    # Keep provider cache writes inside the temporary directory as well.
    for symbol in ('SMCI','NVDA'):
        try:
            cache=MarketCache(folder/'benchmark_market')
            times={}
            start=perf_counter()
            frames=cache.frames(symbol,pd.Timestamp.now(tz='UTC'))
            times['download_cold_s']=perf_counter()-start
            start=perf_counter()
            warm=cache.frames(symbol,pd.Timestamp.now(tz='UTC'))
            times['download_warm_s']=perf_counter()-start
            peer=cross_asset.PEERS[symbol]
            start=perf_counter()
            peer_frames=cross_asset._download_peer(peer)
            times['peer_cold_s']=perf_counter()-start
            repo=isolated_repo(folder/(symbol+'_headless.db'))
            start=perf_counter()
            fundamental, fh=runtime.fundamental_context(repo,symbol,logging.getLogger('audit'))
            times['fundamental_cold_s']=perf_counter()-start
            now=pd.Timestamp.now(tz='UTC').to_pydatetime()
            intraday,daily=warm
            def enriched(a, i, d, **kwargs):
                return cross_asset.enrich_cross_asset(a,i,d,now=now,peer_loader=lambda _:peer_frames)
            with patch.object(cross_asset,'prefetch_cross_asset',return_value=None), patch.object(runtime,'clock',return_value=now):
                original=cross_asset.enrich_cross_asset
                # Resolve recursion by keeping the original explicitly.
                def enriched(a,i,d,**kwargs):
                    return original(a,i,d,now=now,peer_loader=lambda _:peer_frames)
                with patch.object(cross_asset,'enrich_cross_asset',side_effect=enriched):
                    start=perf_counter()
                    head=runtime.analyze_headless(repo,symbol,intraday,daily,fundamental,fh,now,logging.getLogger('audit'))
                    head_zone=build_zone_snapshot(head,now=now)
                    times['headless_compute_and_zones_s']=perf_counter()-start
            ui_repo=isolated_repo(folder/(symbol+'_ui.db'))
            class FixedClock:
                @staticmethod
                def now(*_): return now
            env=dict(symbol=symbol,intraday=intraday,daily=daily,repository=ui_repo,
                     calibrated_parameters={'minimum_probability':.55,'stop_atr_multiple':2.25,'risk_per_trade_pct':1.},
                     datetime=FixedClock,timezone=timezone,macro_memory=macro_memory,
                     analyze_probability=analyze_probability,enrich_cross_asset=enriched,
                     synchronize_position=synchronize_position)
            start=perf_counter()
            for name in call_names:
                env['analysis']=eval(compile(ast.Expression(expressions[name]),'app.py','eval'),env)
                if name=='analyze_probability' and fundamental is not None:
                    env['analysis']=replace(apply_fundamental_filter(env['analysis'],fundamental),fundamental_snapshot_sha256=fh)
            ui=env['analysis']
            ui_zone=build_zone_snapshot(ui,now=now)
            times['ui_extracted_compute_and_zones_s']=perf_counter()-start
            start=perf_counter()
            _=build_zone_snapshot(head,now=now)
            times['zone_repeat_s']=perf_counter()-start
            log_snapshot(repo,head,head_zone,now=now)
            log_snapshot(ui_repo,ui,ui_zone,now=now)
            hs,us=repo.zone_predictions(),ui_repo.zone_predictions()
            cols=[k for k in FORECAST_COLUMNS if k!='id']
            canon=lambda rows: sorted([{k:r[k] for k in cols} for r in rows],key=lambda r:r['zone_key'])
            same_snapshot=asdict(head_zone)==asdict(ui_zone)
            equality=bool(hs) and canon(hs)==canon(us)
            # Repeated warm computation uses identical captured data, not a memoized analysis.
            times['warm_lower_bound_s']=times['download_warm_s']+times['headless_compute_and_zones_s']
            times['cold_serial_measured_s']=times['download_cold_s']+times['peer_cold_s']+times['fundamental_cold_s']+times['headless_compute_and_zones_s']
            results.append(dict(symbol=symbol,status='MEDIDO',timings=times,as_of=now.isoformat(),
                                intraday_bars=len(intraday),daily_bars=len(daily),same_zone_snapshot=same_snapshot,
                                same_forecast_content=equality,headless_records=len(hs),ui_records=len(us),
                                hashes_equal=sorted(r['forecast_sha256'] for r in hs)==sorted(r['forecast_sha256'] for r in us),
                                parameters='defaults, same empty operational/account state',
                                fundamental_available=fundamental is not None,correlation=head.cross_asset_context,
                                scope='AST de las tres llamadas reales UI; no navegador, no calibración de otros horizontes ni renderizado/PDF.',
                                inputs_sha256=fingerprint([f.to_json(orient='split',date_format='iso') for f in (intraday,daily,*peer_frames)])))
        except Exception as exc:
            results.append(dict(symbol=symbol,status='ERROR DE MEDICIÓN',error=repr(exc)))
    return results


def run_tests(out, temp):
    target=out/'pytest.xml'
    env=dict(os.environ,GBM_PORTFOLIO_DATA_DIR=str(temp/'pytest_data'),PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',f'--junitxml={target}','--tb=short'],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    (out/'pytest.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    data={'exit_code':result.returncode,'tail':result.stdout[-500:]}
    if target.exists():
        suite=ET.parse(target).getroot().find('testsuite')
        data.update(suite.attrib)
    return data


def finding(code, priority, title, evidence, action, references):
    return dict(code=code,priority=priority,title=title,evidence=evidence,action=action,references=references)


def findings(data):
    db=data['database']; result=[]
    tasks=data['scheduler']
    if tasks['returncode']==0 and not tasks['tasks']:
        result.append(finding('A01','ALTA','Autopiloto no instalado',
            'Consulta Windows sin ninguna tarea GBM_Forward_*. Los logs muestran ejecuciones manuales, no operación diaria desatendida.',
            'Instalar las tres tareas con credenciales Windows; verificar ejecuciones reales, horarios NY y reintentos.',
            'scripts/install_autopilot_tasks.ps1:30-35; inventario del Programador adjunto'))
    elif tasks['returncode']:
        result.append(finding('A01','ALTA','Instalación del autopiloto no verificable',tasks['error'],
                              'Consultar tareas con permisos de lectura antes de declarar operación desatendida.','Windows Task Scheduler'))
    if not data['backup']['encrypted_exists']:
        result.append(finding('A02','ALTA','No hay respaldo cifrado operativo verificable',
            'Existe implementación AES-256-GCM, pero faltan backups/portfolio.db.aesgcm y/o manifiesto. El runtime no invoca backup antes de escribir.',
            'Programar copia consistente diaria y prueba de restauración con clave custodiada; no confundir código de backup con un backup realizado.',
            'scripts/github_backup.py:44-127; scripts/autopilot_runtime.py:292-331'))
    if not db['metrics']:
        result.append(finding('A03','ALTA','Acierto y calibración todavía no demostrables',
            f"{db['total']} pronósticos, {db['resolved']} resueltos, {db['overdue']} vencidos con gracia de 15 min. Brier real y curva de fiabilidad: N/D, no cero.",
            'Acumular sesiones futuras, resolver con cierres exactos y revisar Brier/fiabilidad por versión; conservar PRELIMINAR.',
            'zone_prediction_log; portfolio_tracker/services/zone_forward.py:351-377'))
    if db['overdue']:
        result.append(finding('A04','ALTA','Predicciones vencidas sin resolver',str(db['overdue'])+' registros vencidos.',
                              'Investigar datos de cierre y ejecutar catch-up autorizado; no imputar precios actuales.', 'zone_prediction_log.expires_at'))
    measured=[r for r in data['benchmark'] if r['status']=='MEDIDO']
    if any(r['timings']['warm_lower_bound_s']>2 or r['timings']['cold_serial_measured_s']>2 for r in measured):
        result.append(finding('A05','MEDIA','No cumple el objetivo de dos segundos por activo',
            '; '.join(f"{r['symbol']}: frío serial {r['timings']['cold_serial_measured_s']:.2f}s, caliente mínimo {r['timings']['warm_lower_bound_s']:.2f}s" for r in measured),
            'Instrumentar tramos; memoizar indicadores por corte, descargar incremental, sacar PDF del ciclo crítico y precargar contexto. Medir p50/p95 antes de prometer SLA.',
            'scripts/autopilot_market_cache.py:45-98; app.py:171-179,1716-1740; benchmark adjunto'))
    result.append(finding('A06','MEDIA','Versión histórica e identidad no reproducibles sólo con el hash',
        f"{len(db['models'])} hashes de motor registrados; {db['current_model_records']} registros coinciden con el código actual. El hash anterior no es corrupción, pero sin manifiesto de fuentes no prueba qué binarios/reglas lo generaron. forecast_sha256 incluye UUID.",
        'Archivar manifiesto por versión, dependencias, parámetros y entradas. Comparar UI/headless por contenido canónico sin id y reloj congelado; no exigir igualdad de firmas de registros diferentes.',
        'portfolio_tracker/services/zone_forward.py:23-30,156-168,220-231'))
    threshold=next(p for p in data['probes'] if p['name'].startswith('Umbral'))
    if not threshold['passed']:
        result.append(finding('A07','MEDIA','El aplicador de correlación confía ciegamente en el contexto',
            'build_cross_context sí exige correlación >0.70; apply_cross_context acepta status=available y proposed_impact=5 incluso con correlación 0.20. Reproducción en frontera pública, no evidencia de bono indebido en datos reales.',
            'Revalidar correlación finita >0.70 y corte sincronizado en el aplicador, o usar un contrato interno validado.',
            'portfolio_tracker/analytics/cross_correlation.py:181-191,205-219'))
    result.append(finding('A08','MEDIA','Recuperación no ilimitada y grupo de seis no atómico',
        'Cada zona se confirma en transacción separada; un corte entre zonas deja grupo parcial íntegro. Catch-up depende de descargar todas las velas 5m y el diario; el proveedor no garantiza disponibilidad intradía más allá de 60 días.',
        'Transacción única por seis zonas o estado explícito de grupo; archivar sesiones completas al cierre y resolver desde evidencia local firmada cuando el proveedor ya no las sirve.',
        'portfolio_tracker/services/zone_forward.py:245-270,289-316; https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html'))
    result.append(finding('A09','MEDIA','La prueba UI/headless existente no valida la ingesta real',
        'El test previo sustituye analyze_probability y usa frames None. Esta auditoría ejecuta llamadas UI extraídas del AST con datos reales y reloj compartido; no prueba estado de cuenta activo ni todas las rutas de calibración. Las descargas y cachés de los dos flujos difieren.',
        'Extraer un servicio de snapshot común con reloj, frames y memoria explícitos; añadir pruebas extremo a extremo con posiciones activas, fallos y límites de vela.',
        'tests/test_autopilot.py:170-190; app.py:1511-1632; scripts/autopilot_runtime.py:165-195'))
    result.append(finding('A10','BAJA','Observabilidad incompleta de los trabajos',
        'Los tres trabajos usan collector.log. No hay resolver.log/catchup.log, duración estructurada por símbolo ni evidencia de un ciclo completo 11:00/17:00/09:05. Las mediciones históricas son de precisión de un segundo.',
        'Añadir job_id, tiempos por etapa, resumen de pendientes/vencidos y heartbeat; mantener log unificado si se documenta y puede filtrarse.',
        'scripts/autopilot_runtime.py:41-62,294-333; logs/collector.log'))
    if db['invalid_hashes'] or db['field_issues'] or db['duplicates'] or db['integrity']!=['ok']:
        result.append(finding('A11','CRITICA','Inconsistencia de datos detectada',
            f"Firmas inválidas {db['invalid_hashes']}; campos {len(db['field_issues'])}; duplicados exactos {db['duplicates']}; SQLite {db['integrity']}",
            'Aislar registros; investigar sin recomputar hashes para ocultar cambios.', 'JSON adjunto, database'))
    return result


def charts(data):
    from reportlab.graphics.shapes import Drawing,Line,Rect,String,Circle
    drawings=[]
    for kind in ('coverage','reliability'):
        d=Drawing(480,190)
        left,bottom,width,height=42,34,420,135
        d.add(Line(left,bottom,left+width,bottom,strokeColor=__import__('reportlab.lib.colors',fromlist=['HexColor']).HexColor('#666666')))
        d.add(Line(left,bottom,left,bottom+height))
        if kind=='coverage':
            counts=Counter()
            for r in data['database']['coverage']:
                counts[(r['day'],r['symbol'])]+=r['n']
            maximum=max(counts.values(),default=1)
            for i,((day,symbol),n) in enumerate(sorted(counts.items())):
                slot=width/max(len(counts),1); x=left+i*slot+8
                d.add(Rect(x,bottom,max(6,slot-16),height*n/maximum,fillColor=__import__('reportlab.lib.colors',fromlist=['HexColor']).HexColor('#34485E')))
                d.add(String(x,bottom+height*n/maximum+5,str(n),fontSize=9))
                d.add(String(x,20,day,fontSize=8)); d.add(String(x,8,symbol,fontSize=8))
        else:
            d.add(Line(left,bottom,left+width,bottom+height,strokeColor=__import__('reportlab.lib.colors',fromlist=['HexColor']).HexColor('#AAAAAA'),strokeDashArray=[3,3]))
            bins=data['database']['bins']
            if not bins:
                d.add(String(125,108,'SIN RESULTADOS RESUELTOS',fontName='Helvetica-Bold',fontSize=11))
                d.add(String(108,87,'No se dibujan frecuencias reales ficticias.',fontSize=10))
            for b in bins:
                color=__import__('reportlab.lib.colors',fromlist=['HexColor']).HexColor('#163F63' if b['event']=='Toque' else '#7B5134')
                d.add(Circle(left+width*b['predicted'],bottom+height*b['observed'],3.5,fillColor=color))
            d.add(String(120,10,'Probabilidad emitida (0 a 1); eje Y: frecuencia observada',fontSize=8))
            d.add(String(45,174,'Toque: azul / Cierre: marrón. Diagonal: referencia, no resultados.',fontSize=8))
        drawings.append(d)
    return drawings


def build_pdf(data,path):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='AuditBody',fontName='Helvetica',fontSize=9,leading=13,spaceAfter=6))
    styles.add(ParagraphStyle(name='AuditSmall',fontName='Helvetica',fontSize=7.5,leading=10,spaceAfter=4))
    styles['Heading1'].textColor=colors.HexColor('#18354A')
    def p(text,small=False):
        return Paragraph(escape(str(text)).replace('\n','<br/>').replace('—','-').replace('–','-'),styles['AuditSmall' if small else 'AuditBody'])
    def table(headers,rows,widths):
        body=[[p(v,True) for v in headers]]+[[p(v,True) for v in row] for row in rows]
        t=Table(body,colWidths=[x*mm for x in widths],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DEE7EF')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#C9D2DA')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return t
    def section(title):
        return Paragraph(title,styles['Heading1'])
    db=data['database']; coverage,reliability=charts(data)
    story=[Spacer(1,20),Paragraph('AUDITORÍA INTEGRAL',styles['Title']),Paragraph('Predicciones, autopiloto y backend',styles['Heading1']),
           p(data['observed_at']+' UTC | '+data['local_time']+' Nueva York'),Spacer(1,14),
           Paragraph(data['verdict'],styles['Title']),p('Criterio: aptitud para recolección desatendida, trazabilidad y validación. No se aprueba sólo porque pasen pruebas unitarias.'),
           table(['Indicador','Resultado'],[['Predicciones / resueltas',f"{db['total']} / {db['resolved']}"],['Firmas inválidas / duplicados exactos',f"{db['invalid_hashes']} / {db['duplicates']}"],['Pendientes / vencidas (+15 min)',f"{db['pending']} / {db['overdue']}"],['Brier real', 'N/D: sin muestras evaluables' if not db['metrics'] else 'Ver sección de validación'],['Pruebas unitarias',str(data['tests'].get('tests','No ejecutadas'))+'; fallos '+str(data['tests'].get('failures','N/D'))],['Motor actual SHA-256',db['current_model']]], [77,97]),
           Spacer(1,14),p('Conclusión ejecutiva: '+('; '.join(f['title'] for f in data['findings'] if f['priority'] in ('ALTA','CRITICA')) or 'Revisar hallazgos documentados.')),
           p('Protección: lectura SQLite mode=ro y snapshot consistente en RAM. Ninguna resolución, mutación ni instalación sobre producción. Pruebas destructivas exclusivamente en bases temporales.'),PageBreak()]
    story += [section('1. Alcance, método y limitaciones'),p('Se revisaron fuente del motor y probabilidades de zonas, colector/resolvedor/catch-up, correlación, registros firmados, interfaz y reportes. Se conservan archivos previos no confirmados en Git. No se corrigen hallazgos durante esta auditoría.'),
              p('Los datos de producción se congelan al inicio. Las consultas externas de rendimiento pueden ocurrir después. Una predicción de hoy todavía no vencida no constituye fallo del resolvedor. Un toque nulo puede ser una exclusión legítima: zona ya alcanzada o vela parcial ambigua.'),
              p('SHA-256 demuestra consistencia de contenido frente a su huella, no autenticidad frente a un administrador que reescribe ambos. Hash del pronóstico y hash del motor representan objetos diferentes. El UUID forma parte del hash del pronóstico.'),
              p('Simulación de apagado: terminación abrupta de proceso con transacción abierta. No se interrumpió la electricidad, el sistema operativo ni el almacenamiento de la PC. No demuestra durabilidad del hardware ante todos los fallos.'),
              p(f"Archivos inventariados: {len(data['sources'])}. Inventario completo y huellas en JSON adjunto. SQLite integrity_check: {db['integrity']}; foreign_key_check: {len(db['foreign_keys'])} incidencias."),
              section('2. Cobertura y calidad de datos'),table(['Símbolo','Fecha','Zona','N'],[[r['symbol'],r['day'],r['zone'],r['n']] for r in db['coverage']],[28,42,40,64]),coverage,
              table(['Comprobación','Resultado'],[['Campos incoherentes',len(db['field_issues'])],['Hashes repetidos',db['repeated_hashes']],['Cambios en contexto detectados por hash',f"{db['mutation_sensitive']} / {db['total']}"],['Emisiones próximas (<=5 min)',len(db['near'])],['Con contexto correlación / valor numérico',f"{db['correlation_records']} / {db['numeric_correlation_records']}"],['Resultados completos (toque y cierre)',db['complete_results']]],[94,80]),
              p('Emisiones cercanas en velas distintas no son duplicados técnicos. No se suman como ensayos independientes: la cohorte estadística usa la primera emisión por zona, sesión y versión.'),
              section('3. Versiones y trazabilidad'),table(['Hash del modelo','Registros','Coincide actual'],[[v,n,'Sí' if v==db['current_model'] else 'No; versión anterior/no catalogada'] for v,n in db['models'].items()],[100,24,50]),
              p('La firma de las nuevas predicciones incluye context_json.cross_asset. Los registros previos sin correlación no se rellenan a posteriori. No se dispone aquí de un manifiesto fuente firmado de cada versión histórica; su semántica exacta no queda demostrada sólo por conservar 64 caracteres.'),PageBreak()]
    story += [section('4. Resolución, Brier y fiabilidad'),p(f"Resueltos: {db['resolved']} de {db['total']} ({db['resolved_percent'] or 0:.2f}%). Pendientes: {db['pending']}; vencidos tras gracia de 15 min: {db['overdue']}."),
              p('Brier binario = promedio de (probabilidad - resultado 0/1)^2, rango [0,1]; menor es mejor. Se separan Toque y Cierre, zonas y versiones. Se muestra además el promedio con igual peso por sesión. Los seis niveles y los dos activos pueden estar correlacionados.'),reliability]
    if db['metrics']:
        story.append(table(['Versión','Evento','Zona','N','Brier'],[[r['version'][:10],r['event'],r['zone'],r['n'],f"{r['brier']:.4f}"] for r in db['metrics']],[35,32,40,20,47]))
    else:
        story.append(table(['Evento','Zonas','N evaluable','Brier'],[['Toque','ENTRY1/2/3, TP1/2, R3',0,'N/D'],['Cierre','ENTRY1/2/3, TP1/2, R3',0,'N/D']],[28,80,30,36]))
        story.append(p('La línea diagonal sólo es referencia matemática. No hay curva empírica calculable sin resultados. Fabricar frecuencias o poner Brier=0 sería engañoso. El sistema sigue siendo PRELIMINAR.'))
    story += [section('5. Automatización, logs y respaldos'),p('Programador Windows: '+('consulta completada; tareas GBM: '+str(data['scheduler']['tasks']) if not data['scheduler']['returncode'] else 'no verificable: '+data['scheduler']['error'])),
              p('Respaldo cifrado encontrado: '+str(data['backup']['encrypted_exists'])+'. Manifiesto: '+str(data['backup']['manifest_exists'])+'. No se leyó ni expuso ninguna clave. No se ejecutó backup/push durante la auditoría.'),
              table(['Log','Existe','Líneas','Errores/avisos'],[[n,r['exists'],r.get('lines',0),len(r.get('errors',[]))] for n,r in data['logs'].items()],[65,28,30,51])]
    runs=[r for v in data['logs'].values() for r in v.get('runs',[])]
    story.append(table(['Trabajo','Inicio NY','Duración (s)'],[[r['job'],r['start'],r['seconds']] for r in runs],[45,91,38]))
    story += [p('Los inicios y finales incluyen check-only, salidas por horario e idempotencia; una duración de cero segundos no significa que se haya calculado un pronóstico completo.'),PageBreak(),section('6. Pruebas controladas de fallos')]
    for probe in data['probes']:
        story.append(KeepTogether([Paragraph(probe['name'],styles['Heading2']),p(('PASÓ' if probe['passed'] else 'FALLÓ')+' - '+probe.get('detail',probe.get('error','')))]))
    story += [p('La prueba del umbral examina defensa de la función pública. El camino habitual build_cross_context sí controla >0.70; no se afirma que haya aplicado un bono incorrecto en producción sin evidencia.'),
              section('7. Consistencia UI/headless y rendimiento')]
    for r in data['benchmark']:
        story.append(Paragraph(r.get('symbol','Benchmark')+' - '+r['status'],styles['Heading2']))
        if r['status']!='MEDIDO':
            story.append(p(r.get('error',r.get('reason','')))); continue
        story.append(table(['Tramo','Segundos'],[[k,f'{v:.3f}'] for k,v in r['timings'].items()],[118,56]))
        story.append(p(f"{r['intraday_bars']} velas 5m; {r['daily_bars']} diarias. Snapshot de zonas idéntico: {r['same_zone_snapshot']}; contenido firmado sin UUID idéntico: {r['same_forecast_content']}; registros headless/UI: {r['headless_records']}/{r['ui_records']}; hashes de registros iguales: {r['hashes_equal']}."))
        story.append(p(r['scope']+' Una corrida fría y una caliente por activo: no es p95 ni garantía de SLA. Frío serial suma descargas medidas; la UI puede solaparlas. El mínimo caliente omite fundamentales, par, calibración y PDF; tampoco es tiempo de render de navegador.'))
    story += [section('8. Hallazgos y acciones correctivas')]
    for f in data['findings']:
        story.append(KeepTogether([Paragraph(f"{f['code']} | {f['priority']} | {f['title']}",styles['Heading2']),p(f['evidence'])]))
        story += [p('Acción: '+f['action']),p('Evidencia: '+f['references'],True)]
    story += [PageBreak(),section('9. Pruebas, integridad y cierre'),p(json.dumps(data['tests'],ensure_ascii=False)),
              p('Manifiesto contable antes/después (lectura): '+('idéntico' if data['ledger_unchanged'] else 'cambió; revisar actividad concurrente de la aplicación; la auditoría no tiene permisos de escritura sobre su conexión origen')),
              table(['Tabla','Filas','SHA-256 del contenido'],[[name,value['count'],value['sha256']] for name,value in data['ledger_before'].items()],[42,18,114]),
              p('Archivos fuente existentes alterados por esta auditoría: '+str(data['changed_sources'])),
              p('Cierre: '+data['verdict']+'. Priorizar instalación y evidencia de ejecución del scheduler, respaldo restaurable, recolección/resolución real y medición de rendimiento. No se corrigen ni descartan registros para forzar aprobación.'),
              section('10. Fuentes y anexos'),p('Código local y SQLite congelados en la fecha indicada; source inventory y métricas completas: audit.json. Salida íntegra de pytest: pytest.log y pytest.xml. Logs de producción consultados: logs/collector.log.'),
              p('SQLite WAL / durabilidad: https://www.sqlite.org/wal.html\nSQLite prevención de corrupción: https://www.sqlite.org/howtocorrupt.html\nyfinance, límites intradía: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html'),
              p('Limitación de retención: la documentación del proveedor limita históricos intradía a los últimos 60 días; no se puede prometer catch-up ilimitado sin archivos locales de mercado. [yfinance]'),
              p('Atomicidad de transacción y durabilidad eléctrica son propiedades diferentes; SQLite depende también de sincronización y almacenamiento. Se probaron transacciones con terminación de proceso, no fallos físicos. [SQLite]')]
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#64748B'))
        canvas.drawString(18*mm,12*mm,'GBM+ | Auditoría de predicciones | Uso interno')
        canvas.drawRightString(192*mm,12*mm,f'{doc.page}')
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=20*mm,
                      title='Auditoría integral de predicciones y backend',author='Auditoría técnica').build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=ROOT/'data/portfolio.db')
    parser.add_argument('--run-tests',action='store_true')
    parser.add_argument('--tests-xml',type=Path,help='Reutiliza evidencia JUnit reciente si ninguna fuente del sistema cambió después')
    parser.add_argument('--evidence-dir',type=Path,help='Reutiliza probes/benchmark recientes si ninguna fuente, salvo este generador, cambió después')
    parser.add_argument('--benchmark-network',action='store_true')
    parser.add_argument('--scheduler-json',type=Path,help='Inventario de sólo lectura obtenido con permisos Windows')
    args=parser.parse_args()
    now=pd.Timestamp.now(tz='UTC')
    out=ROOT/'output'/'audit_predictions'/now.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    inventory=source_inventory()
    conn=snapshot_production(args.database)
    before=ledger_manifest(conn)
    db,rows=inspect_database(conn,now)
    conn.close()
    print(json.dumps({'database':{k:v for k,v in db.items() if k not in ('coverage','near','models','bins','metrics','field_issues')},'coverage':db['coverage']},default=json_safe),flush=True)
    with tempfile.TemporaryDirectory(prefix='gbm-prediction-audit-',ignore_cleanup_errors=True) as directory:
        folder=Path(directory)
        with patch.dict(os.environ,{'GBM_PORTFOLIO_DATA_DIR':str(folder/'runtime_data')}):
            # Config may already be imported by zone_forward; override only local
            # audit process constants, never project files or live process state.
            from portfolio_tracker import config
            import portfolio_tracker.services.forward_market as market_module
            config.DATA_DIR=folder/'runtime_data'; config.RECEIPTS_DIR=config.DATA_DIR/'receipts'
            market_module.DATA_DIR=config.DATA_DIR
            if args.run_tests:
                tests=run_tests(out,folder)
            elif args.tests_xml:
                prior=args.tests_xml.resolve()
                changed=[name for name in inventory if name!='scripts/audit_predictions.py' and
                         (ROOT/name).stat().st_mtime > prior.stat().st_mtime]
                if changed:
                    raise ValueError('Las pruebas deben repetirse: fuentes posteriores a JUnit: '+str(changed))
                suite=ET.parse(prior).getroot().find('testsuite')
                tests={**suite.attrib,'exit_code':int(suite.attrib.get('failures','0'))+int(suite.attrib.get('errors','0')),
                       'reused_source':str(prior),'reused_reason':'Fuentes del sistema sin cambios desde esta ejecución; sólo se ajustó el generador de auditoría.'}
                import shutil
                shutil.copyfile(prior,out/'pytest.xml')
                if prior.with_suffix('.log').exists():
                    shutil.copyfile(prior.with_suffix('.log'),out/'pytest.log')
            else:
                tests={'status':'NO EJECUTADO'}
            if args.evidence_dir:
                evidence=args.evidence_dir.resolve()
                evidence_cut=min((evidence/'probes.json').stat().st_mtime,(evidence/'benchmark.json').stat().st_mtime)
                changed=[name for name in inventory if name!='scripts/audit_predictions.py' and
                         (ROOT/name).stat().st_mtime > evidence_cut]
                if changed:
                    raise ValueError('Los probes/benchmark deben repetirse: fuentes posteriores: '+str(changed))
                checks=json.loads((evidence/'probes.json').read_text(encoding='utf-8'))
                benchmark=json.loads((evidence/'benchmark.json').read_text(encoding='utf-8'))
            else:
                checks=probes(folder)
                benchmark=benchmark_and_parity(folder,args.benchmark_network)
            (out/'probes.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=json_safe),encoding='utf-8')
            (out/'benchmark.json').write_text(json.dumps(benchmark,ensure_ascii=False,indent=2,default=json_safe),encoding='utf-8')
            # sqlite3 context managers commit/rollback, but do not close cycles.
            # Release temporary connection cycles before Windows removes files.
            import gc
            gc.collect()
            if args.benchmark_network:
                # yfinance holds process-global SQLite connections on Windows.
                # Close only this audit process's disposable provider caches.
                import yfinance.cache as yf_cache
                for name in ('_CookieDBManager','_TzDBManager','_ISINDBManager'):
                    manager=getattr(yf_cache,name,None)
                    if manager is not None:
                        manager.close_db()
    current=snapshot_production(args.database)
    after=ledger_manifest(current); current.close()
    scheduler=json.loads(args.scheduler_json.read_text(encoding='utf-8-sig')) if args.scheduler_json else scheduler_inventory()
    data=dict(observed_at=now.isoformat(),local_time=now.tz_convert('America/New_York').isoformat(),database=db,
              logs=inspect_logs(),scheduler=scheduler,tests=tests,probes=checks,benchmark=benchmark,sources=inventory,
              backup={'encrypted_exists':(ROOT/'backups/portfolio.db.aesgcm').is_file(),'manifest_exists':(ROOT/'backups/manifest.json').is_file()},
              ledger_before=before,ledger_after=after,ledger_unchanged=before==after,
              changed_sources=[name for name,meta in source_inventory().items() if inventory.get(name)!=meta])
    data['findings']=findings(data)
    data['verdict']='RECHAZADO' if any(f['priority'] in ('ALTA','CRITICA') for f in data['findings']) or any(not r['passed'] for r in checks) else 'APROBADO'
    (out/'audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=json_safe,allow_nan=False),encoding='utf-8')
    pdf=ROOT/'output/pdf'/('auditoria_integral_'+now.strftime('%Y%m%d')+'.pdf')
    pdf.parent.mkdir(parents=True,exist_ok=True)
    build_pdf(data,pdf)
    print(json.dumps({'report':str(pdf),'evidence':str(out),'verdict':data['verdict'],'probes':checks,'benchmark':benchmark},default=json_safe,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
