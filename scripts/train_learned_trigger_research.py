"""One-off, offline 1h/1D learned-LONG-trigger experiment on audited replay.

Never imports Streamlit, submits an order, opens portfolio.db, or changes the
live AND gate. First preregister, then construct *research-only* possible-fill
labels, then nested chronological development and a one-time internal holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.historical_replay import validate_historical_replay
from portfolio_tracker.analytics.holdout_registry import reserve_holdout_opening
from portfolio_tracker.analytics.learned_trigger_research import (
    FEATURE_NAMES, FEATURE_VERSION, L2_GRID, RESEARCH_VERSION, TARGET_CLASSES,
    embargoed_training, extract_grouped_features, fit_model, predict,
    score_labels, strategy_counts,
)
from portfolio_tracker.analytics.operational_target import TARGET_VERSION
from portfolio_tracker.analytics.execution_path import EXECUTION_VERSION
from portfolio_tracker.services.model_observations import canonical
from scripts.build_historical_replay import load_dataset
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.research_clock_adapter import research_contract, assess_research_execution
from scripts.seal_trigger_experiment_holdout import protect_key
from scripts.audit_trigger_replay import digest_file


INPUT = ROOT / 'output/historical_replay/input/massive_400_2026-09-25'
BASE = ROOT / 'output/historical_replay/pilot_1h_1d'
OUTPUT = ROOT / 'output/historical_replay/learned_trigger_v1'
AAD = b'LEARNED_LONG_TRIGGER_INTERNAL_HOLDOUT_V1'
HORIZONS = ('1 Hora', '1 Día')
HOLDOUT_FRACTION = 0.20


def baseline_paths():
    result = {}
    for symbol in ('SMCI', 'NVDA'):
        matches = list(BASE.glob(symbol + '_*.json'))
        if len(matches) != 1:
            raise ValueError(f'Replay base ambiguo para {symbol}: {len(matches)} archivos.')
        result[symbol] = matches[0]
    return result


def read_baselines():
    result = {}
    for symbol, path in baseline_paths().items():
        replay = json.loads(path.read_text(encoding='utf-8'))
        validate_historical_replay(replay)
        if replay['symbol'] != symbol or len(replay['observations']) != 299:
            raise ValueError('No es la población fija de 299 cortes por activo.')
        result[symbol] = replay
    return result


def protocol():
    path = OUTPUT / 'preregistration.json'
    if path.exists():
        return read_signed(path)
    replays = read_baselines()
    sessions = sorted({pd.Timestamp(c['observed_at']).tz_convert('America/New_York').date().isoformat()
                       for replay in replays.values() for c in replay['observations']})
    if len(sessions) != 299:
        raise ValueError('Sesiones de ambos activos desalineadas.')
    holdout_n = math.ceil(len(sessions) * HOLDOUT_FRACTION)
    source_names = (
        'portfolio_tracker/analytics/learned_trigger_research.py',
        'portfolio_tracker/analytics/execution_path.py',
        'portfolio_tracker/analytics/operational_target.py',
        'scripts/research_clock_adapter.py',
        'scripts/train_learned_trigger_research.py',
    )
    p = {
        'version': RESEARCH_VERSION,
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'symbols': ['SMCI', 'NVDA'], 'horizons': list(HORIZONS),
        'sessions': len(sessions), 'development_start': sessions[0],
        'development_end': sessions[-holdout_n - 1],
        'internal_holdout_start': sessions[-holdout_n],
        'internal_holdout_end': sessions[-1],
        'internal_holdout_sessions': holdout_n,
        'holdout_status': 'PREVIOUSLY_INSPECTED_IN_TRIGGER_AUDITS_NOT_VIRGIN_OOS',
        'population': 'ALL_FROZEN_LONG_PLANS_AT_11_NY; possible OHLC fill independent of current AND',
        'label_policy': 'SIMULATED_RESOLVED_ONLY_TP_FIRST_SL_FIRST_TIMEOUT; no-fill has no class',
        'cost_policy': 'Frozen plan: commission 25bps/side; slippage 5bps/side; spread 4bps full; assumed not observed',
        'model': 'EXISTING_NUMPY_MULTINOMIAL_LOGIT_L2; one pooled LONG model per horizon',
        'l2_grid': list(L2_GRID), 'feature_version': FEATURE_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'feature_policy': 'Closed-cut only. No AND/activation/rejection output, future path, news or accounting input. 20-day peer correlation rebuilt from prior completed daily bars; saved cross_context unavailable in 598/598 cuts.',
        'grouped_momentum': 'One 5m composite of MACD histogram and EMA9/21 spread; separate StochRSI oscillator and 1h/4h/daily trend.',
        'hour_note': '11:00 is constant in this pilot; coefficient is non-identifiable and carries no time-selection evidence.',
        'split_policy': 'One session per symbol, trailing 60-session internal holdout, 3 expanding outer folds, 2 inner folds, one XNYS-session embargo and exit-availability purge.',
        'action_rule': 'LONG research signal iff P(TP)>P(SL) and train-only class-payoff expected net return >0. No tuning of threshold.',
        'comparators': ['CURRENT_AND', 'EMA9_21_CLOSED', 'TRAIN_ONLY_CLASS_FREQUENCY'],
        'holdout_opening': 'Once after all development folds, final l2 and model are frozen; no retune. Durable registry before decrypt.',
        'no_production_promotion': True,
        'source_replays_sha256': {s: digest_file(path) for s, path in baseline_paths().items()},
        'input_manifest_sha256': digest_file(INPUT / 'manifest.json'),
        'sources_sha256': {name: digest_file(ROOT / name) for name in source_names},
    }
    write_signed(path, p)
    return p


def validate_sources(p):
    if digest_file(INPUT / 'manifest.json') != p['input_manifest_sha256']:
        raise ValueError('Histórico auditado reemplazado.')
    for s, path in baseline_paths().items():
        if digest_file(path) != p['source_replays_sha256'][s]:
            raise ValueError('Replay firmado reemplazado: ' + s)
    amendment_path = OUTPUT / 'implementation_amendment.json'
    amendment = read_signed(amendment_path) if amendment_path.exists() else None
    for name, expected in p['sources_sha256'].items():
        actual = digest_file(ROOT / name)
        if actual == expected:
            continue
        if (name != 'scripts/train_learned_trigger_research.py' or not amendment
                or amendment['protocol_sha256'] != sha(p)
                or amendment['source'] != name
                or amendment['preregistered_sha256'] != expected
                or amendment['amended_sha256'] != actual
                or amendment['scope'] != 'RESEARCH_RUNNER_CALENDAR_BUG_ONLY'):
            raise ValueError('Código cambió después del preregistro: ' + name)


def _research_protocol(p):
    return {'protocol_sha256': sha(p), 'cohorts': [{'time_ny': '11:00'}],
            'start': p['development_start'], 'end': p['internal_holdout_end']}


def prepare():
    p = protocol()
    validate_sources(p)
    if (OUTPUT / 'seal.json').exists():
        print('Ya existe muestra sellada; no se recalcula.', flush=True)
        return
    datasets = {s: load_dataset(INPUT, s) for s in p['symbols']}
    replays = read_baselines()
    q = _research_protocol(p)
    rows = []
    statuses = defaultdict(Counter)
    for symbol in p['symbols']:
        peer = 'NVDA' if symbol == 'SMCI' else 'SMCI'
        for index, cut in enumerate(replays[symbol]['observations'], 1):
            at = pd.Timestamp(cut['observed_at'])
            day = at.tz_convert('America/New_York').date().isoformat()
            for horizon in cut['horizons']:
                if horizon['label'] not in HORIZONS:
                    continue
                contract = horizon['operational_contract']
                if not contract or contract['side'] != 'LONG':
                    statuses[horizon['label']]['SHORT_OR_NO_PLAN'] += 1
                    continue
                vector = extract_grouped_features(cut, horizon, datasets[symbol], datasets[peer])
                f = cut['feature_snapshot']['model_features']
                ema_rule = (float(f['ema9']) > float(f['ema21'])
                            and float(f['last_price']) >= float(f['ema9']))
                research = research_contract(contract, eligible=True, protocol=q, cut=at)
                outcome = assess_research_execution(research, datasets[symbol], q)
                status = outcome['status']
                statuses[horizon['label']][status] += 1
                row = {'cut_id': cut['cut_id'], 'symbol': symbol,
                       'session_ny': day, 'observed_at': cut['observed_at'],
                       'horizon': horizon['label'], 'feature_version': FEATURE_VERSION,
                       'features': list(vector), 'plan_contract_sha256': contract['contract_sha256'],
                       'current_and': bool(contract['eligible_at_emission']),
                       'ema_rule': bool(ema_rule), 'research_execution_status': status,
                       'outcome': None, 'label_available_at': None,
                       'net_return': None, 'net_pnl_per_share': None,
                       'fill_price': None, 'exit_price': None}
                if status == 'SIMULATED_RESOLVED':
                    if (outcome['version'] != EXECUTION_VERSION or
                            outcome['outcome'] not in TARGET_CLASSES or
                            not outcome['fill_at'] or not outcome['exit_at']):
                        raise ValueError('Fill resuelto inválido.')
                    if (contract['eligible_at_emission'] and
                            (outcome['outcome'] != horizon['execution_result']['outcome'] or
                             abs(outcome['net_pnl_per_share'] - horizon['execution_result']['net_pnl_per_share']) > 1e-8)):
                        raise ValueError('Control AND no coincide con ejecución congelada.')
                    row.update(outcome=outcome['outcome'],
                               label_available_at=outcome['exit_at'],
                               net_return=float(outcome['net_pnl_per_share']) / float(outcome['fill_price']),
                               net_pnl_per_share=float(outcome['net_pnl_per_share']),
                               fill_price=float(outcome['fill_price']),
                               exit_price=float(outcome['exit_price']))
                rows.append(row)
            if index % 50 == 0 or index == 299:
                print(symbol, index, '/299 cortes extraídos', flush=True)

    dev = [r for r in rows if r['session_ny'] <= p['development_end']]
    holdout = [r for r in rows if r['session_ny'] >= p['internal_holdout_start']]
    if len(dev) + len(holdout) != len(rows):
        raise ValueError('Partición de sesiones con huecos.')
    write_signed(OUTPUT / 'development.json', {
        'protocol_sha256': sha(p), 'rows': dev,
        'status_counts_all_long_candidates': {h: dict(count) for h, count in statuses.items()},
        'development_rows': len(dev), 'holdout_rows_encrypted': len(holdout),
    })
    plaintext = canonical({'protocol_sha256': sha(p), 'rows': holdout}).encode('utf-8')
    key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    encrypted = nonce + AESGCM(key).encrypt(nonce, plaintext, AAD)
    protected = protect_key(key)
    for name, raw in (('holdout.aesgcm', encrypted), ('holdout.key.dpapi', protected)):
        path = OUTPUT / name
        path.write_bytes(raw) if not path.exists() else (_ for _ in ()).throw(ValueError('Holdout cifrado ya existe.'))
    write_signed(OUTPUT / 'seal.json', {
        'protocol_sha256': sha(p), 'sealed_at': datetime.now(timezone.utc).isoformat(),
        'ciphertext_sha256': hashlib.sha256(encrypted).hexdigest(),
        'protected_key_sha256': hashlib.sha256(protected).hexdigest(),
        'plaintext_sha256': hashlib.sha256(plaintext).hexdigest(),
        'holdout_sessions': p['internal_holdout_sessions'],
        'members': {h: [(r['cut_id'], r['observed_at']) for r in holdout if r['horizon'] == h]
                    for h in HORIZONS},
        'labels_examined_for_model_selection': False,
        'not_virgin_oos': True,
    })
    print('SEALED', len(dev), 'development plans;', len(holdout), 'holdout plans encrypted', flush=True)


def _folds(sessions):
    n = len(sessions)
    start = max(100, n // 2)
    tail = n - start
    if tail < 3 * 20:
        raise ValueError('Desarrollo demasiado corto para 3 folds.')
    limits = [start, start + tail // 3, start + 2 * tail // 3, n]
    return [(sessions[:limits[i]], sessions[limits[i]:limits[i + 1]]) for i in range(3)]


def _inner_select(rows, train_sessions, all_sessions):
    n = len(train_sessions)
    boundaries = ((n // 2, 3 * n // 4), (3 * n // 4, n))
    candidates = []
    for l2 in L2_GRID:
        losses = []
        for start, end in boundaries:
            train = embargoed_training(rows, train_sessions[start], sessions=all_sessions)
            val = [r for r in rows if r['session_ny'] in set(train_sessions[start:end])
                   and r.get('outcome') in TARGET_CLASSES]
            if len(train) < 30 or len(val) < 6:
                continue
            try:
                fitted = fit_model(train, l2)
            except ValueError:
                continue
            losses.append(score_labels(val, predict(fitted, val))['log_loss'])
        if len(losses) == 2:
            candidates.append((float(np.mean(losses)), l2))
    if not candidates:
        return None
    return min(candidates)[1]


def _metrics(rows, probs, model):
    baseline = np.repeat(np.asarray(model['prior']).reshape(1, -1), len(rows), axis=0)
    ema = np.asarray([model['ema_prior'][str(bool(r['ema_rule']))] for r in rows])
    return {'model': score_labels(rows, probs),
            'frequency': score_labels(rows, baseline),
            'ema_conditioned_frequency': score_labels(rows, ema),
            'trades': strategy_counts(rows, probs, model)}


def _aggregate(results):
    summary = {}
    for arm in ('model', 'frequency', 'ema_conditioned_frequency'):
        applicable = [r[arm] for r in results if r[arm]['n']]
        total = sum(r['n'] for r in applicable)
        summary[arm] = {'n': total,
                        'brier': sum(r['brier'] * r['n'] for r in applicable) / total if total else None,
                        'log_loss': sum(r['log_loss'] * r['n'] for r in applicable) / total if total else None}
    summary['brier_skill_vs_frequency'] = (
        1.0 - summary['model']['brier'] / summary['frequency']['brier']
        if summary['model']['brier'] is not None and summary['frequency']['brier'] else None)
    trade = {}
    for name in ('LEARNED', 'CURRENT_AND', 'EMA9_21'):
        records = [r['trades'][name] for r in results]
        trade[name] = {'long_signals': sum(r['long_signals'] for r in records),
                       'possible_ohlc_fills': sum(r['possible_ohlc_fills'] for r in records),
                       'outcomes': {c: sum(r['outcomes'][c] for r in records) for c in TARGET_CLASSES},
                       'net_return_sum_hypothetical': sum(r['net_return_sum_hypothetical'] or 0 for r in records)}
    summary['trades'] = trade
    return summary


def _decrypt_holdout(p, seal):
    ciphertext = (OUTPUT / 'holdout.aesgcm').read_bytes()
    protected = (OUTPUT / 'holdout.key.dpapi').read_bytes()
    if (hashlib.sha256(ciphertext).hexdigest() != seal['ciphertext_sha256'] or
            hashlib.sha256(protected).hexdigest() != seal['protected_key_sha256']):
        raise ValueError('Holdout cifrado alterado.')
    key = protect_key(protected, decrypt=True)
    plaintext = AESGCM(key).decrypt(ciphertext[:12], ciphertext[12:], AAD)
    if hashlib.sha256(plaintext).hexdigest() != seal['plaintext_sha256']:
        raise ValueError('Compromiso del holdout inválido.')
    data = json.loads(plaintext)
    if data['protocol_sha256'] != sha(p):
        raise ValueError('Holdout de otro protocolo.')
    return data['rows']


def run_validation():
    p = protocol()
    validate_sources(p)
    if (OUTPUT / 'validation.json').exists():
        print('Validación existente: no se reabre el holdout.', flush=True)
        return
    dev = read_signed(OUTPUT / 'development.json')
    seal = read_signed(OUTPUT / 'seal.json')
    if dev['protocol_sha256'] != sha(p) or seal['protocol_sha256'] != sha(p):
        raise ValueError('Desarrollo/holdout de otro protocolo.')
    rows = dev['rows']
    # Some sessions have no LONG plan in either symbol. Split over the frozen
    # XNYS cut calendar, not over LONG-only rows, or fold dates silently shift.
    sessions = sorted({pd.Timestamp(c['observed_at']).tz_convert('America/New_York').date().isoformat()
                       for replay in read_baselines().values() for c in replay['observations']
                       if pd.Timestamp(c['observed_at']).tz_convert('America/New_York').date().isoformat()
                       <= p['development_end']})
    if len(sessions) != 299 - p['internal_holdout_sessions']:
        raise ValueError('Fechas de desarrollo incompletas.')
    by_horizon = {}
    frozen_models = {}
    for horizon in HORIZONS:
        population = [r for r in rows if r['horizon'] == horizon]
        folds = []
        selections = []
        for train_sessions, test_sessions in _folds(sessions):
            l2 = _inner_select(population, train_sessions, sessions)
            test = [r for r in population if r['session_ny'] in set(test_sessions)]
            train = embargoed_training(population, test_sessions[0], sessions=sessions)
            if l2 is None:
                folds.append({'test_start': test_sessions[0], 'test_end': test_sessions[-1],
                              'status': 'INSUFFICIENT_INNER_CLASSES', 'train_n': len(train),
                              'test_long_plans': len(test)})
                continue
            try:
                model = fit_model(train, l2)
            except ValueError as exc:
                folds.append({'test_start': test_sessions[0], 'test_end': test_sessions[-1],
                              'status': 'INSUFFICIENT_TRAIN_CLASSES', 'reason': str(exc),
                              'train_n': len(train), 'test_long_plans': len(test)})
                continue
            selections.append(l2)
            metric = _metrics(test, predict(model, test), model)
            folds.append({'test_start': test_sessions[0], 'test_end': test_sessions[-1],
                          'status': 'SCORED', 'l2': l2, 'train_n': len(train),
                          'train_class_counts': model['training_class_counts'],
                          'test_long_plans': len(test), **metric})
        scored = [f for f in folds if f['status'] == 'SCORED']
        aggregate = _aggregate(scored) if scored else None
        final_l2 = statistics.mode(selections) if selections else None
        final_train = embargoed_training(population, p['internal_holdout_start'],
                                          sessions=sessions + [p['internal_holdout_start']])
        try:
            final_model = fit_model(final_train, final_l2) if final_l2 is not None else None
        except ValueError:
            final_model = None
        frozen_models[horizon] = final_model
        by_horizon[horizon] = {
            'development_long_plans': len(population),
            'development_possible_ohlc_fills': sum(r['outcome'] in TARGET_CLASSES for r in population),
            'development_classes': dict(Counter(r['outcome'] for r in population if r['outcome'])),
            'development_statuses': dict(Counter(r['research_execution_status'] for r in population)),
            'outer_folds': folds, 'outer_aggregate': aggregate,
            'final_l2_selected_without_holdout': final_l2,
            'final_training_n': len(final_train),
            'model_frozen': final_model is not None,
        }
    frozen = {'protocol_sha256': sha(p), 'development': by_horizon,
              'models': frozen_models, 'frozen_at': datetime.now(timezone.utc).isoformat(),
              'not_approved': True}
    write_signed(OUTPUT / 'development_validation.json', frozen)
    # One opening after all model-selection choices are persisted. The same
    # dates were studied in prior gate audits, so this is an internal check,
    # explicitly NOT a fresh independent final OOS approval.
    registry = OUTPUT / 'internal_holdout_openings.sqlite'
    opening = {}
    for horizon in HORIZONS:
        members = tuple(tuple(v) for v in seal['members'][horizon])
        opening[horizon] = reserve_holdout_opening(
            registry_path=registry,
            dataset_sha256=p['input_manifest_sha256'],
            symbol='SMCI_NVDA', horizon=horizon,
            protocol_sha256=sha(p), commitment_sha256=sha(seal),
            members=members,
        )
    write_signed(OUTPUT / 'holdout_opening.json', opening)
    held_rows = _decrypt_holdout(p, seal)
    holdout = {}
    for horizon in HORIZONS:
        candidates = [r for r in held_rows if r['horizon'] == horizon]
        model = frozen_models[horizon]
        holdout[horizon] = {'long_plans': len(candidates),
                            'possible_ohlc_fills': sum(r['outcome'] in TARGET_CLASSES for r in candidates),
                            'class_counts': dict(Counter(r['outcome'] for r in candidates if r['outcome'])),
                            'status_counts': dict(Counter(r['research_execution_status'] for r in candidates)),
                            'selected_l2': by_horizon[horizon]['final_l2_selected_without_holdout'],
                            'metrics': _metrics(candidates, predict(model, candidates), model)
                            if model else None}
    validation = {'version': RESEARCH_VERSION, 'protocol_sha256': sha(p),
                  'development': by_horizon, 'internal_holdout': holdout,
                  'holdout_opening_sha256': sha(opening),
                  'holdout_is_independent_oos': False,
                  'reason': 'These historical dates were already inspected in previous gate experiments; sealing only prevents reuse by this script.',
                  'production_trigger_changed': False, 'portfolio_db_accessed': False,
                  'model_approval': False, 'generated_at': datetime.now(timezone.utc).isoformat()}
    write_signed(OUTPUT / 'validation.json', validation)
    write_report(validation, p)
    print('VALIDATED_INTERNAL_RESEARCH', OUTPUT / 'reporte.md', flush=True)


def write_report(result, p):
    lines = [
        '# Gatillo aprendido LONG · 1h y 1D · investigación sin despliegue',
        '',
        f"Protocolo `{sha(p)}`. Fuente: 400 sesiones auditadas por activo; 101 de calentamiento y 299 cortes a las 11:00 NY.",
        'Solo planes LONG preexistentes. La puerta AND se reemplaza únicamente en una copia de investigación; TP/SL, zona, costes y evaluador OHLC originales se conservan. Ningún fill de broker fue observado.',
        'El bloque final de 60 sesiones fue cifrado y se abrió una vez con registro duradero después de congelar features, L2 y modelos. **No es holdout virgen:** esas fechas ya se inspeccionaron en otras auditorías. No aprobar para producción con este bloque.',
        'La correlación guardada en los snapshots está N/D en 598/598 cortes; aquí se reconstruyó solo para investigación con los 21 cierres diarios emparejados estrictamente anteriores al corte. Requerirá paridad live/replay antes de cualquier integración.',
        'Las clases son TP primero, SL primero y timeout. Comisión 25 bps/lado, slippage 5 bps/lado y spread total 4 bps son **supuestos congelados** incorporados al P&L neto simulado; no cambian el orden físico de las barreras.',
        '',
        '| Horizonte | Desarrollo planes LONG | Fills OHLC posibles | Brier modelo / frecuencia / EMA | Log loss modelo / frecuencia / EMA | Brier Skill vs frecuencia | Señales LONG modelo / AND / EMA (OOF) | Holdout señales modelo / AND / EMA |',
        '|---|---:|---:|---|---|---:|---|---|',
    ]
    for horizon in HORIZONS:
        d = result['development'][horizon]
        o = d['outer_aggregate']
        h = result['internal_holdout'][horizon]
        hm = h['metrics']
        def values(m, name):
            return f"{m[name]['brier']:.4f}" if m and m[name]['brier'] is not None else 'N/D'
        def loss(m, name):
            return f"{m[name]['log_loss']:.4f}" if m and m[name]['log_loss'] is not None else 'N/D'
        def signals(m):
            return '/'.join(str(m['trades'][k]['long_signals']) for k in ('LEARNED', 'CURRENT_AND', 'EMA9_21')) if m else 'N/D'
        lines.append('| ' + ' | '.join((horizon, str(d['development_long_plans']),
            str(d['development_possible_ohlc_fills']),
            ' / '.join(values(o,k) for k in ('model','frequency','ema_conditioned_frequency')),
            ' / '.join(loss(o,k) for k in ('model','frequency','ema_conditioned_frequency')),
            f"{o['brier_skill_vs_frequency']:.4f}" if o and o['brier_skill_vs_frequency'] is not None else 'N/D',
            signals(o), signals(hm))) + ' |')
    lines += ['',
              'La frecuencia base y el comparador EMA usan solamente las clases del entrenamiento de cada fold. AND y EMA son reglas de selección, no probabilidades; sus columnas Brier no existen. “Fills” son posibles por OHLC, no ejecuciones confirmadas.',
              'Los horizontes 1h y 1D comparten cortes: **no** se suman como operaciones independientes. Los folds dividen por sesión XNYS, purgan etiquetas no conocidas y aplican embargo de una sesión.',
              'La hora es siempre 11:00 y por tanto no aporta variación aprendible. No se usaron fundamentales/noticias sin histórico punto-en-tiempo.',
              'El gate de éxito exige Brier y log loss mejores que frecuencia base, suficientes LONG y retorno neto no peor que AND; con un holdout previamente explorado, incluso cumplirlo no habilitaría producción.',
              'No se entrenó con las 1/0 entradas autorizadas por AND; las etiquetas provienen de planes LONG congelados con fill OHLC contrafactual, claramente separado de observación real.',
              ]
    for horizon in HORIZONS:
        d = result['development'][horizon]
        h = result['internal_holdout'][horizon]
        lines += ['', f'## {horizon}',
                  f"Desarrollo: clases {d['development_classes']}; estados de ejecución {d['development_statuses']}.",
                  f"Holdout interno: {h['long_plans']} planes LONG, {h['possible_ohlc_fills']} fills OHLC posibles, clases {h['class_counts']}."]
        for fold in d['outer_folds']:
            lines.append(f"- {fold['test_start']}–{fold['test_end']}: {fold['status']}, train={fold['train_n']}, planes test={fold['test_long_plans']}" +
                         (f", L2={fold['l2']}" if fold['status'] == 'SCORED' else ''))
        if h['metrics']:
            for name, x in h['metrics']['trades'].items():
                lines.append(f"- Holdout {name}: {x['long_signals']} señales LONG, {x['possible_ohlc_fills']} fills posibles, TP/SL/timeout={x['outcomes']}, suma de retornos netos hipotéticos={x['net_return_sum_hypothetical']:.6f}.")
    path = OUTPUT / 'reporte.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_signed(OUTPUT / 'report_manifest.json', {
        'protocol_sha256': sha(p),
        'validation_sha256': digest_file(OUTPUT / 'validation.json'),
        'development_validation_sha256': digest_file(OUTPUT / 'development_validation.json'),
        'seal_sha256': digest_file(OUTPUT / 'seal.json'),
        'report_sha256': digest_file(path),
        'no_live_or_accounting_mutation': True,
    })


def record_calendar_amendment():
    p = protocol()
    name = 'scripts/train_learned_trigger_research.py'
    if not (OUTPUT / 'seal.json').exists() or (OUTPUT / 'holdout_opening.json').exists():
        raise ValueError('La enmienda solo procede tras sellar y antes de abrir el bloque.')
    amendment = {
        'protocol_sha256': sha(p), 'source': name,
        'preregistered_sha256': p['sources_sha256'][name],
        'amended_sha256': digest_file(ROOT / name),
        'scope': 'RESEARCH_RUNNER_CALENDAR_BUG_ONLY',
        'reason': 'La primera validación rechazó 327 planes LONG porque algunos de los 239 días de desarrollo no tienen un plan LONG. Los folds ahora usan el calendario de los 299 cortes congelados, no los días seleccionados por side. No cambian labels, features, muestras, ventanas, embargo, costes, umbrales ni el bloque cifrado.',
        'prior_failed_validation': 'Fechas de desarrollo incompletas; no hubo métricas ni apertura de holdout.',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    write_signed(OUTPUT / 'implementation_amendment.json', amendment)
    print('CALENDAR_AMENDMENT', sha(amendment))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preregister','prepare','amend-calendar','validate'))
    args = parser.parse_args()
    if args.phase == 'preregister':
        p = protocol()
        print('PREREGISTERED', sha(p), p['development_end'], p['internal_holdout_start'])
    elif args.phase == 'prepare':
        prepare()
    elif args.phase == 'amend-calendar':
        record_calendar_amendment()
    else:
        run_validation()


if __name__ == '__main__':
    main()
