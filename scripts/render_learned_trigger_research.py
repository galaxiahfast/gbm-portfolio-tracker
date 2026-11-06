"""Render an already signed learned-trigger result; NEVER reopen its holdout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_massive_history import read_signed, write_signed, sha
from scripts.audit_trigger_replay import digest_file

OUTPUT = ROOT / 'output/historical_replay/learned_trigger_v1'


def fmt(value):
    return 'N/D' if value is None else f'{value:.4f}'


def main():
    if (OUTPUT / 'reporte_v3.md').exists():
        raise ValueError('Informe existente: no sobrescribir.')
    protocol = read_signed(OUTPUT / 'preregistration.json')
    validation = read_signed(OUTPUT / 'validation.json')
    frozen = read_signed(OUTPUT / 'development_validation.json')
    opening = read_signed(OUTPUT / 'holdout_opening.json')
    amendment = read_signed(OUTPUT / 'implementation_amendment.json')
    if (validation['protocol_sha256'] != sha(protocol)
            or frozen['protocol_sha256'] != sha(protocol)
            or validation['holdout_opening_sha256'] != sha(opening)
            or amendment['scope'] != 'RESEARCH_RUNNER_CALENDAR_BUG_ONLY'):
        raise ValueError('Cadena de investigación inconsistente.')
    lines = [
        '# Ensayo del gatillo LONG aprendido — 1h y 1D',
        '',
        'Resultado: **NO PROMOVER**. El modelo no generó una sola señal LONG bajo la regla de expectativa neta preregistrada y no supera consistentemente la frecuencia base.',
        '',
        '400 sesiones auditadas de 5 minutos por activo: 101 de calentamiento y 299 cortes a las 11:00 NY. Se conservaron planes, zonas, triple barrera y términos de coste ya existentes. La puerta AND solo se ignoró en una copia de investigación para construir etiquetas de fills *posibles por OHLC*, no de operaciones reales.',
        'Las 60 sesiones finales se cifraron y se abrieron una vez **después** de fijar los folds, la penalización y los modelos. Pero esas fechas ya habían sido exploradas en auditorías anteriores: este es un holdout interno de procedimiento, **no evidencia OOS virgen**. No se usó para ajustar el modelo después.',
        'El script de validación tuvo una corrección documentada antes de la apertura: usar el calendario completo de cortes para los folds, pues algunos días no tienen plan LONG. La enmienda conserva features, etiquetas, costes, umbrales y el bloque cifrado.',
        '',
    ]
    for horizon in protocol['horizons']:
        d = validation['development'][horizon]
        h = validation['internal_holdout'][horizon]
        lines += ['', f'## {horizon}',
                  '| Horizonte | Bloque | Planes LONG | Fills OHLC posibles | TP / SL / timeout | Brier modelo / frecuencia / EMA | Brier Skill vs frecuencia | Log loss modelo / frecuencia / EMA | Señales LONG modelo / AND / EMA |',
                  '|---|---|---:|---:|---|---|---:|---|---|']
        for block, plans, fills, classes, metrics in (
                ('walk-forward OOF', sum(x.get('test_long_plans', 0) for x in d['outer_folds']
                                         if x['status'] == 'SCORED'),
                 d['outer_aggregate']['model']['n'] if d['outer_aggregate'] else 0,
                 None, d['outer_aggregate']),
                ('interno final', h['long_plans'], h['possible_ohlc_fills'],
                 h['class_counts'], h['metrics'])):
            counts = classes or (metrics['trades']['LEARNED']['outcomes'] if metrics else {})
            class_text = (f"{counts.get('TP_FIRST', 0)} / {counts.get('SL_FIRST', 0)} / "
                          f"{counts.get('TIMEOUT', 0)}") if classes else 'ver detalle de selección abajo'
            if metrics:
                brier = ' / '.join(fmt(metrics[k]['brier']) for k in
                                   ('model','frequency','ema_conditioned_frequency'))
                skill = (1.0 - metrics['model']['brier'] / metrics['frequency']['brier']
                         if metrics['model']['brier'] is not None
                         and metrics['frequency']['brier'] else None)
                loss = ' / '.join(fmt(metrics[k]['log_loss']) for k in
                                  ('model','frequency','ema_conditioned_frequency'))
                signals = ' / '.join(str(metrics['trades'][k]['long_signals']) for k in
                                     ('LEARNED','CURRENT_AND','EMA9_21'))
            else:
                brier = loss = signals = 'N/D'
                skill = None
            lines.append(f'| {horizon} | {block} | {plans} | {fills} | {class_text} | {brier} | {fmt(skill)} | {loss} | {signals} |')
        lines += ['',
                  f"Desarrollo completo (no solo OOF): {d['development_long_plans']} planes LONG, "
                  f"{d['development_possible_ohlc_fills']} fills OHLC posibles; "
                  f"TP/SL/timeout {d['development_classes']}.",
                  f"Bloque final interno: {h['long_plans']} planes LONG, "
                  f"{h['possible_ohlc_fills']} fills OHLC posibles; "
                  f"TP/SL/timeout {h['class_counts']}.",
                  f"L2 final escogido sin ver el bloque interno: {h['selected_l2']}."]
        for fold in d['outer_folds']:
            lines.append(f"- Fold {fold['test_start']}–{fold['test_end']}: "
                         f"{fold['status']}; train={fold['train_n']}; "
                         f"planes test={fold['test_long_plans']}; L2={fold.get('l2', 'N/D')}.")
        if d['outer_aggregate']:
            for arm, record in d['outer_aggregate']['trades'].items():
                lines.append(f"- Desarrollo OOF {arm}: {record['long_signals']} señales, "
                             f"{record['possible_ohlc_fills']} fills posibles, "
                             f"TP/SL/timeout {record['outcomes']}; "
                             f"suma no ponderada de retornos netos hipotéticos "
                             f"{fmt(record['net_return_sum_hypothetical'])}.")
        if h['metrics']:
            for arm, record in h['metrics']['trades'].items():
                net = record['net_return_sum_hypothetical']
                lines.append(f"- Bloque final {arm}: {record['long_signals']} señales, "
                             f"{record['possible_ohlc_fills']} fills posibles, "
                             f"TP/SL/timeout {record['outcomes']}; "
                             f"suma no ponderada de retornos netos por fill {fmt(net)}."
                             ' No es rendimiento de cartera.')
    lines += [
        '', '## Contratos y límites',
        '- Regresión logística multinomial Ridge/L2 sobre TP_FIRST, SL_FIRST y TIMEOUT; un modelo por horizonte, datos SMCI+NVDA combinados, partición por sesión XNYS. Preprocesamiento y selección de L2 solo con datos anteriores; purga por disponibilidad de la etiqueta y embargo de una sesión.',
        '- El criterio de señal fue fijado antes del holdout: P(TP)>P(SL) y expectativa neta estimada con pagos de las clases del entrenamiento >0. Las probabilidades son scores **sin calibrar**. AND y EMA9/21 son reglas, no probabilidades, por lo que su Brier no existe; la frecuencia histórica y la frecuencia condicionada por EMA son los comparadores probabilísticos.',
        '- El spread de 4 bps, slippage de 5 bps y comisión de 25 bps por lado son **supuestos congelados** del evaluador. Modifican P&L simulado, no el orden temporal TP/SL. No se observó fill GBM ni coste real.',
        '- La correlación original estaba N/D en 598/598 snapshots por un problema de fechas. Solo en este ensayo se reconstruyó a partir de 21 cierres diarios previos y emparejados; falta paridad con live antes de integrarla.',
        '- La hora fue 11:00 para todos los cortes: no hay evidencia para aprender un efecto horario. No se incorporaron fundamentales/noticias sin datos históricos punto-en-tiempo. No se entrenó con las 1/0 entradas que autorizó el AND.',
        '- El 1h contiene muy pocos TP en desarrollo y final. El 1D mostró una mejora marginal de desarrollo, pero en el bloque interno su Brier y log loss fueron peores que la frecuencia base. Tampoco hubo señales LONG accionables del modelo. No hay base para sustituir el gatillo vigente.',
        '- 1h y 1D observan el mismo corte, no dos operaciones independientes. Las señales y retornos de la tabla no se deben sumar entre horizontes. Ningún resultado aprueba una recomendación financiera ni un cambio de producción.',
    ]
    path = OUTPUT / 'reporte_v3.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_signed(OUTPUT / 'report_manifest_v3.json', {
        'protocol_sha256': sha(protocol),
        'validation_sha256': digest_file(OUTPUT / 'validation.json'),
        'development_validation_sha256': digest_file(OUTPUT / 'development_validation.json'),
        'holdout_opening_sha256': digest_file(OUTPUT / 'holdout_opening.json'),
        'implementation_amendment_sha256': digest_file(OUTPUT / 'implementation_amendment.json'),
        'renderer_sha256': digest_file(Path(__file__)),
        'report_sha256': digest_file(path),
        'holdout_reopened': False,
        'supersedes_layout_only': digest_file(OUTPUT / 'reporte_v2.md'),
    })
    print('REPORT', path)


if __name__ == '__main__':
    main()
