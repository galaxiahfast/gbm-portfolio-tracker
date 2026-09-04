from dataclasses import replace
from io import BytesIO

import pytest
import pandas as pd
from pypdf import PdfReader

from portfolio_tracker.analytics.zone_reach import ReachEstimate
from portfolio_tracker.analytics.conditional_zone_reach import calculate_dynamic_visual_zone
from portfolio_tracker.services.price_zones import build_zone_snapshot
from portfolio_tracker.services.pdf_report import (
    build_executive_report, build_technical_report, build_probability_report, build_master_report,
)
from tests.test_pdf_report import _analysis


def snapshot_fixture():
    analysis = _analysis()
    snapshot = build_zone_snapshot(analysis, now='2026-09-02 15:00Z')
    estimates = tuple(ReachEstimate(p, 21, lo, hi, 'Frecuencia histórica; no calibrada OOS')
                      for p, lo, hi in [(76, 55, 89), (38, 21, 59), (5, 1, 23),
                                       (5, 1, 23), (0, 0, 15), (0, 0, 15)])
    return analysis, replace(snapshot, estimates=estimates)


@pytest.mark.parametrize('builder', [build_executive_report, build_technical_report,
                                    build_probability_report, build_master_report])
def test_all_reports_include_shared_six_zone_snapshot(builder):
    analysis, snapshot = snapshot_fixture()
    args = (analysis, None) if builder is build_master_report else (analysis,)
    payload = builder(*args, zone_snapshot=snapshot)
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(payload)).pages)
    assert text.count('Alcance intradía de las zonas') == 1
    assert text.count('Probabilidad estimada de alcance hoy:') == 6
    for label in ('Zona 1', 'Zona 2', 'Zona 3', 'TP1', 'TP2', 'Nivel 3',
                  '76%', '38%', '55-89%', '21 sesiones', 'no calibrada OOS',
                  '2026-09-02T15:00:00+00:00', 'Distancia'):
        assert label in text


def test_unavailable_pdf_does_not_fabricate_probabilities():
    analysis = _analysis()
    snapshot = build_zone_snapshot(analysis, now='2026-08-30 15:00Z')
    payload = build_executive_report(analysis, zone_snapshot=snapshot)
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(payload)).pages)
    assert text.count('Probabilidad estimada de alcance hoy: N/D') == 6
    assert 'Fuera de sesión' in text


def test_pdf_displays_dynamic_classification_without_changing_frozen_snapshot():
    analysis, frozen = snapshot_fixture()
    pairs = tuple((zone, 'BELOW') for zone in frozen.buys) + tuple(
        (zone, 'ABOVE') for zone in frozen.sales
    )
    assessments = calculate_dynamic_visual_zone(
        analysis.buy_levels.take_profit_1 + 1,
        '2026-09-03T17:40:00Z',
        '2026-09-03T20:00:00Z',
        pairs,
        original_estimates=frozen.estimates,
    )
    dynamic = replace(
        frozen,
        evaluated_at=pd.Timestamp('2026-09-03T17:40:00Z'),
        buys=tuple(replace(z, label=a.label) for z, a in zip(frozen.buys, assessments[:3])),
        sales=tuple(replace(z, label=a.label) for z, a in zip(frozen.sales, assessments[3:])),
        estimates=tuple(a.estimate for a in assessments),
    )
    text = '\n'.join(
        page.extract_text() or ''
        for page in PdfReader(BytesIO(build_executive_report(analysis, zone_snapshot=dynamic))).pages
    )
    assert 'Superado' in text and 'Soporte inmediato' in text
    assert 'Estimación visual preliminar' in text
    assert frozen.estimates[3].probability == 5


def test_pdf_shows_closed_market_and_visual_extended_levels():
    analysis, snapshot = snapshot_fixture()
    ceiling = max(zone.high for zone in snapshot.sales if zone.high is not None)
    analysis = replace(analysis, last_price=ceiling + 5)
    snapshot = build_zone_snapshot(analysis, now='2026-09-03T22:00:00Z')
    text = '\n'.join(
        page.extract_text() or ''
        for page in PdfReader(BytesIO(build_executive_report(analysis, zone_snapshot=snapshot))).pages
    )
    assert 'MERCADO CERRADO' in text
    assert 'Niveles extendidos (Proyectados)' in text
    assert text.count('Proyectado') >= 2
    for level in snapshot.extended_levels:
        assert f'${level.price:,.2f}' in text
    assert 'no se guardan en el registro forward' in text
