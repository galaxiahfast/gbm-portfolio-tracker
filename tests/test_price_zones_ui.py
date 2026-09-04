from dataclasses import replace
import inspect
import pickle
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from portfolio_tracker.ui.price_zones import DisplayZone, build_zone_lists, distance_to_zone, price_location
from portfolio_tracker.ui.theme import PREMIUM_CSS
from portfolio_tracker.ui.price_zones import render_price_zones
from portfolio_tracker.analytics.conditional_zone_reach import calculate_dynamic_visual_zone
from portfolio_tracker.analytics.zone_reach import ReachEstimate
from portfolio_tracker.services.price_zones import (
    ZoneSnapshot,
    build_zone_snapshot,
    build_visual_zone_snapshot,
    market_session_status,
    projected_extended_levels,
)
from tests.test_pdf_report import _analysis


def test_existing_levels_are_ordered_without_mutating_analysis():
    analysis = _analysis()
    before = pickle.dumps(analysis)
    buys, sales = build_zone_lists(analysis)
    assert len(buys) == len(sales) == 3
    assert buys[0].low == analysis.buy_levels.entry_low
    assert sales[0].low == analysis.buy_levels.take_profit_1
    assert sales[1].low == analysis.buy_levels.take_profit_2
    support_prices = [z.low for z in buys if z.low is not None]
    sale_prices = [z.low for z in sales if z.low is not None]
    assert support_prices == sorted(set(support_prices), reverse=True)
    assert sale_prices == sorted(sale_prices)
    if analysis.buy_levels.take_profit_1 == analysis.buy_levels.take_profit_2:
        assert "coincide con TP1" in sales[1].source
    assert pickle.dumps(analysis) == before
    assert buys[0].context_score == analysis.probability_up
    assert buys[1].context_score == analysis.probability_down


def test_missing_distinct_levels_are_not_invented():
    analysis = _analysis()
    analysis = replace(analysis, nearest_support=9999, structural_support=9999,
                       weekly_support=9999, weekly_resistance=0, structural_resistance=0,
                       pivots=replace(analysis.pivots, s1=9999, s2=9999, r1=0, r2=0),
                       fibonacci=replace(analysis.fibonacci, level_382=9999, level_500=9999, level_618=9999))
    buys, sales = build_zone_lists(analysis)
    assert buys[1].low is None and buys[2].low is None
    assert sales[2].low is None


def test_distance_and_tracker_handle_inside_outside_and_invalid_prices():
    zone = DisplayZone("Entrada", 90, 95, "test", 40, "bajista")
    assert distance_to_zone(100, zone) == (-5, -5)
    assert distance_to_zone(92, zone) == (0, 0)
    assert distance_to_zone(0, zone) is None
    assert distance_to_zone(float("nan"), zone) is None
    assert price_location(100, (zone,))[2:] == (1.0, "Por encima de las zonas")
    assert price_location(85, (zone,))[2:] == (0.0, "Por debajo de las zonas")
    assert price_location(92.5, (zone,))[2] == 0.5
    assert price_location(float("nan"), (zone,)) is None
    assert price_location(100, ()) is None


def test_display_preserves_active_plan_even_after_price_crosses_it():
    analysis = _analysis()
    active = replace(analysis, position_state="LONG_ACTIVE", last_price=analysis.buy_levels.take_profit_2 + 2)
    buys, sales = build_zone_lists(active)
    assert buys[0].low == analysis.buy_levels.entry_low
    assert sales[0].low == analysis.buy_levels.take_profit_1
    assert distance_to_zone(active.last_price, sales[0])[0] < 0


def test_requested_executive_blocks_are_hidden_only_by_presentation():
    for selector in (".st-key-quant_decision", ".st-key-quant_hidden_score",
                     ".st-key-quant_hidden_activation", ".st-key-quant_hidden_execution"):
        assert selector in PREMIUM_CSS
    assert 'display: none !important' in PREMIUM_CSS


def test_second_view_renders_six_levels_and_a_bounded_tracker():
    app = AppTest.from_string('''
from portfolio_tracker.ui.theme import apply_premium_ui
from portfolio_tracker.ui.price_zones import render_price_zones
from tests.test_pdf_report import _analysis
apply_premium_ui()
render_price_zones(_analysis())
''', default_timeout=30).run()
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    for label in ("Zona 1", "Zona 2", "Zona 3", "TP1", "TP2", "Nivel 3"):
        assert label in text
    captions = "\n".join(item.value for item in app.caption)
    for removed in ("Escala:", "Más cercana:", "Alcance por zona:"):
        assert removed not in captions
    assert len(app.get("progress")) == 0
    assert len(app.get("column")) == 3
    assert "Precio actual" in text
    assert "Zona de compra / Entrada ideal" in text
    assert "Zona de venta / Objetivos y resistencia" in text


def test_three_panels_use_shared_typography_and_hide_old_metrics():
    assert '.st-key-quant_core_metrics,\n.st-key-quant_legacy_zones {\n  display: none !important;' in PREMIUM_CSS
    assert '--quant-panel-font-size: .875rem' in PREMIUM_CSS
    assert '--quant-panel-font-weight: 500' in PREMIUM_CSS
    assert 'font-size: var(--quant-panel-font-size) !important' in PREMIUM_CSS
    assert 'font-weight: var(--quant-panel-font-weight) !important' in PREMIUM_CSS


def test_tracker_is_never_enqueued_even_temporarily():
    source = inspect.getsource(render_price_zones)
    for removed in ("st.progress(", "Ubicación del precio", "Escala:", "Más cercana:", "Alcance por zona:"):
        assert removed not in source
    assert "location = price_location(price, buys + sales)" in source


def test_dynamic_visual_reclassifies_frozen_levels_and_decays_remaining_time():
    zones = (
        (DisplayZone("Entrada", 36.09, 36.16, "forward", None, ""), "BELOW"),
        (DisplayZone("TP1 · Primer objetivo", 37.07, 37.07, "forward", None, ""), "ABOVE"),
        (DisplayZone("TP2 · Segundo objetivo", 37.88, 37.88, "forward", None, ""), "ABOVE"),
    )
    originals = tuple(
        ReachEstimate(value, 21, None, None, "forward", value / 2, close_direction=direction)
        for value, (_, direction) in zip((10, 71, 45), zones)
    )
    result = calculate_dynamic_visual_zone(
        37.70,
        pd.Timestamp("2026-09-03T13:40:00", tz="America/New_York"),
        pd.Timestamp("2026-09-03T16:00:00", tz="America/New_York"),
        zones,
        original_estimates=originals,
    )
    assert result[0].estimate.probability < 5
    assert result[0].warning == "Poco probable en el tiempo restante"
    assert result[1].classification == "Superado (Soporte)"
    assert "TP1 (Superado) - Soporte inmediato" == result[1].label
    assert result[1].estimate.probability < originals[1].probability
    assert result[2].classification == "Objetivo activo"
    assert result[2].minutes_remaining == 140


def test_visual_snapshot_is_read_only_and_app_does_not_log_it():
    analysis = _analysis()
    buys, sales = build_zone_lists(analysis)
    static = ZoneSnapshot(
        pd.Timestamp("2026-08-24T15:00:00Z"),
        buys,
        sales,
        tuple(ReachEstimate(50, 20, 30, 70, "forward") for _ in range(6)),
    )

    class ReadOnlyRepository:
        def zone_predictions(self):
            return []

        def save_prediction(self, *_args, **_kwargs):
            raise AssertionError("La visualización no debe persistir predicciones")

    snapshot = build_visual_zone_snapshot(
        analysis,
        repository=ReadOnlyRepository(),
        now="2026-08-24T18:50:00Z",
        original_snapshot=static,
    )
    assert len(snapshot.buys) == len(snapshot.sales) == 3
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "log_snapshot(repository, analysis, zone_snapshot)" not in app_source
    assert "build_visual_zone_snapshot(analysis, repository=repository)" in app_source


def test_closed_market_preserves_frozen_zone_classification_and_next_cut():
    analysis = _analysis()
    buys, sales = build_zone_lists(analysis)
    frozen = ZoneSnapshot(
        pd.Timestamp("2026-09-03T15:00:00Z"),
        buys,
        sales,
        tuple(ReachEstimate(50, 20, 30, 70, "forward") for _ in range(6)),
    )
    closed = build_visual_zone_snapshot(
        analysis,
        now="2026-09-03T22:00:00Z",
        original_snapshot=frozen,
    )
    assert closed.buys == frozen.buys and closed.sales == frozen.sales
    assert closed.estimates == frozen.estimates
    assert closed.evaluated_at == pd.Timestamp("2026-09-03T22:00:00Z")
    status = market_session_status("2026-09-03T22:00:00Z")
    assert status.is_open is False
    assert "MERCADO CERRADO" in status.message
    assert status.next_collection_at == pd.Timestamp("2026-09-04T15:00:00Z")


def test_projected_levels_extend_beyond_all_original_zones_in_both_directions():
    analysis = _analysis()
    buys, sales = build_zone_lists(analysis)
    snapshot = ZoneSnapshot(
        pd.Timestamp("2026-09-03T18:00:00Z"), buys, sales,
        tuple(ReachEstimate(50, 20, 30, 70, "forward") for _ in range(6)),
    )
    sales_ceiling = max(
        analysis.buy_levels.take_profit_1,
        analysis.buy_levels.take_profit_2,
        analysis.pivots.r1,
    )
    bullish = replace(analysis, last_price=sales_ceiling + 5)
    upside = projected_extended_levels(bullish, snapshot)
    assert 2 <= len(upside) <= 3
    assert all(level.direction == "ABOVE" and level.price > bullish.last_price for level in upside)
    assert all("Proyectado" in level.label for level in upside)

    buy_floor = min(
        analysis.buy_levels.entry_low,
        analysis.pivots.s1,
        analysis.pivots.s2,
    )
    bearish = replace(analysis, last_price=max(1, buy_floor - 5))
    downside = projected_extended_levels(bearish, snapshot)
    assert 2 <= len(downside) <= 3
    assert all(level.direction == "BELOW" and 0 < level.price < bearish.last_price for level in downside)
    assert all("Soporte extendido" in level.label for level in downside)


def test_zone_snapshot_exposes_extended_levels_with_stable_contract():
    analysis = _analysis()
    ceiling = max(
        analysis.buy_levels.take_profit_1,
        analysis.buy_levels.take_profit_2,
        analysis.pivots.r1,
    )
    analysis = replace(analysis, last_price=ceiling + 5.0)

    snapshot = build_zone_snapshot(analysis, now="2026-09-03T22:00:00Z")

    assert 2 <= len(snapshot.extended_levels) <= 3
    assert all(level.price > analysis.last_price for level in snapshot.extended_levels)
    assert all(level.type == "resistencia_extendida" for level in snapshot.extended_levels)
    assert all(level.label and level.source for level in snapshot.extended_levels)


def test_streamlit_renders_extended_level_prices_when_resistances_are_exceeded():
    app = AppTest.from_string('''
from dataclasses import replace
from tests.test_pdf_report import _analysis
from portfolio_tracker.services.price_zones import build_zone_snapshot
from portfolio_tracker.ui.price_zones import render_price_zones

analysis = _analysis()
ceiling = max(analysis.buy_levels.take_profit_1, analysis.buy_levels.take_profit_2, analysis.pivots.r1)
analysis = replace(analysis, last_price=ceiling + 5.0)
snapshot = build_zone_snapshot(analysis, now="2026-09-03T22:00:00Z")
render_price_zones(analysis, zone_snapshot=snapshot)
''').run(timeout=20)

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Niveles extendidos (Proyectados)" in rendered
    for level in build_zone_snapshot(
        replace(_analysis(), last_price=max(
            _analysis().buy_levels.take_profit_1,
            _analysis().buy_levels.take_profit_2,
            _analysis().pivots.r1,
        ) + 5.0),
        now="2026-09-03T22:00:00Z",
    ).extended_levels:
        assert f"${level.price:,.2f}" in rendered


def test_zone_list_renders_unavailable_confidence_without_crashing():
    app = AppTest.from_string('''
from portfolio_tracker.analytics.zone_reach import ReachEstimate
from portfolio_tracker.services.price_zones import DisplayZone
from portfolio_tracker.ui.price_zones import _render_list

zone = DisplayZone("Zona limitada", 99.0, 99.0, "Prueba", None, "alcista")
estimate = ReachEstimate(
    probability=25.0,
    samples=0,
    lower=None,
    upper=None,
    status="Muestra insuficiente",
    close_probability=10.0,
    close_lower=None,
    close_upper=None,
    close_direction="ABOVE",
    model="conditional-v2",
    confidence_available=False,
)
_render_list((zone,), 100.0, (estimate,))
''', default_timeout=30).run()
    assert not app.exception
    captions = "\n".join(item.value for item in app.caption)
    assert "IC 95%: N/D (muestra insuficiente)" in captions


def test_zone_list_survives_partial_levels_and_all_optional_statistics():
    app = AppTest.from_string('''
from portfolio_tracker.analytics.zone_reach import ReachEstimate
from portfolio_tracker.services.price_zones import DisplayZone
from portfolio_tracker.ui.price_zones import _render_list

zone = DisplayZone("Zona fuera de sesión", 99.0, None, "Último corte cerrado", None, "alcista")
estimate = ReachEstimate(
    probability=25.0,
    samples=0,
    lower=None,
    upper=None,
    status="Fuera de sesión",
    close_probability=10.0,
    close_lower=None,
    close_upper=None,
    close_direction="ABOVE",
    model="conditional-v2",
    effective_samples=None,
    confidence_available=True,
)
_render_list((zone,), 100.0, (estimate,))
''', default_timeout=30).run()
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    captions = "\n".join(item.value for item in app.caption)
    assert "Sin nivel disponible" in text
    assert "IC 95%: N/D (muestra insuficiente)" in captions
    assert "Distancia N/D" in captions
