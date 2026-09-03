from dataclasses import replace
import inspect
import pickle

from streamlit.testing.v1 import AppTest

from portfolio_tracker.ui.price_zones import DisplayZone, build_zone_lists, distance_to_zone, price_location
from portfolio_tracker.ui.theme import PREMIUM_CSS
from portfolio_tracker.ui.price_zones import render_price_zones
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
