"""The executive presentation shows one entry reference, never two plans."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from portfolio_tracker.ui.system_decision import _visible_levels


def _decision(**overrides):
    values = dict(
        current_shares=0, average_price=None, entry_low=41.56,
        entry_high=41.63, stop_loss=35.81, take_profit=42.55,
        action="NO_ACCIONABLE",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _signed_plan():
    return SimpleNamespace(entry=40.03, stop=39.28, target=41.36,
                           status="VERIFICAR_GATILLO")


def test_existing_shares_show_actual_purchase_and_frozen_exit_not_new_entry():
    label, entry, _, stop, target_label, target, _, ratio, note, holding = _visible_levels(
        _decision(current_shares=23, average_price=38.90), _signed_plan()
    )
    assert holding
    assert label == "Compra registrada · promedio"
    assert entry == pytest.approx(38.90)
    assert stop == pytest.approx(35.81)
    assert target_label == "Salida objetivo"
    assert target == pytest.approx(42.55)
    assert ratio == pytest.approx((42.55 - 38.90) / (38.90 - 35.81))
    assert "no se muestra otra entrada" in note


def test_flat_portfolio_shows_only_one_frozen_conditional_entry():
    label, entry, _, stop, _, target, _, ratio, note, holding = _visible_levels(
        _decision(), _signed_plan()
    )
    assert not holding
    assert label == "Única entrada condicional"
    assert entry == pytest.approx(40.03)
    assert stop == pytest.approx(39.28)
    assert target == pytest.approx(41.36)
    assert ratio == pytest.approx((41.36 - 40.03) / (40.03 - 39.28))
    assert "no un fill" in note


def test_missing_average_does_not_substitute_a_hypothetical_entry():
    label, entry, _, _, _, _, _, ratio, _, holding = _visible_levels(
        _decision(current_shares=23, average_price=None), _signed_plan()
    )
    assert holding and label == "Compra registrada · promedio"
    assert entry is None and ratio is None


def test_completed_session_opportunity_cannot_look_like_a_new_buy():
    finished = SimpleNamespace(entry=40.03, stop=39.28, target=41.36,
                               status="OBJETIVO_OBSERVADO")
    label, entry, _, stop, _, target, _, ratio, note, holding = _visible_levels(
        _decision(), finished
    )
    assert not holding and label == "Entrada disponible"
    assert entry is stop is target is ratio is None
    assert "ya terminó" in note


def test_open_position_panel_shows_only_four_real_position_metrics():
    app = AppTest.from_string('''
from types import SimpleNamespace
from portfolio_tracker.ui.system_decision import render_system_decision
from portfolio_tracker.services.intraday_plan import IntradayPlan
decision = SimpleNamespace(
    current_shares=23, average_price=38.90, entry_low=41.56, entry_high=41.63,
    stop_loss=35.81, take_profit=42.55, action="NO_ACCIONABLE",
    recommendation_mode="CONSERVADOR", reasons=("Posición entre stop y objetivo.",),
    explanation="R:R de otra entrada", risk_veto=False, waiting_cause="",
    total_capital=1003.54, cash_available=41.10, position_size=0,
    preliminary_bias="alcista", preliminary_horizon="1 Semana",
)
plan = IntradayPlan("OBJETIVO_OBSERVADO", "Hipotético", entry=40.03,
                    stop=39.28, target=41.36)
render_system_decision(decision, intraday_plan=plan)
''').run()
    assert not app.exception
    labels = [item.label for item in app.metric]
    assert labels == [
        "Stop loss móvil", "Precio objetivo de salida",
        "Precio promedio de compra", "Acciones en cartera",
    ]
    assert any("$38.90" == item.value for item in app.metric)


def test_no_position_does_not_render_a_decision_card():
    app = AppTest.from_string('''
from types import SimpleNamespace
from portfolio_tracker.ui.system_decision import render_system_decision
render_system_decision(SimpleNamespace(current_shares=0))
''').run()
    assert not app.exception
    assert not app.metric
    assert not app.markdown


def test_flat_portfolio_has_only_a_concise_waiting_state():
    app = AppTest.from_string('''
from types import SimpleNamespace
from portfolio_tracker.ui.system_decision import render_system_decision
render_system_decision(SimpleNamespace(
    current_shares=0, action="NO_ACCIONABLE",
    waiting_cause="sin evidencia de entradas ejecutables",
))
''').run()
    assert not app.exception
    assert not app.metric
    assert any("ESPERAR" in item.value and "sin evidencia" in item.value
               for item in app.caption)


def test_passed_exit_target_is_not_shown_as_a_future_objective():
    app = AppTest.from_string('''
from types import SimpleNamespace
from portfolio_tracker.ui.system_decision import render_system_decision
decision = SimpleNamespace(
    current_shares=23, average_price=41.50, stop_loss=41.01,
    take_profit=42.55, action="CONFIRMAR_SALIDA",
)
render_system_decision(decision, current_price=43.04)
''').run()
    assert not app.exception
    labels = [item.label for item in app.metric]
    assert labels == [
        "Stop loss · confirmar salida", "Salida pendiente · último cierre",
        "Precio promedio de compra", "Acciones en cartera",
    ]
    assert app.metric[1].value == "$43.04"
    assert "$42.55" not in [item.value for item in app.metric]
