"""The session plan is frozen and first-passage is causal, never a fill claim."""
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

from portfolio_tracker.services import intraday_plan as module


DAY = "2026-09-24"


def _frame():
    index = pd.date_range(f"{DAY} 11:05", periods=3, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {"Open": [102.0, 101.0, 101.0], "High": [102.0, 103.5, 102.0],
         "Low": [100.0, 97.5, 100.0], "Close": [101.0, 102.0, 101.0],
         "Volume": [1000.0, 1000.0, 1000.0]},
        index=index,
    )


def _contract():
    return {
        "observed_at": f"{DAY}T11:00:21-04:00", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit": 103.0, "side": "LONG",
        "eligible_at_emission": False,
    }


def test_first_passage_never_uses_same_bar_exit_and_stop_wins_tie():
    bars = _frame()
    assert module._hypothetical_sequence(bars.iloc[:1], 100, 98, 103)[0] == "TOQUE_SIN_FILL"
    outcome, when = module._hypothetical_sequence(bars.iloc[:2], 100, 98, 103)
    assert outcome == "SL_FIRST_HIPOTETICO"
    assert when == bars.index[1].tz_convert("UTC") or when == bars.index[1]


def test_plan_levels_do_not_move_with_five_minute_price(monkeypatch):
    from portfolio_tracker.analytics import operational_target

    monkeypatch.setattr(module, "_signed_daily_contract", lambda *args: _contract())
    monkeypatch.setattr(operational_target, "validate_operational_contract", lambda contract: None)
    analysis = SimpleNamespace(symbol="SMCI", last_price=101.0, intraday_indicators=_frame())
    repository = SimpleNamespace()
    first = module.build_intraday_plan(analysis, repository, now=f"{DAY}T11:10:00-04:00")
    analysis.last_price = 106.0
    second = module.build_intraday_plan(analysis, repository, now=f"{DAY}T11:15:00-04:00")
    assert (first.entry, first.stop, first.target) == (second.entry, second.stop, second.target) == (100, 98, 103)
    assert first.status == "VERIFICAR_GATILLO"
    assert second.status == "STOP_OBSERVADO"
    assert not first.eligible_at_cut and not second.eligible_at_cut


def test_no_signed_cut_means_no_retroactive_plan(monkeypatch):
    monkeypatch.setattr(module, "_signed_daily_contract", lambda *args: None)
    analysis = SimpleNamespace(symbol="SMCI", intraday_indicators=_frame())
    plan = module.build_intraday_plan(analysis, SimpleNamespace(), now=f"{DAY}T12:00:00-04:00")
    assert plan.status == "SIN_CORTE"
    assert plan.entry is None


def test_missing_closed_bar_does_not_become_no_touch(monkeypatch):
    from portfolio_tracker.analytics import operational_target

    monkeypatch.setattr(module, "_signed_daily_contract", lambda *args: _contract())
    monkeypatch.setattr(operational_target, "validate_operational_contract", lambda contract: None)
    analysis = SimpleNamespace(symbol="SMCI", intraday_indicators=_frame().iloc[1:])
    plan = module.build_intraday_plan(analysis, SimpleNamespace(), now=f"{DAY}T11:15:00-04:00")
    assert plan.status == "SIN_EVIDENCIA"
    assert "Faltan velas" in plan.detail


def test_signed_cut_lookup_requires_verified_complete_record():
    class Repository:
        def live_model_execution_record(self, run_id):
            return None

    assert module._signed_daily_contract(Repository(), "SMCI", DAY) is None


def test_streamlit_displays_sequence_separately_from_zone_touch():
    app = AppTest.from_string('''
import pandas as pd
from portfolio_tracker.services.intraday_plan import IntradayPlan
from portfolio_tracker.ui.intraday_plan import render_intraday_plan
render_intraday_plan(IntradayPlan(
    "VERIFICAR_GATILLO", "Toque sin fill confirmado.",
    pd.Timestamp("2026-09-24T11:00:21-04:00"), 40.0, 39.0, 42.0,
    "TOQUE_SIN_FILL", pd.Timestamp("2026-09-24T11:05:00-04:00"), False,
))
''').run()
    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    assert "Probabilidad de entrada → objetivo antes de stop: N/D" in captions
    assert "no consta fill" in captions or "Toque sin fill confirmado" in captions
