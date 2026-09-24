from streamlit.testing.v1 import AppTest

from portfolio_tracker.services.operational_validation import validation_disclosure


def test_sparse_forward_evidence_is_preliminary_per_horizon():
    status = validation_disclosure(
        {"1 Hora": {"eligible": 18, "resolved": 19},
         "6 Horas": {"eligible": 19, "resolved": 19}},
        approved_horizons=set(),
    )
    assert status["mode"] == "CONSERVADOR"
    assert status["preliminary"] is True
    assert "1 Hora: n=18/300, modelo no aprobado" in status["banner"]
    assert "6 Horas: n=19/300, modelo no aprobado" in status["banner"]
    assert all(row["status"] == "PRELIMINAR" for row in status["rows"])


def test_preliminary_banner_is_visible_in_streamlit():
    app = AppTest.from_string('''
from portfolio_tracker.services.operational_validation import validation_disclosure
from portfolio_tracker.ui.system_decision import render_validation_banner
render_validation_banner(validation_disclosure(
    {"1 Hora": {"eligible": 18, "resolved": 19}}, set()))
''').run(timeout=30)
    assert not app.exception
    assert any("PRELIMINAR" in item.value and "n=18/300, modelo no aprobado" in item.value
               for item in app.warning)
