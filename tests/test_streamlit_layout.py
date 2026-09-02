"""Regresiones de estructura para contenedores Streamlit."""

import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_all_st_tabs_calls_are_unpacked_into_individual_containers() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    tab_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "st"
        and node.value.func.attr == "tabs"
    ]

    assert len(tab_assignments) == 3
    assert all(
        len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
        for node in tab_assignments
    )


def test_no_open_attribute_is_called_on_tabs_tuple() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "tabs.open" not in source


def test_predictor_tabs_are_dynamic_and_guard_individual_containers() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "executive_tab, technical_tab, calibration_tab = st.tabs(" in source
    assert 'on_change="rerun"' in source
    assert "if executive_tab.open:" in source
    assert "with executive_tab:" in source
    assert "if technical_tab.open:" in source
    assert "with technical_tab:" in source
    assert "if calibration_tab.open:" in source
    assert "with calibration_tab:" in source
    assert "backtesting_page(repository, fx_quote, embedded=True)" in source
    assert source.count("st.download_button(") == 4
    assert source.index("actions_slot = st.container()") < source.index(
        "executive_tab, technical_tab, calibration_tab = st.tabs("
    )
    assert "Descargar PDF maestro" in source


def test_predictor_uses_honest_score_nomenclature_until_empirically_calibrated() -> None:
    app_source = APP_PATH.read_text(encoding="utf-8")
    engine_path = APP_PATH.parent / "portfolio_tracker" / "analytics" / "technical_probability.py"
    engine_source = engine_path.read_text(encoding="utf-8")

    assert '"Probabilidad de subida"' not in app_source
    assert '"Probabilidad de bajada"' not in app_source
    assert "Score heurístico alcista" in engine_source
    assert "Pendiente de calibración empírica Brier" in engine_source
