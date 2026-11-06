import ast
from pathlib import Path

import pandas as pd

from portfolio_tracker.ui.charts import _numeric_frame
from portfolio_tracker.ui.theme import PREMIUM_CSS


def test_premium_theme_is_responsive_and_respects_reduced_motion() -> None:
    assert "backdrop-filter" not in PREMIUM_CSS
    assert "radial-gradient" not in PREMIUM_CSS
    assert "linear-gradient" not in PREMIUM_CSS
    assert "#4CC9F0" not in PREMIUM_CSS
    assert "border-left-color: #dedede" in PREMIUM_CSS
    assert "@media (max-width: 768px)" in PREMIUM_CSS
    assert "@media (prefers-reduced-motion: reduce)" in PREMIUM_CSS
    assert "stSidebarNavItems" in PREMIUM_CSS
    assert "<script" not in PREMIUM_CSS.lower()


def test_chart_normalization_never_mutates_business_data() -> None:
    original = pd.DataFrame({"Precio": ["10.25", "11.50"], "Etiqueta": ["a", "b"]})
    untouched = original.copy(deep=True)

    normalized = _numeric_frame(original)

    pd.testing.assert_frame_equal(original, untouched)
    assert normalized["Precio"].tolist() == [10.25, 11.5]
    assert normalized["Etiqueta"].isna().all()


def test_native_theme_keeps_high_contrast_dark_palette() -> None:
    config = (
        Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    ).read_text(encoding="utf-8")
    assert 'backgroundColor = "#080808"' in config
    assert 'secondaryBackgroundColor = "#111111"' in config
    assert 'textColor = "#F7F7F7"' in config
    assert 'borderColor = "#292929"' in config
    assert "Inter" in config


def test_quant_minimal_theme_is_scoped_and_preserves_risk_content() -> None:
    assert '[data-testid="stMain"]:has(.st-key-quant_actions)' in PREMIUM_CSS
    assert '.st-key-quant_buy_zone' in PREMIUM_CSS
    assert '.st-key-quant_sell_zone' in PREMIUM_CSS
    assert '.st-key-quant_disclosure_duplicate' in PREMIUM_CSS
    assert '.st-key-quant_decision [data-testid="stMarkdownContainer"] > p:last-child' in PREMIUM_CSS
    # Never suppress all errors/alerts; only the duplicate decision-level paragraph.
    assert '[data-testid="stAlert"] {\n  display: none' not in PREMIUM_CSS
    assert 'filter: grayscale(1)' in PREMIUM_CSS


def test_secondary_panels_hide_distractions_without_removing_calculations() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for label in ("Condiciones de activación", "Plan de ejecución detallado"):
        assert f'st.expander("{label}", expanded=False)' in source
    assert 'st.expander("Patrones y estructuras"' not in source
    assert '_render_chart_patterns(analysis, compact=' not in source
    assert 'render_price_zones(analysis, zone_snapshot=zone_snapshot, reference_only=True)' in source
    predictor = source.split("def _probability_predictor_content", 1)[1].split(
        "def probability_predictor_page", 1
    )[0]
    assert 'st.expander("Estado de datos y calibración"' not in predictor
    assert 'st.expander("Fundamentales y noticias"' not in predictor
    assert 'render_cross_asset(analysis)' not in predictor
    assert '_render_fundamental_news(analysis' not in predictor
    assert 'st.multiselect("Emisoras"' not in predictor
    assert 'enrich_cross_asset(analysis' in predictor
    assert 'apply_fundamental_filter(analysis' in predictor
    assert 'calibrate_scenarios(contract' in predictor
    assert 'latest_approved_operational_models(' in predictor
    assert '"Activos del motor cuantitativo"' in source
    assert 'key="quant_core_metrics"' in source
    assert 'key="quant_buy_zone"' in source
    assert 'key="quant_sell_zone"' in source


def test_portfolio_summary_shows_gains_and_read_only_trade_history_without_chart() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    dashboard = functions["dashboard"]
    metric_labels = [
        call.args[0].value
        for call in ast.walk(dashboard)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "metric"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ]
    assert len(metric_labels) == 4
    assert set(metric_labels) == {
        "Patrimonio", "Comisiones",
        "Ganancia neta total", "Ganancia neta del mes",
    }
    dashboard_source = ast.get_source_segment(source, dashboard)
    for amount in (
        "summary.equity_usd", "summary.commissions_usd", "total_gain", "month_gain",
    ):
        assert f"usd_mxn_metric({amount}, fx_quote)" in dashboard_source
    assert "render_trade_history(trades, sales)" in dashboard_source
    assert 'page_intro("Resumen del portafolio", "")' in dashboard_source
    assert "Ganancias realizadas al vender" not in dashboard_source
    assert "pendientes de validación" not in dashboard_source
    assert "concentration_warnings(summary)" not in dashboard_source
    assert "Conversión informativa USD/MXN" not in source
    assert "Precio de Yahoo Finance, actualizado" not in source
    assert "render_realized_gain_chart" not in dashboard_source
    assert "portfolio_history_chart" not in dashboard_source
    assert "premium_bar_chart" not in dashboard_source
    assert '"Posiciones abiertas"' not in dashboard_source
    assert '"Distribución"' not in dashboard_source
    assert '"P&L no realizado"' not in dashboard_source
    assert '"Aportaciones netas"' not in dashboard_source

    history_source = ast.get_source_segment(source, functions["render_trade_history"])
    assert 'st.subheader("Historial de operaciones"' in history_source
    assert '"Ganancia neta"' in history_source
    assert "En compras la ganancia queda pendiente" not in history_source
    assert "st.expander(" not in history_source
    operations_source = ast.get_source_segment(source, functions["operations_page"])
    assert "render_trade_history" not in operations_source
    assert 'trade_form(repository, prefix="manual"' in operations_source
    assert "render_trade_management(repository)" in operations_source


def test_summary_currency_amounts_have_one_uniform_font() -> None:
    for panel in ("portfolio_summary_metrics", "portfolio_holdings_metrics"):
        assert f".st-key-{panel} [data-testid=\"stMetricValue\"]" in PREMIUM_CSS
        assert f".st-key-{panel} [data-testid=\"stMetricValue\"] *" in PREMIUM_CSS
    assert "font-size: inherit !important" in PREMIUM_CSS
