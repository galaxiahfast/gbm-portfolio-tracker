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


def test_secondary_panels_collapse_without_lazy_computation_gates() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for label in ("Condiciones de activación", "Plan de ejecución detallado",
                  "Patrones y estructuras", "Fundamentales y noticias", "Estado de datos y calibración"):
        assert f'st.expander("{label}", expanded=False)' in source
    assert 'key="quant_core_metrics"' in source
    assert 'key="quant_buy_zone"' in source
    assert 'key="quant_sell_zone"' in source
