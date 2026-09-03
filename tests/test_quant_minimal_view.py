"""Render the presentation in isolation: no production database or market calls."""
import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_executive_view_keeps_both_zones_and_secondary_details_closed():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_render_probability_executive", "_render_chart_patterns"}
    functions = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
                              if isinstance(node, ast.FunctionDef) and node.name in names)
    script = '''from __future__ import annotations
import streamlit as st
import pandas as pd
from portfolio_tracker.analytics.chart_patterns import PatternDirection
from portfolio_tracker.services.pdf_report import executive_decision
from portfolio_tracker.services.projection_chart import ordered_horizon_projections, build_15_day_projection_figure
from portfolio_tracker.ui.theme import apply_premium_ui
from portfolio_tracker.ui.price_zones import render_price_zones
from tests.test_pdf_report import _analysis
'''
    script += functions
    script += '\napply_premium_ui()\nst.container(key="quant_actions")\n_render_probability_executive(_analysis())\n'
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    for removed in ("Entrada LONG", "TP1 / TP2 del LONG", "Último precio",
                    "Score operativo", "Régimen mayor · S / D / 4h",
                    "Estocástico RSI", "Vigilancia de rebote"):
        assert removed not in labels
    text = "\n".join(item.value for item in app.markdown)
    assert "Precio actual" in text
    assert "Zona 1" in text and "TP1 · Primer objetivo" in text
    assert "Stop loss técnico" in labels
    for expander in app.expander:
        assert not expander.proto.expanded
