"""Valuación informativa de posiciones, sin acceder a la base contable."""

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _render_script(*, with_quote: bool, with_fx: bool = False) -> str:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "render_open_holdings_value"
    )
    quote = (
        "{'SMCI': PriceQuote('SMCI', Decimal('45'), datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc), 'Yahoo Finance intradia')}"
        if with_quote else "{}"
    )
    fx = (
        "FxQuote(Decimal('17'), datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc), 'Yahoo')"
        if with_fx else "None"
    )
    currency_metric = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "usd_mxn_metric"
    )
    return "\n".join((
        "from __future__ import annotations",
        "from datetime import datetime, timezone",
        "from decimal import Decimal",
        "from zoneinfo import ZoneInfo",
        "import pandas as pd",
        "import streamlit as st",
        "from portfolio_tracker.models import FxQuote, Position, PortfolioSummary, PriceQuote, money",
        "LOCAL_TIMEZONE = ZoneInfo('America/Mexico_City')",
        "def usd_metric(value): return f'${money(value):,.2f}'",
        "def mxn(value, fx): return f'${money(value * fx.rate):,.2f} MXN' if fx else 'Tasa USD/MXN pendiente'",
        "def signed_delta(value, suffix=''): return None if value == 0 else f'{value:+,.2f}{suffix}'",
        ast.get_source_segment(source, currency_metric),
        ast.get_source_segment(source, function),
        "position = Position('SMCI', Decimal('3'), Decimal('120.75'), Decimal('40.25'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'))",
        "summary = PortfolioSummary(Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), [position])",
        f"render_open_holdings_value(summary, {quote}, {fx})",
    ))


def test_open_holdings_show_market_value_and_unrealized_gain() -> None:
    app = AppTest.from_string(_render_script(with_quote=True)).run()

    assert not app.exception
    assert [metric.value for metric in app.metric] == [
        "$135.00 USD / N/D MXN", "$14.25 USD / N/D MXN",
    ]
    row = app.dataframe[0].value.iloc[0]
    assert row["Acciones"] == 3
    assert row["Precio disponible (USD)"] == 45
    assert row["Valor actual (USD)"] == 135
    assert row["Ganancia/pérdida sin vender (USD)"] == 14.25


def test_open_holdings_do_not_present_trade_price_as_current_quote() -> None:
    app = AppTest.from_string(_render_script(with_quote=False)).run()

    assert not app.exception
    assert [metric.value for metric in app.metric] == ["N/D", "N/D"]
    assert app.dataframe[0].value.iloc[0]["Corte"] == "Sin cotización"


def test_open_holdings_show_usd_and_mxn_separated_by_slash() -> None:
    app = AppTest.from_string(_render_script(with_quote=True, with_fx=True)).run()

    assert not app.exception
    assert [metric.value for metric in app.metric] == [
        "$135.00 USD / $2,295.00 MXN", "$14.25 USD / $242.25 MXN",
    ]
    assert app.dataframe[0].value.iloc[0]["Valor actual (MXN)"] == 2295
