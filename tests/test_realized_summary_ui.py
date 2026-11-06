"""Smoke test del historial de Resumen sin abrir portfolio.db."""

import ast
from pathlib import Path
from decimal import Decimal

from streamlit.testing.v1 import AppTest


def _function_source(name: str) -> str:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node)


def test_realized_history_renders_without_chart_or_portfolio_database() -> None:
    script = "\n".join((
        "from __future__ import annotations",
        "from datetime import datetime, timezone",
        "from decimal import Decimal",
        "import pandas as pd",
        "import streamlit as st",
        "from portfolio_tracker.models import money",
        "from portfolio_tracker.services.realized_performance import RealizedSale",
        "def local_datetime(value): return datetime.fromisoformat(value)",
        _function_source("render_trade_history"),
        "sales = (RealizedSale(2, datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc), 'SMCI', Decimal('3'), Decimal('30.30'), Decimal('35.88'), Decimal('5.58')),) ",
        "trades = [{'id': 1, 'executed_at': '2026-10-06T15:00:00+00:00', 'symbol': 'SMCI', 'side': 'BUY', 'quantity': Decimal('3'), 'price_usd': Decimal('10'), 'gross_usd': Decimal('30'), 'commission_usd': Decimal('0.30'), 'validation_status': 'VERIFIED'}, {'id': 2, 'executed_at': '2026-10-07T15:00:00+00:00', 'symbol': 'SMCI', 'side': 'SELL', 'quantity': Decimal('3'), 'price_usd': Decimal('12'), 'gross_usd': Decimal('36'), 'commission_usd': Decimal('0.12'), 'validation_status': 'VERIFIED'}]",
        "render_trade_history(trades, sales)",
    ))

    app = AppTest.from_string(script, default_timeout=20).run()

    assert not app.exception
    assert any("Historial de operaciones" in element.value for element in app.subheader)
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value["Ganancia neta"].dropna().tolist() == [5.58]
    assert not app.get("vega_lite_chart")


def test_recent_lot_sale_renders_positive_realized_gain() -> None:
    trades = [
        {"id": 17, "executed_at": "2026-10-08T14:48:00+00:00", "symbol": "SMCI",
         "side": "BUY", "quantity": Decimal("22"), "price_usd": Decimal("43.28"),
         "gross_usd": Decimal("952.10"), "commission_usd": Decimal("2.38"),
         "validation_status": "VERIFIED"},
        {"id": 18, "executed_at": "2026-10-16T15:39:00+00:00", "symbol": "SMCI",
         "side": "BUY", "quantity": Decimal("5"), "price_usd": Decimal("32.56"),
         "gross_usd": Decimal("162.80"), "commission_usd": Decimal("0.40"),
         "validation_status": "VERIFIED"},
        {"id": 19, "executed_at": "2026-10-19T15:46:00+00:00", "symbol": "SMCI",
         "side": "SELL", "quantity": Decimal("5"), "price_usd": Decimal("33.00"),
         "gross_usd": Decimal("165.00"), "commission_usd": Decimal("0.41"),
         "allocated_buy_trade_id": 18, "validation_status": "VERIFIED"},
    ]
    script = "\n".join((
        "from __future__ import annotations",
        "from datetime import datetime, timezone",
        "from decimal import Decimal",
        "import pandas as pd",
        "import streamlit as st",
        "from portfolio_tracker.models import money",
        "from portfolio_tracker.services.realized_performance import RealizedSale, realized_sales",
        "def local_datetime(value): return datetime.fromisoformat(value)",
        _function_source("render_trade_history"),
        f"trades = {trades!r}",
        "sales = realized_sales(trades, timezone=timezone.utc)",
        "render_trade_history(trades, sales)",
    ))
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.dataframe[0].value.iloc[0]["Ganancia neta"] == 1.39
