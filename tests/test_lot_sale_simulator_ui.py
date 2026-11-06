"""El simulador aparece en Resumen sin ofrecer escritura contable."""

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_lot_sale_simulator_displays_selected_and_fifo_outcomes() -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "render_lot_sale_simulator"
    )
    script = "\n".join((
        "from __future__ import annotations",
        "from datetime import datetime",
        "from decimal import Decimal",
        "import streamlit as st",
        "from portfolio_tracker.models import PriceQuote",
        "from portfolio_tracker.services.lot_sale_simulator import open_buy_lots, simulate_lot_sale",
        "def local_datetime(value): return datetime.fromisoformat(value)",
        ast.get_source_segment(source, function),
        "trades = [",
        "{'id': 17, 'executed_at': '2026-10-08T14:00:00+00:00', 'symbol': 'SMCI', 'side': 'BUY', 'quantity': Decimal('22'), 'price_usd': Decimal('43.28'), 'gross_usd': Decimal('952.10'), 'commission_usd': Decimal('2.38')},",
        "{'id': 18, 'executed_at': '2026-10-16T15:39:00+00:00', 'symbol': 'SMCI', 'side': 'BUY', 'quantity': Decimal('5'), 'price_usd': Decimal('32.56'), 'gross_usd': Decimal('162.80'), 'commission_usd': Decimal('0.40')},",
        "]",
        "render_lot_sale_simulator(trades, {})",
    ))

    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.selectbox[0].value == 18
    app.number_input[1].set_value(32.88)
    app.button[0].click().run()

    assert not app.exception
    rendered = " ".join(node.value for node in app.markdown)
    assert "+$0.79 USD" in rendered
    assert "-$52.94 USD" in " ".join(node.value for node in app.info)
    assert "no envía órdenes ni modifica operaciones" in " ".join(
        node.value for node in app.caption
    )
