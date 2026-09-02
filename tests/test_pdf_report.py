import math
import hashlib
import json
from io import BytesIO

import pandas as pd
from pypdf import PdfReader

from portfolio_tracker.analytics.technical_probability import analyze_probability
from portfolio_tracker.services.pdf_report import (
    build_executive_report,
    build_master_report,
    build_probability_report,
    build_technical_report,
    executive_decision,
)
from portfolio_tracker.services.pdf_technical_charts import build_technical_pdf_charts


def _ohlcv(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [price - 0.08 for price in prices],
            "High": [price + 0.45 for price in prices],
            "Low": [price - 0.45 for price in prices],
            "Close": prices,
            "Volume": [1000 + (item % 23) * 70 for item in range(len(prices))],
        },
        index=index,
    )


def _analysis():
    intraday_index = pd.date_range("2026-08-20 13:30", periods=220, freq="5min", tz="UTC")
    intraday_prices = [40 + item * 0.015 + math.sin(item / 4) * 1.2 for item in range(len(intraday_index))]
    daily_index = pd.date_range("2021-01-01", periods=1400, freq="B")
    daily_prices = [30 + item * 0.025 + math.sin(item / 12) * 1.8 for item in range(len(daily_index))]
    return analyze_probability("SMCI", _ohlcv(intraday_index, intraday_prices), _ohlcv(daily_index, daily_prices))


def test_pdf_report_is_valid_and_contains_executive_and_technical_sections() -> None:
    analysis = _analysis()
    payload = build_probability_report(analysis)

    assert payload.startswith(b"%PDF")
    assert len(payload) > 5_000
    reader = PdfReader(BytesIO(payload))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Resumen ejecutivo" in text
    assert "Contexto fundamental y noticias" in text
    assert "Plan de ejecución y riesgo" in text
    assert "Scores y precios por horizonte" in text
    assert "Pendiente de calibraci" in text
    assert "Trayectoria proyectada - proximos 15 dias habiles" in text
    assert "Graficas tecnicas reales" in text
    assert "Bandas de Bollinger y VWAP - 5 min" in text
    assert "Estocastico RSI - 5 min" in text
    assert "MACD (12,26,9) - 5 min" in text
    assert "Estructura diaria - EMA 9/21/50/200" in text
    expected_target_label = "Objetivo bajista 1" if analysis.execution_levels.direction == "SHORT" else "Take profit 1"
    assert expected_target_label in text
    expected_stop_label = "Invalidacion bajista" if analysis.execution_levels.direction == "SHORT" else "Stop loss"
    assert expected_stop_label in text
    assert "Detalle técnico completo" in text
    assert "Validacion de patrones chartistas" in text
    assert "Lectura del motor" in text
    assert "Bloque estructurado para análisis por IA" in text
    assert len(reader.pages) >= 4


def test_pdf_technical_charts_are_built_from_real_indicator_frames() -> None:
    analysis = _analysis()
    charts = build_technical_pdf_charts(analysis, width=493)

    assert len(charts) == 15
    assert {chart.section for chart in charts} == {"Intradia", "Contexto", "Estructural"}
    assert all(chart.observation_count >= 2 for chart in charts)
    assert sum(
        any(item.__class__.__name__ == "PolyLine" for item in chart.drawing.contents)
        for chart in charts
    ) >= 12
    macd_chart = next(chart for chart in charts if chart.title.startswith("MACD"))
    assert sum(item.__class__.__name__ == "Rect" for item in macd_chart.drawing.contents) >= 70


def test_three_pdf_download_scopes_are_independent_and_valid() -> None:
    analysis = _analysis()
    executive = build_executive_report(analysis)
    technical = build_technical_report(analysis)
    complete = build_probability_report(analysis)

    executive_text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(executive)).pages)
    technical_text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(technical)).pages)
    complete_text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(complete)).pages)
    assert all(payload.startswith(b"%PDF") for payload in (executive, technical, complete))
    assert "Resumen ejecutivo" in executive_text
    assert "Graficas tecnicas reales" not in executive_text
    assert "Graficas tecnicas reales" in technical_text
    assert "Resumen ejecutivo" not in technical_text
    assert "Resumen ejecutivo" in complete_text and "Graficas tecnicas reales" in complete_text


def test_master_pdf_adds_audited_calibration_as_third_view() -> None:
    analysis = _analysis()
    result_payload = json.dumps(
        {
            "aggregate": {
                "trades": 12,
                "win_rate": 0.58,
                "profit_factor": 1.42,
                "maximum_drawdown_pct": 6.2,
            },
            "aggregate_decision": "APROBADO",
        },
        sort_keys=True,
    )
    context = {
        "online_stats": {
            "resolved": 8,
            "accuracy": 0.625,
            "brier_score": 0.21,
            "adaptive_threshold": 0.56,
        },
        "backtest_run": {
            "id": 7,
            "status": "APPROVED",
            "engine_version": "oos-test",
            "parameters_json": json.dumps(
                {
                    "minimum_probability": 0.57,
                    "stop_atr_multiple": 2.25,
                    "risk_per_trade_pct": 0.5,
                    "optimization_trials": 18,
                }
            ),
            "dataset_sha256": "d" * 64,
            "payload_json": result_payload,
            "payload_sha256": hashlib.sha256(result_payload.encode()).hexdigest(),
        },
    }
    payload = build_master_report(analysis, context)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)

    assert payload.startswith(b"%PDF")
    assert "Resumen ejecutivo" in text
    assert "Graficas tecnicas reales" in text
    assert "Calibración y backtesting" in text
    assert "Integridad SHA-256" in text
    assert "VALIDA" in text


def test_technical_pdf_contains_every_advanced_chart() -> None:
    text = "\n".join(
        page.extract_text() or ""
        for page in PdfReader(BytesIO(build_technical_report(_analysis()))).pages
    )
    expected_titles = {
        "Bandas de Bollinger y VWAP - 5 min",
        "Estocastico RSI - 5 min",
        "MACD (12,26,9) - 5 min",
        "Confirmacion MACD - 1 hora",
        "ADX y direccion - 5 min",
        "Volumen acumulado OBV - 5 min",
        "Estructura diaria - EMA 9/21/50/200",
        "Estructura semanal - EMA 9/21/50/200",
        "MACD macro - diario",
        "MACD macro - semanal",
        "Ichimoku - diario",
        "Estructura mensual - EMA 9/21/50",
        "MACD estructural - mensual",
        "Patrones chartistas - 5 min",
        "Patrones chartistas - diario",
    }
    assert all(title in text for title in expected_titles)


def test_executive_decision_never_promotes_a_vetoed_trade() -> None:
    analysis = _analysis()
    decision = executive_decision(analysis)
    if analysis.risk_veto or analysis.signal_rejected:
        assert decision.label == "EVITA / ESPERA"
