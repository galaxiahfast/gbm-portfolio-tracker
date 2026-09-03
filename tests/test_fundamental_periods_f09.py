"""F09: analytical-only regressions for period/unit/sign compatibility."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from portfolio_tracker.analytics import fundamental_news as fn


AS_OF = datetime(2026, 9, 2, tzinfo=timezone.utc)


def statement(values, end="2026-06-30", **metadata):
    frame = pd.DataFrame({pd.Timestamp(end): values})
    frame.attrs = {
        "period": "QUARTER", "currency": "USD", "unit": "currency", "scale": 1,
        **metadata,
    }
    return frame


def snapshot(income=None, cash=None, **kwargs):
    return fn.build_fundamental_snapshot(
        "SMCI", observed_at=AS_OF, income_statement=income, cashflow=cash, **kwargs,
    )


def pair(**overrides):
    return (
        statement({"Total Revenue": 100, "Net Income": 10}),
        statement({"Operating Cash Flow": 20, "Free Cash Flow": 15}, **overrides),
    )


def test_info_flows_never_divide_quarterly_denominators():
    income, _ = pair()
    result = snapshot(income, info={"operatingCashflow": 400, "freeCashflow": 300})
    assert result.metrics["cash_conversion"] is None
    assert result.metrics["fcf_margin"] is None
    assert result.fundamental_points == 1.5  # only the independently positive FCF


def test_quarterly_ratios_use_quarterly_numerators_even_when_info_has_totals():
    income, cash = pair()
    result = snapshot(income, cash, info={"operatingCashflow": 400, "freeCashflow": 300})
    assert result.metrics["cash_conversion"] == 2
    assert result.metrics["fcf_margin"] == .15
    assert result.metrics["cash_conversion_numerator"] == 20
    assert "QUARTER al 2026-06-30" in result.metrics["cash_conversion_basis"]
    assert "statement:" in result.metrics["fcf_margin_basis"]


@pytest.mark.parametrize("metadata", [
    {"period": "TTM"}, {"period": "ANNUAL"}, {"period": ""},
    {"currency": "MXN"}, {"currency": ""}, {"unit": "shares"},
    {"scale": 0}, {"scale": -1}, {"scale": float("inf")},
    {"period_start": "2026-04-01"},
])
def test_incompatible_or_unverified_metadata_is_excluded(metadata):
    income, cash = pair(**metadata)
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] is None
    assert result.metrics["fcf_margin"] is None
    assert any("N/D" in reason for reason in result.reasons)


def test_unannotated_statements_are_not_assumed_quarterly_or_usd():
    income, cash = pair()
    income.attrs = {}
    cash.attrs = {}
    assert snapshot(income, cash).metrics["cash_conversion"] is None


def test_explicit_monetary_scale_is_normalized():
    income, cash = pair(scale=1000)
    income *= 1000
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] == 2
    assert result.metrics["fcf_margin"] == .15
    assert result.metrics["cash_conversion_numerator"] == 20000


def test_ttm_info_requires_both_verified_matching_periods():
    source = {"operatingCashflow": 80, "netIncomeToCommon": 40,
              "freeCashflow": 60, "totalRevenue": 400}
    meta = {key: {"period": "TTM", "period_end": "2026-06-30",
                  "currency": "USD", "unit": "currency", "scale": 1}
            for key in source}
    result = snapshot(info=source, financial_metadata=meta)
    assert result.metrics["cash_conversion"] == 2
    assert result.metrics["fcf_margin"] == .15
    assert "TTM" in result.metrics["fcf_margin_basis"]
    meta["netIncomeToCommon"]["period"] = "QUARTER"
    assert snapshot(info=source, financial_metadata=meta).metrics["cash_conversion"] is None
    # Reject missing counterpart metadata even when usable statements exist.
    income, cash = pair()
    result = snapshot(income, cash, info=source,
                      financial_metadata={"operatingCashflow": meta["operatingCashflow"]})
    assert result.metrics["cash_conversion"] is None


def test_mismatched_quarter_end_cannot_reuse_an_older_common_date():
    income, cash = pair()
    cash[pd.Timestamp("2026-09-30")] = [30, 25]
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] is None
    assert "Fechas" in result.metrics["cash_conversion_basis"]


def test_latest_period_selected_by_date_not_column_order_and_missing_stays_missing():
    income, cash = pair()
    # Reverse chronological expectations: old column deliberately first.
    income.insert(0, pd.Timestamp("2026-03-31"), [500, 100])
    cash.insert(0, pd.Timestamp("2026-03-31"), [900, 100])
    assert snapshot(income, cash).metrics["cash_conversion"] == 2
    cash.loc["Operating Cash Flow", pd.Timestamp("2026-06-30")] = float("nan")
    assert snapshot(income, cash).metrics["cash_conversion"] is None


def test_ambiguous_dates_or_rows_are_not_used():
    income, cash = pair()
    duplicate_dates = pd.concat([cash, cash], axis=1)
    duplicate_dates.attrs = cash.attrs.copy()
    assert snapshot(income, duplicate_dates).metrics["cash_conversion"] is None
    duplicate_rows = pd.concat([cash, cash.loc[["Operating Cash Flow"]]])
    duplicate_rows.attrs = cash.attrs.copy()
    assert snapshot(income, duplicate_rows).metrics["cash_conversion"] is None


def test_explicit_info_zeros_are_not_replaced_with_statement_values():
    income, cash = pair()
    result = snapshot(income, cash, info={
        "operatingCashflow": 0, "freeCashflow": 0, "totalDebt": 0, "totalCash": 2,
    }, balance_sheet=statement({"Total Debt": 999}))
    assert result.metrics["operating_cash_flow"] == 0
    assert result.metrics["free_cash_flow"] == 0
    assert result.metrics["total_debt"] == 0
    assert result.metrics["net_debt"] == -2
    assert any("Flujo de caja libre nulo: +0.0" in text for text in result.reasons)
    # This independently documented ratio still uses its quarterly pair.
    assert result.metrics["cash_conversion_numerator"] == 20


def test_zero_numerators_are_valid_and_zero_denominators_are_undefined():
    income, _ = pair()
    cash = statement({"Operating Cash Flow": 0, "Free Cash Flow": 0})
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] == 0
    assert result.metrics["fcf_margin"] == 0
    income *= 0
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] is None
    assert result.metrics["fcf_margin"] is None


def test_double_negative_ratios_never_reward_losses():
    income = statement({"Total Revenue": -100, "Net Income": -10})
    cash = statement({"Operating Cash Flow": -20, "Free Cash Flow": -15})
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] is None
    assert result.metrics["fcf_margin"] is None
    assert result.fundamental_points == -2


def test_negative_cash_with_positive_denominator_retains_penalties():
    income, _ = pair()
    cash = statement({"Operating Cash Flow": -20, "Free Cash Flow": -15})
    result = snapshot(income, cash)
    assert result.metrics["cash_conversion"] == -2
    assert result.metrics["fcf_margin"] == -.15
    assert result.fundamental_points == -5.5


def test_interest_expense_sign_does_not_turn_negative_ebit_bullish():
    income = statement({"EBIT": -20, "Interest Expense": -2})
    result = snapshot(income)
    assert result.metrics["interest_coverage"] == -10
    assert result.fundamental_points == -2
    income.loc["EBIT"] = 20
    assert snapshot(income).metrics["interest_coverage"] == 10


def test_negative_debt_equity_is_not_a_low_leverage_bonus():
    assert snapshot(info={"debtToEquity": -10}).fundamental_points == -2


def test_audit_provenance_is_serialized_without_mutating_input():
    income, cash = pair()
    original = cash.copy(deep=True)
    result = snapshot(income, cash)
    payload = fn.snapshot_to_payload(result)
    assert fn.snapshot_to_payload(fn.snapshot_from_payload(payload)) == payload
    pd.testing.assert_frame_equal(original, cash)
    assert original.attrs == cash.attrs
    assert result.version == "fundamental-news-v3-period-aligned"


@pytest.mark.parametrize("financial_currency", ["USD", None])
def test_yfinance_adapter_declares_quarter_raw_units_not_trading_currency(
    monkeypatch, tmp_path, financial_currency,
):
    import yfinance as yf

    income, cash = pair()
    income.attrs = {}
    cash.attrs = {}
    ticker = SimpleNamespace(
        get_info=lambda: {"financialCurrency": financial_currency, "currency": "USD"},
        quarterly_income_stmt=income, quarterly_balance_sheet=pd.DataFrame(),
        quarterly_cashflow=cash, calendar={}, news=[],
    )
    monkeypatch.setattr(fn, "DATA_DIR", tmp_path)
    monkeypatch.setattr(yf, "Ticker", lambda _: ticker)
    monkeypatch.setattr(yf, "set_tz_cache_location", lambda _: None)
    result = fn.download_fundamental_news("SMCI")
    assert result.metrics["cash_conversion"] == (2 if financial_currency else None)
    assert income.attrs == {}
    assert cash.attrs == {}


def test_empty_snapshot_remains_explicitly_neutral():
    result = snapshot()
    assert result.total_points == 0
    assert any("no entregó métricas" in text for text in result.reasons)

