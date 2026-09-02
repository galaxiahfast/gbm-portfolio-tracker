"""Interfaz Streamlit del portafolio GBM+.

Ejecuta: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from portfolio_tracker.analytics.backtesting import (
    BacktestBatchResult,
    BacktestConfig,
    BacktestOptimizationResult,
    batch_to_payload,
    optimize_backtest_parameters,
    run_backtest_batch,
)
from portfolio_tracker.analytics.chart_patterns import PatternDirection
from portfolio_tracker.analytics.fundamental_news import (
    FundamentalNewsSnapshot,
    apply_fundamental_filter,
    download_fundamental_news,
    snapshot_from_payload,
    snapshot_to_payload,
)
from portfolio_tracker.analytics.probability_calibration import calibrate_probability
from portfolio_tracker.analytics.risk import concentration_warnings
from portfolio_tracker.analytics.multi_timeframe import MacroTrend
from portfolio_tracker.analytics.technical_probability import (
    CandlePattern,
    CloudPosition,
    DailyTrend,
    MomentumState,
    ObvState,
    ProbabilityAnalysis,
    TechnicalSignal,
    analyze_probability,
)
from portfolio_tracker.config import DB_PATH, LOCAL_TIMEZONE, PROJECT_ROOT
from portfolio_tracker.db import Database
from portfolio_tracker.models import (
    CashMovementKind,
    FxQuote,
    PriceQuote,
    ReportedTotalType,
    TradeDraft,
    TradeSide,
    money,
    shares,
)
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.audit import AuditLevel, PortfolioAuditor
from portfolio_tracker.services.market_data import (
    CompositeFxProvider,
    MarketDataError,
    YahooChartProvider,
)
from portfolio_tracker.services.implementation_status import inspect_implementation_status
from portfolio_tracker.services.ocr import GbmOcrExtractor, OcrUnavailableError
from portfolio_tracker.services.portfolio import PortfolioCalculator
from portfolio_tracker.services.pdf_report import (
    build_executive_report,
    build_master_report,
    build_probability_report,
    build_technical_report,
    executive_decision,
)
from portfolio_tracker.services.projection_chart import (
    build_15_day_projection_figure,
    ordered_horizon_projections,
)
from portfolio_tracker.services.quant_market_data import (
    QuantMarketDataError,
    download_backtest_daily,
    download_quant_frames,
    normalize_symbol,
)
from portfolio_tracker.services.receipt_storage import ReceiptStorage
from portfolio_tracker.services.validation import validate_trade
from portfolio_tracker.ui import apply_premium_ui, premium_bar_chart, premium_line_chart


st.set_page_config(
    page_title="Portafolio GBM+",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_premium_ui()


@st.cache_resource
def get_repository() -> PortfolioRepository:
    database = Database(DB_PATH)
    database.initialize()
    repository = PortfolioRepository(database)
    repository.ensure_initial_capital()
    return repository


@st.cache_data(
    ttl="5m", max_entries=4, show_spinner=False, refresh_mode="background"
)
def fetch_live_fx() -> dict[str, str | bool]:
    """Consulta automática con Yahoo y respaldo Frankfurter."""

    quote = CompositeFxProvider().usd_mxn()
    return {
        "rate": str(quote.rate),
        "observed_at": quote.observed_at.isoformat(),
        "provider": quote.provider,
        "is_reference": quote.is_reference,
    }


def _fetch_one_price(symbol: str) -> tuple[str, dict[str, str] | None, str | None]:
    try:
        quote = YahooChartProvider().quote_usd(symbol)
        return (
            symbol,
            {
                "price": str(quote.price_usd),
                "observed_at": quote.observed_at.isoformat(),
                "provider": quote.provider,
            },
            None,
        )
    except MarketDataError as exc:
        return symbol, None, str(exc)


@st.cache_data(
    ttl="5m", max_entries=32, show_spinner=False, refresh_mode="background"
)
def fetch_live_prices(
    symbols: tuple[str, ...],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Cotiza hasta cuatro símbolos a la vez para mantener la interfaz fluida."""

    normalized = tuple(sorted({item.strip().upper() for item in symbols if item.strip()}))
    output: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    if not normalized:
        return output, failures
    with ThreadPoolExecutor(max_workers=min(4, len(normalized))) as executor:
        futures = [executor.submit(_fetch_one_price, symbol) for symbol in normalized]
        for future in as_completed(futures):
            symbol, payload, error = future.result()
            if payload:
                output[symbol] = payload
            elif error:
                failures.append(error)
    return output, failures


@st.cache_data(ttl="4m", max_entries=12, show_spinner=False)
def fetch_probability_frames(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mantiene las velas del predictor acotadas y actualizadas sin bloquear reruns."""

    return download_quant_frames(symbol)


@st.cache_data(ttl="30m", max_entries=12, show_spinner=False)
def fetch_fundamental_snapshot(symbol: str) -> FundamentalNewsSnapshot:
    """Corte fundamental/noticioso acotado; nunca se recalcula en cada rerun."""

    return download_fundamental_news(symbol)


@st.cache_data(ttl=6 * 60 * 60, max_entries=32, show_spinner=False)
def fetch_backtest_frame(symbol: str, period: str) -> pd.DataFrame:
    """Histórico cacheado por emisora y ventana para evitar descargas repetidas."""

    return download_backtest_daily(symbol, period)


def current_fx(repository: PortfolioRepository) -> tuple[FxQuote | None, str | None]:
    error: str | None = None
    try:
        payload = fetch_live_fx()
        live = FxQuote(
            rate=Decimal(str(payload["rate"])),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
            provider=str(payload["provider"]),
            is_reference=bool(payload["is_reference"]),
        )
        if not repository.fx_quote_exists(live):
            repository.add_fx_quote(live)
    except MarketDataError as exc:
        error = str(exc)
    return repository.latest_fx_quote(), error


def current_prices(
    repository: PortfolioRepository, symbols: list[str]
) -> tuple[dict[str, PriceQuote], list[str]]:
    normalized = sorted({item.strip().upper() for item in symbols if item.strip()})
    if not normalized:
        return {}, []
    payload, failures = fetch_live_prices(tuple(normalized))
    for symbol, item in payload.items():
        quote = PriceQuote(
            symbol=symbol,
            price_usd=Decimal(item["price"]),
            observed_at=datetime.fromisoformat(item["observed_at"]),
            provider=item["provider"],
        )
        if not repository.price_quote_exists(quote):
            repository.add_price_quote(quote)
    return repository.latest_price_quotes(normalized), failures


def usd(value: Decimal) -> str:
    return f"${money(value):,.2f} USD"


def usd_metric(value: Decimal) -> str:
    return f"${money(value):,.2f}"


def signed_delta(value: Decimal, suffix: str = "") -> str | None:
    """Evita presentar un cero neutro como si fuera una ganancia."""

    return None if value == 0 else f"{value:+,.2f}{suffix}"


def mxn(value_usd: Decimal, fx: FxQuote | None) -> str:
    if fx is None:
        return "Tasa USD/MXN pendiente"
    return f"${money(value_usd * fx.rate):,.2f} MXN"


def local_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(LOCAL_TIMEZONE)


def page_intro(title: str, description: str) -> None:
    with st.container(key="page_header"):
        st.markdown(":material/monitoring:  GBM+ · PORTFOLIO INTELLIGENCE")
        st.title(title, anchor=False)
        st.caption(description)


def show_market_notice() -> None:
    if fx_error:
        st.warning(
            "No fue posible actualizar en línea. Se está usando la última tasa "
            "guardada; puedes calibrarla manualmente en Configuración."
        )


US_MARKET_TIMEZONE = ZoneInfo("America/New_York")


def us_regular_market_is_open(now: datetime | None = None) -> bool:
    """Horario regular aproximado; Yahoo sigue siendo la fuente de la última vela."""

    current = now or datetime.now(timezone.utc)
    eastern = current.astimezone(US_MARKET_TIMEZONE)
    return eastern.weekday() < 5 and time(9, 30) <= eastern.time() < time(16, 0)


def register_receipt(
    repository: PortfolioRepository, content: bytes, filename: str
) -> int:
    stored = ReceiptStorage().save(content, filename)
    return repository.upsert_receipt(
        sha256=stored.sha256,
        original_filename=stored.original_filename,
        mime_type=stored.mime_type,
        original_path=stored.original_path,
        thumbnail_path=stored.thumbnail_path,
        width=stored.width,
        height=stored.height,
        byte_size=stored.byte_size,
    )


def save_trade(
    repository: PortfolioRepository,
    trade: TradeDraft,
    *,
    receipt_content: bytes | None = None,
    receipt_filename: str = "",
) -> bool:
    report = validate_trade(trade)
    trade.validation_status = report.status
    with st.container(border=True):
        st.subheader("Resultado de validación", anchor=False)
        for check in report.checks:
            st.success(check)
        for warning in report.warnings:
            st.warning(warning)
        for error in report.errors:
            st.error(error)
    if not report.is_valid:
        return False

    calculator = PortfolioCalculator()
    existing = repository.list_trades(ascending=True)
    quantities = calculator.quantities(existing)
    if trade.side is TradeSide.SELL:
        available = quantities.get(trade.symbol.upper(), Decimal("0"))
        if trade.quantity > available:
            st.error(
                f"No se puede vender {trade.quantity}: hay {available} títulos "
                f"registrados de {trade.symbol.upper()}."
            )
            return False
    if trade.side is TradeSide.BUY and repository.cash_balance_usd() + trade.cash_delta_usd < 0:
        st.error("La compra excede el efectivo USD disponible.")
        return False

    try:
        receipt_id = None
        if receipt_content:
            receipt_id = register_receipt(
                repository, receipt_content, receipt_filename or "comprobante"
            )
        trade_id = repository.add_trade(trade, receipt_id)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return False
    st.toast(f"Operación #{trade_id} guardada", icon=":material/check_circle:")
    st.success("La operación quedó conciliada y vinculada a su comprobante.")
    return True


def trade_form(
    repository: PortfolioRepository,
    *,
    prefix: str,
    defaults: dict[str, Any] | None = None,
    fx_quote: FxQuote | None = None,
    receipt_content: bytes | None = None,
    receipt_filename: str = "",
    ocr_text: str = "",
    ocr_confidence: Decimal | None = None,
) -> bool:
    defaults = defaults or {}
    now_local = datetime.now(LOCAL_TIMEZONE)
    default_dt = defaults.get("executed_at") or now_local
    if default_dt.tzinfo is None:
        default_dt = default_dt.replace(tzinfo=LOCAL_TIMEZONE)
    side_index = 1 if str(defaults.get("side") or "").lower() == "venta" else 0
    rate_default = defaults.get("fx_rate") or (fx_quote.rate if fx_quote else Decimal("18"))
    total_modes = {
        "Total bruto mostrado; comisión separada": ReportedTotalType.GROSS,
        "Cargo o abono final; comisión incluida": ReportedTotalType.SETTLEMENT,
    }
    default_total_type = str(defaults.get("reported_total_type") or "GROSS")
    total_index = 1 if default_total_type == ReportedTotalType.SETTLEMENT.value else 0

    with st.form(f"{prefix}_trade_form", clear_on_submit=False, border=True):
        st.subheader("Revisión previa", anchor=False)
        st.caption("Todos los campos son editables. Nada se guarda hasta confirmar.")
        first, second, third = st.columns(3)
        symbol = first.text_input(
            "Emisora / ticker", value=str(defaults.get("symbol") or "").upper()
        )
        products = ["USA", "SIC", "MÉXICO", "OTRO"]
        product_default = str(defaults.get("product") or "USA").upper()
        product = second.selectbox(
            "Producto",
            products,
            index=products.index(product_default) if product_default in products else 0,
        )
        side_label = third.selectbox(
            "Tipo de operación", ["Compra", "Venta"], index=side_index
        )

        first, second, third = st.columns(3)
        quantity = first.number_input(
            "Títulos",
            min_value=0.0,
            value=float(defaults.get("quantity") or 0),
            step=1.0,
            format="%.6f",
        )
        price = second.number_input(
            "Precio por título (USD)",
            min_value=0.0,
            value=float(defaults.get("price_usd") or 0),
            step=0.01,
            format="%.4f",
        )
        order_types = ["Limitada", "Mercado", "Stop limit", "Otra"]
        order_default = str(defaults.get("order_type") or "Limitada")
        order_type = third.selectbox(
            "Tipo de orden",
            order_types,
            index=order_types.index(order_default) if order_default in order_types else 0,
        )

        first, second, third = st.columns(3)
        commission = first.number_input(
            "Comisión (USD)",
            min_value=0.0,
            value=float(defaults.get("commission_usd") or 0),
            step=0.01,
            format="%.2f",
        )
        commission_rate = second.number_input(
            "Tasa de comisión (%)",
            min_value=0.0,
            value=float(defaults.get("commission_rate_pct") or 0),
            step=0.01,
            format="%.4f",
        )
        reported_total = third.number_input(
            "Total visible (USD)",
            min_value=0.0,
            value=float(defaults.get("reported_total_usd") or 0),
            step=0.01,
            format="%.2f",
            help="Deja 0 si el comprobante no muestra un total.",
        )
        total_mode_label = st.selectbox(
            "¿Qué representa el total visible?",
            list(total_modes),
            index=total_index,
            help=(
                "Compra final = títulos × precio + comisión. Venta final = títulos × "
                "precio − comisión. El formato GBM adjunto muestra el bruto y la comisión aparte."
            ),
        )

        first, second, third = st.columns(3)
        execution_date = first.date_input("Fecha", value=default_dt.date())
        execution_time = second.time_input(
            "Hora", value=default_dt.timetz().replace(tzinfo=None)
        )
        fx_rate = third.number_input(
            "USD/MXN aplicado o de referencia",
            min_value=0.0001,
            value=float(rate_default),
            step=0.0001,
            format="%.4f",
        )
        notes = st.text_area("Notas", value=str(defaults.get("notes") or ""))
        confirmed = st.checkbox(
            "Revisé el comprobante y confirmo títulos, precio, lado, total y comisión."
        )
        submitted = st.form_submit_button(
            "Validar y guardar", type="primary", icon=":material/verified:"
        )

    if not submitted:
        return False
    if not confirmed:
        st.error("Debes confirmar la revisión humana antes de guardar.")
        return False

    executed_at = datetime.combine(execution_date, execution_time, tzinfo=LOCAL_TIMEZONE)
    trade = TradeDraft(
        symbol=symbol,
        product=product,
        side=TradeSide.BUY if side_label == "Compra" else TradeSide.SELL,
        order_type=order_type,
        quantity=shares(quantity),
        price_usd=Decimal(str(price)),
        commission_usd=money(commission),
        commission_rate_pct=(
            Decimal(str(commission_rate)) if commission_rate > 0 else None
        ),
        reported_total_usd=money(reported_total) if reported_total > 0 else None,
        reported_total_type=total_modes[total_mode_label],
        fx_rate=Decimal(str(fx_rate)),
        executed_at=executed_at,
        notes=notes,
        ocr_text=ocr_text,
        ocr_confidence=ocr_confidence,
    )
    return save_trade(
        repository,
        trade,
        receipt_content=receipt_content,
        receipt_filename=receipt_filename,
    )


def dashboard(repository: PortfolioRepository, fx_quote: FxQuote | None) -> None:
    page_intro(
        "Resumen del portafolio",
        "Patrimonio, efectivo, posiciones y rendimiento con valuación USD/MXN.",
    )
    show_market_notice()
    trades = repository.list_trades(ascending=True)
    symbols = sorted({str(trade["symbol"]) for trade in trades})
    prices, price_failures = current_prices(repository, symbols)
    summary = PortfolioCalculator().summarize(
        trades=trades,
        cash_movements=repository.list_cash_movements(),
        prices=prices,
    )

    total_pnl = summary.total_pnl_usd
    result_label = (
        "Ganancia total actual"
        if total_pnl > 0
        else "Pérdida total actual"
        if total_pnl < 0
        else "Resultado total actual"
    )
    result_color = "green" if total_pnl > 0 else "red" if total_pnl < 0 else "gray"
    with st.container(border=True):
        st.badge(
            "GANANCIA" if total_pnl > 0 else "PÉRDIDA" if total_pnl < 0 else "SIN CAMBIO",
            color=result_color,
            icon=(
                ":material/trending_up:"
                if total_pnl > 0
                else ":material/trending_down:"
                if total_pnl < 0
                else ":material/trending_flat:"
            ),
        )
        st.metric(
            result_label,
            usd_metric(total_pnl),
            delta=signed_delta(summary.total_return_pct, "% sobre aportaciones"),
            delta_color="normal",
            help=mxn(total_pnl, fx_quote),
        )
        st.caption(
            f"Realizado: {usd_metric(summary.realized_pnl_usd)} · "
            f"Posiciones abiertas: {usd_metric(summary.unrealized_pnl_usd)} · "
            f"Comisiones acumuladas: {usd_metric(summary.commissions_usd)}"
        )

    with st.container(horizontal=True):
        st.metric(
            "Patrimonio",
            usd_metric(summary.equity_usd),
            signed_delta(summary.total_return_pct, "%"),
            border=True,
            icon=":material/account_balance_wallet:",
            help=mxn(summary.equity_usd, fx_quote),
        )
        st.metric(
            "Efectivo", usd_metric(summary.cash_usd), border=True,
            icon=":material/payments:", help=mxn(summary.cash_usd, fx_quote)
        )
        st.metric(
            "Posiciones", usd_metric(summary.holdings_value_usd), border=True,
            icon=":material/candlestick_chart:", help=mxn(summary.holdings_value_usd, fx_quote)
        )
        st.metric(
            "Comisiones", usd_metric(summary.commissions_usd), border=True,
            icon=":material/receipt_long:", help=mxn(summary.commissions_usd, fx_quote)
        )

    with st.container(horizontal=True):
        st.metric(
            "P&L realizado", usd_metric(summary.realized_pnl_usd),
            delta=signed_delta(summary.realized_pnl_usd, " USD"), border=True
        )
        st.metric(
            "P&L no realizado", usd_metric(summary.unrealized_pnl_usd),
            delta=signed_delta(summary.unrealized_pnl_usd, " USD"), border=True
        )
        st.metric("Aportaciones netas", usd_metric(summary.net_contributions_usd), border=True)

    if fx_quote:
        repository.add_portfolio_snapshot(
            cash_usd=summary.cash_usd,
            holdings_value_usd=summary.holdings_value_usd,
            equity_usd=summary.equity_usd,
            fx_rate=fx_quote.rate,
        )

    left, right = st.columns([1.6, 1])
    with left.container(border=True):
        st.subheader("Posiciones abiertas", anchor=False)
        if summary.positions:
            rows: list[dict[str, Any]] = []
            for position in summary.positions:
                weight = (
                    position.market_value_usd / summary.holdings_value_usd
                    if summary.holdings_value_usd else Decimal("0")
                )
                rows.append({
                    "Emisora": position.symbol,
                    "Títulos": float(position.quantity),
                    "Costo promedio": float(position.average_cost_usd),
                    "Precio actual": float(position.current_price_usd),
                    "Valor": float(position.market_value_usd),
                    "P&L abierto": float(position.unrealized_pnl_usd),
                    "Peso": float(weight),
                })
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True,
                column_config={
                    "Títulos": st.column_config.NumberColumn(format="%.6f"),
                    "Costo promedio": st.column_config.NumberColumn(format="$%.2f"),
                    "Precio actual": st.column_config.NumberColumn(format="$%.2f"),
                    "Valor": st.column_config.NumberColumn(format="$%.2f"),
                    "P&L abierto": st.column_config.NumberColumn(format="$%.2f"),
                    "Peso": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                },
            )
        else:
            st.info("Registra una compra para comenzar a construir posiciones.")

    with right.container(border=True):
        st.subheader("Distribución", anchor=False)
        if summary.positions:
            allocation = pd.DataFrame({
                "Emisora": [item.symbol for item in summary.positions],
                "Valor USD": [float(item.market_value_usd) for item in summary.positions],
            }).set_index("Emisora")
            premium_bar_chart(allocation, height=300, key="portfolio_allocation")
        else:
            st.info("Sin distribución disponible.")

    with st.container(border=True):
        st.subheader("Evolución registrada", anchor=False)
        snapshots = repository.list_portfolio_snapshots()
        if len(snapshots) >= 2:
            history = pd.DataFrame({
                "Fecha": [local_datetime(row["observed_at"]) for row in snapshots],
                "Patrimonio": [float(row["equity_usd"]) for row in snapshots],
                "Efectivo": [float(row["cash_usd"]) for row in snapshots],
                "Posiciones": [float(row["holdings_value_usd"]) for row in snapshots],
            }).set_index("Fecha")
            premium_line_chart(history, height=320, key="portfolio_history")
        else:
            st.info("La serie crecerá conforme se registren nuevas valuaciones.")

    for warning in summary.warnings + price_failures + concentration_warnings(summary):
        st.warning(warning)


def cash_page(repository: PortfolioRepository, fx_quote: FxQuote | None) -> None:
    page_intro(
        "Efectivo y divisas",
        "Ingresos MXN → USD, retiros USD → MXN y trazabilidad de cada conversión.",
    )
    show_market_notice()
    default_rate = float(fx_quote.rate) if fx_quote else 18.0
    available = repository.cash_balance_usd()
    with st.container(horizontal=True):
        st.metric(
            "Efectivo disponible", usd_metric(available), border=True,
            icon=":material/account_balance:", help=mxn(available, fx_quote)
        )
        if fx_quote:
            st.metric(
                "USD/MXN", f"{fx_quote.rate:,.4f}", border=True,
                icon=":material/currency_exchange:",
                help=f"{fx_quote.provider} · {fx_quote.observed_at.astimezone(LOCAL_TIMEZONE):%d/%m/%Y %H:%M}",
            )

    deposit_tab, withdrawal_tab, history_tab = st.tabs(
        ["Ingreso desde MXN", "Retiro hacia MXN", "Historial"]
    )

    with deposit_tab:
        with st.form("deposit_form", border=True):
            amount_mxn = st.number_input(
                "Importe depositado (MXN)", min_value=0.0, step=100.0, format="%.2f"
            )
            deposit_rate = st.number_input(
                "Tipo de cambio aplicado (MXN por USD)", min_value=0.0001,
                value=default_rate, step=0.0001, format="%.4f", key="deposit_rate"
            )
            deposit_date = st.date_input("Fecha", value=date.today(), key="deposit_date")
            deposit_notes = st.text_input("Notas", key="deposit_notes")
            deposit_submitted = st.form_submit_button(
                "Registrar ingreso", type="primary", icon=":material/add_card:"
            )
        if amount_mxn > 0:
            equivalent = money(
                Decimal(str(amount_mxn)) / Decimal(str(deposit_rate))
            )
            st.info(f"Se acreditarán aproximadamente {usd(equivalent)}.")
        if deposit_submitted:
            usd_amount = money(
                Decimal(str(amount_mxn)) / Decimal(str(deposit_rate))
            )
            if amount_mxn <= 0:
                st.error("Captura un importe mayor que cero.")
            else:
                repository.add_cash_movement(
                    kind=CashMovementKind.DEPOSIT,
                    original_amount=money(amount_mxn),
                    original_currency="MXN",
                    usd_amount=usd_amount,
                    fx_rate=Decimal(str(deposit_rate)),
                    occurred_at=datetime.combine(
                        deposit_date, time(12, 0), tzinfo=LOCAL_TIMEZONE
                    ),
                    notes=deposit_notes,
                )
                st.success(f"Ingreso registrado: {usd(usd_amount)}.")

    with withdrawal_tab:
        with st.form("withdrawal_form", border=True):
            amount_usd = st.number_input(
                "Dólares a retirar", min_value=0.0, step=10.0, format="%.2f"
            )
            withdrawal_rate = st.number_input(
                "Tipo de cambio aplicado (MXN por USD)", min_value=0.0001,
                value=default_rate, step=0.0001, format="%.4f", key="withdrawal_rate"
            )
            withdrawal_date = st.date_input(
                "Fecha", value=date.today(), key="withdrawal_date"
            )
            withdrawal_notes = st.text_input("Notas", key="withdrawal_notes")
            withdrawal_submitted = st.form_submit_button(
                "Registrar retiro", type="primary", icon=":material/output:"
            )
        amount_mxn = money(
            Decimal(str(amount_usd)) * Decimal(str(withdrawal_rate))
        )
        if amount_usd > 0:
            st.info(f"La salida estimada será ${amount_mxn:,.2f} MXN.")
        if withdrawal_submitted:
            usd_decimal = money(amount_usd)
            if usd_decimal <= 0:
                st.error("Captura un importe mayor que cero.")
            elif usd_decimal > available:
                st.error("El retiro excede el efectivo USD disponible.")
            else:
                repository.add_cash_movement(
                    kind=CashMovementKind.WITHDRAWAL,
                    original_amount=amount_mxn,
                    original_currency="MXN",
                    usd_amount=usd_decimal,
                    fx_rate=Decimal(str(withdrawal_rate)),
                    occurred_at=datetime.combine(
                        withdrawal_date, time(12, 0), tzinfo=LOCAL_TIMEZONE
                    ),
                    notes=withdrawal_notes,
                )
                st.success(
                    f"Retiro registrado: {usd(usd_decimal)} → ${amount_mxn:,.2f} MXN."
                )

    with history_tab:
        movements = repository.list_cash_movements()
        frame = pd.DataFrame([{
            "ID": row["id"],
            "Fecha": local_datetime(row["occurred_at"]).strftime("%d/%m/%Y"),
            "Tipo": {"INITIAL": "Capital inicial", "DEPOSIT": "Ingreso", "WITHDRAWAL": "Retiro"}[row["kind"]],
            "Importe origen": f"{row['original_amount']} {row['original_currency']}",
            "USD": float(row["usd_amount"]),
            "USD/MXN": float(row["fx_rate"]) if row["fx_rate"] else None,
            "Notas": row["notes"],
        } for row in movements])
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={"USD": st.column_config.NumberColumn(format="$%.2f")}
        )


def operations_page(repository: PortfolioRepository, fx_quote: FxQuote | None) -> None:
    page_intro("Operaciones", "Captura manual e historial contable de compras y ventas.")
    new_operation_tab, trade_history_tab = st.tabs(
        ["Nueva operación", "Historial"]
    )
    with new_operation_tab:
        trade_form(repository, prefix="manual", fx_quote=fx_quote)

    with trade_history_tab:
        trades = repository.list_trades()
        if not trades:
            st.info("No hay operaciones registradas.")
            return
        frame = pd.DataFrame([{
            "ID": row["id"],
            "Fecha": local_datetime(row["executed_at"]).strftime("%d/%m/%Y %H:%M"),
            "Emisora": row["symbol"],
            "Operación": "Compra" if row["side"] == "BUY" else "Venta",
            "Títulos": float(row["quantity"]),
            "Precio": float(row["price_usd"]),
            "Bruto": float(row["gross_usd"]),
            "Comisión": float(row["commission_usd"]),
            "Validación": row["validation_status"],
            "Imagen": "Sí" if row["receipt_id"] else "No",
        } for row in trades])
        st.dataframe(
            frame, width="stretch", hide_index=True,
            column_config={
                "Títulos": st.column_config.NumberColumn(format="%.6f"),
                "Precio": st.column_config.NumberColumn(format="$%.2f"),
                "Bruto": st.column_config.NumberColumn(format="$%.2f"),
                "Comisión": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        with st.container(border=True):
            st.subheader("Detalle", anchor=False)
            selected_id = st.selectbox("Operación", [row["id"] for row in trades])
            selected = next(row for row in trades if row["id"] == selected_id)
            first, second = st.columns([1, 1])
            with first:
                st.json({
                    "emisora": selected["symbol"], "producto": selected["product"],
                    "lado": selected["side"], "titulos": str(selected["quantity"]),
                    "precio_usd": str(selected["price_usd"]),
                    "total_visible_usd": str(selected["reported_total_usd"] or ""),
                    "interpretacion_total": selected.get("reported_total_type", "GROSS"),
                    "comision_usd": str(selected["commission_usd"]),
                    "notas": selected["notes"],
                })
            with second:
                if selected.get("thumbnail_path"):
                    image_path = PROJECT_ROOT / selected["thumbnail_path"]
                    if image_path.exists():
                        st.image(str(image_path), caption=selected["original_filename"], width=360)
                else:
                    st.info("Esta operación no tiene imagen vinculada.")
            confirm_delete = st.checkbox(
                "Entiendo que se eliminará la operación; la imagen se conservará para auditoría."
            )
            if st.button(
                "Eliminar operación", disabled=not confirm_delete, icon=":material/delete:"
            ):
                if repository.delete_trade(int(selected_id)):
                    st.toast("Operación eliminada; comprobante conservado")
                    st.rerun()


def receipt_page(repository: PortfolioRepository, fx_quote: FxQuote | None) -> None:
    page_intro(
        "Importar comprobante",
        "OCR local asistido: lectura, edición humana, conciliación y guardado.",
    )
    saved_message = st.session_state.pop("receipt_import_success", None)
    if saved_message:
        st.success(saved_message, icon=":material/check_circle:")
        st.caption("El formulario quedó limpio. Selecciona otro archivo para una nueva importación.")
    upload_nonce = st.session_state.setdefault("receipt_upload_nonce", 0)
    with st.container(border=True):
        st.markdown("**Flujo seguro:** imagen → extracción → corrección → validación → confirmación")
        st.caption("El OCR nunca modifica efectivo ni posiciones por sí solo.")
        uploaded = st.file_uploader(
            "Arrastra o selecciona JPG, PNG o WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"receipt_upload_{upload_nonce}",
        )
    extractor = GbmOcrExtractor()
    if not extractor.is_available():
        st.warning("No hay un motor OCR disponible; puedes usar la captura manual.")
    else:
        st.caption(f"Motor local disponible: {extractor.backend_name}")
    if uploaded is None:
        st.info("La imagen solo se guarda cuando confirmas la operación.")
        return

    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    preview, controls = st.columns([1, 1.4])
    with preview.container(border=True):
        st.image(content, caption=uploaded.name, width="stretch")
        st.caption(f"SHA-256 previo: {digest[:16]}…")
    with controls.container(border=True):
        st.subheader("Lectura", anchor=False)
        if st.button(
            "Analizar comprobante", type="primary", icon=":material/document_scanner:",
            disabled=not extractor.is_available()
        ):
            try:
                with st.status("Analizando la imagen…", expanded=True) as status:
                    st.write("Extrayendo texto y campos del comprobante.")
                    extraction = extractor.extract(content)
                    status.update(label="Lectura terminada", state="complete", expanded=False)
                st.session_state["ocr_digest"] = digest
                st.session_state["ocr_fields"] = extraction.fields
                st.session_state["ocr_text"] = extraction.raw_text
                st.session_state["ocr_confidence"] = extraction.confidence
                st.session_state["ocr_warnings"] = extraction.warnings
                st.toast("Campos extraídos; revísalos antes de guardar")
            except OcrUnavailableError as exc:
                st.error(str(exc))
        if st.session_state.get("ocr_digest") != digest:
            st.info("Analiza la imagen para generar el formulario editable.")

    if st.session_state.get("ocr_digest") != digest:
        return
    fields = st.session_state.get("ocr_fields", {})
    missing = [label for key, label in (
        ("symbol", "emisora"), ("side", "tipo de operación"),
        ("quantity", "títulos"), ("price_usd", "precio")
    ) if not fields.get(key)]
    with st.container(horizontal=True):
        confidence = st.session_state.get("ocr_confidence")
        if confidence is not None:
            color = "green" if confidence >= 80 else "orange" if confidence >= 65 else "red"
            st.badge(f"Confianza OCR {confidence}%", color=color)
        st.badge(
            "Campos críticos completos" if not missing else "Revisión manual requerida",
            color="green" if not missing else "orange",
        )
    for warning in st.session_state.get("ocr_warnings", []):
        st.warning(warning)
    if missing:
        st.warning("Completa manualmente: " + ", ".join(missing) + ".")
    with st.expander("Texto detectado por OCR"):
        st.code(st.session_state.get("ocr_text", ""), language=None)
    saved = trade_form(
        repository, prefix=f"ocr_{digest[:8]}", defaults=fields, fx_quote=fx_quote,
        receipt_content=content, receipt_filename=uploaded.name,
        ocr_text=st.session_state.get("ocr_text", ""), ocr_confidence=confidence,
    )
    if saved:
        for key in ("ocr_digest", "ocr_fields", "ocr_text", "ocr_confidence", "ocr_warnings"):
            st.session_state.pop(key, None)
        st.session_state["receipt_upload_nonce"] = upload_nonce + 1
        st.session_state["receipt_import_success"] = (
            "Operación guardada correctamente. El comprobante anterior ya no puede enviarse otra vez."
        )
        st.rerun()


def market_page(repository: PortfolioRepository) -> None:
    page_intro(
        "Mercado", "Consulta ligera de precios USD con caché y actualización en segundo plano."
    )
    holdings = sorted({trade["symbol"] for trade in repository.list_trades()})
    options = sorted(set(holdings + ["SMCI", "TSLA"]))
    selected = st.multiselect(
        "Emisoras a consultar", options, default=holdings or ["SMCI", "TSLA"],
        accept_new_options=True, help="Escribe un ticker nuevo y presiona Enter."
    )
    if st.button("Actualizar ahora", icon=":material/refresh:"):
        fetch_live_prices.clear()
        st.rerun()
    quotes, failures = current_prices(repository, selected)
    if quotes:
        with st.container(horizontal=True):
            for symbol in selected[:4]:
                quote = quotes.get(symbol)
                if quote:
                    st.metric(
                        symbol, usd_metric(quote.price_usd), border=True,
                        icon=":material/show_chart:",
                        help=f"{quote.provider} · {quote.observed_at.astimezone(LOCAL_TIMEZONE):%d/%m/%Y %H:%M}",
                    )
        if len(selected) > 4:
            frame = pd.DataFrame([{
                "Emisora": symbol, "Precio USD": float(quote.price_usd),
                "Fuente": quote.provider,
                "Fecha": quote.observed_at.astimezone(LOCAL_TIMEZONE),
            } for symbol, quote in quotes.items()])
            st.dataframe(
                frame, width="stretch", hide_index=True,
                column_config={"Precio USD": st.column_config.NumberColumn(format="$%.2f")}
            )
    else:
        st.info("Selecciona al menos una emisora para consultar.")
    for failure in failures:
        st.warning(failure)
    st.caption("Las cotizaciones son informativas y pueden tener retraso.")


def _render_probability_executive(
    analysis: ProbabilityAnalysis,
    *,
    adaptive_probability_threshold: float | None = None,
) -> None:
    """Panel de decisión breve; no vuelve a consultar ni recalcula datos de mercado."""

    decision = executive_decision(analysis)
    decision_label = decision.label
    decision_rationale = decision.rationale
    decision_tone = decision.tone
    if (
        adaptive_probability_threshold is not None
        and analysis.operation_probability / 100 < adaptive_probability_threshold
        and decision_tone == "success"
    ):
        decision_label = "ESPERAR · UMBRAL ONLINE NO ALCANZADO"
        decision_rationale = (
            f"El gatillo técnico existe, pero el score {analysis.operation_probability:.1f}/100 "
            f"no supera el umbral adaptativo de {adaptive_probability_threshold:.1%}."
        )
        decision_tone = "warning"
    levels = analysis.execution_levels
    bearish_plan = levels.direction == "SHORT"
    zone_label = "Zona de salida / venta" if bearish_plan else "Zona de entrada ideal"
    stop_label = "Invalidación bajista" if bearish_plan else "Stop loss técnico"
    target_label = "Objetivo bajista" if bearish_plan else "Take profit"
    icon = "🟢" if decision_tone == "success" else "🔴" if decision_tone == "danger" else "🟠"
    message = (
        f"### {icon} {decision_label}\n\n{decision_rationale}\n\n"
        f"**Estado del plan:** {'CONDICIONAL' if analysis.execution_plan_conditional else 'VALIDADO'}  \n"
        f"**{zone_label}:** \\${levels.entry_low:,.2f} - \\${levels.entry_high:,.2f}  \n"
        f"**{stop_label}:** \\${levels.stop_loss:,.2f}  \n"
        f"**{target_label} 1:** \\${levels.take_profit_1:,.2f} · "
        f"**{target_label} 2:** \\${levels.take_profit_2:,.2f}"
    )
    if decision_tone == "success":
        st.success(message)
    elif decision_tone == "danger":
        st.error(message)
    else:
        st.warning(message)
    score_suffix = "%" if analysis.has_empirical_probability else "/100"
    st.caption(
        f"{analysis.bullish_display_label}: {analysis.probability_up:.1f}{score_suffix} "
        f"({analysis.calibration_disclosure})."
    )

    with st.container(border=True):
        st.markdown("**Detonante cuantitativo de activación**")
        st.write(analysis.activation_trigger)
        with st.container(horizontal=True):
            st.badge(
                "Cumplido" if analysis.activation_trigger_met else "Pendiente",
                icon=":material/check_circle:" if analysis.activation_trigger_met else ":material/schedule:",
                color="green" if analysis.activation_trigger_met else "orange",
            )
            if analysis.tactical_short:
                st.badge("SHORT táctico contra tendencia mensual", color="orange")
            st.caption(f"Factor de exposición relativo: {analysis.exposure_factor:.2f}x")

    if analysis.neckline_heat_warning:
        st.error(analysis.neckline_heat_warning, icon=":material/local_fire_department:")

    with st.container(horizontal=True):
        st.metric(
            zone_label,
            f"{levels.entry_low:,.2f} - {levels.entry_high:,.2f} USD",
            border=True,
            icon=":material/login:",
            help=(
                "Zona informativa. Si el plan es condicional, no ejecutar hasta que el precio y los osciladores validen el retroceso."
            ),
        )
        st.metric(
            stop_label,
            f"${levels.stop_loss:,.2f}",
            border=True,
            icon=":material/shield:",
            help=(
                f"Stop dinámico: precio de entrada conservador - {levels.stop_atr_multiple:.2f} × ATR(14) 5m, "
                f"respetando estructura técnica cuando sea más conservadora. "
                f"ATR actual: ${levels.atr_5m:.4f}."
            ),
        )
        st.metric(
            f"{target_label} 1 · 1 hora",
            f"${levels.take_profit_1:,.2f}",
            border=True,
            icon=":material/looks_one:",
            help=(
                (f"Alineado con {levels.pattern_target_label}. " if levels.pattern_target_applied else "")
                + f"R:R de TP1 = {levels.take_profit_1_reward_risk:.2f}R; mínimo {levels.minimum_reward_risk:.1f}R."
            ),
        )
        st.metric(
            f"{target_label} 2 · 6 horas",
            f"${levels.take_profit_2:,.2f}",
            border=True,
            icon=":material/looks_two:",
        )

    with st.container(horizontal=True):
        st.metric("Último precio", f"${analysis.last_price:,.2f}", border=True)
        st.metric(
            "Score operativo",
            f"{analysis.operation_probability:.1f}/100" if analysis.operation_probability else "Detonante pendiente",
            border=True,
            help=(
                f"{analysis.probability_status}. Muestra resuelta: "
                f"{analysis.calibration_samples}; Brier: "
                f"{analysis.calibration_brier_score:.3f}"
                if analysis.calibration_brier_score is not None
                else f"{analysis.probability_status}. Sin Brier validado todavía."
            ),
        )
        st.metric(
            "Régimen 5 min",
            analysis.market_regime,
            delta=analysis.position_size_policy,
            delta_color="inverse" if analysis.market_regime != "TENDENCIA CONFIRMADA" else "off",
            border=True,
        )
        st.metric(
            "Estocástico RSI",
            f"%K {analysis.stochastic_k:.1f} · %D {analysis.stochastic_d:.1f}",
            delta="LONG bloqueado" if analysis.long_entry_blocked else "Sin bloqueo extremo",
            delta_color="inverse" if analysis.long_entry_blocked else "off",
            border=True,
        )
        st.metric(
            "Vigilancia de rebote",
            "Activa" if analysis.rebound_watch_active else "Inactiva",
            delta=f"Soporte ${analysis.nearest_support:,.2f}",
            delta_color="off",
            border=True,
        )

    _render_chart_patterns(analysis, compact=True)

    ordered_projections = ordered_horizon_projections(analysis.horizon_projections)
    horizon_frame = pd.DataFrame(
        [
            {
                "Horizonte": item.label,
                "Lectura alcista": item.probability_up,
                "Objetivo alcista": item.bullish_target,
                "Lectura lateral": item.probability_range,
                "Rango esperado": f"${item.range_low:,.2f} - ${item.range_high:,.2f}",
                "Lectura bajista": item.probability_down,
                "Objetivo bajista": item.bearish_target,
                "Motor": item.engine_name,
                "Estado": item.probability_status,
                "Muestra": item.calibration_samples,
                "Brier": item.brier_score,
            }
            for item in ordered_projections
        ]
    )
    st.subheader("Mapa de scores por horizonte", anchor=False)
    with st.container(horizontal=True, gap="small"):
        st.badge("Escenario alcista", icon=":material/trending_up:", color="green")
        st.badge("Escenario lateral", icon=":material/trending_flat:", color="blue")
        st.badge("Escenario bajista", icon=":material/trending_down:", color="red")
    st.dataframe(
        horizon_frame,
        hide_index=True,
        width="stretch",
        height="content",
        row_height=40,
        key=f"executive_horizons_{analysis.symbol}",
        column_config={
            "Horizonte": st.column_config.TextColumn(pinned=True, width=92),
            "Lectura alcista": st.column_config.ProgressColumn(
                "Score alcista", format="%.1f/100", min_value=0, max_value=100,
                color="green", width=105,
            ),
            "Objetivo alcista": st.column_config.NumberColumn(
                "Objetivo ↑", format="$%.2f", width=105,
            ),
            "Lectura lateral": st.column_config.ProgressColumn(
                "Score lateral", format="%.1f/100", min_value=0, max_value=100,
                color="blue", width=105,
            ),
            "Rango esperado": st.column_config.TextColumn(width=145),
            "Lectura bajista": st.column_config.ProgressColumn(
                "Score bajista", format="%.1f/100", min_value=0, max_value=100,
                color="red", width=105,
            ),
            "Objetivo bajista": st.column_config.NumberColumn(
                "Objetivo ↓", format="$%.2f", width=105,
            ),
            "Muestra": st.column_config.NumberColumn(format="%d"),
            "Brier": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    with st.container(border=True):
        st.subheader(
            "Velas históricas y trayectoria proyectada · 30 + 15 sesiones",
            anchor=False,
        )
        projection_figure = build_15_day_projection_figure(
            analysis.daily_projection,
            analysis.last_price,
            analysis.daily_indicators,
        )
        st.plotly_chart(
            projection_figure,
            width="stretch",
            height=340,
            key=f"executive_projection_{analysis.symbol}",
            config={
                "displaylogo": False,
                "scrollZoom": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        st.caption(
            "Las primeras 30 velas son sesiones OHLC observadas. Después de la línea ámbar, "
            "las 15 velas son escenarios bootstrap: apertura enlazada, cierre central y "
            "mechas ATR 15–85. La sección futura no representa precios garantizados."
        )


def _render_fundamental_news(
    analysis: ProbabilityAnalysis,
    snapshot: FundamentalNewsSnapshot | None,
    *,
    compact: bool,
) -> None:
    with st.container(border=True):
        st.subheader("Contexto fundamental y noticias", anchor=False)
        if snapshot is None:
            st.info(
                "No existe todavía un corte fundamental verificable. El motor mantuvo "
                "ponderación neutral y no inventó métricas."
            )
            return
        with st.container(horizontal=True):
            st.metric(
                "Ponderación total",
                f"{analysis.fundamental_score:+.1f} pp",
                delta=analysis.fundamental_label,
                delta_color=(
                    "normal" if analysis.fundamental_score > 0
                    else "inverse" if analysis.fundamental_score < 0 else "off"
                ),
                border=True,
            )
            st.metric(
                "Fundamental",
                f"{snapshot.fundamental_points:+.1f} pp",
                border=True,
            )
            st.metric(
                "Noticias",
                f"{snapshot.news_points:+.1f} pp",
                border=True,
            )
            st.metric(
                "Eventos",
                snapshot.event_risk_level,
                delta="Veto activo" if analysis.fundamental_risk_veto else "Sin veto",
                delta_color="inverse" if analysis.fundamental_risk_veto else "off",
                border=True,
            )
        st.caption(
            f"Corte {local_datetime(snapshot.observed_at):%d/%m/%Y %H:%M} · "
            f"{snapshot.provider} · SHA-256 {analysis.fundamental_snapshot_sha256[:16]}…"
        )
        if compact:
            for reason in analysis.fundamental_reasons[:4]:
                st.write("• " + reason)
            return

        metrics = snapshot.metrics
        metric_rows = [
            {"Métrica": "Margen neto", "Valor": metrics.get("profit_margin"), "Formato": "ratio"},
            {"Métrica": "Crecimiento ingresos", "Valor": metrics.get("revenue_growth"), "Formato": "ratio"},
            {"Métrica": "Crecimiento beneficios", "Valor": metrics.get("earnings_growth"), "Formato": "ratio"},
            {"Métrica": "Deuda/capital", "Valor": metrics.get("debt_to_equity"), "Formato": "number"},
            {"Métrica": "Flujo de caja libre", "Valor": metrics.get("free_cash_flow"), "Formato": "usd"},
            {"Métrica": "Flujo operativo", "Valor": metrics.get("operating_cash_flow"), "Formato": "usd"},
            {"Métrica": "Conversión beneficio/caja", "Valor": metrics.get("cash_conversion"), "Formato": "number"},
            {"Métrica": "Margen de flujo libre", "Valor": metrics.get("fcf_margin"), "Formato": "ratio"},
            {"Métrica": "Deuda neta", "Valor": metrics.get("net_debt"), "Formato": "usd"},
            {"Métrica": "Cobertura de intereses", "Valor": metrics.get("interest_coverage"), "Formato": "number"},
            {"Métrica": "Ingresos trimestrales", "Valor": metrics.get("quarterly_revenue"), "Formato": "usd"},
            {"Métrica": "Beneficio trimestral", "Valor": metrics.get("quarterly_net_income"), "Formato": "usd"},
        ]
        display_metrics = pd.DataFrame(
            [
                {
                    "Métrica": row["Métrica"],
                    "Valor": (
                        "Sin dato" if row["Valor"] is None
                        else f"{float(row['Valor']):.1%}" if row["Formato"] == "ratio"
                        else f"${float(row['Valor']):,.0f}" if row["Formato"] == "usd"
                        else f"{float(row['Valor']):,.2f}"
                    ),
                }
                for row in metric_rows
            ]
        )
        st.dataframe(display_metrics, hide_index=True, key=f"fundamentals_{analysis.symbol}")
        if snapshot.events:
            st.markdown("**Eventos relevantes**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Evento": item.kind,
                            "Fecha": local_datetime(item.event_at),
                            "Detalle": item.detail,
                        }
                        for item in snapshot.events
                    ]
                ),
                hide_index=True,
                key=f"fundamental_events_{analysis.symbol}",
                column_config={
                    "Fecha": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                },
            )
        if snapshot.news:
            st.markdown("**Flujo de noticias versionado**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Fecha": local_datetime(item.published_at),
                            "Titular": item.title,
                            "Fuente": item.publisher,
                            "Temas": ", ".join(item.topics) or "General",
                            "Impacto": item.sentiment,
                            "Peso recencia": item.recency_weight,
                            "Clase": item.impact_class,
                            "Enlace": item.url or None,
                        }
                        for item in snapshot.news
                    ]
                ),
                hide_index=True,
                key=f"fundamental_news_{analysis.symbol}",
                column_config={
                    "Fecha": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                    "Impacto": st.column_config.NumberColumn(format="%+.2f"),
                    "Peso recencia": st.column_config.NumberColumn(format="%.3f"),
                    "Enlace": st.column_config.LinkColumn(display_text="Abrir fuente"),
                },
            )
        with st.expander("Justificación completa del ponderador"):
            for reason in analysis.fundamental_reasons:
                st.write("• " + reason)


def _render_chart_patterns(
    analysis: ProbabilityAnalysis,
    *,
    compact: bool,
) -> None:
    """Presenta las mismas evidencias objetivas usadas por el motor y el PDF."""

    patterns = analysis.chart_patterns[: 4 if compact else 10]
    with st.container(border=True):
        st.subheader("Patrones chartistas y estructuras", anchor=False)
        with st.container(horizontal=True, gap="small"):
            st.badge(
                f"{sum(pattern.valid for pattern in analysis.chart_patterns)} confirmados >75%",
                icon=":material/verified:",
                color="green" if any(pattern.valid for pattern in analysis.chart_patterns) else "gray",
            )
            st.badge(
                f"Impacto {analysis.chart_pattern_impact:+.1f} pp",
                icon=":material/balance:",
                color=(
                    "green"
                    if analysis.chart_pattern_impact > 0
                    else "red"
                    if analysis.chart_pattern_impact < 0
                    else "gray"
                ),
            )
            if analysis.chart_pattern_veto:
                st.badge(
                    "Veto chartista activo",
                    icon=":material/gpp_bad:",
                    color="red",
                )
        if not patterns:
            st.caption(
                "No se detectaron estructuras geométricas recientes que superen los filtros Zig-Zag."
            )
            return
        direction_labels = {
            PatternDirection.BULLISH: "Alcista",
            PatternDirection.BEARISH: "Bajista",
            PatternDirection.NEUTRAL: "Neutral",
        }
        rows = []
        for pattern in patterns:
            row = {
                "Marco": pattern.timeframe,
                "Patrón": pattern.label,
                "Dirección": direction_labels[pattern.direction],
                "Confianza": pattern.confidence,
                "Estado": "Confirmado" if pattern.valid else "En formación / descartado",
                "Neckline": pattern.neckline,
                "Objetivo": pattern.target_price,
                "Volumen": pattern.volume_ratio,
            }
            if not compact:
                row["Validación"] = pattern.detail
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            height="content",
            key=f"chart_patterns_{'quick' if compact else 'advanced'}_{analysis.symbol}",
            column_config={
                "Confianza": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100, width=110
                ),
                "Neckline": st.column_config.NumberColumn(format="$%.2f"),
                "Objetivo": st.column_config.NumberColumn(format="$%.2f"),
                "Volumen": st.column_config.NumberColumn("Volumen relativo", format="%.2fx"),
            },
        )
        st.caption(
            "Solo patrones confirmados y con confianza superior a 75% modifican el puntaje; "
            "las figuras incompletas se muestran únicamente para auditoría."
        )

def _probability_predictor_content(*, live_mode: bool) -> None:
    page_intro(
        "Motor cuantitativo · Fase 5",
        "Scores multi-temporales, calibración empírica y veto central de riesgo.",
    )
    st.warning(
        "Las lecturas se muestran como scores heurísticos sobre 100 mientras no exista "
        "una muestra OOS masiva con Brier calibrado. No son probabilidades ni recomendaciones."
    )
    # El contenedor conserva esta posición aunque los bytes se generen después.
    # Así los cuatro botones quedan físicamente encima de st.tabs.
    actions_slot = st.container()
    executive_tab, technical_tab, calibration_tab = st.tabs(
        [
            "Vista Ejecutiva (Modo Rápido)",
            "Vista Técnica Avanzada (Completa)",
            "Calibración y backtesting",
        ],
        on_change="rerun",
    )

    st.session_state.setdefault("predictor_symbol", "SMCI")
    with st.form("probability_symbol_form", border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            symbol_input = st.text_input(
                "Emisora",
                value=st.session_state["predictor_symbol"],
                placeholder="SMCI",
                help="Ticker de Estados Unidos reconocido por Yahoo Finance.",
            )
            analyze_submitted = st.form_submit_button(
                "Analizar", type="primary", icon=":material/analytics:"
            )
    if analyze_submitted:
        try:
            st.session_state["predictor_symbol"] = normalize_symbol(symbol_input)
        except QuantMarketDataError as exc:
            st.error(str(exc))
            return
    symbol = st.session_state["predictor_symbol"]

    if st.button(
        "Actualizar velas ahora",
        icon=":material/refresh:",
        key="refresh_probability_data",
    ):
        fetch_probability_frames.clear()
        fetch_fundamental_snapshot.clear()
        st.rerun()

    output = st.container()
    calibrated_parameters = repository.latest_backtest_parameters() or {
        "minimum_probability": 0.55,
        "stop_atr_multiple": 2.25,
        "risk_per_trade_pct": 1.0,
    }
    try:
        with output.skeleton(height=360):
            intraday, daily = fetch_probability_frames(symbol)
            analysis = analyze_probability(
                symbol,
                intraday,
                daily,
                atr_stop_multiple=float(
                    calibrated_parameters.get("stop_atr_multiple", 2.25)
                ),
            )
    except (QuantMarketDataError, ValueError) as exc:
        output.error(str(exc))
        output.caption(
            "Verifica conexión, ticker y horario de mercado. El resto del portafolio "
            "continúa funcionando aunque esta fuente falle."
        )
        return

    fundamental_snapshot: FundamentalNewsSnapshot | None = None
    fundamental_warning: str | None = None
    fundamental_hash = ""
    try:
        fundamental_snapshot = fetch_fundamental_snapshot(symbol)
        payload_json = snapshot_to_payload(fundamental_snapshot)
        _, fundamental_hash = repository.record_fundamental_news_snapshot(
            symbol=fundamental_snapshot.symbol,
            observed_at=datetime.fromisoformat(fundamental_snapshot.observed_at),
            provider=fundamental_snapshot.provider,
            engine_version=fundamental_snapshot.version,
            payload_json=payload_json,
        )
    except (QuantMarketDataError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        stored = repository.latest_fundamental_news_snapshot(symbol)
        if stored:
            try:
                fundamental_snapshot = snapshot_from_payload(str(stored["payload_json"]))
                fundamental_hash = str(stored["payload_sha256"])
                fundamental_warning = (
                    "La actualización externa falló; se utilizó el último corte local "
                    "cuya huella SHA-256 fue verificada."
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                fundamental_snapshot = None
        if fundamental_snapshot is None:
            fundamental_warning = f"Contexto fundamental neutral: {exc}"
    if fundamental_snapshot is not None:
        try:
            analysis = apply_fundamental_filter(analysis, fundamental_snapshot)
            analysis = replace(
                analysis,
                fundamental_snapshot_sha256=fundamental_hash,
            )
        except ValueError as exc:
            fundamental_warning = f"El corte fundamental fue descartado: {exc}"
            fundamental_snapshot = None

    horizon_minutes = {
        "1 Hora": 60,
        "6 Horas": 390,
        "1 Día": 1_440,
        "1 Semana": 10_080,
        "1 Mes": 43_200,
        "6 Meses": 181_440,
    }
    raw_probability_up = analysis.probability_up
    raw_horizon_probabilities = {
        horizon.label: horizon.probability_up
        for horizon in analysis.horizon_projections
    }
    calibrated_horizons = []
    for horizon in analysis.horizon_projections:
        samples = repository.live_model_calibration_samples(
            analysis.symbol,
            horizon_minutes=horizon_minutes[horizon.label],
        )
        calibration = calibrate_probability(
            horizon.probability_up / 100,
            samples,
        )
        probability_range = horizon.probability_range
        calibrated_up = min(
            100.0 - probability_range,
            calibration.calibrated_probability * 100,
        )
        calibrated_horizons.append(
            replace(
                horizon,
                probability_up=round(calibrated_up, 1),
                probability_down=round(100.0 - probability_range - calibrated_up, 1),
                probability_status=calibration.status,
                calibration_samples=calibration.sample_size,
                brier_score=calibration.brier_score,
            )
        )
    primary_calibration = calibrate_probability(
        raw_probability_up / 100,
        repository.live_model_calibration_samples(
            analysis.symbol,
            horizon_minutes=390,
        ),
    )
    calibrated_probability_up = round(
        primary_calibration.calibrated_probability * 100, 1
    )
    if primary_calibration.empirically_calibrated:
        long_direction = analysis.execution_levels.direction == "LONG"
        directional_probability = (
            calibrated_probability_up
            if long_direction
            else 100.0 - calibrated_probability_up
        )
        operation_probability = (
            min(analysis.operation_probability, directional_probability)
            if analysis.risk_veto
            else directional_probability
        )
    else:
        operation_probability = analysis.operation_probability
    analysis = replace(
        analysis,
        raw_probability_up=raw_probability_up,
        probability_up=calibrated_probability_up,
        probability_down=round(100.0 - calibrated_probability_up, 1),
        operation_probability=round(operation_probability, 1),
        horizon_projections=tuple(calibrated_horizons),
        probability_status=primary_calibration.status,
        calibration_samples=primary_calibration.sample_size,
        calibration_brier_score=primary_calibration.brier_score,
    )

    if live_mode:
        repository.resolve_live_model_observations(
            symbol=analysis.symbol,
            current_price=Decimal(str(analysis.last_price)),
            current_as_of=analysis.as_of,
        )
        for horizon in analysis.horizon_projections:
            repository.record_live_model_observation(
                symbol=analysis.symbol,
                observed_at=analysis.as_of,
                reference_price=Decimal(str(analysis.last_price)),
                raw_probability_up=Decimal(
                    str(raw_horizon_probabilities[horizon.label] / 100)
                ),
                parameters_json=json.dumps(
                    {
                        **calibrated_parameters,
                        "engine": horizon.engine_name,
                        "probability_status": horizon.probability_status,
                    },
                    sort_keys=True,
                ),
                horizon_minutes=horizon_minutes[horizon.label],
            )
    online_stats = repository.live_model_stats(analysis.symbol)

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(
                "Mercado EUA abierto · actualización cada 5 min"
                if live_mode
                else "Mercado EUA cerrado · último corte disponible",
                color="green" if live_mode else "gray",
                icon=":material/sync:" if live_mode else ":material/schedule:",
            )
            st.caption(
                f"Realimentación resuelta: {online_stats['resolved']} observaciones · "
                f"acierto {online_stats['accuracy']:.1%} · "
                f"Brier {online_stats['brier_score']:.3f} · "
                f"umbral adaptativo {online_stats['adaptive_threshold']:.1%}."
            )
            st.badge(
                analysis.probability_status,
                color=(
                    "green"
                    if analysis.probability_status.startswith("Probabilidad empíricamente")
                    else "orange"
                ),
                icon=":material/science:",
            )
        if fundamental_warning:
            st.warning(fundamental_warning)

    executive_pdf = build_executive_report(analysis)
    technical_pdf = build_technical_report(analysis)
    combined_pdf = build_probability_report(analysis)
    calibration_context = {
        "backtest_run": repository.latest_backtest_run(),
        "online_stats": online_stats,
    }
    master_pdf = build_master_report(analysis, calibration_context)
    timestamp = analysis.as_of.strftime("%Y%m%d_%H%M")
    with actions_slot.container(border=True):
        st.markdown("**Descargas del análisis actual**")
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.download_button(
                "Descargar Vista Ejecutiva",
                data=executive_pdf,
                file_name=f"vista_ejecutiva_{analysis.symbol}_{timestamp}.pdf",
                mime="application/pdf",
                icon=":material/description:",
                type="primary",
                help="Decisión, niveles, horizontes y trayectoria bootstrap de 15 sesiones.",
            )
            st.download_button(
                "Descargar Vista Técnica Avanzada",
                data=technical_pdf,
                file_name=f"vista_tecnica_{analysis.symbol}_{timestamp}.pdf",
                mime="application/pdf",
                icon=":material/analytics:",
                help="Todos los paneles técnicos, indicadores y lectura auditable del motor.",
            )
            st.download_button(
                "Descargar PDF combinado",
                data=combined_pdf,
                file_name=f"reporte_combinado_{analysis.symbol}_{timestamp}.pdf",
                mime="application/pdf",
                icon=":material/library_books:",
                help="Combina la vista ejecutiva y la vista técnica avanzada en un único PDF.",
            )
            st.download_button(
                "Descargar PDF maestro",
                data=master_pdf,
                file_name=f"reporte_maestro_{analysis.symbol}_{timestamp}.pdf",
                mime="application/pdf",
                icon=":material/summarize:",
                help="Incluye las vistas ejecutiva, técnica y la última calibración/backtesting auditada.",
            )
        st.caption(
            "Los cuatro archivos se generan desde el mismo corte de datos; el maestro añade la calibración local auditada."
        )

    if executive_tab.open:
        with executive_tab:
            _render_probability_executive(
                analysis,
                adaptive_probability_threshold=float(
                    online_stats["adaptive_threshold"]
                ),
            )
            _render_fundamental_news(analysis, fundamental_snapshot, compact=True)
    if technical_tab.open:
        with technical_tab:
            _render_fundamental_news(analysis, fundamental_snapshot, compact=False)
            signal_labels = {
                TechnicalSignal.BUY: "Compra confirmada",
                TechnicalSignal.SELL: "Venta confirmada",
                TechnicalSignal.WATCH_BUY: "Vigilar compra",
                TechnicalSignal.WATCH_SELL: "Vigilar venta",
                TechnicalSignal.NEUTRAL: "Sin señal",
            }
            signal_colors = {
                TechnicalSignal.BUY: "green",
                TechnicalSignal.SELL: "red",
                TechnicalSignal.WATCH_BUY: "blue",
                TechnicalSignal.WATCH_SELL: "orange",
                TechnicalSignal.NEUTRAL: "gray",
            }
            trend_labels = {
                DailyTrend.BULLISH: "Tendencia diaria alcista",
                DailyTrend.BEARISH: "Tendencia diaria bajista",
                DailyTrend.NEUTRAL: "Tendencia diaria neutral",
            }
            trend_colors = {
                DailyTrend.BULLISH: "green",
                DailyTrend.BEARISH: "red",
                DailyTrend.NEUTRAL: "gray",
            }
            macro_labels = {
                MacroTrend.STRONG_BULLISH: "Alcista firme",
                MacroTrend.BULLISH: "Alcista",
                MacroTrend.NEUTRAL: "Neutral",
                MacroTrend.BEARISH: "Bajista",
                MacroTrend.STRONG_BEARISH: "Bajista firme",
            }
            macro_colors = {
                MacroTrend.STRONG_BULLISH: "green",
                MacroTrend.BULLISH: "blue",
                MacroTrend.NEUTRAL: "gray",
                MacroTrend.BEARISH: "orange",
                MacroTrend.STRONG_BEARISH: "red",
            }

            with st.container(horizontal=True):
                st.badge(signal_labels[analysis.signal], color=signal_colors[analysis.signal])
                st.badge(
                    trend_labels[analysis.daily_trend],
                    color=trend_colors[analysis.daily_trend],
                )
                st.badge(
                    "Volumen confirmado" if analysis.volume_confirmed else "Volumen sin confirmar",
                    color="green" if analysis.volume_confirmed else "orange",
                )
                st.badge(
                    f"MACD 5m {analysis.macd_state_5m.value.lower()}",
                    color={
                        MomentumState.BULLISH: "green",
                        MomentumState.BEARISH: "red",
                        MomentumState.NEUTRAL: "gray",
                    }[analysis.macd_state_5m],
                )
                st.badge(
                    analysis.market_regime,
                    color="green" if analysis.market_regime == "TENDENCIA CONFIRMADA" else "red" if analysis.range_market else "orange",
                )
                st.badge(
                    f"Semanal: {macro_labels[analysis.weekly_trend]}",
                    color=macro_colors[analysis.weekly_trend],
                )
                st.badge(
                    f"Mensual: {macro_labels[analysis.monthly_trend]}",
                    color=macro_colors[analysis.monthly_trend],
                )

            with st.container(horizontal=True):
                st.metric(
                    analysis.bullish_display_label,
                    f"{analysis.probability_up:.1f}{'%' if analysis.has_empirical_probability else '/100'}",
                    delta=analysis.calibration_disclosure,
                    delta_color="off",
                    border=True,
                    icon=":material/trending_up:",
                )
                st.metric(
                    analysis.bearish_display_label,
                    f"{analysis.probability_down:.1f}{'%' if analysis.has_empirical_probability else '/100'}",
                    delta=analysis.calibration_disclosure,
                    delta_color="off",
                    border=True,
                    icon=":material/trending_down:",
                )
                st.metric(
                    "Último precio",
                    f"${analysis.last_price:,.2f}",
                    border=True,
                    icon=":material/attach_money:",
                )
                st.metric(
                    "Nivel técnico sugerido",
                    f"${analysis.suggested_level:,.2f}",
                    border=True,
                    icon=":material/my_location:",
                    help="Nivel de vigilancia derivado de Bollinger, VWAP, pivotes y Fibonacci; no es una orden.",
                )
                st.metric(
                    "Score de operación",
                    (
                        f"{analysis.operation_probability:.1f}/100"
                        if analysis.signal in (TechnicalSignal.BUY, TechnicalSignal.SELL)
                        else "Sin gatillo"
                    ),
                    border=True,
                    icon=":material/security:",
                    help="Score direccional después del veto macro; no se presenta como probabilidad sin calibración OOS masiva.",
                )

            obv_labels = {
                ObvState.ACCUMULATION: "Acumulación",
                ObvState.DISTRIBUTION: "Distribución",
                ObvState.CONFIRMING_UP: "Confirma subida",
                ObvState.CONFIRMING_DOWN: "Confirma bajada",
                ObvState.NEUTRAL: "Neutral",
            }
            with st.container(horizontal=True):
                st.metric(
                    "MACD · 5 min",
                    f"{analysis.macd_5m:+.3f}",
                    delta=f"Histograma {analysis.macd_histogram_5m:+.3f}",
                    border=True,
                )
                st.metric(
                    "VWAP de sesión",
                    f"${analysis.vwap:,.2f}",
                    delta=f"Precio {analysis.price_vs_vwap_pct:+.2f}%",
                    border=True,
                )
                st.metric(
                    "ADX · 5 min",
                    f"{analysis.adx:.1f}",
                    delta="Rango / lateral" if analysis.range_market else "Tendencia activa",
                    delta_color="off" if analysis.range_market else "normal",
                    border=True,
                )
                st.metric(
                    "Flujo OBV",
                    obv_labels[analysis.obv_state],
                    delta=f"Precio 12 velas {analysis.obv_price_change_pct:+.2f}%",
                    delta_color="off",
                    border=True,
                )

            cloud_labels = {
                CloudPosition.ABOVE: "Sobre la nube",
                CloudPosition.INSIDE: "Dentro de la nube",
                CloudPosition.BELOW: "Bajo la nube",
            }
            candle_labels = {
                CandlePattern.BULLISH_ENGULFING: "Envolvente alcista",
                CandlePattern.BEARISH_ENGULFING: "Envolvente bajista",
                CandlePattern.HAMMER: "Martillo",
                CandlePattern.SHOOTING_STAR: "Estrella fugaz",
                CandlePattern.BULLISH_CONTINUATION: "Continuación alcista",
                CandlePattern.BEARISH_CONTINUATION: "Continuación bajista",
                CandlePattern.NONE: "Sin patrón",
            }
            with st.container(horizontal=True):
                st.metric(
                    "Fibonacci más cercano",
                    f"${analysis.fibonacci.nearest_level:,.2f}",
                    delta=(
                        f"{analysis.fibonacci.nearest_ratio} · {analysis.fibonacci.distance_pct:.2f}% · "
                        f"{analysis.fibonacci.role.lower()}"
                    ),
                    delta_color="normal" if analysis.fibonacci.near_zone else "off",
                    border=True,
                )
                st.metric(
                    "Ichimoku · 5 min",
                    cloud_labels[analysis.ichimoku_5m],
                    delta=f"Tenkan {analysis.tenkan_5m:.2f} · Kijun {analysis.kijun_5m:.2f}",
                    delta_color="off",
                    border=True,
                )
                st.metric(
                    "Ichimoku · diario",
                    cloud_labels[analysis.ichimoku_daily],
                    delta=f"Tenkan {analysis.tenkan_daily:.2f} · Kijun {analysis.kijun_daily:.2f}",
                    delta_color="off",
                    border=True,
                )
                st.metric(
                    "Última vela · 5 min",
                    candle_labels[analysis.candle_pattern],
                    delta="Última vela cerrada",
                    delta_color="off",
                    border=True,
                )

            _render_chart_patterns(analysis, compact=False)

            if analysis.risk_veto:
                st.error(analysis.risk_alert)
                for reason in analysis.risk_reasons:
                    st.error(reason, icon=":material/gpp_bad:")
            elif analysis.signal_rejected or "no vale la pena" in analysis.verdict.lower():
                st.error(analysis.verdict)
            elif analysis.range_market:
                st.warning(analysis.verdict)
            elif analysis.signal in (TechnicalSignal.BUY, TechnicalSignal.SELL):
                st.success(analysis.verdict)
            else:
                st.info(analysis.verdict)

            panel = st.segmented_control(
                "Horizonte de análisis",
                ["Intradiaria · 5 min–1 h", "Contexto · diario–semanal", "Estructural · mensual–anual"],
                default="Intradiaria · 5 min–1 h",
                key="probability_timeframe_panel",
            )

            if panel == "Intradiaria · 5 min–1 h":
                price_frame = analysis.intraday_indicators.tail(78).loc[
                    :, ["Close", "VWAP", "BB_upper", "BB_middle", "BB_lower"]
                ].rename(columns={"Close": "Precio", "VWAP": "VWAP sesión", "BB_upper": "Banda superior", "BB_middle": "Media Bollinger", "BB_lower": "Banda inferior"})
                oscillator_frame = analysis.intraday_indicators.tail(78).loc[:, ["StochRSI_K", "StochRSI_D"]].rename(columns={"StochRSI_K": "%K", "StochRSI_D": "%D"})
                oscillator_frame["Sobrecompra"] = 80.0
                oscillator_frame["Sobreventa"] = 20.0
                macd_5m_frame = analysis.intraday_indicators.tail(78).loc[:, ["MACD", "MACD_signal", "MACD_histogram"]].rename(columns={"MACD_signal": "Señal 9", "MACD_histogram": "Histograma"})
                macd_hourly_frame = analysis.hourly_indicators.tail(40).loc[:, ["MACD", "MACD_signal", "MACD_histogram"]].rename(columns={"MACD_signal": "Señal 9", "MACD_histogram": "Histograma"})
                strength_frame = analysis.intraday_indicators.tail(78).loc[:, ["ADX14", "Plus_DI14", "Minus_DI14"]].rename(columns={"ADX14": "ADX 14", "Plus_DI14": "+DI", "Minus_DI14": "-DI"})
                obv_frame = analysis.intraday_indicators.tail(78).loc[:, ["OBV"]]

                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("Gatillo operativo · 5 min", anchor=False)
                    premium_line_chart(price_frame, height=300, key="quant_intraday_price")
                with second.container(border=True):
                    st.subheader("Estocástico RSI · 5 min", anchor=False)
                    premium_line_chart(oscillator_frame, height=300, key="quant_stoch_rsi")
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("MACD rápido · 5 min", anchor=False)
                    premium_line_chart(macd_5m_frame, height=270, key="quant_macd_5m")
                with second.container(border=True):
                    st.subheader("Confirmación · 1 hora", anchor=False)
                    if macd_hourly_frame.dropna(how="all").empty:
                        st.info("Aún no hay suficientes barras horarias para MACD completo.")
                    else:
                        premium_line_chart(macd_hourly_frame, height=270, key="quant_macd_hourly")
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("ADX y dirección · 5 min", anchor=False)
                    premium_line_chart(strength_frame, height=250, key="quant_adx")
                    st.caption("ADX <20 activa el veto si existe una señal operativa.")
                with second.container(border=True):
                    st.subheader("Volumen acumulado OBV · 5 min", anchor=False)
                    premium_line_chart(obv_frame, height=250, key="quant_obv")
                    st.write(f"**Última vela:** {analysis.candle_detail}")

            elif panel == "Contexto · diario–semanal":
                daily_frame = analysis.daily_indicators.tail(260).loc[:, ["Close", "EMA9", "EMA21", "EMA50", "EMA200"]].rename(columns={"Close": "Cierre", "EMA9": "EMA 9", "EMA21": "EMA 21", "EMA50": "EMA 50", "EMA200": "EMA 200"})
                weekly_frame = analysis.weekly_indicators.tail(156).loc[:, ["Close", "EMA9", "EMA21", "EMA50", "EMA200"]].rename(columns={"Close": "Cierre", "EMA9": "EMA 9", "EMA21": "EMA 21", "EMA50": "EMA 50", "EMA200": "EMA 200"})
                macd_daily_frame = analysis.daily_indicators.tail(180).loc[:, ["MACD", "MACD_signal", "MACD_histogram"]].rename(columns={"MACD_signal": "Señal 9", "MACD_histogram": "Histograma"})
                macd_weekly_frame = analysis.weekly_indicators.tail(104).loc[:, ["MACD", "MACD_signal", "MACD_histogram"]].rename(columns={"MACD_signal": "Señal 9", "MACD_histogram": "Histograma"})
                ichimoku_daily_frame = analysis.daily_indicators.tail(180).loc[:, ["Close", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B"]].rename(columns={"Close": "Precio", "Ichimoku_Tenkan": "Tenkan 9", "Ichimoku_Kijun": "Kijun 26", "Ichimoku_Senkou_A": "Nube A", "Ichimoku_Senkou_B": "Nube B"})

                with st.container(horizontal=True):
                    st.metric("Tendencia semanal", macro_labels[analysis.weekly_trend], border=True)
                    st.metric("Tendencia mensual", macro_labels[analysis.monthly_trend], border=True)
                    st.metric("EMA 50 diaria", f"${analysis.ema50:,.2f}", border=True)
                    st.metric("EMA 200 diaria", f"${analysis.ema200:,.2f}", border=True)
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("Estructura diaria · EMA 9/21/50/200", anchor=False)
                    premium_line_chart(daily_frame, height=300, key="quant_daily_ema")
                with second.container(border=True):
                    st.subheader("Estructura semanal · EMA 9/21/50/200", anchor=False)
                    premium_line_chart(weekly_frame, height=300, key="quant_weekly_ema")
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("MACD macro · diario", anchor=False)
                    premium_line_chart(macd_daily_frame, height=270, key="quant_macd_daily")
                with second.container(border=True):
                    st.subheader("MACD macro · semanal", anchor=False)
                    premium_line_chart(macd_weekly_frame, height=270, key="quant_macd_weekly")
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("Ichimoku · diario", anchor=False)
                    premium_line_chart(ichimoku_daily_frame, height=280, key="quant_ichimoku_daily")
                with second.container(border=True):
                    st.subheader("Soportes y resistencias · diario/semanal", anchor=False)
                    pivot_frame = pd.DataFrame([
                        {"Nivel": "R2", "Precio": analysis.pivots.r2}, {"Nivel": "R1", "Precio": analysis.pivots.r1},
                        {"Nivel": "Pivote", "Precio": analysis.pivots.pivot}, {"Nivel": "S1", "Precio": analysis.pivots.s1},
                        {"Nivel": "S2", "Precio": analysis.pivots.s2},
                        {"Nivel": "Resistencia 20 semanas", "Precio": analysis.weekly_resistance},
                        {"Nivel": "Soporte 20 semanas", "Precio": analysis.weekly_support},
                    ])
                    st.dataframe(pivot_frame, hide_index=True, width="stretch", column_config={"Precio": st.column_config.NumberColumn(format="$%.2f")})
                    st.caption(f"Sesión base: {analysis.pivots.source_date}.")

            else:
                monthly_frame = analysis.monthly_indicators.tail(60).loc[:, ["Close", "EMA9", "EMA21", "EMA50"]].rename(columns={"Close": "Cierre mensual", "EMA9": "EMA 9", "EMA21": "EMA 21", "EMA50": "EMA 50"})
                monthly_macd_frame = analysis.monthly_indicators.tail(60).loc[:, ["MACD", "MACD_signal", "MACD_histogram"]].rename(columns={"MACD_signal": "Señal 9", "MACD_histogram": "Histograma"})

                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("Estructura mensual", anchor=False)
                    premium_line_chart(monthly_frame, height=300, key="quant_monthly_ema")
                with second.container(border=True):
                    st.subheader("MACD mensual", anchor=False)
                    premium_line_chart(monthly_macd_frame, height=300, key="quant_macd_monthly")
                first, second = st.columns(2)
                with first.container(border=True):
                    st.subheader("Fibonacci · 22 sesiones", anchor=False)
                    fibonacci_frame = pd.DataFrame([
                        {"Nivel": "Máximo", "Precio": analysis.fibonacci.high}, {"Nivel": "0.382", "Precio": analysis.fibonacci.level_382},
                        {"Nivel": "0.500", "Precio": analysis.fibonacci.level_500}, {"Nivel": "0.618", "Precio": analysis.fibonacci.level_618},
                        {"Nivel": "Mínimo", "Precio": analysis.fibonacci.low},
                    ])
                    st.dataframe(fibonacci_frame, hide_index=True, width="stretch", column_config={"Precio": st.column_config.NumberColumn(format="$%.2f")})
                    st.caption(f"{analysis.fibonacci.source_start} a {analysis.fibonacci.source_end}.")
                with second.container(border=True):
                    st.subheader("Fibonacci anual · 252 sesiones", anchor=False)
                    annual_fibonacci_frame = pd.DataFrame([
                        {"Nivel": "Máximo", "Precio": analysis.annual_fibonacci.high}, {"Nivel": "0.382", "Precio": analysis.annual_fibonacci.level_382},
                        {"Nivel": "0.500", "Precio": analysis.annual_fibonacci.level_500}, {"Nivel": "0.618", "Precio": analysis.annual_fibonacci.level_618},
                        {"Nivel": "Mínimo", "Precio": analysis.annual_fibonacci.low},
                    ])
                    st.dataframe(annual_fibonacci_frame, hide_index=True, width="stretch", column_config={"Precio": st.column_config.NumberColumn(format="$%.2f")})
                    st.caption(f"{analysis.annual_fibonacci.source_start} a {analysis.annual_fibonacci.source_end}.")
                with st.container(border=True):
                    st.subheader("Grandes zonas de liquidez · aproximación anual", anchor=False)
                    if analysis.liquidity_zones:
                        liquidity_frame = pd.DataFrame([
                            {"Zona inferior": zone.lower, "Centro": zone.center, "Zona superior": zone.upper, "Volumen anual": zone.volume_share_pct}
                            for zone in analysis.liquidity_zones
                        ])
                        st.dataframe(
                            liquidity_frame, hide_index=True, width="stretch",
                            column_config={
                                "Zona inferior": st.column_config.NumberColumn(format="$%.2f"),
                                "Centro": st.column_config.NumberColumn(format="$%.2f"),
                                "Zona superior": st.column_config.NumberColumn(format="$%.2f"),
                                "Volumen anual": st.column_config.NumberColumn(format="%.1f%%"),
                            },
                        )
                    else:
                        st.info("No existe volumen suficiente para estimar zonas.")
                    st.caption("Nodos aproximados por volumen diario agregado; no representan el libro de órdenes de GBM.")

            with st.container(border=True):
                st.subheader("Lectura del motor", anchor=False)
                st.caption("Puntaje base: 50%. Cada fila muestra su aporte en puntos porcentuales.")
                score_frame = pd.DataFrame([
                    {
                        "Filtro": component.name,
                        "Impacto": component.impact_points,
                        "Justificación": component.detail,
                    }
                    for component in analysis.score_breakdown
                ])
                st.dataframe(
                    score_frame,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Impacto": st.column_config.NumberColumn(format="%+.1f pp"),
                    },
                )
                for observation in analysis.observations:
                    st.write("• " + observation)
                for warning in analysis.warnings:
                    st.warning(warning)
                st.caption(
                    f"Última vela: {analysis.as_of.astimezone(LOCAL_TIMEZONE):%d/%m/%Y %H:%M} · "
                    "Fuente: Yahoo Finance mediante yfinance. Datos posiblemente retrasados."
                )
    if calibration_tab.open:
        with calibration_tab:
            backtesting_page(repository, fx_quote, embedded=True)


@st.fragment(run_every="5m")
def _live_probability_predictor() -> None:
    """Actualiza solo el predictor durante la sesión regular estadounidense."""

    _probability_predictor_content(live_mode=True)


def probability_predictor_page() -> None:
    if us_regular_market_is_open():
        _live_probability_predictor()
    else:
        _probability_predictor_content(live_mode=False)


def _backtest_account_defaults(
    repository: PortfolioRepository,
) -> tuple[float, float, list[str]]:
    movements = repository.list_cash_movements()
    net_contributions = sum(
        (
            -float(item["usd_amount"])
            if item["kind"] == CashMovementKind.WITHDRAWAL.value
            else float(item["usd_amount"])
        )
        for item in movements
    )
    trades = repository.list_trades(ascending=True)
    commission_rates = [
        float(item["commission_usd"] / item["gross_usd"] * 10_000)
        for item in trades
        if item.get("gross_usd") and item["gross_usd"] > 0
    ]
    commission_bps = (
        sum(commission_rates) / len(commission_rates)
        if commission_rates
        else 25.0
    )
    holdings = sorted({str(item["symbol"]).upper() for item in trades})
    return max(100.0, net_contributions), commission_bps, holdings


def _render_backtest_result(batch: BacktestBatchResult) -> None:
    aggregate = batch.aggregate
    approved = batch.aggregate_decision.startswith("APROBADO")
    st.badge(
        "Setups OOS aceptables" if approved else "Preservación de capital: setups rechazados",
        color="green" if approved else "red",
        icon=":material/verified:" if approved else ":material/gpp_bad:",
    )
    with st.container(horizontal=True):
        st.metric(
            "Acierto fuera de muestra",
            f"{aggregate.win_rate:.1%}",
            delta=f"Límite conservador {aggregate.win_rate_lower_bound:.1%}",
            border=True,
            icon=":material/target:",
        )
        st.metric(
            "Profit factor neto",
            "∞" if aggregate.profit_factor is None and aggregate.trades else
            f"{(aggregate.profit_factor or 0):.2f}",
            border=True,
            icon=":material/balance:",
        )
        st.metric(
            "Drawdown máximo",
            f"{aggregate.maximum_drawdown_pct:.2f}%",
            delta_color="inverse",
            border=True,
            icon=":material/trending_down:",
        )
        st.metric(
            "Costos simulados",
            f"${aggregate.costs_usd:,.2f}",
            border=True,
            icon=":material/receipt_long:",
        )
    with st.container(horizontal=True):
        st.metric("Operaciones OOS", aggregate.trades, border=True)
        st.metric("Setups rechazados", aggregate.rejected, border=True)
        st.metric("Retorno neto OOS", f"{aggregate.net_return_pct:+.2f}%", border=True)
        st.metric(
            "Error Brier",
            "Sin muestra" if aggregate.brier_score is None else f"{aggregate.brier_score:.3f}",
            border=True,
            help="0 es calibración perfecta; 0.25 equivale aproximadamente a predecir 50/50.",
        )

    benchmark_rows = []
    for result in batch.results:
        if result.benchmarks is None:
            continue
        benchmark_rows.append(
            {
                "Emisora": result.symbol,
                "Bot neto": result.validation.net_return_pct / 100,
                "Buy & Hold neto": result.benchmarks.buy_hold_net_return_pct / 100,
                "Cruce EMA neto": result.benchmarks.ema_crossover_net_return_pct / 100,
                "Exceso vs B&H": result.benchmarks.bot_excess_vs_buy_hold_pct / 100,
                "Exceso vs EMA": result.benchmarks.bot_excess_vs_ema_pct / 100,
            }
        )
    if benchmark_rows:
        with st.container(border=True):
            st.subheader("Benchmarks netos y comparables", anchor=False)
            st.dataframe(
                pd.DataFrame(benchmark_rows),
                hide_index=True,
                key="backtest_benchmarks",
                column_config={
                    column: st.column_config.NumberColumn(format="percent")
                    for column in (
                        "Bot neto", "Buy & Hold neto", "Cruce EMA neto",
                        "Exceso vs B&H", "Exceso vs EMA",
                    )
                },
            )

    fold_rows = [
        {
            "Emisora": result.symbol,
            "Ventana": fold.fold,
            "Inicio": pd.Timestamp(fold.start_date).date(),
            "Fin": pd.Timestamp(fold.end_date).date(),
            "Muestra calibración": fold.calibration_samples,
            "Señales": fold.metrics.setups,
            "Trades": fold.metrics.trades,
            "Acierto": fold.metrics.win_rate,
            "Brier": fold.metrics.brier_score,
        }
        for result in batch.results
        for fold in result.walk_forward_folds
    ]
    if fold_rows:
        with st.container(border=True):
            st.subheader("Validación walk-forward expansiva", anchor=False)
            st.dataframe(
                pd.DataFrame(fold_rows),
                hide_index=True,
                key="backtest_walk_forward",
                column_config={
                    "Inicio": st.column_config.DateColumn(format="DD/MM/YYYY"),
                    "Fin": st.column_config.DateColumn(format="DD/MM/YYYY"),
                    "Acierto": st.column_config.NumberColumn(format="percent"),
                    "Brier": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    regime_rows = [
        {
            "Emisora": result.symbol,
            "Régimen": regime,
            "Señales": metrics.setups,
            "Trades": metrics.trades,
            "Acierto": metrics.win_rate,
            "Profit factor": metrics.profit_factor,
            "Retorno": metrics.net_return_pct / 100,
        }
        for result in batch.results
        for regime, metrics in result.regime_metrics
    ]
    if regime_rows:
        with st.container(border=True):
            st.subheader("Robustez por régimen de mercado", anchor=False)
            st.dataframe(
                pd.DataFrame(regime_rows),
                hide_index=True,
                key="backtest_regimes",
                column_config={
                    "Acierto": st.column_config.NumberColumn(format="percent"),
                    "Profit factor": st.column_config.NumberColumn(format="%.2f"),
                    "Retorno": st.column_config.NumberColumn(format="percent"),
                },
            )

    rows = []
    for result in batch.results:
        rows.append(
            {
                "Emisora": result.symbol,
                "Corte OOS": pd.Timestamp(result.split_date).date(),
                "Trades entrenamiento": result.training.trades,
                "Trades OOS": result.validation.trades,
                "Acierto OOS": result.validation.win_rate,
                "Límite inferior": result.validation.win_rate_lower_bound,
                "Profit factor": result.validation.profit_factor,
                "Drawdown": result.validation.maximum_drawdown_pct / 100,
                "Retorno neto": result.validation.net_return_pct / 100,
                "Costos USD": result.validation.costs_usd,
                "Decisión": result.decision,
            }
        )
    with st.container(border=True):
        st.subheader("Validación por emisora", anchor=False)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            key="backtest_symbol_results",
            column_config={
                "Corte OOS": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "Acierto OOS": st.column_config.NumberColumn(format="percent"),
                "Límite inferior": st.column_config.NumberColumn(format="percent"),
                "Profit factor": st.column_config.NumberColumn(format="%.2f"),
                "Drawdown": st.column_config.NumberColumn(format="percent"),
                "Retorno neto": st.column_config.NumberColumn(format="percent"),
                "Costos USD": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

    equity_rows = []
    for result in batch.results:
        equity_rows.extend(
            {
                "Fecha": pd.Timestamp(date_value),
                "Emisora": result.symbol,
                "Capital simulado": equity,
            }
            for date_value, equity in result.validation_equity_curve
        )
    if equity_rows:
        with st.container(border=True):
            st.subheader("Curva de capital fuera de muestra", anchor=False)
            equity_frame = pd.DataFrame(equity_rows).pivot(
                index="Fecha", columns="Emisora", values="Capital simulado"
            )
            premium_line_chart(equity_frame, height=320, key="backtest_equity_curve")

    rejection_rows = [
        {"Emisora": result.symbol, "Motivo": reason, "Cantidad": count}
        for result in batch.results
        for reason, count in result.rejected_reasons
        if count
    ]
    with st.container(border=True):
        st.subheader("Veto y preservación de capital", anchor=False)
        for result in batch.results:
            st.markdown(f"**{result.symbol} · {result.decision}**")
            for reason in result.decision_reasons:
                st.write("• " + reason)
        if rejection_rows:
            st.dataframe(
                pd.DataFrame(rejection_rows),
                hide_index=True,
                key="backtest_rejections",
                column_config={
                    "Cantidad": st.column_config.NumberColumn(format="%d"),
                },
            )


def _render_optimization_result(result: BacktestOptimizationResult) -> None:
    best = result.best_config
    st.subheader("Parámetros óptimos encontrados", anchor=False)
    with st.container(horizontal=True):
        st.metric("Umbral mínimo", f"{best.minimum_probability:.0%}", border=True)
        st.metric("Distancia del stop", f"{best.stop_atr_multiple:.2f} × ATR", border=True)
        st.metric("Riesgo por operación", f"{best.risk_per_trade_pct:.2f}%", border=True)
        st.metric("Combinaciones evaluadas", len(result.trials), border=True)
    trial_frame = pd.DataFrame(
        [
            {
                "Umbral": item.minimum_probability,
                "Stop ATR": item.stop_atr_multiple,
                "Riesgo": item.risk_per_trade_pct / 100,
                "Trades": item.trades,
                "Compras ganadoras": item.long_wins,
                "Ventas ganadoras": item.short_wins,
                "Acierto": item.win_rate,
                "Profit factor": item.profit_factor,
                "Drawdown": item.maximum_drawdown_pct / 100,
                "Brier": item.brier_score,
                "Puntaje objetivo": item.objective_score,
            }
            for item in result.trials
        ]
    )
    st.dataframe(
        trial_frame,
        hide_index=True,
        key="backtest_optimization_trials",
        column_config={
            "Umbral": st.column_config.NumberColumn(format="percent"),
            "Riesgo": st.column_config.NumberColumn(format="percent"),
            "Acierto": st.column_config.NumberColumn(format="percent"),
            "Drawdown": st.column_config.NumberColumn(format="percent"),
            "Profit factor": st.column_config.NumberColumn(format="%.2f"),
            "Brier": st.column_config.NumberColumn(format="%.3f"),
            "Puntaje objetivo": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "La tabla ordena la validación interna del entrenamiento. El resultado inferior "
        "se calcula después sobre el tramo OOS que la búsqueda nunca vio."
    )


def backtesting_page(
    repository: PortfolioRepository,
    fx_quote: FxQuote | None,
    *,
    embedded: bool = False,
) -> None:
    if not embedded:
        page_intro(
            "Calibración estadística y backtesting",
            "Validación fuera de muestra del motor Fase 4 con costos, ATR y veto de capital.",
        )
    else:
        st.header("Calibración estadística y backtesting", anchor=False)
        st.caption(
            "Optimización de umbral, stop ATR y riesgo con validación fuera de muestra."
        )
    st.warning(
        "El tramo de validación nunca calibra el modelo. Un resultado aprobado describe "
        "el histórico probado; no garantiza rendimiento futuro."
    )
    account_capital, ledger_commission_bps, holdings = _backtest_account_defaults(repository)
    available = sorted(set(["TSLA", "NVDA", "SMCI", "GME"] + holdings))
    defaults = sorted(set(["TSLA", "NVDA", "SMCI", "GME"] + holdings))

    with st.form("backtesting_configuration", border=True):
        st.subheader("Configuración reproducible", anchor=False)
        symbols = st.multiselect(
            "Emisoras",
            available,
            default=defaults,
            accept_new_options=True,
            help="Incluye automáticamente las posiciones registradas en el libro local.",
        )
        first, second, third = st.columns(3)
        period = first.selectbox("Histórico diario", ["5y", "10y", "max"], index=1)
        training_pct = second.slider(
            "Entrenamiento", min_value=55, max_value=85, value=70, step=5,
            help="El resto del histórico se reserva cronológicamente para validación OOS.",
        )
        starting_capital = third.number_input(
            "Capital de referencia (USD)", min_value=100.0,
            value=float(round(account_capital, 2)), step=100.0,
            help="Precargado desde las aportaciones netas del libro contable.",
        )
        first, second, third = st.columns(3)
        stop_multiple = first.slider("Stop ATR de referencia", 2.0, 2.5, 2.25, 0.05)
        reward_risk = second.slider("Objetivo riesgo/beneficio", 1.5, 3.0, 2.0, 0.25)
        holding_sessions = third.number_input(
            "Máximo de sesiones", min_value=2, max_value=30, value=10, step=1
        )
        first, second, third = st.columns(3)
        risk_pct = first.slider("Riesgo de referencia", 0.25, 2.0, 1.0, 0.25, format="%.2f%%")
        commission_bps = second.number_input(
            "Comisión por lado (bps)", min_value=0.0, max_value=100.0,
            value=float(round(ledger_commission_bps, 2)), step=1.0,
            help="Precargada con el promedio observado en las operaciones reales.",
        )
        slippage_bps = third.number_input(
            "Deslizamiento por lado (bps)", min_value=0.0, max_value=50.0,
            value=5.0, step=1.0,
        )
        minimum_probability = st.slider(
            "Umbral de referencia",
            0.50, 0.75, 0.55, 0.01, format="%.0f%%",
        )
        search_mode = st.selectbox(
            "Profundidad de búsqueda automática",
            ["Rápida · 8 combinaciones", "Amplia · 18 combinaciones"],
            help="La búsqueda amplia tarda más, pero explora una rejilla mayor sin tocar el OOS final.",
        )
        submitted = st.form_submit_button(
            "Ejecutar y registrar backtest",
            type="primary",
            icon=":material/science:",
        )

    if submitted:
        try:
            normalized_symbols = tuple(sorted({normalize_symbol(item) for item in symbols}))
            if not normalized_symbols:
                raise ValueError("Selecciona al menos una emisora.")
            config = BacktestConfig(
                training_fraction=training_pct / 100,
                stop_atr_multiple=stop_multiple,
                reward_risk=reward_risk,
                holding_sessions=int(holding_sessions),
                risk_per_trade_pct=risk_pct,
                commission_bps_per_side=commission_bps,
                slippage_bps_per_side=slippage_bps,
                minimum_probability=minimum_probability,
            )
            frames: dict[str, pd.DataFrame] = {}
            failures: list[str] = []
            with st.status("Descargando históricos y ejecutando reglas…", expanded=True) as status:
                with ThreadPoolExecutor(max_workers=min(4, len(normalized_symbols))) as executor:
                    futures = {
                        executor.submit(fetch_backtest_frame, symbol, period): symbol
                        for symbol in normalized_symbols
                    }
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            frames[symbol] = future.result()
                            st.write(f"{symbol}: histórico disponible.")
                        except (QuantMarketDataError, ValueError) as exc:
                            failures.append(str(exc))
                if failures:
                    raise QuantMarketDataError(" ".join(failures))
                optimization = optimize_backtest_parameters(
                    frames,
                    config,
                    starting_capital_usd=float(starting_capital),
                    probability_grid=(0.52, 0.57)
                    if search_mode.startswith("Rápida")
                    else (0.50, 0.55, 0.60),
                    atr_grid=(2.0, 2.25)
                    if search_mode.startswith("Rápida")
                    else (2.0, 2.25, 2.5),
                    risk_grid=(0.50, 1.00),
                )
                batch = optimization.final_oos
                payload = batch_to_payload(batch)
                parameters = {
                    **asdict(optimization.best_config),
                    "period": period,
                    "account_reference_usd": float(starting_capital),
                    "fx_usd_mxn": float(fx_quote.rate) if fx_quote else None,
                    "optimization_trials": len(optimization.trials),
                    "search_mode": search_mode,
                }
                run_id = repository.record_backtest_run(
                    engine_version=batch.engine_version,
                    symbols_json=json.dumps(normalized_symbols),
                    parameters_json=json.dumps(parameters, sort_keys=True),
                    dataset_sha256=batch.dataset_sha256,
                    payload_json=payload,
                    status=(
                        "APPROVED"
                        if batch.aggregate_decision.startswith("APROBADO")
                        else "REJECTED"
                    ),
                )
                status.update(
                    label=f"Backtest #{run_id} terminado y auditado",
                    state="complete",
                    expanded=False,
                )
            st.session_state["latest_backtest_batch"] = batch
            st.session_state["latest_backtest_optimization"] = optimization
            st.toast(f"Backtest #{run_id} registrado", icon=":material/verified:")
        except (QuantMarketDataError, ValueError) as exc:
            st.error(str(exc))

    optimization = st.session_state.get("latest_backtest_optimization")
    if isinstance(optimization, BacktestOptimizationResult):
        _render_optimization_result(optimization)
    batch = st.session_state.get("latest_backtest_batch")
    if isinstance(batch, BacktestBatchResult):
        _render_backtest_result(batch)
    else:
        st.info("Configura y ejecuta la primera validación fuera de muestra.")

    history = repository.list_backtest_runs()
    with st.container(border=True):
        st.subheader("Historial estadístico auditable", anchor=False)
        if history:
            history_frame = pd.DataFrame(
                [
                    {
                        "ID": row["id"],
                        "Fecha": local_datetime(row["created_at"]),
                        "Motor": row["engine_version"],
                        "Estado": row["status"],
                        "Emisoras": ", ".join(json.loads(row["symbols_json"])),
                        "SHA-256": str(row["payload_sha256"])[:16] + "…",
                    }
                    for row in history
                ]
            )
            st.dataframe(
                history_frame,
                hide_index=True,
                key="backtest_history",
                column_config={
                    "Fecha": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm"),
                },
            )
        else:
            st.caption("Aún no existen ejecuciones registradas.")


def audit_page(repository: PortfolioRepository) -> None:
    page_intro(
        "Auditoría y validación",
        "Control autónomo del libro de efectivo, posiciones, SQLite y huellas SHA-256.",
    )
    report = PortfolioAuditor(repository).run()
    if st.button(
        "Ejecutar y registrar auditoría", type="primary", icon=":material/fact_check:"
    ):
        report = PortfolioAuditor(repository).run()
        repository.record_audit_run(
            status=report.status.value, passed=report.passed,
            warnings=report.warnings, errors=report.errors,
            details_json=report.to_json(),
        )
        st.toast("Auditoría registrada", icon=":material/verified:")
    status_text = {
        AuditLevel.PASS: "Integridad confirmada",
        AuditLevel.WARNING: "Integridad con observaciones",
        AuditLevel.ERROR: "Se detectaron inconsistencias",
    }[report.status]
    status_color = {
        AuditLevel.PASS: "green", AuditLevel.WARNING: "orange", AuditLevel.ERROR: "red"
    }[report.status]
    st.badge(status_text, color=status_color)
    with st.container(horizontal=True):
        st.metric("Controles aprobados", report.passed, border=True, icon=":material/check_circle:")
        st.metric("Advertencias", report.warnings, border=True, icon=":material/warning:")
        st.metric("Errores", report.errors, border=True, icon=":material/error:")
    frame = pd.DataFrame([{
        "Área": item.area,
        "Estado": {AuditLevel.PASS: "Aprobado", AuditLevel.WARNING: "Advertencia", AuditLevel.ERROR: "Error"}[item.level],
        "Resultado": item.message, "Referencia": item.reference,
    } for item in report.findings])
    st.dataframe(frame, width="stretch", hide_index=True)
    runs = repository.list_audit_runs()
    with st.expander("Historial de auditorías registradas"):
        if runs:
            history = pd.DataFrame([{
                "ID": row["id"],
                "Fecha": local_datetime(row["created_at"]).strftime("%d/%m/%Y %H:%M"),
                "Estado": row["status"], "Aprobados": row["passed"],
                "Advertencias": row["warnings"], "Errores": row["errors"],
            } for row in runs])
            st.dataframe(history, width="stretch", hide_index=True)
        else:
            st.info("Aún no hay auditorías persistidas.")


def implementation_control_page() -> None:
    page_intro(
        "Control de implementación y complejidad",
        "Inventario local del avance funcional, pruebas y tamaño actual del software.",
    )
    snapshot = inspect_implementation_status(PROJECT_ROOT)
    complete_count = sum(item.status == "complete" for item in snapshot.milestones)
    pending_count = len(snapshot.milestones) - complete_count

    with st.container(border=True):
        st.subheader("Progreso general", anchor=False)
        st.progress(
            snapshot.completion_percent / 100,
            text=f"{snapshot.completion_percent}% del alcance técnico registrado",
        )
        st.caption(
            "El porcentaje pondera hitos funcionales. No representa precisión predictiva ni rendimiento financiero."
        )

    with st.container(horizontal=True):
        st.metric(
            "Última suite aprobada",
            f"{snapshot.verified_tests}/{snapshot.discovered_tests}",
            border=True,
            icon=":material/check_circle:",
            help=f"Ejecución completa registrada el {snapshot.verified_date}.",
        )
        st.metric(
            "Módulos Python activos",
            snapshot.active_modules,
            border=True,
            icon=":material/account_tree:",
        )
        st.metric(
            "Líneas Python útiles",
            f"{snapshot.python_lines:,}",
            border=True,
            icon=":material/code:",
            help="Líneas no vacías de app.py y portfolio_tracker/.",
        )
        st.metric(
            "Hitos pendientes",
            pending_count,
            border=True,
            icon=":material/pending_actions:",
        )

    st.subheader("Mapa de hitos", anchor=False)
    for milestone in snapshot.milestones:
        completed = milestone.status == "complete"
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.badge(
                "Implementado" if completed else "Pendiente",
                icon=":material/check:" if completed else ":material/schedule:",
                color="green" if completed else "orange",
            )
            with st.container(gap=None):
                st.markdown(f"**{milestone.name}**")
                st.caption(milestone.detail)

    st.caption(
        "Las métricas de archivos y pruebas se recalculan al abrir la página. "
        "La cifra de pruebas aprobadas corresponde a la última suite completa registrada."
    )


def settings_page(repository: PortfolioRepository, fx_quote: FxQuote | None) -> None:
    page_intro(
        "Configuración",
        "Fuentes de mercado, calibración manual y estado del almacenamiento local.",
    )
    with st.container(border=True):
        st.subheader("Tipo de cambio USD/MXN", anchor=False)
        if fx_quote:
            st.metric(
                "Tasa activa", f"{fx_quote.rate:,.4f}", border=True,
                icon=":material/currency_exchange:",
                help=f"{fx_quote.provider} · {fx_quote.observed_at.astimezone(LOCAL_TIMEZONE):%d/%m/%Y %H:%M}",
            )
        if st.button("Actualizar fuentes ahora", icon=":material/refresh:"):
            fetch_live_fx.clear()
            fetch_live_prices.clear()
            st.rerun()
        with st.form("manual_fx", border=False):
            manual_rate = st.number_input(
                "Tasa aplicada por GBM", min_value=0.0001,
                value=float(fx_quote.rate) if fx_quote else 18.0,
                step=0.0001, format="%.4f",
            )
            submitted = st.form_submit_button("Guardar calibración manual")
        if submitted:
            repository.add_fx_quote(FxQuote(
                rate=Decimal(str(manual_rate)), observed_at=datetime.now(timezone.utc),
                provider="Captura manual GBM", is_reference=False,
            ))
            st.toast("Tasa manual guardada")
            st.rerun()

    with st.container(border=True):
        st.subheader("Precio manual", anchor=False)
        with st.form("manual_price", border=False):
            symbol = st.text_input("Ticker").upper().strip()
            price = st.number_input("Precio actual USD", min_value=0.0, step=0.01, format="%.4f")
            submitted_price = st.form_submit_button("Guardar precio")
        if submitted_price:
            if not symbol or price <= 0:
                st.error("Captura ticker y precio válidos.")
            else:
                repository.add_price_quote(PriceQuote(
                    symbol=symbol, price_usd=Decimal(str(price)),
                    observed_at=datetime.now(timezone.utc), provider="Captura manual",
                ), is_manual=True)
                st.toast("Precio manual guardado")

    with st.container(border=True):
        st.subheader("Estado local", anchor=False)
        st.write(f"**Migración de esquema:** {repository.database.schema_version()}")
        st.write(f"**Integridad SQLite:** {repository.database.integrity_check()}")
        st.write(f"**Base de datos:** `{DB_PATH}`")
        st.write(f"**OCR:** {GbmOcrExtractor().backend_name}")
        st.caption(
            "La carpeta data/ está excluida de Git. Los registros y comprobantes "
            "permanecen en este equipo y deben respaldarse juntos."
        )
    with st.expander("Contratos para análisis y ML futuro"):
        st.markdown("""
        - Separar contabilidad, señales y decisiones de inversión.
        - Medir margen de seguridad y mostrar supuestos fundamentales.
        - Aplicar límites de pérdida, riesgo/recompensa y concentración.
        - Validar fuera de muestra y versionar modelo, horizonte y fecha de corte.
        - Mostrar scores honestos y, solo con muestra masiva OOS, probabilidades calibradas con Brier.
        """)


repository = get_repository()
fx_quote, fx_error = current_fx(repository)


def show_dashboard() -> None:
    dashboard(repository, fx_quote)


def show_operations() -> None:
    operations_page(repository, fx_quote)


def show_receipts() -> None:
    receipt_page(repository, fx_quote)


def show_cash() -> None:
    cash_page(repository, fx_quote)


def show_market() -> None:
    market_page(repository)


def show_probability_predictor() -> None:
    probability_predictor_page()


def show_audit() -> None:
    audit_page(repository)


def show_implementation_control() -> None:
    implementation_control_page()


def show_settings() -> None:
    settings_page(repository, fx_quote)


pages = [
    st.Page(show_dashboard, title="Resumen", icon=":material/space_dashboard:", default=True),
    st.Page(show_operations, title="Operaciones", icon=":material/swap_horiz:"),
    st.Page(show_receipts, title="Importar comprobante", icon=":material/document_scanner:"),
    st.Page(show_cash, title="Efectivo", icon=":material/account_balance_wallet:"),
    st.Page(show_market, title="Mercado", icon=":material/query_stats:"),
    st.Page(
        show_probability_predictor,
        title="Motor cuantitativo",
        icon=":material/neurology:",
    ),
    st.Page(show_audit, title="Auditoría", icon=":material/fact_check:"),
    st.Page(
        show_implementation_control,
        title="Control de implementación",
        icon=":material/checklist:",
    ),
    st.Page(show_settings, title="Configuración", icon=":material/settings:"),
]

navigation = st.navigation(pages, position="sidebar", expanded=True)
with st.sidebar:
    with st.container(key="sidebar_market_status"):
        st.caption("ESTADO DE MERCADO")
        if fx_quote:
            st.metric("USD/MXN", f"{fx_quote.rate:,.4f}", border=True)
            st.caption(fx_quote.provider)
        else:
            st.warning("Tasa pendiente")
        st.caption("Datos privados · almacenamiento local")

navigation.run()
