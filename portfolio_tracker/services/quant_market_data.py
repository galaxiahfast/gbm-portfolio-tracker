"""Descarga ligera de velas para el predictor técnico de Fases 1, 2 y 3."""

from __future__ import annotations

import re

import pandas as pd

from portfolio_tracker.config import DATA_DIR
from portfolio_tracker.analytics.closed_bars import select_last_closed_bar


SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class QuantMarketDataError(RuntimeError):
    """Error recuperable al consultar o normalizar datos de mercado."""


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise QuantMarketDataError("Captura una emisora válida, por ejemplo SMCI o TSLA.")
    return normalized


def _normalize_frame(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise QuantMarketDataError(f"Yahoo Finance no devolvió velas para {symbol}.")
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        # yfinance puede conservar un nivel de ticker aun para una sola emisora.
        ticker_levels = normalized.columns.get_level_values(-1)
        if symbol in ticker_levels:
            normalized = normalized.xs(symbol, axis=1, level=-1)
        else:
            normalized.columns = normalized.columns.get_level_values(0)
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise QuantMarketDataError(
            "La respuesta de mercado no contiene: " + ", ".join(missing) + "."
        )
    normalized = normalized.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized["Volume"] = normalized["Volume"].fillna(0)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    if normalized.empty:
        raise QuantMarketDataError(f"Las velas recibidas para {symbol} no son utilizables.")
    return normalized


def download_quant_frames(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga 1 mes a 5 minutos y 5 años diarios en solo dos llamadas.

    El import es diferido para que el resto del portafolio pueda iniciar incluso
    si aún no se ha instalado la dependencia opcional.
    """

    normalized_symbol = normalize_symbol(symbol)
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise QuantMarketDataError(
            "Falta yfinance. Ejecuta nuevamente iniciar_app.bat para instalarlo."
        ) from exc

    # Evita que yfinance intente escribir sus bases de caché en AppData u otra
    # ubicación sin permisos. `data/` es persistente y ya está ignorada por Git.
    yfinance_cache = DATA_DIR / "yfinance_cache"
    yfinance_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yfinance_cache))

    common = {
        "tickers": normalized_symbol,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
        "timeout": 12,
        "multi_level_index": False,
    }
    try:
        intraday = yf.download(period="1mo", interval="5m", prepost=False, **common)
        daily = yf.download(period="5y", interval="1d", prepost=False, **common)
    except Exception as exc:
        raise QuantMarketDataError(
            f"No fue posible descargar datos para {normalized_symbol}."
        ) from exc

    return (
        select_last_closed_bar(_normalize_frame(intraday, normalized_symbol), "5m"),
        select_last_closed_bar(_normalize_frame(daily, normalized_symbol), "1d"),
    )


def download_backtest_daily(symbol: str, period: str = "10y") -> pd.DataFrame:
    """Descarga un histórico diario ajustado para backtesting reproducible."""

    normalized_symbol = normalize_symbol(symbol)
    if period not in {"5y", "10y", "max"}:
        raise QuantMarketDataError("El periodo de backtesting no es válido.")
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise QuantMarketDataError(
            "Falta yfinance. Ejecuta nuevamente iniciar_app.bat para instalarlo."
        ) from exc
    yfinance_cache = DATA_DIR / "yfinance_cache"
    yfinance_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yfinance_cache))
    try:
        frame = yf.download(
            tickers=normalized_symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=False,
            timeout=20,
            multi_level_index=False,
        )
    except Exception as exc:
        raise QuantMarketDataError(
            f"No fue posible descargar el histórico de {normalized_symbol}."
        ) from exc
    return _normalize_frame(frame, normalized_symbol)
