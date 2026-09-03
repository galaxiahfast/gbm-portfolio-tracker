"""Proveedores reemplazables de divisas y precios con fallos controlados."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from urllib.parse import quote

import requests

from ..models import FxQuote, PriceQuote


class MarketDataError(RuntimeError):
    pass


def download_daily_history(symbol="SMCI", **kwargs):
    """Diarios cacheados desde 2024 para contexto auditable del forward."""
    from .forward_market import download_daily_history as download
    return download(symbol, **kwargs)


class FxProvider(ABC):
    @abstractmethod
    def usd_mxn(self) -> FxQuote:
        raise NotImplementedError


class QuoteProvider(ABC):
    @abstractmethod
    def quote_usd(self, symbol: str) -> PriceQuote:
        raise NotImplementedError


def _utc_from_timestamp(value: int | float | None) -> datetime:
    if value:
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return datetime.now(timezone.utc)


class YahooChartProvider(FxProvider, QuoteProvider):
    """Cotizacion intradia sin SDK pesado; se puede sustituir por otro API."""

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, timeout: float = 7.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "GBM-Portfolio-Tracker/1.0 (+local personal app)"}
        )

    def _metadata(self, symbol: str) -> dict[str, Any]:
        url = f"{self.BASE_URL}/{quote(symbol, safe='')}"
        try:
            response = self.session.get(
                url,
                params={"interval": "1m", "range": "1d"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()["chart"]["result"]
            if not result:
                raise KeyError("result")
            return result[0]["meta"]
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"No fue posible cotizar {symbol}.") from exc

    def usd_mxn(self) -> FxQuote:
        metadata = self._metadata("MXN=X")
        rate = Decimal(str(metadata.get("regularMarketPrice", "0")))
        if rate <= 0:
            raise MarketDataError("El proveedor devolvio un tipo de cambio invalido.")
        return FxQuote(
            rate=rate,
            observed_at=_utc_from_timestamp(metadata.get("regularMarketTime")),
            provider="Yahoo Finance intradia",
            is_reference=True,
        )

    def quote_usd(self, symbol: str) -> PriceQuote:
        normalized = symbol.strip().upper()
        metadata = self._metadata(normalized)
        price = Decimal(str(metadata.get("regularMarketPrice", "0")))
        currency = str(metadata.get("currency", "USD")).upper()
        if price <= 0:
            raise MarketDataError(f"Precio invalido para {normalized}.")
        if currency != "USD":
            raise MarketDataError(
                f"{normalized} cotiza en {currency}; captura el precio USD manualmente."
            )
        return PriceQuote(
            symbol=normalized,
            price_usd=price,
            observed_at=_utc_from_timestamp(metadata.get("regularMarketTime")),
            provider="Yahoo Finance intradia",
            currency=currency,
        )


class FrankfurterFxProvider(FxProvider):
    """Respaldo abierto de tasas de referencia publicadas por bancos centrales."""

    URL = "https://api.frankfurter.dev/v2/rate/USD/MXN"

    def __init__(self, timeout: float = 7.0) -> None:
        self.timeout = timeout

    def usd_mxn(self) -> FxQuote:
        try:
            response = requests.get(self.URL, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            rate = Decimal(str(payload["rate"]))
            observed_at = datetime.fromisoformat(payload["date"]).replace(
                tzinfo=timezone.utc
            )
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            raise MarketDataError("No fue posible consultar Frankfurter.") from exc
        return FxQuote(
            rate=rate,
            observed_at=observed_at,
            provider="Frankfurter (referencia diaria)",
            is_reference=True,
        )


class CompositeFxProvider(FxProvider):
    def __init__(self) -> None:
        self.providers: tuple[FxProvider, ...] = (
            YahooChartProvider(),
            FrankfurterFxProvider(),
        )

    def usd_mxn(self) -> FxQuote:
        failures: list[str] = []
        for provider in self.providers:
            try:
                return provider.usd_mxn()
            except MarketDataError as exc:
                failures.append(str(exc))
        raise MarketDataError(" ".join(failures))
