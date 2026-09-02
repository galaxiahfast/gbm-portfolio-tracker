"""Contexto fundamental y noticias versionado para el motor cuantitativo.

No intenta predecir titulares. Convierte datos observables en un ajuste simétrico,
limitado y explicable; los vetos solo se activan ante eventos o deterioro severo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from portfolio_tracker.analytics.technical_probability import (
    ProbabilityAnalysis,
    ScoreComponent,
    TechnicalSignal,
    validate_probability_analysis,
)
from portfolio_tracker.config import DATA_DIR
from portfolio_tracker.services.quant_market_data import QuantMarketDataError, normalize_symbol


SNAPSHOT_VERSION = "fundamental-news-v2"
SUPPORTED_PRIMARY_SYMBOLS = ("TSLA", "NVDA", "SMCI", "GME")

POSITIVE_TERMS = {
    "beat", "beats", "record revenue", "raises guidance", "upgrade", "upgraded",
    "strong demand", "profit growth", "partnership", "contract win", "approval",
    "expands", "launches", "outperform", "buyback", "rate cut",
}
NEGATIVE_TERMS = {
    "miss", "misses", "cuts guidance", "downgrade", "downgraded", "lawsuit",
    "investigation", "recall", "weak demand", "delay", "dilution", "offering",
    "accounting concern", "default", "layoffs", "underperform", "rate hike",
}
TOPIC_TERMS = {
    "Inteligencia artificial": ("artificial intelligence", " ai ", "ai chip", "data center"),
    "Semiconductores": ("semiconductor", "chip", "gpu", "foundry"),
    "Tasas de interés": ("interest rate", "fed ", "federal reserve", "rate cut", "rate hike"),
    "Resultados": ("earnings", "quarterly results", "revenue", "guidance"),
}


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    publisher: str
    url: str
    published_at: str
    summary: str
    sentiment: float
    topics: tuple[str, ...]
    recency_weight: float = 0.0
    impact_class: str = "BAJO"


@dataclass(frozen=True, slots=True)
class CorporateEvent:
    kind: str
    event_at: str
    detail: str


@dataclass(frozen=True, slots=True)
class FundamentalNewsSnapshot:
    symbol: str
    observed_at: str
    provider: str
    version: str
    metrics: dict[str, float | str | None]
    events: tuple[CorporateEvent, ...]
    news: tuple[NewsItem, ...]
    fundamental_points: float
    news_points: float
    event_points: float
    total_points: float
    label: str
    risk_veto: bool
    veto_scope: str
    reasons: tuple[str, ...]
    event_risk_level: str = "BAJO"
    risk_window_until: str = ""
    event_risk_reasons: tuple[str, ...] = ()


def snapshot_to_payload(snapshot: FundamentalNewsSnapshot) -> str:
    return json.dumps(
        asdict(snapshot), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def snapshot_from_payload(payload_json: str) -> FundamentalNewsSnapshot:
    payload = json.loads(payload_json)
    return FundamentalNewsSnapshot(
        symbol=str(payload["symbol"]),
        observed_at=str(payload["observed_at"]),
        provider=str(payload["provider"]),
        version=str(payload["version"]),
        metrics=dict(payload.get("metrics", {})),
        events=tuple(CorporateEvent(**item) for item in payload.get("events", [])),
        news=tuple(
            NewsItem(
                **{
                    **item,
                    "topics": tuple(item.get("topics", [])),
                    "recency_weight": float(item.get("recency_weight", 0.0)),
                    "impact_class": str(item.get("impact_class", "BAJO")),
                }
            )
            for item in payload.get("news", [])
        ),
        fundamental_points=float(payload["fundamental_points"]),
        news_points=float(payload["news_points"]),
        event_points=float(payload["event_points"]),
        total_points=float(payload["total_points"]),
        label=str(payload["label"]),
        risk_veto=bool(payload["risk_veto"]),
        veto_scope=str(payload.get("veto_scope", "ALL" if payload["risk_veto"] else "NONE")),
        reasons=tuple(str(item) for item in payload.get("reasons", [])),
        event_risk_level=str(payload.get("event_risk_level", "BAJO")),
        risk_window_until=str(payload.get("risk_window_until", "")),
        event_risk_reasons=tuple(
            str(item) for item in payload.get("event_risk_reasons", [])
        ),
    )


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _statement_value(frame: pd.DataFrame | None, aliases: Sequence[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    normalized = {re.sub(r"[^a-z0-9]", "", str(index).lower()): index for index in frame.index}
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]", "", alias.lower())
        if key not in normalized:
            continue
        values = pd.to_numeric(frame.loc[normalized[key]], errors="coerce").dropna()
        if not values.empty:
            return _finite(values.iloc[0])
    return None


def _first_number(info: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _finite(info.get(key))
        if value is not None:
            return value
    return None


def _parse_datetime(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                return None
            parsed = timestamp.to_pydatetime()
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _calendar_events(calendar: object, observed_at: datetime) -> tuple[CorporateEvent, ...]:
    events: list[CorporateEvent] = []
    if isinstance(calendar, pd.DataFrame):
        raw = calendar.to_dict()
    elif isinstance(calendar, Mapping):
        raw = dict(calendar)
    else:
        raw = {}
    for key, value in raw.items():
        if "earning" not in str(key).lower() and "dividend" not in str(key).lower():
            continue
        candidates = value.values() if isinstance(value, Mapping) else value if isinstance(value, (list, tuple)) else (value,)
        for candidate in candidates:
            event_at = _parse_datetime(candidate)
            if event_at and event_at >= observed_at - timedelta(days=1):
                kind = "Resultados" if "earning" in str(key).lower() else "Dividendo"
                events.append(CorporateEvent(kind, event_at.isoformat(), str(key)))
    return tuple(sorted(events, key=lambda item: item.event_at)[:6])


def _nested(mapping: Mapping[str, object], *path: str) -> object:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _news_items(raw_news: Sequence[Mapping[str, object]] | None, observed_at: datetime) -> tuple[NewsItem, ...]:
    output: list[NewsItem] = []
    for raw in list(raw_news or [])[:30]:
        content = raw.get("content") if isinstance(raw.get("content"), Mapping) else raw
        assert isinstance(content, Mapping)
        title = str(content.get("title") or raw.get("title") or "").strip()
        if not title:
            continue
        summary = str(content.get("summary") or content.get("description") or "").strip()
        publisher = str(
            _nested(content, "provider", "displayName")
            or raw.get("publisher") or "Fuente no indicada"
        )
        url = str(
            _nested(content, "clickThroughUrl", "url")
            or _nested(content, "canonicalUrl", "url")
            or raw.get("link") or ""
        )
        published = _parse_datetime(
            content.get("pubDate") or raw.get("providerPublishTime") or raw.get("published_at")
        ) or observed_at
        if published > observed_at + timedelta(minutes=5):
            continue
        text = f" {title} {summary} ".lower()
        positive = sum(term in text for term in POSITIVE_TERMS)
        negative = sum(term in text for term in NEGATIVE_TERMS)
        sentiment = max(-1.0, min(1.0, (positive - negative) / max(1, positive + negative)))
        topics = tuple(
            topic for topic, terms in TOPIC_TERMS.items() if any(term in text for term in terms)
        )
        age_hours = max(0.0, (observed_at - published).total_seconds() / 3600)
        recency_weight = math.exp(-math.log(2) * age_hours / 72)
        high_impact_terms = (
            "earnings", "guidance", "investigation", "accounting", "offering",
            "federal reserve", "rate hike", "rate cut", "contract", "partnership",
        )
        impact_class = (
            "ALTO"
            if any(term in text for term in high_impact_terms)
            else "MEDIO" if topics or abs(sentiment) >= 0.5 else "BAJO"
        )
        output.append(
            NewsItem(
                title=title[:300], publisher=publisher[:120], url=url[:1000],
                published_at=published.isoformat(), summary=summary[:600],
                sentiment=round(sentiment, 4), topics=topics,
                recency_weight=round(recency_weight, 6),
                impact_class=impact_class,
            )
        )
    unique = {f"{item.published_at}|{item.title}": item for item in output}
    return tuple(sorted(unique.values(), key=lambda item: item.published_at, reverse=True)[:20])


def build_fundamental_snapshot(
    symbol: str,
    *,
    info: Mapping[str, object] | None = None,
    income_statement: pd.DataFrame | None = None,
    balance_sheet: pd.DataFrame | None = None,
    cashflow: pd.DataFrame | None = None,
    calendar: object = None,
    raw_news: Sequence[Mapping[str, object]] | None = None,
    observed_at: datetime | None = None,
    provider: str = "Yahoo Finance / yfinance",
) -> FundamentalNewsSnapshot:
    observed = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    normalized_symbol = normalize_symbol(symbol)
    source = dict(info or {})
    quarterly_revenue = _statement_value(income_statement, ("Total Revenue", "Operating Revenue"))
    quarterly_net_income = _statement_value(income_statement, ("Net Income", "Net Income Common Stockholders"))
    quarterly_debt = _statement_value(balance_sheet, ("Total Debt", "Long Term Debt And Capital Lease Obligation"))
    quarterly_equity = _statement_value(balance_sheet, ("Stockholders Equity", "Total Equity Gross Minority Interest"))
    quarterly_fcf = _statement_value(cashflow, ("Free Cash Flow",))
    quarterly_operating_cash = _statement_value(
        cashflow,
        ("Operating Cash Flow", "Total Cash From Operating Activities"),
    )
    quarterly_capex = _statement_value(
        cashflow,
        ("Capital Expenditure", "Capital Expenditures"),
    )
    quarterly_ebit = _statement_value(income_statement, ("EBIT", "Operating Income"))
    quarterly_interest = _statement_value(
        income_statement,
        ("Interest Expense", "Interest Expense Non Operating"),
    )
    operating_cash = _first_number(source, "operatingCashflow") or quarterly_operating_cash
    total_cash = _first_number(source, "totalCash")
    total_debt = _first_number(source, "totalDebt") or quarterly_debt
    cash_conversion = (
        operating_cash / quarterly_net_income
        if operating_cash is not None and quarterly_net_income not in (None, 0)
        else None
    )
    fcf_value = _first_number(source, "freeCashflow") or quarterly_fcf
    fcf_margin = (
        fcf_value / quarterly_revenue
        if fcf_value is not None and quarterly_revenue not in (None, 0)
        else None
    )
    net_debt = (
        total_debt - total_cash
        if total_debt is not None and total_cash is not None
        else None
    )
    interest_coverage = (
        quarterly_ebit / abs(quarterly_interest)
        if quarterly_ebit is not None and quarterly_interest not in (None, 0)
        else None
    )
    metrics: dict[str, float | str | None] = {
        "sector": str(source.get("sector") or ""),
        "industry": str(source.get("industry") or ""),
        "market_cap": _first_number(source, "marketCap"),
        "trailing_pe": _first_number(source, "trailingPE"),
        "forward_pe": _first_number(source, "forwardPE"),
        "price_to_book": _first_number(source, "priceToBook"),
        "debt_to_equity": _first_number(source, "debtToEquity"),
        "profit_margin": _first_number(source, "profitMargins"),
        "operating_margin": _first_number(source, "operatingMargins"),
        "revenue_growth": _first_number(source, "revenueGrowth"),
        "earnings_growth": _first_number(source, "earningsGrowth"),
        "return_on_equity": _first_number(source, "returnOnEquity"),
        "free_cash_flow": fcf_value,
        "operating_cash_flow": operating_cash,
        "capital_expenditure": quarterly_capex,
        "cash_conversion": cash_conversion,
        "fcf_margin": fcf_margin,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "net_debt": net_debt,
        "interest_coverage": interest_coverage,
        "quarterly_revenue": quarterly_revenue,
        "quarterly_net_income": quarterly_net_income,
        "quarterly_equity": quarterly_equity,
    }

    reasons: list[str] = []
    fundamental = 0.0
    margin = _finite(metrics["profit_margin"])
    growth = _finite(metrics["revenue_growth"])
    earnings_growth = _finite(metrics["earnings_growth"])
    free_cash_flow = _finite(metrics["free_cash_flow"])
    debt_to_equity = _finite(metrics["debt_to_equity"])
    if margin is not None:
        impact = 2.0 if margin >= 0.15 else -3.0 if margin < 0 else 0.5
        fundamental += impact
        reasons.append(f"Margen neto {margin:.1%}: {impact:+.1f} pp.")
    if growth is not None:
        impact = 2.0 if growth >= 0.15 else -2.0 if growth < 0 else 0.0
        fundamental += impact
        reasons.append(f"Crecimiento de ingresos {growth:.1%}: {impact:+.1f} pp.")
    if earnings_growth is not None:
        impact = 2.0 if earnings_growth >= 0.10 else -2.0 if earnings_growth <= -0.10 else 0.0
        fundamental += impact
        reasons.append(f"Crecimiento de beneficios {earnings_growth:.1%}: {impact:+.1f} pp.")
    if free_cash_flow is not None:
        impact = 1.5 if free_cash_flow > 0 else -2.0
        fundamental += impact
        reasons.append(f"Flujo de caja libre {'positivo' if free_cash_flow > 0 else 'negativo'}: {impact:+.1f} pp.")
    if debt_to_equity is not None:
        impact = 1.0 if debt_to_equity < 100 else -2.0 if debt_to_equity > 250 else 0.0
        fundamental += impact
        reasons.append(f"Deuda/capital {debt_to_equity:.1f}: {impact:+.1f} pp.")
    if cash_conversion is not None:
        impact = 2.0 if cash_conversion >= 1.0 else -2.0 if cash_conversion < 0.6 else 0.0
        fundamental += impact
        reasons.append(
            f"Conversión de beneficio a caja {cash_conversion:.2f}x: {impact:+.1f} pp."
        )
    if fcf_margin is not None:
        impact = 1.5 if fcf_margin >= 0.10 else -1.5 if fcf_margin < 0 else 0.0
        fundamental += impact
        reasons.append(f"Margen de flujo libre {fcf_margin:.1%}: {impact:+.1f} pp.")
    if interest_coverage is not None:
        impact = 1.0 if interest_coverage >= 5 else -2.0 if interest_coverage < 2 else 0.0
        fundamental += impact
        reasons.append(f"Cobertura de intereses {interest_coverage:.2f}x: {impact:+.1f} pp.")
    fundamental = max(-8.0, min(8.0, fundamental))

    news = _news_items(raw_news, observed)
    weighted_sentiment = 0.0
    weight_total = 0.0
    negative_relevant = 0
    for item in news:
        published = _parse_datetime(item.published_at) or observed
        age_hours = max(0.0, (observed - published).total_seconds() / 3600)
        weight = item.recency_weight * (1.5 if item.impact_class == "ALTO" else 1.25 if item.topics else 1.0)
        weighted_sentiment += item.sentiment * weight
        weight_total += weight
        if item.sentiment < 0 and item.topics:
            negative_relevant += 1
    news_points = max(-6.0, min(6.0, 4.0 * weighted_sentiment / weight_total)) if weight_total else 0.0
    reasons.append(
        f"Noticias ponderadas por recencia: {news_points:+.1f} pp en {len(news)} titular(es)."
    )

    events = _calendar_events(calendar, observed)
    earnings_close = any(
        item.kind == "Resultados"
        and timedelta(0) <= (_parse_datetime(item.event_at) - observed) <= timedelta(days=5)
        for item in events
        if _parse_datetime(item.event_at)
    )
    event_points = -2.0 if earnings_close else 0.0
    event_risk_reasons: list[str] = []
    risk_window_until = ""
    if earnings_close:
        earnings_dates = [
            _parse_datetime(item.event_at)
            for item in events
            if item.kind == "Resultados" and _parse_datetime(item.event_at)
        ]
        risk_window_until = (min(earnings_dates) + timedelta(days=1)).isoformat()
        event_risk_reasons.append("Resultados en ≤5 días: riesgo de gap no modelable por ATR.")
    recent_high_impact = [
        item for item in news
        if item.impact_class == "ALTO"
        and (_parse_datetime(item.published_at) is not None)
        and observed - _parse_datetime(item.published_at) <= timedelta(hours=24)
    ]
    macro_critical = any(
        "Tasas de interés" in item.topics and item.impact_class == "ALTO"
        for item in recent_high_impact
    )
    if macro_critical:
        event_risk_reasons.append("Titular macro crítico de tasas/Fed dentro de las últimas 24 h.")
        risk_window_until = max(
            risk_window_until,
            (observed + timedelta(hours=24)).isoformat(),
        )
    if recent_high_impact:
        event_risk_reasons.append(
            f"{len(recent_high_impact)} anuncio(s) institucional(es) de impacto alto en 24 h."
        )
    reasons.extend(event_risk_reasons)

    total = round(max(-12.0, min(10.0, fundamental + news_points + event_points)), 2)
    severe_negative = bool(
        (news_points <= -3.0 and negative_relevant >= 2)
        or (fundamental <= -6.0 and news_points < 0)
    )
    severe_positive = fundamental >= 6.0 and news_points >= 3.0
    event_risk_level = "CRÍTICO" if earnings_close or macro_critical else "ALTO" if recent_high_impact else "BAJO"
    high_impact_sentiment = sum(item.sentiment for item in recent_high_impact)
    veto_scope = (
        "ALL" if earnings_close or macro_critical
        else "LONG" if high_impact_sentiment < 0 or severe_negative
        else "SHORT" if high_impact_sentiment > 0 or severe_positive
        else "NONE"
    )
    risk_veto = veto_scope != "NONE"
    label = "Favorable" if total >= 3 else "Adverso" if total <= -3 else "Neutral"
    if not any(value is not None and value != "" for value in metrics.values()) and not news:
        reasons.append("La fuente no entregó métricas ni noticias utilizables; ponderación neutral.")
    return FundamentalNewsSnapshot(
        symbol=normalized_symbol, observed_at=observed.isoformat(), provider=provider,
        version=SNAPSHOT_VERSION, metrics=metrics, events=events, news=news,
        fundamental_points=round(fundamental, 2), news_points=round(news_points, 2),
        event_points=event_points, total_points=total, label=label,
        risk_veto=risk_veto, veto_scope=veto_scope, reasons=tuple(reasons),
        event_risk_level=event_risk_level,
        risk_window_until=risk_window_until,
        event_risk_reasons=tuple(event_risk_reasons),
    )


def download_fundamental_news(symbol: str) -> FundamentalNewsSnapshot:
    normalized_symbol = normalize_symbol(symbol)
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise QuantMarketDataError("Falta yfinance para consultar fundamentales y noticias.") from exc
    cache = DATA_DIR / "yfinance_cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    ticker = yf.Ticker(normalized_symbol)

    def safe_read(getter, default):  # type: ignore[no-untyped-def]
        try:
            value = getter()
            return default if value is None else value
        except Exception:
            return default

    info = safe_read(ticker.get_info, {})
    income = safe_read(lambda: ticker.quarterly_income_stmt, pd.DataFrame())
    balance = safe_read(lambda: ticker.quarterly_balance_sheet, pd.DataFrame())
    cashflow = safe_read(lambda: ticker.quarterly_cashflow, pd.DataFrame())
    calendar = safe_read(lambda: ticker.calendar, {})
    news = safe_read(lambda: ticker.news, [])
    income = income if isinstance(income, pd.DataFrame) else pd.DataFrame()
    balance = balance if isinstance(balance, pd.DataFrame) else pd.DataFrame()
    cashflow = cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame()
    info = info if isinstance(info, Mapping) else {}
    news = news if isinstance(news, (list, tuple)) else []
    if not info and income.empty and balance.empty and cashflow.empty and not news:
        raise QuantMarketDataError(
            f"No fue posible actualizar fundamentales/noticias de {normalized_symbol}."
        )
    return build_fundamental_snapshot(
        normalized_symbol, info=info, income_statement=income,
        balance_sheet=balance, cashflow=cashflow, calendar=calendar,
        raw_news=news,
    )


def apply_fundamental_filter(
    analysis: ProbabilityAnalysis,
    snapshot: FundamentalNewsSnapshot,
    *,
    now: datetime | None = None,
) -> ProbabilityAnalysis:
    if analysis.symbol != snapshot.symbol:
        raise ValueError("El contexto fundamental no corresponde a la emisora analizada.")
    observed = _parse_datetime(snapshot.observed_at)
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_days = max(0.0, (reference - observed).total_seconds() / 86400) if observed else 999.0
    freshness = 1.0 if age_days <= 1 else max(0.25, 1.0 - (age_days - 1) / 8)
    impact = round(snapshot.total_points * freshness, 2)
    long_signal = analysis.signal in (TechnicalSignal.BUY, TechnicalSignal.WATCH_BUY)
    short_signal = analysis.signal in (TechnicalSignal.SELL, TechnicalSignal.WATCH_SELL)
    risk_window_until = _parse_datetime(snapshot.risk_window_until)
    event_window_active = bool(risk_window_until and reference <= risk_window_until)
    veto = bool(
        snapshot.risk_veto
        and (age_days <= 3 or event_window_active)
        and (
            snapshot.veto_scope == "ALL"
            or (snapshot.veto_scope == "LONG" and long_signal)
            or (snapshot.veto_scope == "SHORT" and short_signal)
        )
    )
    probability_up = round(max(15.0, min(85.0, analysis.probability_up + impact)), 1)
    probability_down = round(100.0 - probability_up, 1)
    if analysis.operation_probability and (long_signal or short_signal):
        directional_impact = impact if long_signal else -impact
        operation_probability = round(
            max(0.0, min(100.0, analysis.operation_probability + directional_impact)), 1
        )
        if analysis.risk_veto:
            operation_probability = min(operation_probability, analysis.operation_probability)
    else:
        operation_probability = 0.0
    reasons = tuple(snapshot.reasons) + ((f"Corte reutilizado con antigüedad de {age_days:.1f} días.",) if age_days > 1 else ())
    risk_reasons = analysis.risk_reasons
    scenario = analysis.scenario
    risk_alert = analysis.risk_alert
    if veto:
        operation_probability = min(operation_probability, 35.0)
        risk_reasons = (*risk_reasons, "Contexto fundamental/noticioso activó veto de riesgo.")
        scenario = "RIESGO FUNDAMENTAL/EVENTO: esperar nueva confirmación después del catalizador."
        risk_alert = "RIESGO ALTO · CATALIZADOR FUNDAMENTAL O NOTICIOSO"
    horizon_scale = {
        "1 Hora": 0.00,
        "6 Horas": 0.00,
        "1 Día": 0.15,
        "1 Semana": 0.50,
        "1 Mes": 1.00,
        "6 Meses": 1.00,
    }
    adjusted_horizons = []
    for horizon in analysis.horizon_projections:
        shift = impact * horizon_scale.get(horizon.label, 0.5)
        probability_range = horizon.probability_range
        probability_up_horizon = round(
            max(0.0, min(100.0 - probability_range, horizon.probability_up + shift)), 1
        )
        adjusted_horizons.append(
            replace(
                horizon,
                probability_up=probability_up_horizon,
                probability_down=round(100.0 - probability_range - probability_up_horizon, 1),
            )
        )
    adjusted = replace(
        analysis,
        probability_up=probability_up,
        probability_down=probability_down,
        operation_probability=operation_probability,
        risk_veto=analysis.risk_veto or veto,
        signal_rejected=analysis.signal_rejected or veto,
        execution_plan_conditional=analysis.execution_plan_conditional or veto,
        execution_plan_label=(
            "PLAN CONDICIONAL · esperar que desaparezca el catalizador fundamental/noticioso"
            if veto else analysis.execution_plan_label
        ),
        activation_trigger_met=False if veto else analysis.activation_trigger_met,
        long_entry_blocked=analysis.long_entry_blocked or (veto and long_signal),
        risk_reasons=risk_reasons,
        risk_alert=risk_alert,
        scenario=scenario,
        score_breakdown=(
            *analysis.score_breakdown,
            ScoreComponent(
                "Fundamentales y noticias",
                impact,
                f"{snapshot.label}; fundamental {snapshot.fundamental_points:+.1f}, "
                f"noticias {snapshot.news_points:+.1f}, eventos {snapshot.event_points:+.1f} pp; "
                f"riesgo de evento {snapshot.event_risk_level}.",
            ),
        ),
        horizon_projections=tuple(adjusted_horizons),
        fundamental_score=impact,
        fundamental_label=snapshot.label,
        fundamental_reasons=reasons,
        fundamental_risk_veto=veto,
        fundamental_snapshot_sha256="",
        fundamental_as_of=snapshot.observed_at,
        event_risk_level=snapshot.event_risk_level,
        event_risk_window_until=snapshot.risk_window_until,
        fundamental_news_audit=tuple(
            f"{item.published_at} | {item.publisher} | {item.impact_class} | "
            f"peso {item.recency_weight:.3f} | sentimiento {item.sentiment:+.2f} | {item.title}"
            for item in snapshot.news
        ),
    )
    validate_probability_analysis(adjusted)
    return adjusted
