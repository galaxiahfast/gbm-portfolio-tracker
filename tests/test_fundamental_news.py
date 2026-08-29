from dataclasses import replace
from datetime import datetime, timedelta, timezone

from portfolio_tracker.analytics.fundamental_news import (
    SNAPSHOT_VERSION,
    apply_fundamental_filter,
    build_fundamental_snapshot,
    snapshot_from_payload,
    snapshot_to_payload,
)
from portfolio_tracker.analytics.technical_probability import TechnicalSignal
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from test_pdf_report import _analysis


OBSERVED = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)


def _positive_snapshot():
    return build_fundamental_snapshot(
        "SMCI",
        observed_at=OBSERVED,
        info={
            "profitMargins": 0.20,
            "revenueGrowth": 0.25,
            "earningsGrowth": 0.18,
            "freeCashflow": 1_000_000,
            "debtToEquity": 50,
            "sector": "Technology",
        },
        raw_news=[
            {
                "title": "AI chip demand beats forecasts and company raises guidance",
                "publisher": "Example Wire",
                "providerPublishTime": int(OBSERVED.timestamp()),
                "link": "https://example.test/positive",
            }
        ],
    )


def _negative_snapshot(*, earnings_event: bool = True):
    return build_fundamental_snapshot(
        "SMCI",
        observed_at=OBSERVED,
        info={
            "profitMargins": -0.08,
            "revenueGrowth": -0.20,
            "earningsGrowth": -0.25,
            "freeCashflow": -500_000,
            "debtToEquity": 320,
        },
        calendar=(
            {"Earnings Date": [OBSERVED + timedelta(days=2)]}
            if earnings_event else {}
        ),
        raw_news=[
            {
                "title": "AI chip accounting concern triggers investigation",
                "providerPublishTime": int(OBSERVED.timestamp()),
            },
            {
                "title": "Semiconductor weak demand and lawsuit pressure guidance",
                "providerPublishTime": int((OBSERVED - timedelta(hours=4)).timestamp()),
            },
        ],
    )


def test_fundamental_scoring_is_symmetric_and_event_veto_is_objective() -> None:
    positive = _positive_snapshot()
    negative = _negative_snapshot()

    assert positive.total_points > 0
    assert positive.risk_veto is True
    assert positive.veto_scope == "SHORT"
    assert negative.total_points < 0
    assert negative.risk_veto is True
    assert negative.veto_scope == "ALL"
    assert any(item.kind == "Resultados" for item in negative.events)
    assert all(item.topics for item in negative.news)


def test_snapshot_roundtrip_is_canonical_and_lossless() -> None:
    snapshot = _positive_snapshot()
    payload = snapshot_to_payload(snapshot)
    restored = snapshot_from_payload(payload)

    assert snapshot_to_payload(restored) == payload
    assert restored.version == SNAPSHOT_VERSION
    assert restored.news[0].url == "https://example.test/positive"


def test_negative_context_vetoes_setup_and_never_promotes_existing_risk() -> None:
    analysis = _analysis()
    adjusted = apply_fundamental_filter(
        analysis,
        _negative_snapshot(),
        now=OBSERVED,
    )

    assert adjusted.fundamental_score < 0
    assert adjusted.fundamental_risk_veto is True
    assert adjusted.risk_veto is True
    assert adjusted.signal_rejected is True
    assert adjusted.operation_probability <= 35
    assert adjusted.execution_plan_conditional is True
    assert adjusted.activation_trigger_met is False
    assert adjusted.probability_up <= analysis.probability_up
    assert any(item.name == "Fundamentales y noticias" for item in adjusted.score_breakdown)


def test_directional_veto_blocks_long_but_can_validate_tactical_short() -> None:
    base = _analysis()
    snapshot = _negative_snapshot(earnings_event=False)
    long_case = replace(
        base,
        signal=TechnicalSignal.BUY,
        operation_probability=60.0,
        risk_veto=False,
        signal_rejected=False,
        risk_reasons=(),
    )
    short_case = replace(
        base,
        signal=TechnicalSignal.SELL,
        operation_probability=60.0,
        risk_veto=False,
        signal_rejected=False,
        risk_reasons=(),
    )

    long_adjusted = apply_fundamental_filter(long_case, snapshot, now=OBSERVED)
    short_adjusted = apply_fundamental_filter(short_case, snapshot, now=OBSERVED)

    assert snapshot.veto_scope == "LONG"
    assert long_adjusted.fundamental_risk_veto is True
    assert long_adjusted.operation_probability <= 35
    assert short_adjusted.fundamental_risk_veto is False
    assert short_adjusted.operation_probability > short_case.operation_probability


def test_fundamental_snapshot_persistence_detects_tampering(tmp_path) -> None:
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    snapshot = _positive_snapshot()
    payload = snapshot_to_payload(snapshot)
    snapshot_id, digest = repository.record_fundamental_news_snapshot(
        symbol=snapshot.symbol,
        observed_at=OBSERVED,
        provider=snapshot.provider,
        engine_version=snapshot.version,
        payload_json=payload,
    )

    latest = repository.latest_fundamental_news_snapshot("SMCI")
    valid, invalid = repository.verify_fundamental_news_snapshots()
    assert latest is not None and latest["payload_sha256"] == digest
    assert valid == 1 and invalid == ()

    with repository.database.transaction() as connection:
        connection.execute(
            "UPDATE fundamental_news_snapshots SET payload_json = '{}' WHERE id = ?",
            (snapshot_id,),
        )
    valid, invalid = repository.verify_fundamental_news_snapshots()
    assert valid == 0 and invalid == (snapshot_id,)
    assert repository.latest_fundamental_news_snapshot("SMCI") is None
