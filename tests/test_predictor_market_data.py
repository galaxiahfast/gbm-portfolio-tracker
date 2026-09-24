"""The predictor must stay visible without treating stale bars as a trade."""
from dataclasses import dataclass

import pandas as pd

from portfolio_tracker.services.predictor_market_data import (
    complete_daily_from_intraday,
    informational_decision,
    is_regular_nyse_session,
    load_predictor_market_frames,
)


def _session_bars(day="2026-09-22", count=78):
    index = pd.date_range(f"{day} 09:30", periods=count, freq="5min", tz="America/New_York")
    return pd.DataFrame({
        "Open": [40.0] * count,
        "High": [40.5] * count,
        "Low": [39.5] * count,
        "Close": [40.1] * count,
        "Volume": [100] * count,
    }, index=index)


def _old_daily():
    return pd.DataFrame({
        "Open": [39.0], "High": [40.0], "Low": [38.0],
        "Close": [39.5], "Volume": [1000],
    }, index=pd.DatetimeIndex(["2026-09-21"]))


def test_missing_daily_is_reconstructed_only_from_a_complete_closed_session():
    daily, recovered = complete_daily_from_intraday(
        _session_bars(), _old_daily(), now="2026-09-23T15:00:00Z"
    )
    assert recovered == ("2026-09-22",)
    assert daily.loc["2026-09-22", "Close"] == 40.1
    assert daily.loc["2026-09-22", "Volume"] == 7800

    incomplete, recovered = complete_daily_from_intraday(
        _session_bars(count=77), _old_daily(), now="2026-09-23T15:00:00Z"
    )
    assert not recovered
    assert incomplete.index[-1] == pd.Timestamp("2026-09-21")


def test_in_progress_session_is_never_reconstructed():
    daily, recovered = complete_daily_from_intraday(
        _session_bars(day="2026-09-23", count=18),
        _old_daily(), now="2026-09-23T15:00:00Z",
    )
    assert recovered == ()
    assert daily.index[-1] == pd.Timestamp("2026-09-21")


def test_failed_download_uses_verified_cache_and_reports_its_source():
    def unavailable(_symbol):
        raise RuntimeError("sin internet")

    frames = load_predictor_market_frames(
        "SMCI", now=pd.Timestamp("2026-09-23T15:00:00Z"),
        downloader=unavailable,
        cache_loader=lambda _symbol: (_session_bars(), _old_daily()),
    )
    assert frames.source == "respaldo local SHA-256"
    assert frames.reconstructed_sessions == ("2026-09-22",)
    assert "sin internet" in frames.source_warning


def test_exchange_schedule_handles_holidays_and_early_close():
    assert is_regular_nyse_session("2026-09-23T15:00:00Z")
    assert not is_regular_nyse_session("2026-11-26T15:00:00Z")
    assert is_regular_nyse_session("2026-11-27T17:00:00Z")
    assert not is_regular_nyse_session("2026-11-27T19:00:00Z")


@dataclass(frozen=True)
class _Decision:
    action: str = "COMPRAR"
    position_size: int = 10
    monetary_risk: float = 20.0
    expected_value_total: float = 40.0
    trigger_met: bool = True
    risk_veto: bool = False
    reasons: tuple[str, ...] = ()
    explanation: str = "Entrada autorizada."


def test_stale_display_cannot_recommend_an_immediate_trade():
    result = informational_decision(_Decision(), "Datos retrasados.")
    assert result.action == "ESPERAR"
    assert result.position_size == 0
    assert result.monetary_risk == 0.0
    assert result.expected_value_total is None
    assert result.risk_veto
    assert "Datos retrasados" in result.explanation


def test_stale_display_keeps_persistent_exit_alert_visible():
    result = informational_decision(
        _Decision(action="CONFIRMAR_SALIDA"), "Mercado cerrado."
    )
    assert result.action == "CONFIRMAR_SALIDA"
    assert result.position_size == 0
    assert "fill real" in result.explanation
