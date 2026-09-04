"""Cross-asset contracts; synthetic evidence stays in temporary databases."""
from dataclasses import replace
from io import BytesIO
import json
import logging
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from portfolio_tracker.analytics.closed_bars import _calendar
from portfolio_tracker.analytics.cross_correlation import (
    apply_cross_context, build_cross_context, calculate_rolling_correlation,
    calculate_price_ratio, detect_divergence, get_relative_strength, unavailable,
)
from portfolio_tracker.services.cross_asset import enrich_cross_asset

NOW = pd.Timestamp("2026-09-03T15:30:00Z")


def ohlcv(index, close):
    close = np.asarray(close)
    return pd.DataFrame(dict(Open=close, High=close*1.001, Low=close*.999,
                             Close=close, Volume=1000.), index=index)


@pytest.fixture
def frames():
    days = _calendar(2025, 2027).sessions_in_range("2026-01-02", "2026-09-02")
    returns = .003 * np.sin(np.arange(len(days)) * .71)
    a = ohlcv(days, 40*np.cumprod(1+returns))
    b = ohlcv(days, 180*np.cumprod(1+returns*1.1))
    times = pd.date_range("2026-09-03T13:30:00Z", periods=24, freq="5min")
    intra_a = ohlcv(times, 40 + .01*np.sin(np.arange(24)))
    intra_b = ohlcv(times, 180 + .01*np.sin(np.arange(24)))
    intra_b.loc[times[-1], ["Open", "High", "Low", "Close", "Volume"]] = [180, 182.1, 179.9, 182, 1500]
    return a, b, intra_a, intra_b


@pytest.fixture(scope="module")
def analysis():
    from tests.test_pdf_report import _analysis
    return replace(_analysis(), risk_veto=False, long_entry_blocked=False,
                   position_state="FLAT", probability_up=60., probability_down=40.)


def context(frames, **kwargs):
    return build_cross_context("SMCI", *frames, as_of=kwargs.get("as_of", NOW))


def test_returns_not_price_levels_and_ratio(frames):
    a, b, *_ = frames
    histories = {"SMCI": a, "NVDA": b}
    assert calculate_rolling_correlation("SMCI", "NVDA", histories=histories) == pytest.approx(1.)
    r = calculate_price_ratio(histories=histories)
    expected = a.Close / b.Close
    assert r["value"] == pytest.approx(expected.iloc[-1])
    assert r["mean50"] == pytest.approx(expected.tail(50).mean())
    assert set(get_relative_strength(histories=histories)) == {"SMCI", "NVDA"}


@pytest.mark.parametrize("fault", ["constant", "missing", "short", "nan"])
def test_correlation_has_no_silent_fill_or_fake_zero(frames, fault):
    a, b, *_ = frames
    if fault == "constant":
        b.loc[:, "Close"] = 10.
    elif fault == "missing":
        b = b.drop(b.index[-4])
    elif fault == "short":
        a = a.tail(10)
    else:
        a.iloc[-3, a.columns.get_loc("Close")] = np.nan
    assert calculate_rolling_correlation("SMCI", "NVDA", histories={"SMCI": a, "NVDA": b}) is None


def test_leader_breakout_and_signed_inputs(frames):
    result = context(frames)
    assert result["correlation"] > .7
    assert result["proposed_impact"] == 5
    assert "probable seguimiento de SMCI" in " ".join(result["alerts"])
    assert result["intraday_as_of"] == NOW.isoformat()
    assert len(result["input_sha256"]) == 64
    json.dumps(result, allow_nan=False)


def test_bearish_mirror_and_reverse_symbol(frames):
    a, b, ia, ib = frames
    ib.iloc[-1] = [180, 180.1, 177.9, 178, 1500]
    result = context(frames)
    assert result["proposed_impact"] == -5
    reverse = build_cross_context("NVDA", b, a, ib, ia, as_of=NOW)
    assert reverse["correlation"] == pytest.approx(result["correlation"])
    assert reverse["ratio"] == result["ratio"]  # Always SMCI/NVDA, not inverted.


def test_no_volume_confirmation_no_breakout_bonus(frames):
    frames[-1].iloc[-1, frames[-1].columns.get_loc("Volume")] = 1000
    result = context(frames)
    assert not result["intraday"]["NVDA"]["up"]
    assert abs(result["proposed_impact"]) < 5


def test_collector_at_11_uses_prior_session_baseline_not_overnight_momentum(frames):
    a, b, *_ = frames
    times = pd.date_range("2026-09-02T19:45:00Z", periods=3, freq="5min").append(
        pd.date_range("2026-09-03T13:30:00Z", periods=18, freq="5min"))
    ia, ib = ohlcv(times, np.full(21, 40.)), ohlcv(times, np.full(21, 180.))
    ib.iloc[-1] = [180, 182.1, 179.9, 182, 1500]
    result = context((a, b, ia, ib), as_of="2026-09-03T15:00:00Z")
    assert result["proposed_impact"] == 5
    assert result["intraday_as_of"] == "2026-09-03T15:00:00+00:00"


@pytest.mark.parametrize("fault", ["stale", "missing", "forming", "future"])
def test_intraday_unavailable_never_uses_an_older_cut(frames, fault):
    a, b, ia, ib = frames
    if fault == "stale":
        ib = ib.iloc[:-1]
    elif fault == "missing":
        ib = ib.drop(ib.index[-3])
    elif fault == "forming":
        return_result = context(frames, as_of=NOW-pd.Timedelta(minutes=2))
        assert return_result["proposed_impact"] != 5
        return
    else:
        ib.index = ib.index + pd.Timedelta(days=1)
    result = context((a, b, ia, ib))
    assert result["proposed_impact"] == 0
    assert result["intraday_as_of"] is None


def test_future_daily_bar_excluded_and_stale_daily_rejected(frames):
    a, b, ia, ib = frames
    future = b.iloc[-1:].copy()
    future.index = pd.DatetimeIndex(["2026-09-03"])
    future *= 100
    result = context((a, pd.concat([b, future]), ia, ib))
    assert result["daily_as_of"] == "2026-09-02"
    with pytest.raises(ValueError, match="desactualizados"):
        context((a, b.iloc[:-1], ia, ib))


def test_divergence_extremes_and_opposing_momentum(frames):
    *_, ia, ib = frames
    ia.iloc[-1] = [40, 40.0, 38.9, 39, 1500]
    d = detect_divergence(histories={"SMCI": ia, "NVDA": ib})
    assert any("máximo" in s for s in d["alerts"])
    assert any("mínimo" in s for s in d["alerts"])
    assert any("posible frenazo" in s for s in d["alerts"])
    daily = detect_divergence(histories={"SMCI": ia, "NVDA": ib}, timeframe="1d")
    assert any("6 sesiones" in s for s in daily["alerts"])
    assert all("30 min" not in s for s in daily["alerts"])


def test_low_or_negative_correlation_does_not_reward_breakout(frames):
    a, b, ia, ib = frames
    change = a.Close.pct_change(fill_method=None).fillna(0)
    b.loc[:, "Close"] = 180 * (1-change).cumprod()
    b.loc[:, "Open"] = b.Close
    b.loc[:, "High"] = b.Close*1.001
    b.loc[:, "Low"] = b.Close*.999
    result = context(frames)
    assert result["correlation"] < -.7
    assert result["proposed_impact"] == 0


def test_apply_cross_context_defensively_rejects_low_correlation(analysis):
    forged_context = {
        "status": "available",
        "correlation": 0.20,
        "proposed_impact": 5.0,
        "detail": "Contexto no confiable inyectado para comprobar la defensa.",
    }
    result = apply_cross_context(analysis, forged_context)
    assert result.probability_up == analysis.probability_up
    assert result.cross_asset_context["applied_impact"] == 0.0


def test_touch_close_estimates_and_price_levels_are_unchanged(analysis, frames):
    from portfolio_tracker.services.price_zones import build_zone_snapshot
    before = build_zone_snapshot(analysis, now=NOW)
    after = build_zone_snapshot(apply_cross_context(analysis, context(frames)), now=NOW)
    assert before.estimates == after.estimates
    assert [(z.low, z.high) for z in (*before.buys, *before.sales)] == [
        (z.low, z.high) for z in (*after.buys, *after.sales)]


def test_score_is_bounded_idempotent_and_never_changes_execution(analysis, frames):
    cross = context(frames)
    result = apply_cross_context(analysis, cross)
    assert result.probability_up == 65 and result.probability_down == 35
    assert apply_cross_context(result, cross).probability_up == 65
    assert len([c for c in result.score_breakdown if c.name == "Correlación cross-asset"]) == 1
    for attr in ("execution_levels", "buy_levels", "sell_levels", "horizon_projections", "daily_projection",
                 "activation_trigger_met", "operation_probability", "exposure_factor", "risk_veto"):
        if hasattr(analysis, attr):
            assert getattr(result, attr) == getattr(analysis, attr)


@pytest.mark.parametrize("change", [dict(risk_veto=True), dict(long_entry_blocked=True), dict(position_state="LONG_ACTIVE")])
def test_no_bonus_when_blocked_or_active(analysis, frames, change):
    result = apply_cross_context(replace(analysis, **change), context(frames))
    assert result.cross_asset_context["applied_impact"] == 0


def test_existing_direction_cannot_flip_and_failure_removes_old_bonus(analysis, frames):
    base = replace(analysis, probability_up=48., probability_down=52.)
    assert apply_cross_context(base, context(frames)).probability_up < 50
    shifted = apply_cross_context(analysis, context(frames))
    result = apply_cross_context(shifted, unavailable("SMCI", "sin red"))
    assert result.probability_up == analysis.probability_up


def test_download_failure_and_timeout_do_not_crash(analysis, frames, monkeypatch):
    from portfolio_tracker.services import cross_asset as service
    def fail(_):
        raise OSError("offline")
    a, _, ia, _ = frames
    result = enrich_cross_asset(analysis, ia, a, now=NOW, peer_loader=fail)
    assert result.cross_asset_context["status"] == "unavailable"
    assert result.probability_up == analysis.probability_up
    monkeypatch.setattr(service, "prefetch_cross_asset", lambda _: Future())
    result = enrich_cross_asset(analysis, ia, a, now=NOW, timeout=0)
    assert "pendiente" in result.cross_asset_context["detail"]


def test_cache_single_download_per_peer(monkeypatch):
    from portfolio_tracker.services import cross_asset as service
    monkeypatch.setattr(service, "_CACHE", {})
    calls = []
    monkeypatch.setattr(service, "_download_peer", lambda symbol: calls.append(symbol) or (None, None))
    first = service.prefetch_cross_asset("SMCI")
    first.result(timeout=2)
    assert service.prefetch_cross_asset("SMCI") is first
    assert calls == ["NVDA"]
    assert service.prefetch_cross_asset("TSLA") is None


def test_signed_six_zone_context_and_tamper_detection(tmp_path, analysis, frames):
    from portfolio_tracker.db import Database
    from portfolio_tracker.repository import PortfolioRepository
    from portfolio_tracker.services.zone_forward import log_snapshot
    from portfolio_tracker.services.price_zones import DisplayZone, ZoneSnapshot
    from portfolio_tracker.analytics.zone_reach import ReachEstimate
    repo = PortfolioRepository(Database(tmp_path / "only-analytics.db"))
    repo.ensure_zone_forward_schema()
    c = context(frames)
    zone = DisplayZone("test", 39., 39., "test", 60., "alcista")
    est = ReachEstimate(probability=71., samples=21, lower=50., upper=90., status="preliminar", close_probability=40.)
    snap = ZoneSnapshot(NOW, (zone,)*3, (replace(zone, low=41., high=41.),)*3, (est,)*6)
    source = SimpleNamespace(symbol="SMCI", last_price=40., source_bar_closed_at=NOW, cross_asset_context=c)
    assert log_snapshot(repo, source, snap, now=NOW)["saved"] == 6
    rows = repo.zone_predictions()
    assert all(json.loads(r["context_json"])["cross_asset"] == c and r["integrity_ok"] for r in rows)
    with repo.database.transaction() as conn:
        # Test-only bypass of immutability; production triggers remain unchanged.
        triggers = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='zone_prediction_log'").fetchall()
        for row in triggers:
            conn.execute('DROP TRIGGER "' + row[0] + '"')
        conn.execute("UPDATE zone_prediction_log SET context_json='{}'")
    assert all(not r["integrity_ok"] for r in repo.zone_predictions())


def test_global_validation_keeps_versions_events_and_invalid_first(tmp_path):
    from portfolio_tracker.db import Database
    from portfolio_tracker.repository import PortfolioRepository
    from portfolio_tracker.services.cross_validation import global_validation_summary
    from tests.test_zone_forward import prediction, market, emit
    repo = PortfolioRepository(Database(tmp_path / "evidence.db"))
    repo.ensure_zone_forward_schema()
    for symbol in ("SMCI", "NVDA"):
        emit(repo, prediction(symbol=symbol))
    emit(repo, prediction(symbol="NVDA", model_version_hash="b"*64))
    repo.resolve_predictions(lambda *_: market(), now="2026-08-31T20:15:00Z")
    rows = repo.zone_predictions()
    summary = global_validation_summary(rows)
    global_touch = next(s for s in summary if s["Emisora"] == "GLOBAL" and s["Evento"] == "Toque" and s["Versión"] == "a"*64)
    assert global_touch["Evaluadas"] == 2 and global_touch["Sesiones"] == 1
    rows[0]["integrity_ok"] = False
    assert sum(s["Evaluadas"] for s in global_validation_summary(rows) if s["Emisora"] == "GLOBAL" and s["Evento"] == "Toque") == 2


@pytest.mark.parametrize("scope", ["executive", "technical", "combined", "master"])
def test_all_four_pdf_scopes_include_identical_cross_context(analysis, frames, scope):
    from portfolio_tracker.services.pdf_report import (build_executive_report, build_technical_report,
                                                       build_probability_report, build_master_report)
    build = dict(executive=build_executive_report, technical=build_technical_report,
                 combined=build_probability_report, master=build_master_report)[scope]
    result = apply_cross_context(analysis, context(frames))
    pdf = build(result, {}) if scope == "master" else build(result)
    text = "\n".join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert "Relación con NVDA" in text and "Correlación con NVDA" in text
    assert "Ratio SMCI/NVDA" in text and "probable seguimiento de SMCI" in text
    assert result.cross_asset_context["input_sha256"] in text


def test_ui_context_available_and_failure(analysis, frames):
    from streamlit.testing.v1 import AppTest
    code = """
from portfolio_tracker.ui.cross_asset import render_cross_asset
from types import SimpleNamespace
import streamlit as st
render_cross_asset(SimpleNamespace(cross_asset_context=st.session_state['cross']))
"""
    app = AppTest.from_string(code)
    app.session_state["cross"] = context(frames)
    app.run(timeout=20)
    assert not app.exception and len(app.table) == 1
    app.session_state["cross"] = unavailable("SMCI", "sin red")
    app.run(timeout=20)
    assert not app.exception and any("Correlación no disponible" in c.value for c in app.caption)
