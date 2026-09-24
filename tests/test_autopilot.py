"""Autopilot tests isolate databases and market providers; no real orders."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import logging
import sqlite3

import pandas as pd
import pytest

from scripts import autopilot_runtime as runtime
from scripts.autopilot_market_cache import MarketCache
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.price_zones import DisplayZone, ZoneSnapshot
from portfolio_tracker.analytics.zone_reach import ReachEstimate
from portfolio_tracker.services.zone_forward import ZonePrediction
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES, fixed_cut_forecasts, record_fixed_directional
from portfolio_tracker.services.model_observations import POLICY, VERSION
from tests.test_zone_forward import market, prediction

UTC_TIME = pd.Timestamp("2026-09-03T15:00:10Z").to_pydatetime()


@pytest.fixture
def repo(tmp_path):
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    repository.ensure_initial_capital()
    repository.ensure_zone_forward_schema()
    return repository


@pytest.mark.parametrize("job,stamp,scheduled,want", [
    ("collect", "2026-09-03T15:00:10Z", True, True),
    ("collect", "2026-12-01T16:00:10Z", True, True),
    ("collect", "2026-12-01T15:00:10Z", True, False),
    ("collect", "2026-09-03T16:00:10Z", True, False),
    ("collect", "2026-09-07T15:00:10Z", False, False),  # Labor Day
    ("collect", "2026-09-05T15:00:10Z", False, False),
    ("collect", "2026-09-03T14:00:10Z", False, True),
    ("collect", "2026-09-03T21:00:10Z", False, False),
    ("resolve", "2026-09-03T21:00:10Z", True, True),
    ("resolve", "2026-12-01T22:00:10Z", True, True),
    ("resolve", "2026-09-03T20:14:59Z", False, False),
    ("resolve", "2026-11-27T18:15:00Z", False, True),  # early close
    ("resolve", "2026-09-07T21:00:00Z", True, False),
    ("catchup", "2026-09-03T13:05:00Z", True, True),
    ("catchup", "2026-12-01T14:05:00Z", True, True),
    ("catchup", "2026-12-01T13:05:00Z", True, False),
    ("catchup", "2026-09-03T21:20:00Z", True, False),
])
def test_ny_dst_holidays_windows(job, stamp, scheduled, want):
    assert runtime.allowed(job, pd.Timestamp(stamp).to_pydatetime(), scheduled) is want


def test_refuse_new_accounting_database(tmp_path):
    target = tmp_path / "wrong.db"
    with pytest.raises(ValueError):
        runtime.open_repository(target)
    assert not target.exists()


def test_refuse_outdated_operational_schema_before_scheduled_work(tmp_path):
    target = tmp_path / "outdated.db"
    database = Database(target)
    database.initialize()
    PortfolioRepository(database).ensure_zone_forward_schema()
    with database.transaction() as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version=12")
    with pytest.raises(ValueError, match="se requiere v13"):
        runtime.open_repository(target)


def test_os_lock_released_after_exception(tmp_path):
    path = tmp_path / "jobs.lock"
    with pytest.raises(RuntimeError):
        with runtime.exclusive_job(path):
            raise RuntimeError("power loss simulation")
    with runtime.exclusive_job(path):
        with pytest.raises(OSError):
            with runtime.exclusive_job(path):
                pass


def test_scheduled_collector_retries_lock_inside_window(caplog):
    moments = iter(pd.Timestamp(value).to_pydatetime() for value in (
        "2026-09-03T15:00:10Z", "2026-09-03T15:00:10Z",
        "2026-09-03T15:00:40Z", "2026-09-03T15:00:40Z",
    ))
    attempts = []
    def attempt():
        attempts.append(1)
        if len(attempts) == 1:
            raise BlockingIOError("lock")
        return 0
    assert runtime.retry_scheduled_collection(
        attempt, logging.getLogger("retry-test"), now_fn=lambda: next(moments),
        sleep_fn=lambda _: None,
    ) == 0
    assert len(attempts) == 2
    assert "ALERTA_CORTE_EN_RIESGO" in caplog.text


def test_scheduled_collector_never_retries_after_window(caplog):
    moments = iter(pd.Timestamp(value).to_pydatetime() for value in (
        "2026-09-03T15:19:40Z", "2026-09-03T15:19:40Z",
    ))
    attempts = []
    assert runtime.retry_scheduled_collection(
        lambda: attempts.append(1) or 1, logging.getLogger("retry-test"),
        now_fn=lambda: next(moments), sleep_fn=lambda _: pytest.fail("late retry"),
    ) == 1
    assert len(attempts) == 1
    assert "ALERTA_CORTE_PERDIDO" in caplog.text


def synthetic_snapshot(now):
    buy = DisplayZone("entry", 99., 99., "test", None, "alcista")
    sell = DisplayZone("tp", 101., 101., "test", None, "alcista")
    estimate = ReachEstimate(71., 21, 50., 90., "test", close_probability=40.)
    return ZoneSnapshot(now, (buy,)*3, (sell,)*3, (estimate,)*6)


def synthetic_analysis(symbol, source="2026-09-03T15:00:00Z"):
    horizons = tuple(SimpleNamespace(
        label=label, probability_up=60., probability_range=30., probability_down=10.,
        range_low=99., range_high=101., engine_name="test",
    ) for label in HORIZON_MINUTES)
    return SimpleNamespace(symbol=symbol, last_price=100.,
                           market_regime="TREND", macro_permission="LONG_ONLY",
                           source_bar_closed_at=pd.Timestamp(source),
                           horizon_projections=horizons,
                           execution_levels=SimpleNamespace(
                               direction="LONG", stop_loss=95., take_profit_1=105.,
                           ), activation_trigger_met=True,
                           execution_plan_conditional=False, risk_veto=False,
                           signal_rejected=False)


def test_boot_audits_missing_session_without_backdating(repo, caplog):
    assert record_fixed_directional(repo, synthetic_analysis("SMCI"), {}, UTC_TIME) == 6
    with repo.database.connect() as connection:
        before = connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0]
    missing = runtime.audit_recent_missing_cuts(
        repo, ["SMCI"], logging.getLogger("boot-audit"),
        now=pd.Timestamp("2026-09-08T13:05:00Z").to_pydatetime(), sessions=2,
    )
    assert missing == (("SMCI", "2026-09-04"),)
    assert "ALERTA_CORTE_PERDIDO SMCI 2026-09-04" in caplog.text
    with repo.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0] == before


def test_collection_six_each_idempotent_and_no_ledger_writes(repo, tmp_path, monkeypatch):
    import portfolio_tracker.services.price_zones as zones
    log = logging.getLogger("test")
    before = repo.cash_balance_usd()
    monkeypatch.setattr(runtime, "fundamental_context", lambda *_: (None, ""))
    monkeypatch.setattr(MarketCache, "frames", lambda *_: (None, None))
    monkeypatch.setattr(runtime, "analyze_headless", lambda r,s,*a: synthetic_analysis(s))
    monkeypatch.setattr(zones, "build_zone_snapshot", lambda _,now: synthetic_snapshot(now))
    code = runtime.collect(repo, ["SMCI","NVDA"], tmp_path, log, now_fn=lambda: UTC_TIME)
    assert code == 0
    assert len(repo.zone_predictions()) == 12
    with repo.database.connect() as connection:
        rows = connection.execute("SELECT * FROM live_model_observations ORDER BY symbol,horizon_minutes").fetchall()
    assert len(rows) == 12
    assert {row["symbol"] for row in rows} == {"SMCI", "NVDA"}
    assert {row["horizon_minutes"] for row in rows} == set(HORIZON_MINUTES.values())
    assert all(row["integrity_version"] == VERSION and row["horizon_policy"] == POLICY for row in rows)
    assert repo.verify_live_model_observations() == (12, ())
    assert runtime.collect(repo, ["SMCI","NVDA"], tmp_path, log, now_fn=lambda: UTC_TIME) == 0
    assert len(repo.zone_predictions()) == 12
    with repo.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0] == 12
    assert repo.cash_balance_usd() == before
    with repo.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_failure_one_symbol_continues_other(repo, tmp_path, monkeypatch):
    import portfolio_tracker.services.price_zones as zones
    def fundamental(r, symbol, log):
        if symbol == "SMCI":
            raise ConnectionError("offline")
        return None, ""
    monkeypatch.setattr(runtime, "fundamental_context", fundamental)
    monkeypatch.setattr(MarketCache, "frames", lambda *_: (None, None))
    monkeypatch.setattr(runtime, "analyze_headless", lambda r,s,*a: synthetic_analysis(s))
    monkeypatch.setattr(zones, "build_zone_snapshot", lambda _,now: synthetic_snapshot(now))
    assert runtime.collect(repo, ["SMCI","NVDA"], tmp_path, logging.getLogger("test"), now_fn=lambda: UTC_TIME) == 1
    assert {r["symbol"] for r in repo.zone_predictions()} == {"NVDA"}
    with repo.database.connect() as connection:
        assert {r["symbol"] for r in connection.execute("SELECT symbol FROM live_model_observations")} == {"NVDA"}


def test_fixed_cut_rejects_open_or_stale_bar_and_closed_market(repo):
    parameters = {"stop_atr_multiple": 2.25}
    for source, now in [
        ("2026-09-03T15:05:00Z", "2026-09-03T15:00:10Z"),
        ("2026-09-03T14:55:00Z", "2026-09-03T15:00:10Z"),
        ("2026-09-03T15:00:00Z", "2026-09-07T15:00:10Z"),
    ]:
        with pytest.raises(ValueError):
            fixed_cut_forecasts(synthetic_analysis("SMCI", source), parameters,
                                pd.Timestamp(now).to_pydatetime())
    with repo.database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM live_model_observations").fetchone()[0] == 0


def test_fixed_cut_est_and_edt_are_same_ny_wall_clock():
    from portfolio_tracker.services.directional_collection import fixed_cut_forecasts
    for source, now in [
        ("2026-09-03T15:00:00Z", "2026-09-03T15:00:10Z"),
        ("2026-12-01T16:00:00Z", "2026-12-01T16:00:10Z"),
    ]:
        rows = fixed_cut_forecasts(synthetic_analysis("SMCI", source), {},
                                   pd.Timestamp(now).to_pydatetime())
        assert len(rows) == 6
        assert all('"scheduled_cut_ny": "11:00"' in row["parameters_json"] for row in rows)


def test_fixed_cut_later_retry_does_not_create_another_cohort(repo):
    from portfolio_tracker.services.directional_collection import record_fixed_directional
    first = synthetic_analysis("SMCI")
    assert record_fixed_directional(repo, first, {}, UTC_TIME) == 6
    later = synthetic_analysis("SMCI", "2026-09-03T15:05:00Z")
    retry_at = pd.Timestamp("2026-09-03T15:05:10Z").to_pydatetime()
    with pytest.raises(ValueError, match="Corte no accionable"):
        record_fixed_directional(repo, later, {}, retry_at)
    assert repo.verify_live_model_observations() == (6, ())


def test_directional_resolver_uses_exact_historical_close_without_zone_rows(repo, monkeypatch):
    from portfolio_tracker.services.directional_collection import record_fixed_directional
    assert record_fixed_directional(repo, synthetic_analysis("NVDA"), {}, UTC_TIME) == 6
    cash_before = repo.cash_balance_usd()
    bars = pd.DataFrame(
        {"Open": [100., 120.], "High": [101., 121.], "Low": [99., 119.],
         "Close": [101., 120.], "Volume": [100., 100.]},
        index=pd.DatetimeIndex(["2026-09-03T15:55:00Z", "2026-09-03T16:00:00Z"]),
    )
    monkeypatch.setattr(MarketCache, "frames", lambda *_: (bars, pd.DataFrame()))
    as_of = pd.Timestamp("2026-09-03T16:10:00Z").to_pydatetime()
    assert runtime.resolve(repo, ["SMCI", "NVDA"], logging.getLogger("test"), now=as_of) == 0
    with repo.database.connect() as connection:
        row = connection.execute("""SELECT * FROM live_model_observations
                                    WHERE symbol='NVDA' AND horizon_minutes=60""").fetchone()
        assert row["outcome_price"] == "120.0"
        assert row["outcome_bar_at"] == "2026-09-03T16:05:00+00:00"
        assert connection.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert repo.verify_live_model_observations() == (6, ())
    assert repo.cash_balance_usd() == cash_before


def test_boot_resolves_friday_not_monday_spot_and_rerun_noop(repo, monkeypatch):
    from portfolio_tracker.services import forward_market
    item = prediction(timestamp_prediction="2026-08-28T15:00:00Z",
                      source_bar_closed_at="2026-08-28T15:00:00Z")
    repo.save_prediction(item, now=item.timestamp_prediction)
    calls = []
    def provider(symbol, day):
        calls.append((symbol, day))
        return market(day)
    monkeypatch.setattr(forward_market, "resolution_frames", provider)
    now = pd.Timestamp("2026-08-31T13:05:00Z").to_pydatetime()
    assert runtime.resolve(repo, ["SMCI","NVDA"], logging.getLogger("test"), catchup=True, now=now) == 0
    assert calls == [("SMCI","2026-08-28")]
    assert repo.zone_predictions()[0]["actual_close_price"] == 100.
    assert runtime.resolve(repo, ["SMCI","NVDA"], logging.getLogger("test"), catchup=True, now=now) == 0
    assert len(calls) == 1


def test_provider_arbitrary_exception_is_logged_and_pending(repo, monkeypatch):
    from portfolio_tracker.services import forward_market
    item = prediction()
    repo.save_prediction(item, now=item.timestamp_prediction)
    monkeypatch.setattr(forward_market, "resolution_frames", lambda *_: (_ for _ in ()).throw(Exception("internet")))
    code = runtime.resolve(repo, ["SMCI"], logging.getLogger("test"), now=pd.Timestamp("2026-08-31T21:00:00Z").to_pydatetime())
    assert code == 1
    assert repo.zone_predictions()[0]["resolved_at"] is None


def test_incremental_cache_does_not_download_month_again(tmp_path):
    calls = []
    def download(symbol, **kwargs):
        calls.append(kwargs)
        if kwargs["interval"] == "1d":
            return pd.DataFrame(dict(Open=100., High=101., Low=99., Close=100., Volume=1000.),
                                index=pd.bdate_range("2021-01-01", "2026-09-02"))
        index = pd.date_range("2026-09-03T13:30:00Z", "2026-09-03T15:05:00Z", freq="5min")
        return pd.DataFrame(dict(Open=100., High=101., Low=99., Close=100., Volume=1000.), index=index)
    cache = MarketCache(tmp_path, download=download)
    first, _ = cache.frames("SMCI", UTC_TIME)
    second, _ = cache.frames("SMCI", pd.Timestamp("2026-09-03T15:05:10Z"))
    intra = [c for c in calls if c["interval"]=="5m"]
    assert intra[0]["period"] == "1mo"
    assert "start" in intra[1] and "period" not in intra[1]
    assert not second.index.has_duplicates
    assert len(second) == len(first) + 1
    assert len([c for c in calls if c["interval"]=="1d"]) == 1
    assert len(list(tmp_path.glob("SMCI/archive/*.json"))) == 3


def test_cache_hash_fails_closed(tmp_path):
    cache = MarketCache(tmp_path)
    runtime.atomic_json(tmp_path/"broken.json", {"payload": {"frame":"bad"}, "sha256":"a"*64})
    with pytest.raises(ValueError):
        cache._read(tmp_path/"broken.json")


def test_headless_pipeline_uses_real_ui_functions(repo, monkeypatch):
    from tests.test_pdf_report import _analysis
    from portfolio_tracker.analytics import technical_probability as technical
    from portfolio_tracker.analytics import fundamental_news
    from portfolio_tracker.services.operational_state import synchronize_position
    from portfolio_tracker.services.price_zones import build_zone_snapshot
    raw = _analysis()
    seen = {}
    def analyze(*args, **kwargs):
        seen.update(kwargs)
        return raw
    monkeypatch.setattr(technical, "analyze_probability", analyze)
    monkeypatch.setattr(fundamental_news, "apply_fundamental_filter", lambda analysis, _: analysis)
    produced = runtime.analyze_headless(repo, "SMCI", None, None, object(), "signed", UTC_TIME, logging.getLogger("test"))
    expected = synchronize_position(repo.database, raw)
    a = build_zone_snapshot(produced, now=UTC_TIME)
    b = build_zone_snapshot(expected, now=UTC_TIME)
    assert a.buys == b.buys and a.sales == b.sales and a.estimates == b.estimates
    assert seen["require_fresh"] is True
    assert seen["atr_stop_multiple"] == 2.25


def test_entrypoints_do_not_import_streamlit_or_start_server():
    import ast
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob("scripts/*collector.py"), root/"scripts/auto_resolver.py",
                 root/"scripts/boot_catchup.py", root/"scripts/autopilot_runtime.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert "app" not in imports and not any((s or "").startswith("streamlit") for s in imports)


def test_streamlit_reruns_do_not_write_directional_observations():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "repository.record_live_model_observation(" not in source


def test_scheduler_has_forward_and_backup_tasks_without_plaintext_credentials():
    text = (Path(__file__).resolve().parents[1]/"scripts/install_autopilot_tasks.ps1").read_text()
    for name in ("Collector", "Resolver", "Catchup"):
        assert f'Name = "GBM_Forward_{name}"' in text
    assert 'Name = "GBM_Backup_Daily"' in text
    assert 'Script = "github_backup.py"' in text and 'Arguments = "--encrypt"' in text
    assert "<StartWhenAvailable>true" in text
    assert "<BootTrigger>" in text and "<LogonTrigger>" in text
    assert 'Get-Credential' in text and '<LogonType>$mode' in text
    assert 'S4U' not in text and 'Set-TimeZone' not in text
    assert "pythonw.exe" in text and "--scheduled --symbols" in text
    assert "[switch]$CollectorOnly" in text

def test_catchup_does_not_fetch_current_unexpired_session(repo, monkeypatch):
    from portfolio_tracker.services import forward_market
    item = prediction(timestamp_prediction="2026-09-03T15:00:00Z",
                      source_bar_closed_at="2026-09-03T15:00:00Z")
    repo.save_prediction(item, now=item.timestamp_prediction)
    monkeypatch.setattr(forward_market, "resolution_frames", lambda *_: pytest.fail("not due"))
    assert runtime.resolve(repo, ["SMCI"], logging.getLogger("test"), catchup=True, now=UTC_TIME) == 0
    assert repo.zone_predictions()[0]["resolved_at"] is None


def test_cached_price_basis_revision_forces_month_reload(tmp_path):
    requests = []
    factor = [1.]
    def download(symbol, **kwargs):
        requests.append(kwargs)
        if kwargs["interval"] == "1d":
            index = pd.bdate_range("2021-01-01", "2026-09-02")
        else:
            index = pd.date_range("2026-09-03T13:30:00Z", "2026-09-03T15:05:00Z", freq="5min")
        return pd.DataFrame(dict(Open=100*factor[0], High=101*factor[0], Low=99*factor[0],
                                 Close=100*factor[0], Volume=1000.), index=index)
    cache = MarketCache(tmp_path, download=download)
    cache.frames("SMCI", UTC_TIME)
    factor[0] = .1
    result, _ = cache.frames("SMCI", pd.Timestamp("2026-09-03T15:05:10Z"))
    assert requests[-1] == {"period": "1mo", "interval": "5m"}
    assert result.Close.eq(10.).all()


def test_incremental_after_long_shutdown_rebootstraps(tmp_path):
    requests = []
    def download(symbol, **kwargs):
        requests.append(kwargs)
        if kwargs["interval"] == "1d":
            index = pd.bdate_range("2021-01-01", "2026-09-02")
        else:
            index = pd.date_range("2026-09-03T13:30:00Z", "2026-09-03T15:00:00Z", freq="5min")
        return pd.DataFrame(dict(Open=100., High=101., Low=99., Close=100., Volume=1000.), index=index)
    cache = MarketCache(tmp_path, download=download)
    old = pd.DataFrame(dict(Open=100., High=101., Low=99., Close=100., Volume=1000.),
                       index=pd.date_range("2026-01-02T14:30Z", periods=10, freq="5min"))
    cache._write(tmp_path/"SMCI/5m.json", old, pd.Timestamp("2026-01-02T16:00Z"), {})
    cache.frames("SMCI", UTC_TIME)
    assert requests[0] == {"interval": "5m", "period": "1mo"}
