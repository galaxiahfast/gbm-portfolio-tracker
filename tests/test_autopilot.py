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


def test_os_lock_released_after_exception(tmp_path):
    path = tmp_path / "jobs.lock"
    with pytest.raises(RuntimeError):
        with runtime.exclusive_job(path):
            raise RuntimeError("power loss simulation")
    with runtime.exclusive_job(path):
        with pytest.raises(OSError):
            with runtime.exclusive_job(path):
                pass


def synthetic_snapshot(now):
    buy = DisplayZone("entry", 99., 99., "test", None, "alcista")
    sell = DisplayZone("tp", 101., 101., "test", None, "alcista")
    estimate = ReachEstimate(71., 21, 50., 90., "test", close_probability=40.)
    return ZoneSnapshot(now, (buy,)*3, (sell,)*3, (estimate,)*6)


def test_collection_six_each_idempotent_and_no_ledger_writes(repo, tmp_path, monkeypatch):
    import portfolio_tracker.services.price_zones as zones
    log = logging.getLogger("test")
    before = repo.cash_balance_usd()
    monkeypatch.setattr(runtime, "fundamental_context", lambda *_: (None, ""))
    monkeypatch.setattr(MarketCache, "frames", lambda *_: (None, None))
    monkeypatch.setattr(runtime, "analyze_headless", lambda r,s,*a: SimpleNamespace(
        symbol=s, last_price=100., source_bar_closed_at=pd.Timestamp("2026-09-03T15:00:00Z")))
    monkeypatch.setattr(zones, "build_zone_snapshot", lambda _,now: synthetic_snapshot(now))
    code = runtime.collect(repo, ["SMCI","NVDA"], tmp_path, log, now_fn=lambda: UTC_TIME)
    assert code == 0
    assert len(repo.zone_predictions()) == 12
    assert runtime.collect(repo, ["SMCI","NVDA"], tmp_path, log, now_fn=lambda: UTC_TIME) == 0
    assert len(repo.zone_predictions()) == 12
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
    monkeypatch.setattr(runtime, "analyze_headless", lambda r,s,*a: SimpleNamespace(
        symbol=s, last_price=100., source_bar_closed_at=pd.Timestamp("2026-09-03T15:00:00Z")))
    monkeypatch.setattr(zones, "build_zone_snapshot", lambda _,now: synthetic_snapshot(now))
    assert runtime.collect(repo, ["SMCI","NVDA"], tmp_path, logging.getLogger("test"), now_fn=lambda: UTC_TIME) == 1
    assert {r["symbol"] for r in repo.zone_predictions()} == {"NVDA"}


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
