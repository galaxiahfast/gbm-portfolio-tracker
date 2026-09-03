"""F01/F02 regressions; all databases and bars are isolated/synthetic."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3

import pandas as pd
import pytest

from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.model_observations import maturity, is_regular_close

AT = datetime(2026, 8, 28, 19, tzinfo=timezone.utc)  # Friday 15:00 NY


@pytest.fixture
def repo(tmp_path):
    repository = PortfolioRepository(Database(tmp_path / "portfolio.db"))
    repository.database.initialize()
    return repository


def record(repo, at=AT, minutes=60):
    return repo.record_live_model_observation(
        symbol="SMCI", observed_at=at, source_bar_at=pd.Timestamp(at).floor("5min").to_pydatetime(),
        reference_price=Decimal("100"), raw_probability_up=Decimal("0.8"),
        parameters_json='{"version":"test"}', horizon_minutes=minutes,
    )


def bars(*pairs):
    return pd.DataFrame(
        [{"Open": p, "High": p+1, "Low": p-1, "Close": p, "Volume": 100} for _,p in pairs],
        index=pd.DatetimeIndex([pd.Timestamp(t)-pd.Timedelta(minutes=5) for t,_ in pairs]),
    )


def row(repo):
    with repo.database.connect() as c:
        return dict(c.execute("SELECT * FROM live_model_observations ORDER BY id DESC LIMIT 1").fetchone())


def resolve(repo):
    return repo.resolve_live_model_observations(
        symbol="SMCI", current_as_of=AT+timedelta(days=3),
        historical_bars=bars((AT+timedelta(hours=1),101), (AT+timedelta(days=3),90)),
    )


def test_friday_resolves_friday_close_not_monday_price(repo):
    record(repo)
    forecast_hash = row(repo)["observation_sha256"]
    assert resolve(repo) == 1
    result = row(repo)
    assert Decimal(result["outcome_price"]) == Decimal(101)
    assert result["outcome_bar_at"] == result["available_at"]
    assert result["resolved_at"] != result["outcome_bar_at"]
    assert result["observation_sha256"] == forecast_hash
    assert result["resolution_sha256"]
    assert repo.verify_live_model_observations() == (1, ())
    assert resolve(repo) == 0
    assert row(repo) == result


@pytest.mark.parametrize("field,value", [
    ("outcome_price", "90"), ("outcome_up", 0), ("successful", 0),
    ("resolved_at", "2026-08-28T20:01:00+00:00"),
    ("outcome_bar_at", "2026-08-31T19:00:00+00:00"),
    ("outcome_source", "other"), ("available_at", "2026-08-31T19:00:00+00:00"),
    ("source_bar_at", "2026-08-28T18:55:00+00:00"),
    ("resolution_sha256", "bad"), ("resolution_status", "PENDING"),
])
def test_any_result_tamper_is_rejected_and_excluded(repo, field, value):
    record(repo)
    resolve(repo)
    with repo.database.transaction() as c:
        # Model disk/out-of-band corruption, bypass normal immutable API guards.
        c.execute("DROP TRIGGER live_resolution_immutable")
        c.execute("DROP TRIGGER live_forecast_immutable")
        c.execute(f"UPDATE live_model_observations SET {field}=?", (value,))
    assert repo.verify_live_model_observations() == (0, (1,))
    assert repo.live_model_calibration_samples("SMCI", horizon_minutes=60) == ()
    assert repo.live_model_stats("SMCI")["resolved"] == 0


def test_database_blocks_rewrites_and_deletes(repo):
    record(repo)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with repo.database.transaction() as c:
            c.execute("UPDATE live_model_observations SET reference_price='80'")
    resolve(repo)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with repo.database.transaction() as c:
            c.execute("UPDATE live_model_observations SET outcome_price='80'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with repo.database.transaction() as c:
            c.execute("DELETE FROM live_model_observations")


def test_pending_tamper_never_gets_signed(repo):
    record(repo)
    with repo.database.transaction() as c:
        c.execute("UPDATE live_model_observations SET outcome_up=0")
    assert resolve(repo) == 0
    assert row(repo)["resolution_sha256"] is None
    assert repo.verify_live_model_observations() == (0, (1,))


def test_no_spot_no_nearest_no_future_no_duplicate_fill(repo):
    record(repo)
    with pytest.raises(ValueError, match="precio actual"):
        repo.resolve_live_model_observations(symbol="SMCI", current_price=Decimal(110), current_as_of=AT+timedelta(days=3))
    due = AT+timedelta(hours=1)
    assert repo.resolve_live_model_observations(symbol="SMCI", current_as_of=due-timedelta(seconds=1), historical_bars=bars((due,101))) == 0
    assert repo.resolve_live_model_observations(symbol="SMCI", current_as_of=due, historical_bars=bars((due-timedelta(minutes=5),101))) == 0
    assert repo.resolve_live_model_observations(symbol="SMCI", current_as_of=due, historical_bars=bars((due,101),(due,102))) == 0
    assert row(repo)["resolution_status"] == "PENDING"
    assert resolve(repo) == 1


@pytest.mark.parametrize("at,minutes", [
    (datetime(2026,9,4,19,tzinfo=timezone.utc), 4320),  # Monday Labor Day
    (AT, 1440),  # Saturday
    (datetime(2026,11,27,17,tzinfo=timezone.utc), 120),  # after 13:00 NY early close
    (AT, 120),  # after regular close
])
def test_closed_market_target_is_invalid_never_rolled(repo, at, minutes):
    record(repo, at, minutes)
    due = maturity(at, minutes)
    assert not is_regular_close(due)
    assert repo.resolve_live_model_observations(symbol="SMCI", current_as_of=due.to_pydatetime(), historical_bars=bars((due,110))) == 0
    assert row(repo)["resolution_status"] == "INVALID_MARKET_CLOSED"
    assert row(repo)["outcome_price"] is None
    assert row(repo)["resolution_sha256"]
    assert repo.verify_live_model_observations() == (1, ())
    assert repo.live_model_calibration_samples("SMCI", horizon_minutes=minutes) == ()


def test_emission_maturity_and_idempotency(repo):
    emitted = AT+timedelta(seconds=17)
    assert record(repo, emitted)
    assert not record(repo, emitted+timedelta(seconds=30))
    result = row(repo)
    assert result["observed_at"] == emitted.isoformat()
    assert result["source_bar_at"] == AT.isoformat()
    assert result["available_at"] == (AT+timedelta(minutes=65)).isoformat()


def test_timezone_and_dst_contract(repo):
    with pytest.raises(ValueError, match="zona horaria"):
        record(repo, AT.replace(tzinfo=None))
    assert is_regular_close("2026-03-06T15:00:00Z")  # EST
    assert is_regular_close("2026-03-09T14:00:00Z")  # EDT
    with pytest.raises(ValueError, match="cerrada reciente"):
        repo.record_live_model_observation(symbol="SMCI", observed_at=AT,
            source_bar_at=AT+timedelta(minutes=5), reference_price=Decimal(100),
            raw_probability_up=Decimal('.8'), parameters_json='{}')


def test_legacy_migration_preserves_rows_and_accounting(repo):
    repo.ensure_initial_capital()
    with repo.database.transaction() as c:
        c.execute("""INSERT INTO live_model_observations(symbol,observed_at,horizon_minutes,
            reference_price,raw_probability_up,predicted_direction,parameters_json,
            observation_sha256,outcome_price,outcome_up,successful,resolved_at,created_at)
            VALUES ('OLD','2026-08-28T19:00:00+00:00',60,'100','.8','UP','{}',
                    'oldhash','999',1,1,'2026-08-31T19:00:00+00:00','old')""")
        before = tuple(c.execute("SELECT * FROM live_model_observations").fetchone())
        cash = [tuple(x) for x in c.execute("SELECT * FROM cash_movements")]
        # Rerun the additive migration as on a previously installed database.
        c.execute("DELETE FROM schema_migrations WHERE version=9")
    repo.database.initialize()
    repo.database.initialize()
    with repo.database.connect() as c:
        assert tuple(c.execute("SELECT * FROM live_model_observations").fetchone()) == before
        assert [tuple(x) for x in c.execute("SELECT * FROM cash_movements")] == cash
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert repo.live_model_calibration_samples("OLD", horizon_minutes=60) == ()
    assert repo.verify_live_model_observations() == (0, (1,))
    assert repo.cash_balance_usd() == Decimal('921.05')


def test_upgrade_actual_v8_columns_preserves_legacy(tmp_path):
    from portfolio_tracker.db import BASE_SCHEMA, MIGRATIONS
    path = tmp_path / 'v8.db'
    with sqlite3.connect(path) as c:
        c.executescript(BASE_SCHEMA)
        c.execute('CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)')
        c.executemany('INSERT INTO schema_migrations VALUES(?,?,?)',
                      [(v,n,'old') for v,n in MIGRATIONS if v < 9])
        c.execute("""INSERT INTO live_model_observations(symbol,observed_at,horizon_minutes,
            reference_price,raw_probability_up,predicted_direction,parameters_json,
            observation_sha256,outcome_price,outcome_up,successful,resolved_at,created_at)
            VALUES ('OLD','2026-08-28T19:00:00+00:00',60,'100','.8','UP','{}',
                    'oldhash','999',1,1,'2026-08-31T19:00:00+00:00','old')""")
        old_columns = [r[1] for r in c.execute('PRAGMA table_info(live_model_observations)')]
        old_row = c.execute('SELECT * FROM live_model_observations').fetchone()
    repository = PortfolioRepository(Database(path))
    repository.database.initialize()
    with repository.database.connect() as c:
        assert tuple(c.execute('SELECT '+','.join(old_columns)+' FROM live_model_observations').fetchone()) == old_row
    assert row(repository)['resolution_status'] == 'LEGACY_UNVERIFIED'
    assert row(repository)['resolution_sha256'] is None
    assert repository.live_model_calibration_samples('OLD', horizon_minutes=60) == ()
    assert repository.database.schema_version() == 9
    assert list((tmp_path/'backups').glob('*before-v9*'))


def test_concurrent_resolvers_finalize_only_once(repo):
    from concurrent.futures import ThreadPoolExecutor
    record(repo)
    with ThreadPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(lambda _: resolve(repo), range(2)))
    assert sorted(counts) == [0, 1]
    assert repo.verify_live_model_observations() == (1, ())


def test_analysis_cut_keeps_open_label_and_exposes_known_close():
    from tests.test_pdf_report import _analysis
    analysis = _analysis()
    original = analysis.as_of
    assert analysis.source_bar_closed_at == original+timedelta(minutes=5)
    assert analysis.as_of == original
