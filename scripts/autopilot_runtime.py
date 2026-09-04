"""Shared headless job plumbing. No Streamlit import, orders or cash writes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, time, timezone, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")


def clock():
    return datetime.now(timezone.utc)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as output:
            temporary = output.name
            json.dump(value, output, ensure_ascii=False, sort_keys=True, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


class NYFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, NY).strftime("%Y-%m-%d %H:%M:%S %Z")


def logger(log_dir, job="collect"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output = logging.getLogger("gbm.autopilot")
    output.setLevel(logging.INFO)
    output.propagate = False
    for handler in output.handlers[:]:
        handler.close()
        output.removeHandler(handler)
    names = {"collect": "collector.log", "resolve": "resolver.log", "catchup": "catchup.log"}
    file = RotatingFileHandler(log_dir / names.get(job, "collector.log"), maxBytes=2_000_000,
                               backupCount=5, encoding="utf-8")
    file.setFormatter(NYFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    output.addHandler(file)
    # pythonw has no stdout/stderr; durable file is always present.
    import sys
    if sys.stderr is not None:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        console = logging.StreamHandler()
        console.setFormatter(file.formatter)
        output.addHandler(console)
    return output


@contextmanager
def exclusive_job(path):
    """OS advisory lock; released automatically after kill/power loss. No stale PID lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def allowed(job, now, scheduled=False):
    from portfolio_tracker.services.zone_forward import session_bounds
    local = now.astimezone(NY)
    if job == "catchup":
        # Boot can recover Friday on a Monday holiday. It never emits predictions.
        return not scheduled or time(9, 5) <= local.time() < time(17, 20)
    bounds = session_bounds(now)
    if bounds is None:
        return False
    if job == "collect":
        if not bounds[1] <= now < bounds[2]:
            return False
        return not scheduled or time(11) <= local.time() < time(11, 20)
    if job == "resolve":
        from datetime import timedelta
        if now < bounds[2] + timedelta(minutes=15):
            return False
        return not scheduled or time(17) <= local.time() < time(17, 20)
    raise ValueError("Trabajo desconocido")


def open_repository(path):
    """Never run Database.initialize(): existing ledger/schema must already exist."""
    from portfolio_tracker.db import Database
    from portfolio_tracker.repository import PortfolioRepository
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Base existente no encontrada: {path}. No se creará otro portafolio.")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        required = {"trades", "operational_events", "backtest_runs", "fundamental_news_snapshots"}
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if required - tables:
            raise ValueError(f"Falta estructura existente: {sorted(required-tables)}. Iniciar/migrar con el flujo habitual.")
    finally:
        connection.close()
    repository = PortfolioRepository(Database(path))
    repository.ensure_zone_forward_schema()
    return repository


def fundamental_context(repository, symbol, log):
    from portfolio_tracker.analytics.fundamental_news import (
        download_fundamental_news, snapshot_to_payload, snapshot_from_payload,
    )
    try:
        snapshot = download_fundamental_news(symbol)
        _, fingerprint = repository.record_fundamental_news_snapshot(
            symbol=snapshot.symbol, observed_at=datetime.fromisoformat(snapshot.observed_at),
            provider=snapshot.provider, engine_version=snapshot.version,
            payload_json=snapshot_to_payload(snapshot),
        )
        return snapshot, fingerprint
    except Exception as exc:
        # Same signed fallback as UI, with the error visible in durable logs.
        log.warning("%s: fundamentales externos fallaron: %s", symbol, exc)
        stored = repository.latest_fundamental_news_snapshot(symbol)
        if stored:
            try:
                return snapshot_from_payload(str(stored["payload_json"])), str(stored["payload_sha256"])
            except (ValueError, TypeError, KeyError):
                log.exception("%s: corte fundamental local descartado", symbol)
        log.warning("%s: contexto fundamental neutral, igual que fallback UI", symbol)
        return None, ""


def analyze_headless(repository, symbol, intraday, daily, fundamental, fundamental_hash, now, log):
    """Exact dependencies of build_zone_snapshot in app.py, not a second quant model.

    Horizon multiclass calibration and live-horizon observation writes are omitted:
    neither supplies the six zone bounds/touch/close estimates.
    Operational memory is shared with UI; synchronize_position only writes
    operational_events and NEVER creates fills/orders/cash movements.
    """
    from portfolio_tracker.analytics.backtesting import ENGINE_VERSION
    from portfolio_tracker.analytics.technical_probability import analyze_probability
    from portfolio_tracker.analytics.fundamental_news import apply_fundamental_filter
    from portfolio_tracker.services.operational_state import macro_memory, synchronize_position
    from portfolio_tracker.services.cross_asset import prefetch_cross_asset
    if intraday is not None and daily is not None and not intraday.empty and not daily.empty:
        prefetch_cross_asset(symbol)
    parameters = repository.latest_backtest_parameters(symbol=symbol, engine_version=ENGINE_VERSION) or {
        "minimum_probability": .55, "stop_atr_multiple": 2.25, "risk_per_trade_pct": 1.,
    }
    analysis = analyze_probability(
        symbol, intraday, daily, previous_macro_trending=macro_memory(repository.database, symbol),
        require_fresh=True, as_of_time=now,
        atr_stop_multiple=float(parameters.get("stop_atr_multiple", 2.25)),
    )
    if fundamental is not None:
        try:
            analysis = apply_fundamental_filter(analysis, fundamental)
            analysis = replace(analysis, fundamental_snapshot_sha256=fundamental_hash)
        except ValueError as exc:
            log.warning("%s: corte fundamental descartado como en UI: %s", symbol, exc)
    from portfolio_tracker.services.cross_asset import enrich_cross_asset
    analysis = enrich_cross_asset(analysis, intraday, daily, now=now)
    return synchronize_position(repository.database, analysis)


def completed_group(repository, symbol, day, earliest=None):
    """Crash recovery: six intact zone records from one actual cut, not fabricated status."""
    from portfolio_tracker.services.zone_forward import model_version_hash
    groups = {}
    for row in repository.zone_predictions():
        if (not row["integrity_ok"] or row["symbol"] != symbol or row["session_date"] != day
                or row["model_version_hash"] != model_version_hash()):
            continue
        if earliest and datetime.fromisoformat(row["timestamp_prediction"]) < earliest:
            continue
        groups.setdefault(row["source_bar_closed_at"], set()).add(row["zone_key"])
    return any(keys >= {"ENTRY1", "ENTRY2", "ENTRY3", "TP1", "TP2", "R3"} for keys in groups.values())


def collect(repository, symbols, state_dir, log, *, scheduled=False, now_fn=clock):
    from portfolio_tracker.services.price_zones import build_zone_snapshot
    from portfolio_tracker.services.zone_forward import log_snapshot
    from scripts.autopilot_market_cache import MarketCache
    failed = False
    cache = MarketCache(Path(state_dir) / "market")
    for symbol in symbols:
        try:
            now = now_fn()
            if not allowed("collect", now, scheduled):
                log.info("%s: fuera de ventana/sesión; no se retrofechan predicciones", symbol)
                continue
            day = now.astimezone(NY).date().isoformat()
            # Scheduled collection starts at 11 NY; early manual UI cuts must not
            # suppress it. Manual runs use a daily cohort, still idempotent.
            earliest = (datetime.combine(now.astimezone(NY).date(), time(11), NY)
                        if now.astimezone(NY).time() >= time(11) else None)
            if completed_group(repository, symbol, day, earliest):
                log.info("%s: ya hay seis zonas íntegras de hoy; sin duplicar", symbol)
                continue
            fundamental, fundamental_hash = fundamental_context(repository, symbol, log)
            intraday, daily = cache.frames(symbol, now_fn())
            analysis = analyze_headless(repository, symbol, intraday, daily,
                                        fundamental, fundamental_hash, now_fn(), log)
            # Freeze at real emission, not 11:00 if execution arrived late.
            snapshot = build_zone_snapshot(analysis, now=now_fn())
            if not allowed("collect", now_fn(), scheduled):
                raise ValueError("Terminó la ventana antes de emitir; no se guardan pronósticos tardíos.")
            result = log_snapshot(repository, analysis, snapshot, now=now_fn())
            log.info("%s: predicción guardada. %s zonas. %s", symbol, result["saved"], result["reason"])
            if not completed_group(repository, symbol, day, earliest):
                failed = True
                log.warning("%s: recolección incompleta; no se inventan zonas o porcentajes N/D", symbol)
        except Exception:
            failed = True
            log.exception("%s: error de colección; se continúa con los demás activos", symbol)
    return 1 if failed else 0


def resolve(repository, symbols, log, *, catchup=False, now=None):
    from portfolio_tracker.services.forward_market import resolution_frames
    now = now or clock()
    rows = repository.zone_predictions()
    invalid = sum(not r["integrity_ok"] for r in rows)
    if invalid:
        log.error("%s registros con firmas inválidas; excluidos y requieren revisión", invalid)
    unresolved = [r for r in rows if r["integrity_ok"] and r["resolved_at"] is None]
    pending = [r for r in unresolved if now >= datetime.fromisoformat(r["expires_at"]) + timedelta(minutes=15)]
    # Include retired symbols with unresolved evidence, not just current watchlist.
    symbols = list(dict.fromkeys([*symbols, *(r["symbol"] for r in pending)]))
    log.info("Activos configurados/pendientes: %s", ",".join(symbols))
    errors = []
    def provider(symbol, day):
        try:
            frames = resolution_frames(symbol, day)
            log.info("%s %s: datos históricos recuperados para resolución", symbol, day)
            return frames
        except Exception as exc:
            # Existing resolver catches ValueError per symbol/session and continues.
            log.exception("%s %s: proveedor sin datos; permanece pendiente", symbol, day)
            raise ValueError(str(exc)) from exc
    if not pending:
        log.info("%s: no hay sesiones vencidas pendientes; %s registros aún no vencen; sin cambios",
                 "Catch-up" if catchup else "Resolución", len(unresolved))
        return 1 if invalid else 0
    # Existing API resolves only due groups. At boot no current-day group is
    # due. A manual catch-up after close may also resolve today's overdue rows.
    result = repository.resolve_predictions(provider, now=now)
    for warning in result.get("warnings", []):
        log.warning("%s", warning)
    for error in result["errors"]:
        errors.append(error)
        log.error("%s", error)
    log.info("Resolución: %s registros resueltos; %s pendientes; %s firmas inválidas",
             result["resolved"], result["pending"], result["invalid_hashes"])
    return 1 if errors or result["invalid_hashes"] else 0


def cli(job):
    parser = argparse.ArgumentParser(description="Autopiloto de evidencia, nunca ejecución de órdenes.")
    parser.add_argument("--symbols", nargs="+", default=["SMCI", "NVDA"])
    parser.add_argument("--database", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--scheduled", action="store_true", help="Respeta la ventana NY del programador.")
    parser.add_argument("--check-only", action="store_true", help="Comprueba horario y base en solo lectura, sin descargar ni emitir.")
    args = parser.parse_args()
    log = logger(args.log_dir, job)
    try:
        from portfolio_tracker.config import DB_PATH, DATA_DIR
        from portfolio_tracker.services.quant_market_data import normalize_symbol
        symbols = list(dict.fromkeys(normalize_symbol(s) for s in args.symbols))
        database = args.database or DB_PATH
        state_dir = args.state_dir or DATA_DIR / "autopilot"
        now = clock()
        permitted = allowed(job, now, args.scheduled)
        log.info("Inicio %s: símbolos=%s; programado=%s; permitido=%s", job, ",".join(symbols), args.scheduled, permitted)
        if args.check_only:
            if not database.is_file():
                raise ValueError(f"Base inexistente: {database}")
            log.info("Comprobación sin descargas/escrituras DB. Próximo paso: ejecutar sin --check-only.")
            return 0
        if not permitted:
            log.info("Sin trabajo: fuera de sesión/ventana NY o festivo.")
            return 0
        with exclusive_job(Path(state_dir) / "jobs.lock"):
            repository = open_repository(database)
            if job == "collect":
                return collect(repository, symbols, state_dir, log, scheduled=args.scheduled)
            return resolve(repository, symbols, log, catchup=job == "catchup", now=now)
    except (BlockingIOError, PermissionError) as exc:
        log.error("No se adquirió acceso/lock: %s; reintento seguro", exc)
        return 1
    except Exception:
        log.exception("Trabajo fallido sin cierre abrupto; se podrá reintentar")
        return 1
    finally:
        log.info("Fin %s", job)
