"""Prospective zone evidence, isolated from portfolio accounting.

SHA-256 detects corruption; it is not protection against an administrator who
can replace both data and hashes. Exchange-calendar timestamps are UTC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ..analytics.closed_bars import NY, _calendar, utc

FORECAST_COLUMNS = (
    "id", "symbol", "timestamp_prediction", "source_bar_closed_at", "session_date",
    "expires_at", "zone_key", "zone_type", "zone_price", "zone_low", "zone_high",
    "reference_price", "predicted_touch_probability",
    "predicted_close_above_probability", "predicted_close_probability",
    "close_direction", "timeframe", "model_version_hash", "model_name",
    "context_json", "touch_eligible",
)
RESULT_COLUMNS = (
    "actual_touch_occurred", "actual_close_price", "actual_close_relation",
    "resolved_at", "resolution_note", "evidence_sha256",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def aware(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Se exige un timestamp válido con zona horaria.")
    return stamp.tz_convert("UTC")


def session_bounds(value):
    stamp = aware(value)
    day = pd.Timestamp(stamp.tz_convert(NY).date())
    schedule = _calendar(day.year - 1, day.year + 1).schedule
    if day not in schedule.index:
        return None
    row = schedule.loc[day]
    return str(day.date()), utc(row["open"]), utc(row["close"])


@dataclass(frozen=True)
class ZonePrediction:
    symbol: str
    timestamp_prediction: str
    source_bar_closed_at: str
    zone_key: str
    zone_type: str
    zone_low: float
    zone_high: float
    reference_price: float
    predicted_touch_probability: float
    predicted_close_probability: float | None
    close_direction: str
    model_version_hash: str
    model_name: str
    context_json: str = "{}"


def ensure_schema(database):
    """Add ONLY three analytic tables/triggers; never initialize the ledger."""
    statements = [
        """CREATE TABLE IF NOT EXISTS zone_prediction_log (
            id TEXT PRIMARY KEY, symbol TEXT NOT NULL,
            timestamp_prediction TEXT NOT NULL, source_bar_closed_at TEXT NOT NULL,
            session_date TEXT NOT NULL, expires_at TEXT NOT NULL,
            zone_key TEXT NOT NULL, zone_type TEXT NOT NULL,
            zone_price REAL NOT NULL, zone_low REAL NOT NULL, zone_high REAL NOT NULL,
            reference_price REAL NOT NULL,
            predicted_touch_probability REAL NOT NULL CHECK(predicted_touch_probability BETWEEN 0 AND 1),
            predicted_close_above_probability REAL,
            predicted_close_probability REAL, close_direction TEXT NOT NULL,
            timeframe TEXT NOT NULL, model_version_hash TEXT NOT NULL, model_name TEXT NOT NULL,
            context_json TEXT NOT NULL, touch_eligible INTEGER NOT NULL,
            forecast_sha256 TEXT NOT NULL,
            actual_touch_occurred INTEGER, actual_close_price REAL,
            actual_close_relation TEXT, resolved_at TEXT, resolution_note TEXT,
            evidence_sha256 TEXT, resolution_sha256 TEXT,
            UNIQUE(symbol, model_version_hash, source_bar_closed_at, zone_key)
        )""",
        """CREATE TABLE IF NOT EXISTS zone_market_evidence (
            sha256 TEXT PRIMARY KEY, payload_json TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS zone_daily_validation (
            sha256 TEXT PRIMARY KEY, session_date TEXT NOT NULL,
            symbol TEXT NOT NULL, model_version_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, payload_json TEXT NOT NULL)""",
        """CREATE INDEX IF NOT EXISTS ix_zone_pending ON zone_prediction_log(resolved_at, expires_at)""",
    ]
    with database.transaction() as conn:
        for sql in statements:
            conn.execute(sql)
        changed = " OR ".join(f"NEW.{name} IS NOT OLD.{name}" for name in (*FORECAST_COLUMNS, "forecast_sha256"))
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS zone_forecast_immutable
            BEFORE UPDATE ON zone_prediction_log WHEN {changed}
            BEGIN SELECT RAISE(ABORT, 'immutable zone forecast'); END""")
        conn.execute("""CREATE TRIGGER IF NOT EXISTS zone_resolution_immutable
            BEFORE UPDATE ON zone_prediction_log WHEN OLD.resolved_at IS NOT NULL
            BEGIN SELECT RAISE(ABORT, 'immutable zone resolution'); END""")
        for table in ("zone_prediction_log", "zone_market_evidence", "zone_daily_validation"):
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END""")
            if table != "zone_prediction_log":
                conn.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
                    BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END""")


def save_prediction(database, prediction: ZonePrediction, *, now=None):
    """First emission per closed candle/zone/version wins. No historical backfill."""
    emitted = aware(prediction.timestamp_prediction)
    clock = utc(now)
    closed = aware(prediction.source_bar_closed_at)
    if not pd.Timedelta(0) <= clock - emitted <= pd.Timedelta(seconds=60):
        raise ValueError("No se permiten predicciones retrospectivas o futuras.")
    bounds = session_bounds(emitted)
    if bounds is None or not bounds[1] <= closed <= emitted < bounds[2]:
        raise ValueError("No hay una sesión abierta válida para emitir.")
    if emitted - closed >= pd.Timedelta(minutes=5):
        raise ValueError("La última vela cerrada está vencida.")
    if closed != closed.floor("5min"):
        raise ValueError("El corte debe corresponder al cierre exacto de una vela 5m.")
    if not prediction.symbol or prediction.zone_type not in ("ENTRY", "TP1", "TP2", "RESISTANCE", "STOP"):
        raise ValueError("Símbolo/tipo de zona inválido.")
    if prediction.close_direction not in ("ABOVE", "BELOW"):
        raise ValueError("Dirección de cierre inválida.")
    if len(prediction.model_version_hash) != 64 or any(c not in "0123456789abcdef" for c in prediction.model_version_hash):
        raise ValueError("Versión de modelo SHA-256 inválida.")
    for p in (prediction.predicted_touch_probability, prediction.predicted_close_probability):
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("Probabilidad fuera de [0, 1].")
    for price in (prediction.zone_low, prediction.zone_high, prediction.reference_price):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Nivel de precio inválido.")
    if prediction.zone_low > prediction.zone_high:
        raise ValueError("Zona invertida.")
    row = asdict(prediction)
    for key in ("zone_low", "zone_high", "reference_price", "predicted_touch_probability", "predicted_close_probability"):
        if row[key] is not None:
            row[key] = float(row[key])
    row.update(id=str(uuid4()), symbol=prediction.symbol.upper(),
               timestamp_prediction=emitted.isoformat(), source_bar_closed_at=closed.isoformat(),
               session_date=bounds[0], expires_at=bounds[2].isoformat(),
               zone_price=float(prediction.zone_high if prediction.reference_price > prediction.zone_high else prediction.zone_low),
               predicted_close_above_probability=row["predicted_close_probability"] if prediction.close_direction == "ABOVE" else None,
               timeframe="INTRADAY",
               touch_eligible=int(not prediction.zone_low <= prediction.reference_price <= prediction.zone_high))
    row["context_json"] = canonical(json.loads(row["context_json"]))
    row["forecast_sha256"] = digest({k: row[k] for k in FORECAST_COLUMNS})
    with database.transaction() as conn:
        columns = (*FORECAST_COLUMNS, "forecast_sha256")
        cursor = conn.execute(
            f"INSERT INTO zone_prediction_log ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            "ON CONFLICT(symbol,model_version_hash,source_bar_closed_at,zone_key) DO NOTHING",
            [row[k] for k in columns],
        )
        return bool(cursor.rowcount)


@lru_cache(maxsize=256)
def _evidence_valid(expected, payload):
    return payload is not None and digest(json.loads(payload)) == expected


def verified(row, evidence=None):
    try:
        if digest({k: row[k] for k in FORECAST_COLUMNS}) != row["forecast_sha256"]:
            return False
        if row["resolved_at"] is not None:
            payload = {k: row[k] for k in RESULT_COLUMNS}
            payload["forecast_sha256"] = row["forecast_sha256"]
            if digest(payload) != row["resolution_sha256"]:
                return False
            if not _evidence_valid(row["evidence_sha256"], evidence):
                return False
        elif any(row.get(k) is not None for k in (*RESULT_COLUMNS, "resolution_sha256")):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def read_predictions(database):
    connection = database.connect()
    try:
        rows = connection.execute("SELECT * FROM zone_prediction_log ORDER BY timestamp_prediction,id")
        evidence_cache = {}
        result = []
        for original in rows:
            row = dict(original)
            key = row["evidence_sha256"]
            # Cache one small payload per session/download, not one copy per forecast.
            if key and key not in evidence_cache:
                stored = connection.execute("SELECT payload_json FROM zone_market_evidence WHERE sha256=?", (key,)).fetchone()
                evidence_cache[key] = stored[0] if stored else None
            row["integrity_ok"] = verified(row, evidence_cache.get(key))
            result.append(row)
        return result
    finally:
        connection.close()


@lru_cache(maxsize=1)
def model_version_hash():
    root = Path(__file__).resolve().parents[2]
    names = [
        "portfolio_tracker/services/zone_forward.py",
        "portfolio_tracker/services/price_zones.py",
        "portfolio_tracker/services/forward_market.py",
    ] + [str(path.relative_to(root)).replace("\\", "/")
         for path in sorted((root / "portfolio_tracker/analytics").glob("*.py"))]
    return digest({name: sha256((root / name).read_bytes()).hexdigest() for name in names})


def log_snapshot(repository, analysis, snapshot, *, now=None):
    """UI and all four PDFs share this exact frozen snapshot and one log."""
    clock = utc(now)
    emitted = aware(snapshot.evaluated_at)
    bounds = session_bounds(emitted)
    if bounds is None or not bounds[1] <= emitted < bounds[2]:
        return {"saved": 0, "reason": "Mercado cerrado: no se inventan predicciones intradía."}
    if emitted - aware(analysis.source_bar_closed_at) >= pd.Timedelta(minutes=5):
        return {"saved": 0, "reason": "Velas vencidas: snapshot excluido del forward."}
    saved = 0
    skipped = 0
    for i, (zone, estimate) in enumerate(zip((*snapshot.buys, *snapshot.sales), snapshot.estimates)):
        if estimate.probability is None or zone.low is None or zone.high is None:
            skipped += 1
            continue
        item = ZonePrediction(
            symbol=analysis.symbol, timestamp_prediction=emitted.isoformat(),
            source_bar_closed_at=aware(analysis.source_bar_closed_at).isoformat(),
            zone_key=("ENTRY1", "ENTRY2", "ENTRY3", "TP1", "TP2", "R3")[i],
            zone_type=("ENTRY", "ENTRY", "ENTRY", "TP1", "TP2", "RESISTANCE")[i],
            zone_low=float(zone.low), zone_high=float(zone.high),
            reference_price=float(analysis.last_price),
            predicted_touch_probability=float(estimate.probability) / 100,
            predicted_close_probability=None if estimate.close_probability is None else float(estimate.close_probability) / 100,
            close_direction="BELOW" if i < 3 else "ABOVE",
            model_version_hash=model_version_hash(), model_name=estimate.model,
            context_json=canonical({"samples": estimate.samples, "status": estimate.status,
                                    "detail": estimate.detail, "zone_label": zone.label,
                                    "source": zone.source, "matching": "weighted", "min_sessions": 12}),
        )
        saved += repository.save_prediction(item, now=clock)
    return {"saved": saved, "skipped": skipped,
            "reason": f"{skipped} zonas sin nivel/estimación utilizable; no registradas." if skipped else
                      "Registro prospectivo; recargas del mismo corte no se duplican."}


def _market_payload(bars, daily, symbol, fetched_at):
    return {"provider": "yfinance/raw-unadjusted", "symbol": symbol,
            "fetched_at": utc(fetched_at).isoformat(),
            "bars": [{"at": stamp.isoformat(), **{k: float(row[k]) for k in ("Open","High","Low","Close","Volume")}}
                     for stamp, row in bars.iterrows()],
            "daily": {k: float(daily[k]) for k in ("Open","High","Low","Close","Volume")}}


def resolve_predictions(database, provider, *, now=None):
    """provider(symbol, session_date) -> (5m frame, 1D frame).
    No market data call before actual exchange close + 15m. Fail closed.
    """
    clock = utc(now)
    rows = read_predictions(database)
    pending = [r for r in rows if r["integrity_ok"] and r["resolved_at"] is None
               and clock >= aware(r["expires_at"]) + pd.Timedelta(minutes=15)]
    result = {"resolved": 0, "pending": 0, "errors": [], "invalid_hashes": sum(not r["integrity_ok"] for r in rows)}
    groups = {}
    for row in pending:
        groups.setdefault((row["symbol"], row["session_date"]), []).append(row)
    for (symbol, day), group in groups.items():
        try:
            bars, daily = provider(symbol, day)
            bars = bars.copy()
            index = pd.DatetimeIndex(bars.index)
            if index.tz is None:
                raise ValueError("Las velas intradía no tienen zona horaria.")
            bars.index = index.tz_convert("UTC")
            bounds = session_bounds(group[0]["timestamp_prediction"])
            expected = pd.date_range(bounds[1], bounds[2], freq="5min", inclusive="left")
            bars = bars.loc[(bars.index >= bounds[1]) & (bars.index < bounds[2])].sort_index()
            if bars.index.has_duplicates or not bars.index.equals(expected):
                raise ValueError("Faltan velas 5m de la sesión; no se imputa el resultado.")
            fields = ["Open", "High", "Low", "Close", "Volume"]
            values = bars[fields].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
                raise ValueError("OHLCV incompleto/inválido.")
            if ((bars.High < bars[["Open", "Close", "Low"]].max(axis=1)) |
                (bars.Low > bars[["Open", "Close"]].min(axis=1))).any():
                raise ValueError("Geometría OHLC inválida.")
            daily_day = daily.loc[[str(pd.Timestamp(s).date()) == day for s in daily.index]]
            if len(daily_day) != 1:
                raise ValueError("No existe exactamente una vela diaria del vencimiento.")
            last = daily_day.iloc[0]
            if not np.isfinite(last[fields].to_numpy(dtype=float)).all() or float(last.Close) <= 0:
                raise ValueError("Cierre diario inválido.")
            if not math.isclose(float(last.Close), float(bars.Close.iloc[-1]), rel_tol=0.0001, abs_tol=0.01):
                raise ValueError("Cierre diario y última vela 5m discrepan; revisión pendiente.")
            evidence = _market_payload(bars, last, symbol, clock)
            evidence_hash = digest(evidence)
            with database.transaction() as conn:
                conn.execute("INSERT OR IGNORE INTO zone_market_evidence VALUES (?,?)",
                             (evidence_hash, canonical(evidence)))
                for row in group:
                    emitted = aware(row["timestamp_prediction"])
                    full = bars.loc[bars.index >= emitted]
                    partial = bars.loc[(bars.index < emitted) & (bars.index + pd.Timedelta(minutes=5) > emitted)]
                    def touched(frame):
                        return bool((frame.Low <= row["zone_high"]).any()) if row["reference_price"] > row["zone_high"] else bool((frame.High >= row["zone_low"]).any())
                    hit = touched(full)
                    ambiguous = not hit and touched(partial)
                    actual = None if ambiguous or not row["touch_eligible"] else int(hit)
                    close_price = float(last.Close)
                    resolution = dict(
                        actual_touch_occurred=actual, actual_close_price=close_price,
                        actual_close_relation="BELOW" if close_price < row["zone_low"] else "ABOVE" if close_price > row["zone_high"] else "INSIDE",
                        resolved_at=clock.isoformat(), evidence_sha256=evidence_hash,
                        resolution_note="YA ALCANZADA AL CORTE: toque excluido" if not row["touch_eligible"] else
                                        "AMBIGUA: cruce solo en vela parcialmente anterior" if ambiguous else "RESUELTA",
                    )
                    resolution_hash = digest({**resolution, "forecast_sha256": row["forecast_sha256"]})
                    updates = {**resolution, "resolution_sha256": resolution_hash}
                    cursor = conn.execute(
                        "UPDATE zone_prediction_log SET " + ",".join(k+"=?" for k in updates) +
                        " WHERE id=? AND resolved_at IS NULL AND forecast_sha256=?",
                        [*updates.values(), row["id"], row["forecast_sha256"]])
                    result["resolved"] += cursor.rowcount
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            result["pending"] += len(group)
            result["errors"].append(f"{symbol} {day}: {exc}")
    store_daily_summaries(database, now=clock)
    return result


def validation_data(rows):
    """Primary cohort fixed BEFORE inspecting outcomes: first emission/day/zone/version.
    Six zones within one session are correlated, not six independent sessions.
    """
    first = {}
    for row in rows:
        key = (row["symbol"], row["model_version_hash"], row["session_date"], row["zone_key"])
        first.setdefault(key, row)
    evaluated = []
    for row in first.values():
        if not row["integrity_ok"] or row["resolved_at"] is None:
            continue
        touch = row["actual_touch_occurred"]
        close_target = (row["actual_close_price"] >= row["zone_high"] if row["close_direction"] == "ABOVE"
                        else row["actual_close_price"] <= row["zone_low"])
        for event, probability, actual in (
            ("Toque", row["predicted_touch_probability"], touch if row["touch_eligible"] else None),
            ("Cierre", row["predicted_close_probability"], int(close_target)),
        ):
            if probability is None or actual is None:
                continue
            evaluated.append({**row, "event": event, "prediction": probability, "actual": actual,
                              "brier": (probability - actual) ** 2,
                              "bin": min(9, int(probability * 10))})
    return evaluated


def store_daily_summaries(database, *, now=None):
    rows = read_predictions(database)
    groups = {}
    for row in validation_data(rows):
        groups.setdefault((row["symbol"], row["model_version_hash"], row["session_date"]), []).append(row)
    with database.transaction() as conn:
        for (symbol, model, day), group in groups.items():
            payload = {"symbol": symbol, "model": model, "session": day,
                       "cohort": "first-emission-per-zone-per-session-v1",
                       "members": sorted({r["resolution_sha256"] for r in group}),
                       "metrics": {event: {"n": sum(r["event"] == event for r in group),
                                           "brier": float(np.mean([r["brier"] for r in group if r["event"] == event]))}
                                   for event in ("Toque", "Cierre") if any(r["event"] == event for r in group)}}
            conn.execute("INSERT OR IGNORE INTO zone_daily_validation VALUES (?,?,?,?,?,?)",
                         (digest(payload), day, symbol, model, utc(now).isoformat(), canonical(payload)))
