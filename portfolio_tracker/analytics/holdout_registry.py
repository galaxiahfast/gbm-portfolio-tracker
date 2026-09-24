"""Durable, independent audit trail for one-time final-holdout openings.

This database contains no portfolio transactions. Reserving every member of a
holdout in one SQLite transaction prevents a second run from testing on any of
the same observations, even if the dataset fingerprint later changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3


DEFAULT_HOLDOUT_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "model_validation" / "holdout_openings.sqlite"


class HoldoutAlreadyOpenedError(ValueError):
    """The final holdout (or any of its observations) was already consumed."""


def reserve_holdout_opening(
    *, registry_path: str | Path, dataset_sha256: str, symbol: str,
    horizon: str, protocol_sha256: str, commitment_sha256: str,
    members: tuple[tuple[str, str], ...],
) -> dict:
    """Persist opening *before* reading holdout labels or fitting final scores.

    The unique (symbol, horizon, cut_id) constraint is stricter than a dataset
    hash: a re-export or a slightly extended dataset cannot recycle old cuts.
    """
    if not members or not all(cut_id and observed_at for cut_id, observed_at in members):
        raise ValueError("El holdout necesita cortes identificables para registrarse.")
    if len({cut_id for cut_id, _ in members}) != len(members):
        raise ValueError("El holdout contiene cut_id duplicados.")
    if len(dataset_sha256) != 64 or len(protocol_sha256) != 64 or len(commitment_sha256) != 64:
        raise ValueError("Hashes SHA-256 del holdout inválidos.")
    destination = Path(registry_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    opened_at = datetime.now(timezone.utc).isoformat()
    opening_id = hashlib.sha256(
        f"{dataset_sha256}|{symbol}|{horizon}|{protocol_sha256}|{opened_at}".encode("utf-8")
    ).hexdigest()
    with sqlite3.connect(destination, timeout=30, isolation_level=None) as connection:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS holdout_openings (
                    opening_id TEXT PRIMARY KEY,
                    dataset_sha256 TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    protocol_sha256 TEXT NOT NULL,
                    commitment_sha256 TEXT NOT NULL,
                    opened_at_utc TEXT NOT NULL,
                    first_observed_at TEXT NOT NULL,
                    last_observed_at TEXT NOT NULL,
                    UNIQUE(dataset_sha256, symbol, horizon)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS holdout_opening_members (
                    opening_id TEXT NOT NULL REFERENCES holdout_openings(opening_id),
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    cut_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    PRIMARY KEY(symbol, horizon, cut_id)
                )
            """)
            connection.execute(
                "INSERT INTO holdout_openings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (opening_id, dataset_sha256, symbol, horizon, protocol_sha256,
                 commitment_sha256, opened_at, members[0][1], members[-1][1]),
            )
            connection.executemany(
                "INSERT INTO holdout_opening_members VALUES (?, ?, ?, ?, ?)",
                [(opening_id, symbol, horizon, cut_id, observed_at)
                 for cut_id, observed_at in members],
            )
            connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            connection.execute("ROLLBACK")
            raise HoldoutAlreadyOpenedError(
                f"Holdout ya abierto: {symbol}/{horizon}; dataset o cortes reutilizados."
            ) from exc
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return {
        "opening_id": opening_id,
        "dataset_sha256": dataset_sha256,
        "opened_at_utc": opened_at,
        "registry_path": str(destination),
        "member_count": len(members),
    }


def verify_holdout_opening(
    opening: dict, *, symbol: str, horizon: str,
    protocol_sha256: str, commitment_sha256: str,
) -> bool:
    """Check that a signed artifact's receipt still exists in the local ledger."""
    try:
        destination = Path(opening["registry_path"]).resolve(strict=True)
        with sqlite3.connect(f"file:{destination.as_posix()}?mode=ro", uri=True) as connection:
            stored = connection.execute(
                """SELECT dataset_sha256, symbol, horizon, protocol_sha256,
                          commitment_sha256, opened_at_utc
                   FROM holdout_openings WHERE opening_id = ?""",
                (opening["opening_id"],),
            ).fetchone()
            member_count = connection.execute(
                "SELECT count(*) FROM holdout_opening_members WHERE opening_id = ?",
                (opening["opening_id"],),
            ).fetchone()[0]
        return stored == (
            opening["dataset_sha256"], symbol, horizon, protocol_sha256,
            commitment_sha256, opening["opened_at_utc"],
        ) and member_count == opening["member_count"]
    except (OSError, KeyError, sqlite3.Error, TypeError, ValueError):
        return False
