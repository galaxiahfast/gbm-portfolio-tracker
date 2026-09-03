"""Resolve only analytic forward logs; does not initialize or migrate accounting."""
import argparse
import json
import hashlib
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio_tracker.config import DB_PATH
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.forward_market import resolution_frames
from portfolio_tracker.services.market_data import download_daily_history


def existing_table_fingerprints(path, names=None):
    """Read-only logical fingerprints, excluding only this module's own tables."""
    if not path.exists():
        return {}
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if names is None:
            names = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                     if r[0] not in {"zone_prediction_log", "zone_market_evidence", "zone_daily_validation"}]
        result = {}
        for name in sorted(names):
            quoted = '"' + name.replace('"', '""') + '"'
            # Order-independent comparison; nothing is inserted into the source.
            row_hashes = sorted(hashlib.sha256(repr(tuple(r)).encode()).hexdigest()
                                for r in connection.execute("SELECT * FROM " + quoted))
            schema = connection.execute("SELECT type,name,sql FROM sqlite_master WHERE tbl_name=? ORDER BY type,name", (name,)).fetchall()
            result[name] = {"rows": len(row_hashes), "sha256": hashlib.sha256(
                (repr(schema) + ''.join(row_hashes)).encode()).hexdigest()}
        return result
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--download-daily", metavar="SYMBOL")
    args = parser.parse_args()
    before = existing_table_fingerprints(args.database)
    repository = PortfolioRepository(Database(args.database))
    repository.ensure_zone_forward_schema()
    after = existing_table_fingerprints(args.database, before)
    if before != after:
        raise RuntimeError("El estado previo cambió durante la inicialización; revisar concurrencia/integridad.")
    output = {"database": str(args.database), "status": "PRELIMINAR"}
    output["preexisting_tables_unchanged"] = True
    output["preexisting_table_fingerprints"] = after
    failed = False
    if not args.initialize_only:
        output["resolution"] = repository.resolve_predictions(resolution_frames)
        failed = bool(output["resolution"]["errors"] or output["resolution"]["invalid_hashes"])
    if args.download_daily:
        try:
            _, meta = download_daily_history(args.download_daily)
            output["daily_history"] = {k: v for k, v in meta.items() if k != "frame"}
        except (ValueError, RuntimeError, OSError) as exc:
            output["daily_history_error"] = str(exc)
            failed = True
    rows = repository.zone_predictions()
    output["registered"] = len(rows)
    output["resolved"] = sum(row["resolved_at"] is not None for row in rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
