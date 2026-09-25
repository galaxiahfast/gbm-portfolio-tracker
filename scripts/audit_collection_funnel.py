"""Read-only daily coverage audit for the scheduled forward collector.

Usage: python scripts/audit_collection_funnel.py --sessions 10 --symbols SMCI NVDA
No market download, migrations, accounting writes or retrospective predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_tracker.config import DB_PATH
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.collection_funnel import collection_funnel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--symbols", nargs="+", default=["SMCI", "NVDA"])
    args = parser.parse_args(argv)
    database = args.database.resolve()
    if not database.is_file():
        parser.error(f"Base de datos no encontrada: {database}")
    # Validate openability without allowing SQLite to create another database.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True):
        pass
    repository = PortfolioRepository(Database(database))
    print(json.dumps(collection_funnel(repository, args.symbols, sessions=args.sessions),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
