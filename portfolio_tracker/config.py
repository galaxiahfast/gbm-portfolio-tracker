"""Configuracion centralizada y rutas de almacenamiento local."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.getenv("GBM_PORTFOLIO_DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser()
DB_PATH = DATA_DIR / "portfolio.db"
RECEIPTS_DIR = DATA_DIR / "receipts"

LOCAL_TIMEZONE = ZoneInfo("America/Mexico_City")
INITIAL_CAPITAL_USD = Decimal("921.05")
FX_CACHE_MINUTES = 15
QUOTE_CACHE_MINUTES = 15
MAX_RECEIPT_BYTES = 12 * 1024 * 1024


def ensure_data_directories() -> None:
    """Crea solo las carpetas privadas que la aplicacion necesita."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

