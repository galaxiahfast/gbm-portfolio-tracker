"""Contrato independiente para validación de toque/cierre de zonas diarias."""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from portfolio_tracker.services.zone_forward import validation_data


@dataclass(frozen=True, slots=True)
class ZoneTouchValidation:
    brier_touch: float | None
    brier_close: float | None
    independent_sessions: int
    model_version: str | None


def load_zone_touch_validation(repository, symbol: str) -> ZoneTouchValidation:
    """Read zone evidence for zone reporting only, never directional EV."""
    try:
        rows = [
            row for row in repository.zone_predictions()
            if row.get("symbol") == symbol.strip().upper() and row.get("integrity_ok")
        ]
    except sqlite3.OperationalError:
        return ZoneTouchValidation(None, None, 0, None)
    resolved = [row for row in rows if row.get("resolved_at")]
    if not resolved:
        return ZoneTouchValidation(None, None, 0, None)
    latest = max(resolved, key=lambda row: row["timestamp_prediction"])
    version = str(latest["model_version_hash"])
    scored = validation_data([row for row in resolved if row["model_version_hash"] == version])
    values, counts = {}, {}
    for event in ("Toque", "Cierre"):
        per_day = {}
        for row in scored:
            if row["event"] == event:
                per_day.setdefault(row["session_date"], []).append(float(row["brier"]))
        values[event] = (
            sum(sum(day) / len(day) for day in per_day.values()) / len(per_day)
            if per_day else None
        )
        counts[event] = len(per_day)
    return ZoneTouchValidation(
        values["Toque"], values["Cierre"], min(counts.values(), default=0), version,
    )
