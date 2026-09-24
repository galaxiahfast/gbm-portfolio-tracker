"""Honest display contract for sparse forward evidence by horizon."""
from __future__ import annotations

from collections.abc import Mapping

from portfolio_tracker.analytics.horizon_models import PROFESSIONAL_MINIMUM_SAMPLES
from portfolio_tracker.services.directional_collection import HORIZON_MINUTES


def validation_disclosure(
    counts: Mapping[str, Mapping[str, int]], approved_horizons: set[str] | frozenset[str],
) -> dict[str, object]:
    """An approval requires a separate sealed model; forward rows alone never approve it."""
    required = PROFESSIONAL_MINIMUM_SAMPLES
    rows = []
    for label in HORIZON_MINUTES:
        found = counts.get(label, {})
        eligible = max(0, int(found.get("eligible", 0)))
        resolved = max(0, int(found.get("resolved", 0)))
        approved = label in approved_horizons and eligible >= required
        rows.append({
            "horizon": label, "eligible": eligible, "resolved": resolved,
            "required": required, "approved": approved,
            "status": "APROBADO" if approved else "PRELIMINAR",
        })
    missing = [row for row in rows if not row["approved"]]
    any_approved = any(row["approved"] for row in rows)
    mode = "VALIDADO" if not missing else "RESTRINGIDO" if any_approved else "CONSERVADOR"
    detail = "; ".join(
        f"{row['horizon']}: n={row['eligible']}/{required}, modelo no aprobado "
        f"({row['resolved']} resueltas)"
        for row in missing
    )
    return {
        "mode": mode,
        "preliminary": bool(missing), "rows": tuple(rows),
        "banner": f"PRELIMINAR · modo {mode}. {detail}"
        if missing else "Modelos aprobados por horizonte; mantener filtros de riesgo y ejecución.",
    }
