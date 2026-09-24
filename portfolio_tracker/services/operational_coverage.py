"""Coverage of independent first-passage events, not horizon-row wins.

The signed six-horizon resolutions remain untouched. Horizons with the same
entry/barriers belong to one hypothetical trade; their timeouts are separate
views of that trade, never independent fills or independent successes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def summarize_operational_events(records: Iterable[Mapping]) -> dict[str, int | float | None]:
    groups: dict[tuple, list[Mapping]] = {}
    horizon_resolutions = 0
    verified_cuts = 0
    for record in records:
        verified_cuts += 1
        for prediction in record.get("predictions", ()):
            target = prediction.get("operational_target")
            result = prediction.get("operational_result")
            if not isinstance(target, Mapping) or not isinstance(result, Mapping):
                continue
            key = (
                record.get("symbol"), target.get("entry_at"), target.get("side"),
                target.get("entry_price"), target.get("stop_loss"), target.get("take_profit"),
            )
            groups.setdefault(key, []).append(result)
            horizon_resolutions += result.get("resolution_status") == "RESOLVED"
    counts = {"TP_FIRST": 0, "SL_FIRST": 0, "TIMEOUT": 0, "PENDING": 0}
    for results in groups.values():
        barrier_hits = [r for r in results if r.get("outcome") in ("TP_FIRST", "SL_FIRST")]
        if barrier_hits:
            first_at = min(str(r["exit_at"]) for r in barrier_hits)
            first_hits = [r for r in barrier_hits if str(r["exit_at"]) == first_at]
            # A same-bar conflict is conservatively a loss, never an extra win.
            outcome = "SL_FIRST" if any(r["outcome"] == "SL_FIRST" for r in first_hits) else "TP_FIRST"
        elif all(r.get("resolution_status") == "RESOLVED" for r in results):
            outcome = "TIMEOUT"
        else:
            outcome = "PENDING"
        counts[outcome] += 1
    resolved = sum(counts[k] for k in ("TP_FIRST", "SL_FIRST", "TIMEOUT"))
    return {
        "verified_cuts": verified_cuts, "independent_events": len(groups),
        "horizon_resolutions": horizon_resolutions,
        "resolved_events": resolved, "tp_first_events": counts["TP_FIRST"],
        "sl_first_events": counts["SL_FIRST"], "timeout_events": counts["TIMEOUT"],
        "pending_events": counts["PENDING"],
        "tp_first_rate": counts["TP_FIRST"] / resolved if resolved else None,
    }
