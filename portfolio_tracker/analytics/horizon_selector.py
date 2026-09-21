"""Select a horizon only when it has enough independent validation evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math

from portfolio_tracker.analytics.directional_probability import (
    MIN_EFFECTIVE_SAMPLES,
    MIN_HOLDOUT_SAMPLES,
    assess_directional_projection,
)


@dataclass(frozen=True, slots=True)
class HorizonSelection:
    label: str
    direction: str
    directional_edge: float
    samples: int
    projection: object


def _field(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def select_best_horizon(
    scores_dict,
    *,
    min_validated_samples=MIN_EFFECTIVE_SAMPLES,
    min_holdout_samples=MIN_HOLDOUT_SAMPLES,
    direction="AUTO",
):
    """Choose a dominant, calibrated class that beats its OOS baseline."""
    direction = str(direction).upper()
    if direction not in {"AUTO", "LONG", "SHORT"}:
        raise ValueError("direction debe ser AUTO, LONG o SHORT.")
    items = scores_dict.items() if isinstance(scores_dict, dict) else (
        (_field(item, "label", "N/D"), item) for item in scores_dict
    )
    candidates = []
    for label, item in items:
        try:
            up = float(_field(item, "probability_up"))
            down = float(_field(item, "probability_down"))
            samples = int(_field(item, "calibration_holdout_samples", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in (up, down)):
            continue
        evidence = assess_directional_projection(
            item,
            minimum_effective_samples=int(min_validated_samples),
            minimum_holdout_samples=int(min_holdout_samples),
        )
        if not evidence.eligible:
            continue
        edge = up - down
        selected_direction = "LONG" if evidence.dominant_class == "UP" else "SHORT"
        if direction == "LONG" and edge <= 0 or direction == "SHORT" and edge >= 0:
            continue
        rank = abs(edge) if direction == "AUTO" else edge if direction == "LONG" else -edge
        candidates.append((rank, str(label), selected_direction, edge, samples, item))
    if not candidates:
        return None
    _, label, selected_direction, edge, samples, item = max(candidates, key=lambda row: row[0])
    return HorizonSelection(label, selected_direction, edge, samples, item)
