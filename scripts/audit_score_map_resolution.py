"""Read-only six-horizon score-map audit over archived IEX market inputs."""
from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.multi_timeframe import SCORE_MAP_VERSION
from portfolio_tracker.analytics.technical_probability import analyze_probability
from scripts.prospective_archive_common import ARCHIVE, DATA_SOURCE, signed_payload


def audit():
    rows, errors = [], []
    for path in sorted((ARCHIVE / "market_inputs").glob("*/*.json")):
        item = signed_payload(path)
        if item.get("data_source", DATA_SOURCE) != DATA_SOURCE:
            raise ValueError("Fuente mixta en auditoría de score.")
        try:
            intraday = pd.read_json(StringIO(item["intraday_frame"]), orient="table")
            daily = pd.read_json(StringIO(item["daily_frame"]), orient="table")
            cut = datetime.fromisoformat(item["scheduled_cut_at_utc"])
            analysis = analyze_probability(item["symbol"], intraday, daily,
                as_of_time=cut + timedelta(seconds=1), require_fresh=False)
            for projection in analysis.horizon_projections:
                scores = (projection.probability_up, projection.probability_range,
                          projection.probability_down)
                rows.append({"symbol": item["symbol"], "cut": cut.isoformat(),
                             "horizon": projection.label, "scores": scores})
        except (ValueError, RuntimeError, KeyError) as exc:
            errors.append({"market_input": str(path.relative_to(ARCHIVE)),
                           "reason": f"{type(exc).__name__}: {exc}"})
    triples = [tuple(row["scores"]) for row in rows]
    return {"score_map_version": SCORE_MAP_VERSION, "data_source": DATA_SOURCE,
            "rows": len(rows), "distinct_score_triples": len(set(triples)),
            "plateau_70_26_4": sum(triple == (70.0, 26.0, 4.0) for triple in triples),
            "errors": errors}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False))
