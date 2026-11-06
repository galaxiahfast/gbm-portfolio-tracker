"""Build an immutable, fail-closed news sidecar for 598 signed 11:00 cuts.

This does not edit the original replay or train a model. An optional signed
snapshot archive must contain one record per cut with provenance timestamps.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.news_sentiment_replay import sentiment_at_cut
from scripts.download_massive_history import read_signed, write_signed


REPLAY_DIR = ROOT / "output/historical_replay/pilot_1h_1d"
OUTPUT = ROOT / "output/historical_replay/news_sentiment_v1"


def cuts():
    result = []
    for symbol in ("SMCI", "NVDA"):
        paths = list(REPLAY_DIR.glob(symbol + "_*.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one signed {symbol} replay.")
        replay = json.loads(paths[0].read_text(encoding="utf-8"))
        if replay["symbol"] != symbol or len(replay["observations"]) != 299:
            raise ValueError(f"Expected 299 fixed {symbol} cuts.")
        result.extend({"cut_id": cut["cut_id"], "symbol": symbol,
                       "observed_at": cut["observed_at"]}
                      for cut in replay["observations"])
    if len({cut["cut_id"] for cut in result}) != 598:
        raise ValueError("Duplicate replay cut id.")
    return result


def build(snapshot_path: Path | None):
    archive = read_signed(snapshot_path) if snapshot_path else {"snapshots": []}
    snapshots = archive.get("snapshots", [])
    by_cut = {record["cut_id"]: record for record in snapshots}
    if len(by_cut) != len(snapshots):
        raise ValueError("Duplicate news snapshot cut id.")
    all_cuts = cuts()
    if set(by_cut) - {cut["cut_id"] for cut in all_cuts}:
        raise ValueError("News snapshot references an unknown cut.")
    points = []
    for cut in all_cuts:
        snapshot = by_cut.get(cut["cut_id"])
        point = sentiment_at_cut(cut["symbol"], cut["observed_at"], snapshot)
        points.append({"cut_id": cut["cut_id"], **point})
    counts = dict(Counter(point["status"] for point in points))
    eligible = sum(point["sentiment_score"] is not None for point in points)
    source_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest() if snapshot_path else "NO_ARCHIVE"
    result = {"version": "NEWS_SENTIMENT_REPLAY_COVERAGE_V1",
              "source_archive_sha256": source_digest,
              "cuts_total": len(points), "valid_pit_scores": eligible,
              "status_counts": counts,
              "training_allowed": eligible == len(points),
              "points": points}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = "coverage_" + source_digest[:16] + ".json"
    destination = OUTPUT / name
    if destination.exists():
        previous = read_signed(destination)
        if {k: v for k, v in previous.items() if k != "generated_at"} != result:
            raise ValueError("Existing signed news coverage differs from source.")
        result = previous
    else:
        write_signed(destination, result)
    summary = ["# Cobertura PIT de sentimiento para el replay", "",
               "Fuente elegida: SentiSense Documents/News. El score histórico debe estar "
               "disponible en el instante del corte, no solo tener publicación anterior.",
               f"Cortes: {len(points)}; scores PIT válidos: {eligible}; entrenamiento permitido: {eligible == len(points)}.",
               f"Estados: {counts}.",
               "No se ha imputado N/D como cero, no se han alterado TP/SL ni costes, y no se ha abierto holdout."]
    (OUTPUT / ("reporte_" + source_digest[:16] + ".md")).write_text(
        "\n".join(summary) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-file", type=Path)
    args = parser.parse_args()
    result = build(args.snapshot_file)
    print("PIT_NEWS_COVERAGE", result["cuts_total"], result["valid_pit_scores"],
          result["status_counts"], "training_allowed=" + str(result["training_allowed"]))
