"""Score the signed six-date Alpha pilot with frozen FinBERT, exploratorily.

This is descriptive headline scoring, not 1D model training or OOS validation.
Alpha's publication timestamp does not certify first availability. No API
credentials, live decisions, or accounting files are used.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.analytics.headline_sentiment_replay import (
    FINBERT_REPOSITORY, FINBERT_REVISION, FrozenFinBert,
)
from scripts.download_massive_history import read_signed, sha, write_signed
from scripts.pilot_massive_news import cut_utc


SOURCE = ROOT / "output/historical_replay/headline_finbert_alpha_v1/pilot_report.json"
OUT = ROOT / "output/historical_replay/headline_finbert_alpha_exploratory_v1"
REQUIRED_FILES = ("config.json", "pytorch_model.bin", "special_tokens_map.json",
                  "tokenizer_config.json", "vocab.txt")


def select_cut_articles(row: dict, hours: int) -> list[dict]:
    if hours not in (4, 24):
        raise ValueError("Only preregistered 4h/24h windows are allowed")
    cut = cut_utc(row["day"])
    lower = cut - timedelta(hours=hours)
    selected = []
    for article in row["articles"]:
        stamp = datetime.fromisoformat(article["published_utc"])
        if stamp.tzinfo is None:
            raise ValueError("Publication timestamp lacks timezone")
        if lower < stamp.astimezone(timezone.utc) <= cut:
            selected.append(article)
    return selected


def snapshot() -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download

    location = Path(snapshot_download(
        repo_id=FINBERT_REPOSITORY, revision=FINBERT_REVISION,
        allow_patterns=list(REQUIRED_FILES), local_dir=str(OUT / "finbert_snapshot"),
    ))
    hashes = {}
    for name in REQUIRED_FILES:
        path = location / name
        if not path.is_file():
            raise ValueError(f"Missing frozen FinBERT asset: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return location, hashes


def build() -> dict:
    pilot = read_signed(SOURCE)
    if pilot["version"] != "ALPHA_VANTAGE_NEWS_PILOT_V1" or len(pilot["rows"]) != 12:
        raise ValueError("Unexpected signed Alpha pilot")
    # Select only already-published-before-cut titles; no zero imputation.
    title_by_hash = {}
    observations = []
    for row in pilot["rows"]:
        if row["http_status"] != 200 or row["error"] is not None or not row["complete"]:
            raise ValueError("Alpha pilot acquisition was incomplete")
        windows = {}
        for hours in (4, 24):
            selected = select_cut_articles(row, hours)
            ids = []
            for article in selected:
                title = article["title"]
                article_id = hashlib.sha256(title.encode("utf-8")).hexdigest()
                title_by_hash[article_id] = title
                ids.append(article_id)
            windows[str(hours)] = {"article_count": len(selected), "title_ids": ids}
        observations.append({"day": row["day"], "symbol": row["symbol"],
                             "cut_at_utc": cut_utc(row["day"]).isoformat(),
                             "windows": windows})

    location, hashes = snapshot()
    manifest = {"repository": FINBERT_REPOSITORY, "revision": FINBERT_REVISION,
                "files_sha256": hashes, "runtime": "isolated-research-venv",
                "source": "ALPHA_PUBLICATION_PROXY_NOT_CERTIFIED_PIT"}
    write_signed(OUT / "model_manifest.json", manifest)
    scorer = FrozenFinBert(location)
    ids = sorted(title_by_hash)
    scores = scorer.score([title_by_hash[article_id] for article_id in ids])
    score_by_id = dict(zip(ids, scores))
    for observation in observations:
        for hours in (4, 24):
            window = observation["windows"][str(hours)]
            values = [score_by_id[article_id] for article_id in window.pop("title_ids")]
            window["status"] = "AVAILABLE" if values else "N/D"
            window["score"] = sum(values) / len(values) if values else None
    result = {"version": "ALPHA_FINBERT_PILOT_EXPLORATORY_V1",
              "source_report_sha256": sha(pilot), "model_manifest_sha256": sha(manifest),
              "unique_headlines_scored": len(ids),
              "article_cut_observations_24h": sum(x["windows"]["24"]["article_count"]
                                                   for x in observations),
              "cuts": len(observations), "points": observations,
              "missing_semantics": "N/D, never zero-imputed",
              "first_availability_certified": False,
              "walk_forward_eligible": False,
              "production_changed": False, "portfolio_db_accessed": False}
    write_signed(OUT / "pilot_sentiment_sidecar.json", result)
    return {key: value for key, value in result.items() if key != "points"}


if __name__ == "__main__":
    print(build())
