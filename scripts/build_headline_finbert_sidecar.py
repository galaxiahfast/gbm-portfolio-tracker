"""Score signed historical headlines with one frozen local FinBERT snapshot.

Run only after the 299-session archive exists. The snapshot is pinned to a
specific Hugging Face commit, all loaded model/tokenizer files are SHA-256
recorded, and the output retains N/D when a cut has no eligible headlines.
No portfolio or live decision state is read or changed.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_tracker.analytics.headline_sentiment_replay import (
    FEATURE_VERSION, FINBERT_REPOSITORY, FINBERT_REVISION, WINDOW_HOURS,
    FrozenFinBert, aggregate_at_cut, canonical_sha256,
)
from scripts.collect_benzinga_headlines import OUTPUT, SYMBOLS, _cut
from scripts.download_massive_history import read_signed, write_signed


REQUIRED_FILES = (
    "config.json", "pytorch_model.bin", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.txt",
)


def _model_snapshot():
    from huggingface_hub import snapshot_download

    location = snapshot_download(repo_id=FINBERT_REPOSITORY, revision=FINBERT_REVISION,
                                 allow_patterns=list(REQUIRED_FILES),
                                 local_dir=str(OUTPUT / "finbert_snapshot"))
    file_hashes = {}
    for name in REQUIRED_FILES:
        path = Path(location) / name
        if not path.is_file():
            raise ValueError(f"Missing frozen FinBERT asset: {name}")
        file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return location, file_hashes


def build():
    archive = read_signed(OUTPUT / "archive_manifest.json")
    if archive["cuts"] != 598 or archive["sessions_per_symbol"] != 299:
        raise ValueError("Incomplete fixed 299-session headline archive")
    unique = {}
    per_cut = {}
    for entry in archive["coverage"]:
        symbol, day = entry["symbol"], entry["session_ny"]
        if symbol not in SYMBOLS or (symbol, day) in per_cut:
            raise ValueError("Duplicate or unexpected news cut")
        payload = read_signed(OUTPUT / "raw" / f"{symbol}_{day}.json")
        if canonical_sha256(payload) != entry["archive_sha256"]:
            raise ValueError("Signed headline archive changed")
        per_cut[symbol, day] = payload["articles"]
        for article in payload["articles"]:
            previous = unique.get(article["id"])
            if previous is not None and previous != article:
                raise ValueError("Article changed between immutable daily snapshots")
            unique[article["id"]] = article
    if len(per_cut) != 598:
        raise ValueError("Missing news cuts")
    location, hashes = _model_snapshot()
    model_manifest = {"repository": FINBERT_REPOSITORY, "revision": FINBERT_REVISION,
                      "files_sha256": hashes, "feature_version": FEATURE_VERSION}
    write_signed(OUTPUT / "finbert_model_manifest.json", model_manifest)
    scorer = FrozenFinBert(location)
    ids = sorted(unique)
    scores = scorer.score([unique[article_id]["title"] for article_id in ids])
    by_id = dict(zip(ids, scores))
    scored = {"feature_version": FEATURE_VERSION,
              "model_manifest_sha256": canonical_sha256(model_manifest),
              "headline_set_sha256": canonical_sha256([unique[article_id] for article_id in ids]),
              "scored_articles": len(by_id), "score_by_article_id": by_id}
    write_signed(OUTPUT / "frozen_article_scores.json", scored)
    points = []
    for symbol, day in sorted(per_cut):
        cut_at = _cut(day)
        windows = {str(hours): aggregate_at_cut(per_cut[symbol, day], by_id, symbol,
                                                cut_at, hours) for hours in WINDOW_HOURS}
        points.append({"symbol": symbol, "session_ny": day,
                       "cut_at": cut_at.isoformat(), "windows": windows})
    result = {"version": FEATURE_VERSION, "archive_sha256": canonical_sha256(archive),
              "article_scores_sha256": canonical_sha256(scored),
              "model_manifest_sha256": canonical_sha256(model_manifest),
              "cuts": len(points), "available_4h": sum(p["windows"]["4"]["status"] == "AVAILABLE" for p in points),
              "available_24h": sum(p["windows"]["24"]["status"] == "AVAILABLE" for p in points),
              "missing_semantics": "N/D, never zero-imputed", "provenance": "PIT_PROXY_NOT_CERTIFIED",
              "points": points, "portfolio_db_accessed": False,
              "generated_at": datetime.now(timezone.utc).isoformat()}
    write_signed(OUTPUT / "signed_cut_sidecar.json", result)
    return {k: v for k, v in result.items() if k != "points"}


if __name__ == "__main__":
    print(build())
