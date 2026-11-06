"""Offline, conservative headline selection for the fixed 11:00 NY replay.

This module is never imported by the live decision path. Benzinga ``created``
is a provenance *proxy*, not proof of exchange-time delivery. An article whose
latest title was edited after the cut cannot be reconstructed and is excluded.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math


FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
FINBERT_REPOSITORY = "ProsusAI/finbert"
WINDOW_HOURS = (4, 24)
FEATURE_VERSION = "BENZINGA_HEADLINE_FINBERT_PIT_PROXY_V1"


def _utc(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing headline timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("Headline timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def normalize_benzinga(article):
    """Keep only fields needed for audit; never trust a query ticker alone."""
    title = article.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Article has no headline")
    created = _utc(article.get("created"))
    updated = _utc(article.get("updated"))
    if updated < created:
        raise ValueError("Article updated before creation")
    symbols = sorted({str(item.get("name", "")).upper() for item in article.get("stocks", [])
                      if isinstance(item, dict) and item.get("name")})
    if not symbols:
        raise ValueError("Article has no ticker attribution")
    return {
        "id": str(article["id"]), "title": title.strip(),
        "created_at": created.isoformat(), "updated_at": updated.isoformat(),
        "tickers": symbols,
    }


def select_at_cut(articles, symbol, cut_at, hours):
    """Select *current immutable title versions* within a trailing window.

    A valid empty list is N/D, never a neutral score of zero. Completeness of
    archive acquisition must be checked separately before calling this.
    """
    if hours not in WINDOW_HOURS:
        raise ValueError("Unregistered headline window")
    cut = _utc(cut_at) if isinstance(cut_at, str) else cut_at.astimezone(timezone.utc)
    lower = cut - timedelta(hours=hours)
    selected = {}
    for item in articles:
        if symbol.upper() not in item["tickers"]:
            continue
        created, updated = _utc(item["created_at"]), _utc(item["updated_at"])
        if lower < created <= cut and updated <= cut:
            previous = selected.get(item["id"])
            if previous is not None and previous != item:
                raise ValueError("Conflicting versions of the same article")
            selected[item["id"]] = item
    return sorted(selected.values(), key=lambda item: (item["created_at"], item["id"]))


def aggregate_at_cut(articles, scores_by_article_id, symbol, cut_at, hours):
    selected = select_at_cut(articles, symbol, cut_at, hours)
    if not selected:
        return {"status": "N/D", "score": None, "article_count": 0,
                "article_ids": [], "window_hours": hours}
    scores = []
    for item in selected:
        value = scores_by_article_id.get(item["id"])
        if value is None or not math.isfinite(float(value)) or abs(float(value)) > 1:
            raise ValueError("Missing or invalid frozen FinBERT score")
        scores.append(float(value))
    return {"status": "AVAILABLE", "score": sum(scores) / len(scores),
            "article_count": len(scores), "article_ids": [item["id"] for item in selected],
            "window_hours": hours}


class FrozenFinBert:
    """Lazy inference; no model package/download needed for provenance pilots."""

    def __init__(self, snapshot_path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            snapshot_path, local_files_only=True, trust_remote_code=False).eval()
        self.labels = {int(index): label.lower() for index, label in self.model.config.id2label.items()}
        if set(self.labels.values()) != {"positive", "neutral", "negative"}:
            raise ValueError("Unexpected FinBERT label mapping")

    def score(self, headlines):
        if not headlines:
            return []
        output = []
        for start in range(0, len(headlines), 32):
            batch = headlines[start:start + 32]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**tokens).logits, dim=-1)
            positive = next(index for index, label in self.labels.items() if label == "positive")
            negative = next(index for index, label in self.labels.items() if label == "negative")
            output.extend((probabilities[:, positive] - probabilities[:, negative]).tolist())
        return [float(value) for value in output]
