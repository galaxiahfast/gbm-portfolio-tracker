"""Point-in-time news-sentiment sidecar for offline replay research.

The signed price replay is immutable. A news score can join it only when the
provider proves that both the article and the *scored sentiment* were
available no later than the emission cut. A historical API response fetched
today is not, on its own, that proof.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Mapping


SENTIMENT_FEATURE_VERSION = "PIT_NEWS_SENTIMENT_24H_V1"
SENTIMENT_SOURCE = "SENTISENSE_DOCUMENTS_NEWS"


def _instant(value):
    if isinstance(value, (int, float)):
        moment = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("News timestamp without timezone.")
    return moment.astimezone(timezone.utc)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def sentiment_at_cut(symbol: str, cut_at, snapshot: Mapping | None):
    """Return explicit MISSING/UNVERIFIED, or a causal score in [-1, 1].

    Snapshot contracts:
      queried_at <= cut_at: actual prospective capture, score availability
        proven by the observed response time; or
      queried_at > cut_at: every retained historical document must carry a
        provider-authenticated sentiment_available_at <= cut_at, and the
        provider must certify complete historical coverage for a true zero.
    """
    cut = _instant(cut_at)
    base = {"version": SENTIMENT_FEATURE_VERSION, "source": SENTIMENT_SOURCE,
            "symbol": symbol, "cut_at": cut.isoformat(), "sentiment_score": None,
            "document_count": 0, "source_sha256": _digest(snapshot) if snapshot is not None else None}
    if snapshot is None:
        return {**base, "status": "NO_SOURCE_SNAPSHOT"}
    if snapshot.get("symbol") != symbol or snapshot.get("source") != SENTIMENT_SOURCE:
        raise ValueError("News snapshot belongs to another symbol/source.")
    queried = _instant(snapshot["queried_at"])
    prospective = queried <= cut
    if not prospective and snapshot.get("pit_coverage_certified") is not True:
        return {**base, "status": "HISTORICAL_AVAILABILITY_UNVERIFIED"}
    if snapshot.get("coverage_complete") is not True:
        return {**base, "status": "INCOMPLETE_NEWS_COVERAGE"}
    docs = snapshot.get("documents")
    if not isinstance(docs, list):
        raise ValueError("News documents must be an array.")
    selected = []
    unavailable_score = False
    for doc in docs:
        if doc.get("source") != "NEWS":
            continue
        published = _instant(doc["published"])
        if published > cut or published < cut - timedelta(hours=24):
            continue
        if prospective:
            score_available = _instant(doc.get("sentiment_available_at", queried))
        else:
            if not doc.get("sentiment_available_at"):
                return {**base, "status": "HISTORICAL_SCORE_TIME_MISSING"}
            score_available = _instant(doc["sentiment_available_at"])
        if score_available > cut:
            unavailable_score = True
            continue
        score = float(doc["averageSentiment"])
        if not math.isfinite(score) or score < -1 or score > 1:
            raise ValueError("News score outside [-1, 1].")
        selected.append(score)
    if unavailable_score:
        return {**base, "status": "NEWS_SCORE_NOT_AVAILABLE_AT_CUT"}
    if not selected:
        return {**base, "status": "VERIFIED_NO_NEWS", "sentiment_score": 0.0}
    return {**base, "status": "VERIFIED_NEWS", "sentiment_score": sum(selected) / len(selected),
            "document_count": len(selected)}


def attach_sentiment_feature(row: Mapping, point: Mapping):
    """Append exactly one feature; fail rather than imputing missing history."""
    if row["symbol"] != point["symbol"] or row["observed_at"] != point["cut_at"]:
        if _instant(row["observed_at"]) != _instant(point["cut_at"]) or row["symbol"] != point["symbol"]:
            raise ValueError("Sentiment sidecar cut mismatch.")
    if point["version"] != SENTIMENT_FEATURE_VERSION or point["status"] not in {
        "VERIFIED_NEWS", "VERIFIED_NO_NEWS"
    }:
        raise ValueError("PIT sentiment unavailable: no 0-imputation allowed.")
    score = point["sentiment_score"]
    if score is None or not math.isfinite(float(score)) or not -1 <= float(score) <= 1:
        raise ValueError("Invalid PIT sentiment score.")
    return {**row, "features": [*row["features"], float(score)],
            "sentiment_feature_version": SENTIMENT_FEATURE_VERSION,
            "sentiment_source_sha256": point["source_sha256"]}
