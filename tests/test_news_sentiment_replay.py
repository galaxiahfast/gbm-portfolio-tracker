from datetime import datetime, timezone

import numpy as np
import pytest

from portfolio_tracker.analytics.news_sentiment_model_research import (
    NEWS_FEATURE_NAMES, fit_augmented_model, predict_augmented,
)
from portfolio_tracker.analytics.news_sentiment_replay import (
    SENTIMENT_SOURCE, attach_sentiment_feature, sentiment_at_cut,
)
from scripts.download_massive_history import write_signed
from scripts.evaluate_news_sentiment_trigger import _validated_sidecar


CUT = "2026-05-05T15:00:21+00:00"


def _snapshot(*, queried="2026-05-05T14:59:00+00:00", documents=None, **other):
    return {"source": SENTIMENT_SOURCE, "symbol": "SMCI",
            "queried_at": queried, "coverage_complete": True,
            "documents": documents if documents is not None else [], **other}


def test_missing_news_is_not_silently_zero():
    point = sentiment_at_cut("SMCI", CUT, None)
    assert point["status"] == "NO_SOURCE_SNAPSHOT"
    assert point["sentiment_score"] is None
    with pytest.raises(ValueError, match="PIT sentiment unavailable"):
        attach_sentiment_feature({"symbol": "SMCI", "observed_at": CUT,
                                  "features": [0.0] * 19}, point)


def test_historical_response_fetched_later_is_not_point_in_time():
    doc = {"source": "NEWS", "published": "2026-05-05T14:00:00Z",
           "averageSentiment": 0.8}
    point = sentiment_at_cut("SMCI", CUT, _snapshot(
        queried="2026-09-28T18:00:00Z", documents=[doc]))
    assert point["status"] == "HISTORICAL_AVAILABILITY_UNVERIFIED"
    point = sentiment_at_cut("SMCI", CUT, _snapshot(
        queried="2026-09-28T18:00:00Z", documents=[doc],
        pit_coverage_certified=True))
    assert point["status"] == "HISTORICAL_SCORE_TIME_MISSING"


def test_closed_cut_filters_future_news_and_preserves_true_no_news_zero():
    valid = {"source": "NEWS", "published": "2026-05-05T14:00:00Z",
             "averageSentiment": -0.5}
    future = {"source": "NEWS", "published": "2026-05-05T15:10:00Z",
              "averageSentiment": 1.0}
    point = sentiment_at_cut("SMCI", CUT, _snapshot(documents=[valid, future]))
    assert point["status"] == "VERIFIED_NEWS"
    assert point["sentiment_score"] == -0.5
    assert point["document_count"] == 1
    no_news = sentiment_at_cut("SMCI", CUT, _snapshot(documents=[]))
    assert no_news["status"] == "VERIFIED_NO_NEWS"
    assert no_news["sentiment_score"] == 0.0


def test_score_created_after_cut_is_vetoed_even_if_article_was_earlier():
    doc = {"source": "NEWS", "published": "2026-05-05T14:00:00Z",
           "sentiment_available_at": "2026-05-05T15:05:00Z",
           "averageSentiment": 0.9}
    point = sentiment_at_cut("SMCI", CUT, _snapshot(documents=[doc]))
    assert point["status"] == "NEWS_SCORE_NOT_AVAILABLE_AT_CUT"
    assert point["sentiment_score"] is None


def test_attach_appends_exactly_one_verified_feature():
    point = sentiment_at_cut("SMCI", CUT, _snapshot(documents=[
        {"source": "NEWS", "published": "2026-05-05T14:00:00Z",
         "averageSentiment": 0.4}]))
    row = {"symbol": "SMCI", "observed_at": CUT, "features": [0.1] * 19}
    attached = attach_sentiment_feature(row, point)
    assert len(attached["features"]) == len(NEWS_FEATURE_NAMES) == 20
    assert attached["features"][-1] == pytest.approx(0.4)
    assert len(row["features"]) == 19


def test_ablation_fails_closed_without_598_verified_cuts(tmp_path):
    path = tmp_path / "coverage.json"
    write_signed(path, {"cuts_total": 598, "valid_pit_scores": 0,
                        "training_allowed": False, "points": []})
    with pytest.raises(ValueError, match="training forbidden"):
        _validated_sidecar(path, [])


def test_augmented_model_uses_same_three_class_logit_with_20_features():
    classes = ("TP_FIRST", "SL_FIRST", "TIMEOUT")
    rows = []
    for index in range(60):
        category = classes[index % 3]
        vector = [float(index % 7) / 7] * 19 + [(-1, 0, 1)[index % 3] * 0.6]
        rows.append({"features": vector, "outcome": category,
                     "net_return": (.01, -.02, -.003)[index % 3]})
    model = fit_augmented_model(rows, l2=1.0)
    p = predict_augmented(model, rows[:3])
    assert p.shape == (3, 3)
    assert np.allclose(p.sum(axis=1), 1)
    assert model["training_n"] == 60
