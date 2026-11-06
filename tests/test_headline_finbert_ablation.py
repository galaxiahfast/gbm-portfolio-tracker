import pytest

from scripts.evaluate_headline_finbert_trigger import _total_counts, join_news


def _sidecar(score=None, status="N/D"):
    points = [{"symbol": "SMCI", "session_ny": "2025-08-05",
               "cut_at": "2025-08-05T15:00:00+00:00",
               "windows": {"4": {"status": status, "score": score,
                                 "article_count": int(score is not None)},
                           "24": {"status": status, "score": score,
                                  "article_count": int(score is not None)}}}]
    points.extend({"symbol": "NVDA", "session_ny": f"dummy-{n}",
                   "cut_at": "2025-08-05T15:00:00+00:00",
                   "windows": points[0]["windows"]} for n in range(597))
    return {"cuts": 598, "points": points}


def _row():
    return {"symbol": "SMCI", "session_ny": "2025-08-05",
            "observed_at": "2025-08-05T15:00:00+00:00",
            "features": [0.1] * 19, "outcome": "TP_FIRST"}


def test_no_news_excluded_from_both_arms_not_zero_imputed():
    pairs, missing = join_news([_row()], _sidecar(), 24)
    assert pairs == []
    assert missing == {"SMCI": 1}


def test_headline_score_paired_to_identical_base_candidate():
    pairs, missing = join_news([_row()], _sidecar(0.4, "AVAILABLE"), 4)
    assert not missing
    assert len(pairs) == 1
    base, news = pairs[0]
    assert base["features"] == news["features"][:-1]
    assert news["features"][-1] == 0.4
    assert base["outcome"] == news["outcome"]


def test_wrong_timestamp_fails_closed():
    row = _row()
    row["observed_at"] = "2025-08-05T15:05:00+00:00"
    with pytest.raises(ValueError, match="mismatch"):
        join_news([row], _sidecar(0.4, "AVAILABLE"), 24)


def test_trade_totals_keep_classes_separate():
    items = [
        {"long_signals": 2, "possible_ohlc_fills": 1,
         "outcomes": {"TP_FIRST": 1, "SL_FIRST": 0, "TIMEOUT": 0},
         "net_return_sum_hypothetical": 0.01},
        {"long_signals": 1, "possible_ohlc_fills": 1,
         "outcomes": {"TP_FIRST": 0, "SL_FIRST": 1, "TIMEOUT": 0},
         "net_return_sum_hypothetical": -0.02},
    ]
    total = _total_counts(items)
    assert total["long_signals"] == 3
    assert total["possible_ohlc_fills"] == 2
    assert total["outcomes"] == {"TP_FIRST": 1, "SL_FIRST": 1, "TIMEOUT": 0}
