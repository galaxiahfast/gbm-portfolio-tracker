from datetime import datetime, timezone

import pytest

from portfolio_tracker.analytics.headline_sentiment_replay import (
    FINBERT_REVISION, aggregate_at_cut, normalize_benzinga, select_at_cut,
)
from scripts.collect_benzinga_headlines import _cut, _get_page, collect_day


def _article(article_id, created, updated=None, symbol="SMCI", title="Company outlook improves"):
    return normalize_benzinga({
        "id": article_id, "created": created, "updated": updated or created,
        "title": title, "stocks": [{"name": symbol}],
    })


def test_only_prior_unrevised_headlines_are_selected():
    rows = [
        _article(1, "2025-08-05T10:30:00-04:00"),
        _article(2, "2025-08-05T10:55:00-04:00", "2025-08-05T11:01:00-04:00"),
        _article(3, "2025-08-05T11:02:00-04:00"),
        _article(4, "2025-08-05T06:00:00-04:00"),
        _article(5, "2025-08-05T10:45:00-04:00", symbol="NVDA"),
    ]
    cut = "2025-08-05T11:00:00-04:00"
    assert [r["id"] for r in select_at_cut(rows, "SMCI", cut, 4)] == ["1"]
    assert [r["id"] for r in select_at_cut(rows, "SMCI", cut, 24)] == ["4", "1"]


def test_no_headline_is_nd_not_neutral_zero():
    answer = aggregate_at_cut([], {}, "SMCI", "2025-08-05T11:00:00-04:00", 4)
    assert answer["status"] == "N/D"
    assert answer["score"] is None
    assert answer["article_count"] == 0


def test_missing_finbert_score_rejected_and_duplicates_not_counted_twice():
    article = _article(1, "2025-08-05T10:30:00-04:00")
    with pytest.raises(ValueError, match="FinBERT"):
        aggregate_at_cut([article], {}, "SMCI", "2025-08-05T11:00:00-04:00", 4)
    answer = aggregate_at_cut([article, article], {"1": 0.7}, "SMCI",
                              "2025-08-05T11:00:00-04:00", 4)
    assert answer["score"] == 0.7
    assert answer["article_count"] == 1


def test_timezone_required_and_dst_cut_respected():
    with pytest.raises(ValueError, match="timezone"):
        _article(1, "2025-08-05T10:30:00")
    assert _cut("2025-11-05") == datetime(2025, 11, 5, 16, tzinfo=timezone.utc)
    assert _cut("2025-08-05") == datetime(2025, 8, 5, 15, tzinfo=timezone.utc)


def test_model_revision_is_immutable_commit():
    assert len(FINBERT_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in FINBERT_REVISION)


def test_benzinga_failure_does_not_echo_token():
    class FakeResponse:
        status_code = 403

    class FakeSession:
        def get(self, url, **kwargs):
            assert kwargs["params"]["token"] == "sensitive_key"
            return FakeResponse()

    with pytest.raises(RuntimeError) as error:
        _get_page(FakeSession(), "sensitive_key", "SMCI", datetime(2025, 8, 5).date(), 0)
    assert "sensitive_key" not in str(error.value)


def test_pilot_download_paginates_and_retains_signed_headlines(tmp_path):
    class FakeResponse:
        status_code = 200

        def __init__(self, rows):
            self.rows = rows

        def json(self):
            return self.rows

    class FakeSession:
        calls = 0

        def get(self, url, **kwargs):
            page = kwargs["params"]["page"]
            self.calls += 1
            offset = page * 100
            count = 100 if page == 0 else 1
            return FakeResponse([{
                "id": offset + index, "title": f"Headline {offset + index}",
                "created": "2025-08-05T10:00:00-04:00",
                "updated": "2025-08-05T10:00:00-04:00",
                "stocks": [{"name": "SMCI"}],
            } for index in range(count)])

    http = FakeSession()
    result = collect_day(http, "test_key", "SMCI", "2025-08-05", folder=tmp_path)
    assert http.calls == 2
    assert result["normalized_count"] == 101
    assert result["raw_response_sha256"]
    assert collect_day(http, "test_key", "SMCI", "2025-08-05", folder=tmp_path) == result
    assert http.calls == 2
