from datetime import datetime, timezone

from scripts.collect_finnhub_headlines import (
    _cut, audit_articles, fetch, pilot_dates,
)


def _article(timestamp, *, related="SMCI", headline="Shares rise", article_id=1):
    return {"id": article_id, "datetime": timestamp, "headline": headline,
            "related": related, "source": "Example"}


def test_six_dates_are_fixed_before_provider_response():
    assert pilot_dates() == (
        "2025-08-05", "2025-09-15", "2026-09-24",
        "2026-02-20", "2026-04-13", "2026-07-02",
    )


def test_cut_is_dst_aware_and_unix_time_is_utc():
    assert _cut("2025-08-05") == datetime(2025, 8, 5, 15, tzinfo=timezone.utc)
    assert _cut("2025-11-05") == datetime(2025, 11, 5, 16, tzinfo=timezone.utc)


def test_after_cut_excluded_and_missing_not_zero_imputed():
    before = int(datetime(2025, 8, 5, 14, 30, tzinfo=timezone.utc).timestamp())
    after = int(datetime(2025, 8, 5, 15, 10, tzinfo=timezone.utc).timestamp())
    result = audit_articles([_article(before), _article(after, article_id=2)],
                            "SMCI", "2025-08-05")
    assert result["raw_count"] == 2
    assert result["valid_timestamps"] == 2
    assert result["eligible_24h_before_cut"] == 1
    assert result["published_after_cut"] == 1
    assert audit_articles([], "SMCI", "2025-08-05")["eligible_24h_before_cut"] == 0


def test_timestamp_and_related_ticker_problems_counted():
    outside = int(datetime(2025, 8, 1, 15, tzinfo=timezone.utc).timestamp())
    result = audit_articles([_article(outside, related="NVDA"),
                             _article("not-a-unix-timestamp", article_id=2)],
                            "SMCI", "2025-08-05")
    assert result["timestamp_outside_requested_dates"] == 1
    assert result["related_ticker_mismatch"] == 1
    assert result["invalid_timestamps_or_headlines"] == 1


def test_utc_midnight_is_not_a_date_filter_violation():
    # The provider queries calendar dates, while an article just after UTC
    # midnight may still belong to the prior New York evening.
    stamp = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
    result = audit_articles([_article(stamp)], "SMCI", "2026-07-02")
    assert result["valid_timestamps"] == 1
    assert result["timestamp_outside_requested_dates"] == 0


def test_non_200_provider_error_is_reported_without_token():
    class FakeResponse:
        status_code = 403

        def json(self):
            return {"error": "key PRIVATE_KEY has no access"}

    class FakeSession:
        def get(self, url, **kwargs):
            assert kwargs["params"]["token"] == "PRIVATE_KEY"
            return FakeResponse()

    result = fetch(FakeSession(), "PRIVATE_KEY", "SMCI", "2025-08-05")
    assert result["http_status"] == 403
    assert result["error"] == "key [REDACTED] has no access"
    assert "PRIVATE_KEY" not in str(result)
