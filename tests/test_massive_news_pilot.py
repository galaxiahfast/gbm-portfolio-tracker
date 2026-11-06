"""News pilot audit tests; all data and HTTP responses are synthetic."""
from datetime import datetime, timezone

from scripts.pilot_massive_news import (
    _next_params, audit_articles, cut_utc, fetch_articles,
)


def _row(stamp, *, symbol="SMCI", title="Raw headline", item_id="1"):
    return {"id": item_id, "published_utc": stamp, "title": title,
            "tickers": [symbol], "publisher": {"name": "Example"}}


def test_fixed_new_york_cut_is_dst_aware():
    assert cut_utc("2025-08-05") == datetime(2025, 8, 5, 15, tzinfo=timezone.utc)
    assert cut_utc("2025-11-05") == datetime(2025, 11, 5, 16, tzinfo=timezone.utc)


def test_after_cut_excluded_from_24_hour_window():
    result = audit_articles([
        _row("2025-08-05T14:59:59.999999999Z"),
        _row("2025-08-05T15:10:00Z", item_id="2"),
    ], "SMCI", "2025-08-05")
    assert result["valid_timestamps_and_titles"] == 2
    assert result["eligible_24h_before_cut"] == 1
    assert result["published_after_cut"] == 1


def test_invalid_timestamp_and_ticker_are_reported():
    result = audit_articles([
        _row("not-a-timestamp"),
        _row("2025-08-05T14:00:00Z", symbol="NVDA", item_id="2"),
    ], "SMCI", "2025-08-05")
    assert result["invalid_timestamps_or_titles"] == 1
    assert result["ticker_mismatch"] == 1


def test_provider_key_never_in_result_or_pagination_params():
    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    class Session:
        def __init__(self):
            self.calls = []

        def get(self, url, *, params, timeout):
            self.calls.append((url, params))
            if len(self.calls) == 1:
                return Response({"results": [_row("2025-08-05T14:00:00Z")],
                                 "next_url": "https://api.massive.com/v2/reference/news?cursor=xyz&apiKey=SECRET"})
            return Response({"results": []})

    session = Session()
    result = fetch_articles(session, "SECRET", "SMCI", "2025-08-05", sleep=lambda _: None)
    assert result["complete"] is True and result["pages"] == 2
    assert "SECRET" not in str(result)
    assert session.calls[1][1]["cursor"] == "xyz"
    assert _next_params("https://api.massive.com/v2/reference/news?cursor=x&apiKey=SECRET") == {"cursor": "x"}


def test_http_error_has_status_without_body_or_secret():
    class Response:
        status_code = 403

    class Session:
        def get(self, url, *, params, timeout):
            return Response()

    result = fetch_articles(Session(), "SECRET", "SMCI", "2025-08-05")
    assert result["http_status"] == 403
    assert result["error"] == "HTTP_403"
    assert "SECRET" not in str(result)


def test_http_200_provider_denial_does_not_masquerade_as_empty_news():
    class Response:
        status_code = 200

        def json(self):
            return {"status": "ERROR", "message": "Key SECRET lacks news access"}

    class Session:
        def get(self, url, *, params, timeout):
            return Response()

    result = fetch_articles(Session(), "SECRET", "SMCI", "2025-08-05")
    assert result["complete"] is False
    assert result["error"] == "Key [REDACTED] lacks news access"
