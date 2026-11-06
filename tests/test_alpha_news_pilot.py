from scripts.pilot_alpha_news import audit_articles, fetch_articles


def test_alpha_publication_time_and_after_cut_filter():
    result = audit_articles([
        {"title": "Before", "time_published": "20250805T145959"},
        {"title": "After", "time_published": "20250805T151000"},
    ], "SMCI", "2025-08-05")
    assert result["raw_count"] == 2
    assert result["valid_timestamps_and_titles"] == 2
    assert result["eligible_24h_before_cut"] == 1
    assert result["published_after_cut"] == 1


def test_alpha_http_200_rate_limit_is_error_not_empty_news():
    class Response:
        status_code = 200

        def json(self):
            return {"Information": "API key SECRET has reached its daily limit"}

    class Session:
        def get(self, url, *, params, timeout):
            return Response()

    result = fetch_articles(Session(), "SECRET", "SMCI", "2025-08-05")
    assert result["http_status"] == 200
    assert result["complete"] is False
    assert result["error"] == "API key [REDACTED] has reached its daily limit"


def test_alpha_1000_row_response_is_not_treated_as_complete():
    class Response:
        status_code = 200

        def json(self):
            return {"feed": [{"title": "x", "time_published": "20250805T140000"}] * 1000}

    class Session:
        def get(self, url, *, params, timeout):
            return Response()

    result = fetch_articles(Session(), "SECRET", "SMCI", "2025-08-05")
    assert result["complete"] is False
    assert result["error"] == "POSSIBLY_TRUNCATED_AT_1000"
