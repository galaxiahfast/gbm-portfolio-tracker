from scripts.score_alpha_finbert_pilot import select_cut_articles


def test_alpha_sidecar_uses_publication_proxy_before_cut_only():
    row = {"day": "2025-08-05", "articles": [
        {"title": "older", "published_utc": "2025-08-05T10:30:00+00:00"},
        {"title": "recent", "published_utc": "2025-08-05T14:30:00+00:00"},
        {"title": "future", "published_utc": "2025-08-05T15:05:00+00:00"},
    ]}
    assert [x["title"] for x in select_cut_articles(row, 4)] == ["recent"]
    assert [x["title"] for x in select_cut_articles(row, 24)] == ["older", "recent"]
