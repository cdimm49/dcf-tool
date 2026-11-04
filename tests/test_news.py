from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import data.news as news
from data.news import get_recent_news, parse_news


def test_parse_news_supports_old_new_and_broken_items():
    old_published = datetime.fromtimestamp(1_762_200_000, tz=timezone.utc)
    new_published = datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc)
    items = [
        {
            "title": "Older headline",
            "publisher": "Old publisher",
            "link": "https://example.com/old",
            "providerPublishTime": 1_762_200_000,
        },
        {
            "content": {
                "title": "Newer headline",
                "summary": "A short article summary.",
                "pubDate": "2026-11-03T18:00:00Z",
                "provider": {"displayName": "New publisher"},
                "canonicalUrl": {"url": "https://example.com/new"},
            }
        },
        {
            "content": {
                "title": "Missing link",
                "summary": "This entry cannot be opened.",
            }
        },
    ]

    articles = parse_news(items)

    assert articles == [
        {
            "title": "Newer headline",
            "source": "New publisher",
            "published": new_published,
            "summary": "A short article summary.",
            "url": "https://example.com/new",
        },
        {
            "title": "Older headline",
            "source": "Old publisher",
            "published": old_published,
            "summary": "",
            "url": "https://example.com/old",
        },
    ]


def test_parse_news_search_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "news_search_item.json"
    item = json.loads(fixture_path.read_text(encoding="utf-8"))

    articles = parse_news([item])

    assert len(articles) == 1
    article = articles[0]
    assert article["title"] == "Apple: 3 Key Takeaways for Long-Term Investors"
    assert article["source"] == "Motley Fool"
    assert article["url"] == (
        "https://finance.yahoo.com/markets/stocks/articles/"
        "apple-3-key-takeaways-long-004850159.html"
    )
    assert article["published"].year == 2026
    assert article["published"].month == 11


def test_get_recent_news_falls_back_to_ticker(monkeypatch):
    item = {
        "title": "Fallback headline",
        "publisher": "Publisher",
        "link": "https://example.com/fallback",
        "providerPublishTime": 1_793_753_330,
    }
    search = Mock(side_effect=RuntimeError("Search failed"))
    ticker = Mock(return_value=SimpleNamespace(news=[item]))
    monkeypatch.setattr(news.yf, "Search", search)
    monkeypatch.setattr(news.yf, "Ticker", ticker)
    get_recent_news.clear()

    try:
        articles = get_recent_news("AAPL")
    finally:
        get_recent_news.clear()

    search.assert_called_once_with("AAPL", news_count=15)
    ticker.assert_called_once_with("AAPL")
    assert articles[0]["title"] == "Fallback headline"


def test_get_recent_news_does_not_cache_failures(monkeypatch):
    search = Mock(side_effect=RuntimeError("Search failed"))
    ticker = Mock(side_effect=RuntimeError("Ticker failed"))
    monkeypatch.setattr(news.yf, "Search", search)
    monkeypatch.setattr(news.yf, "Ticker", ticker)
    get_recent_news.clear()

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Search error: Search failed"):
                get_recent_news("FAIL")
        assert search.call_count == 2
        assert ticker.call_count == 2
    finally:
        get_recent_news.clear()
