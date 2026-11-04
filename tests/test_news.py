from datetime import datetime, timezone

from data.news import parse_news


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
