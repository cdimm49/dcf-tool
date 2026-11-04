from datetime import datetime, timezone

import streamlit as st
import yfinance as yf


def _published_datetime(value):
    if isinstance(value, datetime):
        published = value
    elif isinstance(value, (int, float)):
        try:
            published = datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, TypeError, ValueError):
            return None
    elif isinstance(value, str):
        try:
            published = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if published.tzinfo is None:
        return published.replace(tzinfo=timezone.utc)
    return published.astimezone(timezone.utc)


def _url_value(value):
    if isinstance(value, dict):
        value = value.get("url")
    return value.strip() if isinstance(value, str) else ""


def parse_news(items):
    """Convert old and new yfinance news entries to a consistent format."""
    articles = []
    if not isinstance(items, (list, tuple)):
        return articles

    for item in items:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        content = content if isinstance(content, dict) else item
        title = content.get("title") or item.get("title")
        title = title.strip() if isinstance(title, str) else ""

        url = (
            _url_value(content.get("canonicalUrl"))
            or _url_value(content.get("clickThroughUrl"))
            or _url_value(item.get("link"))
        )
        if not title or not url:
            continue

        provider = content.get("provider")
        if isinstance(provider, dict):
            provider = provider.get("displayName") or provider.get("name")
        source = item.get("publisher") or provider
        source = source.strip() if isinstance(source, str) else "Unknown source"

        summary = content.get("summary") or item.get("summary")
        summary = summary.strip() if isinstance(summary, str) else ""
        published_value = content.get("pubDate") or item.get(
            "providerPublishTime"
        )

        articles.append(
            {
                "title": title,
                "source": source,
                "published": _published_datetime(published_value),
                "summary": summary,
                "url": url,
            }
        )

    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        articles,
        key=lambda article: article["published"] or oldest,
        reverse=True,
    )


@st.cache_data(ttl=900)
def get_recent_news(ticker):
    """Fetch and normalize recent headlines for a ticker."""
    symbol = ticker.strip().upper() if isinstance(ticker, str) else ""
    if not symbol:
        return []
    return parse_news(yf.Ticker(symbol).news)
