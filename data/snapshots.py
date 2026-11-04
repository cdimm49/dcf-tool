"""Saved copies of company data, used when live Yahoo Finance data is unavailable.

Yahoo Finance often blocks requests from shared cloud servers (such as Streamlit
Community Cloud). To keep the deployed app usable, `scripts/make_snapshots.py`
saves real data for a set of popular tickers to `data/snapshots/*.json` from a
machine Yahoo doesn't block. The app uses those files only when a live fetch
fails, and always labels them with the date they were saved.

This module has no Streamlit or network code so it is easy to test.
"""

import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
_SCALAR_FIELDS = (
    "company_name",
    "current_price",
    "diluted_shares_outstanding",
    "diluted_shares_source",
    "beta",
    "risk_free_rate",
    "sector",
    "industry",
)
_STATEMENT_FIELDS = ("income_statement", "balance_sheet", "cash_flow_statement")


def _snapshot_path(symbol):
    """Return the file for a ticker, or None if the symbol isn't a plain ticker."""
    cleaned = str(symbol).strip().upper()
    if not _SYMBOL_PATTERN.match(cleaned):
        return None
    return SNAPSHOT_DIR / f"{cleaned}.json"


def _clean_scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy numbers
        value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
    return value


def frame_to_dict(frame):
    """Convert a statement DataFrame to plain JSON-friendly lists."""
    if frame is None:
        return None
    values = frame.astype(object).where(frame.notna(), None)
    return {
        "index": [str(label) for label in frame.index],
        "columns": [
            column.isoformat() if hasattr(column, "isoformat") else str(column)
            for column in frame.columns
        ],
        "data": [[_clean_scalar(cell) for cell in row] for row in values.values],
    }


def dict_to_frame(payload):
    """Rebuild a statement DataFrame (rows = line items, columns = dates)."""
    columns = pd.to_datetime(payload["columns"], errors="coerce")
    if columns.isna().any():
        columns = payload["columns"]
    frame = pd.DataFrame(payload["data"], index=payload["index"], columns=columns)
    return frame.apply(pd.to_numeric, errors="coerce")


def _news_to_json(articles):
    saved = []
    for article in articles or []:
        published = article.get("published")
        saved.append(
            {
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "published": published.isoformat() if published else None,
                "summary": article.get("summary", ""),
                "url": article.get("url", ""),
            }
        )
    return saved


def _news_from_json(saved):
    articles = []
    for article in saved or []:
        published = None
        if article.get("published"):
            try:
                published = datetime.fromisoformat(article["published"])
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
            except ValueError:
                published = None
        articles.append({**article, "published": published})
    return articles


def save_snapshot(symbol, company_data, news=None, as_of=None):
    """Write one ticker's data to data/snapshots/<SYMBOL>.json and return the path."""
    path = _snapshot_path(symbol)
    if path is None:
        raise ValueError(f"Not a valid ticker symbol: {symbol!r}")

    payload = {
        "symbol": str(symbol).strip().upper(),
        "as_of": as_of or date.today().isoformat(),
        "company": {
            field: _clean_scalar(company_data.get(field)) for field in _SCALAR_FIELDS
        },
        "statements": {
            field: frame_to_dict(company_data.get(field)) for field in _STATEMENT_FIELDS
        },
        "news": _news_to_json(news),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    return path


def _read_payload(symbol):
    path = _snapshot_path(symbol)
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_snapshot(symbol):
    """Return saved company data in the same shape as live data, or None."""
    payload = _read_payload(symbol)
    if payload is None:
        return None
    try:
        company_data = dict(payload["company"])
        for field in _STATEMENT_FIELDS:
            company_data[field] = dict_to_frame(payload["statements"][field])
        company_data["snapshot_date"] = payload["as_of"]
    except (KeyError, TypeError, ValueError):
        return None

    if not company_data.get("company_name") or company_data.get("current_price") is None:
        return None
    return company_data


def load_saved_news(symbol):
    """Return (articles, saved_date) from a snapshot, or (None, None)."""
    payload = _read_payload(symbol)
    if payload is None or not payload.get("news"):
        return None, None
    return _news_from_json(payload["news"]), payload.get("as_of")


def available_snapshots():
    """Sorted list of tickers that have a saved snapshot."""
    if not SNAPSHOT_DIR.is_dir():
        return []
    return sorted(path.stem for path in SNAPSHOT_DIR.glob("*.json"))
