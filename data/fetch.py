import os
import time

import pandas as pd
import streamlit as st
import yfinance as yf

from data import snapshots

# After a live fetch fails for a ticker we have a snapshot of, skip live attempts
# for this long so visitors aren't made to wait on a blocked connection each time.
BLOCK_SECONDS = 600
_live_blocked_until = 0.0


class LiveDataError(Exception):
    """Live Yahoo Finance data could not be fetched or was incomplete."""


def _get_risk_free_yield():
    try:
        history = yf.Ticker("^TNX").history(period="5d")
        if history is not None and not history.empty:
            value = history["Close"].dropna()
            if not value.empty:
                return float(value.iloc[-1])
    except Exception:
        return None
    return None


def _latest_diluted_shares(income_statement):
    if income_statement is not None:
        share_rows = {
            "dilutedaverageshares",
            "dilutedaveragesharesoutstanding",
            "dilutedshares",
        }
        for row in income_statement.index:
            normalized = "".join(
                character for character in str(row).lower() if character.isalnum()
            )
            if normalized in share_rows:
                values = pd.to_numeric(income_statement.loc[row], errors="coerce").dropna()
                if not values.empty and values.iloc[0] > 0:
                    return float(values.iloc[0]), "Most recent diluted average shares"
    return None, None


def fetch_live(symbol):
    """Fetch company data from Yahoo Finance for an upper-case ticker.

    Raises LiveDataError (or any network error) on failure. It never returns an
    error dict, so a failure can't be mistaken for data and cached.
    """
    company = yf.Ticker(symbol)
    income_statement = company.income_stmt
    balance_sheet = company.balance_sheet
    cash_flow_statement = company.cashflow
    info_error = None
    try:
        info = company.info or {}
    except Exception as error:
        info = {}
        info_error = error

    # Try each price source separately so an unavailable source can fall back.
    current_price = None
    try:
        price = company.fast_info["last_price"]
        if price is not None and not pd.isna(price):
            current_price = price
    except Exception:
        pass

    if current_price is None:
        try:
            history = company.history(period="5d")
            if history is not None and not history.empty:
                price = history["Close"].iloc[-1]
                if price is not None and not pd.isna(price):
                    current_price = price
        except Exception:
            pass

    if current_price is None:
        try:
            price = company.info.get("currentPrice")
            if price is not None and not pd.isna(price):
                current_price = price
        except Exception:
            pass

    diluted_shares, diluted_shares_source = _latest_diluted_shares(income_statement)
    try:
        if diluted_shares is None:
            shares = info["sharesOutstanding"]
            if shares is not None and not pd.isna(shares):
                diluted_shares = shares
                diluted_shares_source = "Most recent yfinance shares outstanding"
    except Exception:
        pass

    if diluted_shares is None:
        try:
            shares = company.fast_info["shares"]
            if shares is not None and not pd.isna(shares):
                diluted_shares = shares
                diluted_shares_source = "Most recent yfinance shares outstanding"
        except Exception:
            pass

    company_name = info.get("longName") or info.get("shortName")
    statements = {
        "income statement": income_statement,
        "balance sheet": balance_sheet,
        "cash flow statement": cash_flow_statement,
    }
    missing_data = [
        name
        for name, statement in statements.items()
        if statement is None or statement.empty
    ]

    if current_price is None or pd.isna(current_price):
        missing_data.append("current price")
    if not company_name:
        missing_data.append("company name")

    if missing_data:
        message = (
            f"Could not fetch complete data for {symbol}. "
            f"Missing: {', '.join(missing_data)}."
        )
        if info_error is not None:
            message = f"{message} Details: {info_error}"
        raise LiveDataError(message)

    return {
        "company_name": company_name,
        "current_price": float(current_price),
        "diluted_shares_outstanding": (
            float(diluted_shares) if diluted_shares is not None else None
        ),
        "diluted_shares_source": diluted_shares_source,
        "beta": info.get("beta"),
        "risk_free_rate": _get_risk_free_yield(),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow_statement": cash_flow_statement,
    }


@st.cache_data(ttl=3600)
def _cached_live(symbol):
    # Failures raise, and Streamlit doesn't cache exceptions, so a temporary
    # Yahoo problem is retried on the next request instead of stuck for an hour.
    return fetch_live(symbol)


def get_company_data(ticker):
    """Return company data, from Yahoo Finance when possible.

    If Yahoo is unreachable and a saved snapshot exists for the ticker, return the
    snapshot (marked with a "snapshot_date" key). Otherwise return {"error": ...}.
    Set the DCF_FORCE_SNAPSHOT environment variable to skip live data for tickers
    that have a snapshot, which is handy for testing the fallback locally.
    """
    global _live_blocked_until

    symbol = ticker.strip().upper() if isinstance(ticker, str) else ""
    if not symbol:
        return {"error": "Enter a ticker symbol to fetch company data."}

    saved = snapshots.load_snapshot(symbol)
    skip_live = saved is not None and (
        bool(os.environ.get("DCF_FORCE_SNAPSHOT"))
        or time.time() < _live_blocked_until
    )

    live_error = None
    if not skip_live:
        try:
            return _cached_live(symbol)
        except Exception as error:
            live_error = error
            if saved is not None:
                # A ticker we hold a snapshot of is a real company, so a failure
                # means Yahoo is unreachable, not that the symbol is wrong.
                _live_blocked_until = time.time() + BLOCK_SECONDS

    if saved is not None:
        return saved

    if isinstance(live_error, LiveDataError):
        message = str(live_error)
    else:
        message = f"Could not fetch data for {symbol!r}. Details: {live_error}"
    return {"error": message}
