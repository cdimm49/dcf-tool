import os
import time

import pandas as pd
import streamlit as st
import yfinance as yf

from data import sec, snapshots

# Once Yahoo fails for a ticker that turns out to be real (SEC or a snapshot has
# it), skip Yahoo statements for this long so visitors don't wait on a blocked
# connection every time.
BLOCK_SECONDS = 600
# How long a price and Treasury yield are reused on the SEC path.
QUOTE_TTL_SECONDS = 600
_live_blocked_until = 0.0
SEC_SOURCE_NOTE = (
    "Financial statements from SEC 10-K filings; price and beta from Yahoo Finance."
)


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


def _sec_user_agent():
    """The SEC asks for a User-Agent with contact details, e.g. "DCF Tool you@email.com"."""
    agent = os.environ.get("SEC_USER_AGENT")
    if agent:
        return agent.strip()
    try:
        agent = st.secrets.get("SEC_USER_AGENT")
    except Exception:  # no secrets file
        return None
    return agent.strip() if isinstance(agent, str) and agent.strip() else None


def _latest_close(symbol):
    history = yf.Ticker(symbol).history(period="5d")
    if history is None or history.empty:
        return None
    closes = history["Close"].dropna()
    return float(closes.iloc[-1]) if not closes.empty else None


def estimate_beta(stock_closes, market_closes, minimum_months=24):
    """Beta from monthly closing prices: cov(stock, market) / var(market)."""
    frame = pd.concat(
        [pd.Series(stock_closes, dtype="float64"), pd.Series(market_closes, dtype="float64")],
        axis=1,
        join="inner",
    ).dropna()
    returns = frame.pct_change(fill_method=None).dropna()
    if len(returns) < minimum_months:
        return None
    market_variance = returns.iloc[:, 1].var()
    if not market_variance or pd.isna(market_variance):
        return None
    beta = returns.iloc[:, 0].cov(returns.iloc[:, 1]) / market_variance
    return round(float(beta), 2) if not pd.isna(beta) else None


def _monthly_closes(symbol):
    history = yf.Ticker(symbol).history(period="5y", interval="1mo")
    if history is None or history.empty:
        return None
    closes = history["Close"].dropna()
    # Align on month, since index timestamps can differ slightly between symbols.
    closes.index = pd.to_datetime(closes.index).tz_localize(None).to_period("M")
    return closes[~closes.index.duplicated(keep="last")]


def _yahoo_beta(symbol):
    try:
        stock, market = _monthly_closes(symbol), _monthly_closes("^GSPC")
        if stock is None or market is None:
            return None
        return estimate_beta(stock, market)
    except Exception:
        return None


def _sec_statement_data(symbol, user_agent):
    """SEC statements plus a beta estimate. Both change slowly, so they're cached
    for hours."""
    filings = sec.fetch_company(symbol, user_agent)
    beta = _yahoo_beta(symbol)
    return {**filings, "beta": beta}


def _quote(symbol):
    """Latest close and Treasury yield. These move during the day, so they're
    cached for minutes, not hours."""
    try:
        current_price = _latest_close(symbol)
    except Exception as error:
        raise LiveDataError(f"Price for {symbol} is unavailable: {error}") from error
    if current_price is None:
        raise LiveDataError(f"Price for {symbol} is unavailable.")
    return {"current_price": current_price, "risk_free_rate": _get_risk_free_yield()}


def _combine_sec(statement_data, quote):
    diluted_shares, diluted_shares_source = _latest_diluted_shares(
        statement_data["income_statement"]
    )
    if diluted_shares_source:
        diluted_shares_source += " (SEC 10-K)"
    elif statement_data.get("shares_outstanding"):
        diluted_shares = statement_data["shares_outstanding"]
        diluted_shares_source = "Shares outstanding from the latest SEC filing's cover page"
    beta = statement_data["beta"]

    return {
        "company_name": statement_data["company_name"],
        "current_price": quote["current_price"],
        "diluted_shares_outstanding": diluted_shares,
        "diluted_shares_source": diluted_shares_source,
        "beta": beta,
        "beta_source": (
            "Raw beta estimated from 5 years of monthly returns against the S&P 500."
            if beta is not None
            else None
        ),
        "risk_free_rate": quote["risk_free_rate"],
        "sector": statement_data["sector"],
        "industry": statement_data["industry"],
        "income_statement": statement_data["income_statement"],
        "balance_sheet": statement_data["balance_sheet"],
        "cash_flow_statement": statement_data["cash_flow_statement"],
        "data_source": "sec",
    }


def fetch_sec_hybrid(symbol, user_agent):
    """Statements from SEC filings plus price, beta and Treasury yield from Yahoo's
    price data, which Yahoo still serves to cloud servers."""
    statement_data = _sec_statement_data(symbol, user_agent)
    return _combine_sec(statement_data, _quote(symbol))


@st.cache_data(ttl=6 * 3600)
def _cached_sec_statements(symbol):
    return _sec_statement_data(symbol, _sec_user_agent())


@st.cache_data(ttl=QUOTE_TTL_SECONDS)
def _cached_quote(symbol):
    return _quote(symbol)


def _cached_sec(symbol):
    # Statements first, so an unknown ticker fails before any price request.
    statement_data = _cached_sec_statements(symbol)
    return _combine_sec(statement_data, _cached_quote(symbol))


def get_company_data(ticker):
    """Return company data from the best source that works.

    1. Yahoo Finance (everything).
    2. SEC filings for statements plus Yahoo prices, when SEC_USER_AGENT is set.
       Yahoo blocks statement requests from cloud servers but not prices.
    3. A saved snapshot (marked with a "snapshot_date" key).
    Otherwise return {"error": ...}.

    Testing switches: DCF_SKIP_YAHOO=1 imitates the cloud block by skipping step 1;
    DCF_FORCE_SNAPSHOT=1 goes straight to step 3 for tickers that have a snapshot.
    """
    global _live_blocked_until

    symbol = ticker.strip().upper() if isinstance(ticker, str) else ""
    if not symbol:
        return {"error": "Enter a ticker symbol to fetch company data."}

    saved = snapshots.load_snapshot(symbol)
    if saved is not None and os.environ.get("DCF_FORCE_SNAPSHOT"):
        return saved

    user_agent = _sec_user_agent()
    have_fallback = bool(user_agent) or saved is not None
    skip_yahoo = bool(os.environ.get("DCF_SKIP_YAHOO")) or (
        have_fallback and time.time() < _live_blocked_until
    )

    errors = []
    yahoo_failed = False
    if not skip_yahoo:
        try:
            return _cached_live(symbol)
        except Exception as error:
            errors.append(error)
            yahoo_failed = True

    if user_agent:
        try:
            data = _cached_sec(symbol)
        except Exception as error:
            errors.append(error)
        else:
            if yahoo_failed:
                # The SEC knows this company, so Yahoo is the problem, not the symbol.
                _live_blocked_until = time.time() + BLOCK_SECONDS
            return data

    if saved is not None:
        if yahoo_failed:
            _live_blocked_until = time.time() + BLOCK_SECONDS
        return saved

    messages = []
    for error in errors:
        if isinstance(error, (LiveDataError, sec.SecDataError)):
            messages.append(str(error))
        else:
            messages.append(f"Could not fetch data for {symbol!r}. Details: {error}")
    return {"error": " | ".join(messages) or f"No data source is available for {symbol}."}
