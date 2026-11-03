import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600)
def get_company_data(ticker):
    try:
        symbol = ticker.strip().upper()
        if not symbol:
            return {"error": "Enter a ticker symbol to fetch company data."}

        company = yf.Ticker(symbol)
        income_statement = company.income_stmt
        balance_sheet = company.balance_sheet
        cash_flow_statement = company.cashflow
        try:
            info = company.info or {}
        except Exception:
            info = {}

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

        diluted_shares = None
        try:
            shares = info["sharesOutstanding"]
            if shares is not None and not pd.isna(shares):
                diluted_shares = shares
        except Exception:
            pass

        if diluted_shares is None:
            try:
                shares = company.fast_info["shares"]
                if shares is not None and not pd.isna(shares):
                    diluted_shares = shares
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

        if pd.isna(current_price):
            missing_data.append("current price")
        if not company_name:
            missing_data.append("company name")

        if missing_data:
            return {
                "error": (
                    f"Could not fetch complete data for {symbol}. "
                    f"Missing: {', '.join(missing_data)}."
                )
            }

        return {
            "company_name": company_name,
            "current_price": float(current_price),
            "diluted_shares_outstanding": (
                float(diluted_shares) if diluted_shares is not None else None
            ),
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow_statement": cash_flow_statement,
        }
    except Exception as error:
        return {
            "error": (
                f"Could not fetch data for {ticker!r}. "
                f"Check the ticker and try again. Details: {error}"
            )
        }
