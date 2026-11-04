import math

import pandas as pd


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _latest_value(statement, row):
    if statement is None or row not in statement.index:
        return None

    values = statement.loc[row]
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    values = pd.to_numeric(values, errors="coerce").dropna()
    return _number(values.iloc[0]) if not values.empty else None


def _cash_from_balance_sheet(balance_sheet, include_securities):
    cash = _latest_value(balance_sheet, "Cash")
    combined_cash = _latest_value(
        balance_sheet, "Cash and short-term investments total"
    )
    if cash is not None and cash < 0:
        return None
    if combined_cash is not None and combined_cash < 0:
        return None
    if cash is None and combined_cash is None:
        return None
    cash_uses_combined_total = cash is None and combined_cash is not None
    if cash_uses_combined_total:
        cash = combined_cash

    short_term_securities = _latest_value(
        balance_sheet, "Short-term marketable securities"
    )
    if include_securities and short_term_securities is not None:
        if short_term_securities < 0:
            return None
    if cash_uses_combined_total:
        short_term_securities = 0.0
    elif short_term_securities is None and combined_cash is not None:
        short_term_securities = max(combined_cash - cash, 0.0)
    short_term_securities = (
        short_term_securities if short_term_securities is not None else 0.0
    )

    long_term_securities = _latest_value(
        balance_sheet, "Long-term marketable securities"
    )
    if include_securities and long_term_securities is not None:
        if long_term_securities < 0:
            return None
    long_term_securities = (
        long_term_securities if long_term_securities is not None else 0.0
    )
    if include_securities:
        cash += short_term_securities + long_term_securities
    return cash


def calculate_multiples(
    income_statement,
    balance_sheet,
    cash_flow_statement,
    market_data,
    include_securities=True,
):
    """Calculate trailing valuation multiples from cleaned USD-million statements."""
    market_data = market_data or {}
    price = _number(market_data.get("current_price"))
    shares = _number(market_data.get("diluted_shares_outstanding"))
    market_cap = (
        price * shares / 1_000_000
        if price is not None and price > 0 and shares is not None and shares > 0
        else None
    )

    diluted_eps = _latest_value(income_statement, "Diluted EPS")
    pe = (
        price / diluted_eps
        if price is not None and price > 0 and diluted_eps is not None and diluted_eps > 0
        else float("nan")
    )

    debt = _latest_value(balance_sheet, "Total Debt")
    cash = _cash_from_balance_sheet(balance_sheet, include_securities)
    enterprise_value = (
        market_cap + debt - cash
        if market_cap is not None
        and debt is not None
        and debt >= 0
        and cash is not None
        and cash >= 0
        and market_cap + debt - cash > 0
        else None
    )

    ebitda = _latest_value(income_statement, "EBITDA")
    revenue = _latest_value(income_statement, "Revenue")
    equity = _latest_value(balance_sheet, "Shareholders’ Equity")

    free_cash_flow = _latest_value(cash_flow_statement, "Free Cash Flow")
    if free_cash_flow is None:
        operating_cash_flow = _latest_value(
            cash_flow_statement, "Operating Cash Flow"
        )
        capex = _latest_value(cash_flow_statement, "CapEx")
        if operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow - abs(capex)

    return {
        "P/E": pe,
        "EV/EBITDA": (
            enterprise_value / ebitda
            if enterprise_value is not None and ebitda is not None and ebitda > 0
            else float("nan")
        ),
        "EV/Sales": (
            enterprise_value / revenue
            if enterprise_value is not None and revenue is not None and revenue > 0
            else float("nan")
        ),
        "P/B": (
            market_cap / equity
            if market_cap is not None and equity is not None and equity > 0
            else float("nan")
        ),
        "FCF yield": (
            free_cash_flow / market_cap
            if market_cap is not None
            and free_cash_flow is not None
            and free_cash_flow >= 0
            else float("nan")
        ),
    }
