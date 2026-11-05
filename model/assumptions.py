import math
import re

import pandas as pd

from model.wacc import adjusted_beta, cost_of_equity, wacc


DCF_INPUTS = (
    "base_revenue",
    "growth_rates",
    "ebit_margin",
    "tax_rate",
    "da_pct_revenue",
    "capex_pct_revenue",
    "delta_nwc_pct_revenue_change",
    "wacc",
    "terminal_method",
    "terminal_growth",
    "exit_ebitda_multiple",
    "debt",
    "cash",
    "diluted_shares",
    "mid_year",
)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_values(frame, row):
    if frame is None or row not in frame.index:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame.loc[row], errors="coerce")


def _recent_value(frame, row):
    values = _row_values(frame, row).dropna()
    return _number(values.iloc[0]) if not values.empty else None


def _recent_average_ratio(income, numerator_frame, numerator_row, *, absolute=False):
    revenue = _row_values(income, "Revenue")
    numerator = _row_values(numerator_frame, numerator_row)
    ratios = []
    for column in income.columns:
        revenue_value = _number(revenue.get(column))
        numerator_value = _number(numerator.get(column))
        if revenue_value and numerator_value is not None:
            ratio = numerator_value / revenue_value
            ratios.append(abs(ratio) if absolute else ratio)
        if len(ratios) == 3:
            break
    return sum(ratios) / len(ratios) if ratios else None


def _tax_rate(income):
    pretax = _row_values(income, "Pre-Tax Income")
    provision = _row_values(income, "Tax Provision")
    rates = []
    for column in income.columns:
        pretax_value = _number(pretax.get(column))
        provision_value = _number(provision.get(column))
        if pretax_value and provision_value is not None:
            rates.append(provision_value / pretax_value)
        if len(rates) == 3:
            break
    if not rates:
        return 0.21, "Assumed 21%; no usable tax provision / pre-tax income history."
    average_rate = sum(rates) / len(rates)
    return min(max(average_rate, 0.10), 0.30), (
        f"{len(rates)}-year average tax provision / pre-tax income, clamped to 10%-30%."
    )


GROWTH_PROFILES = ("fade", "flat", "custom")
FORECAST_YEARS = 5


def historical_starting_growth(income):
    """Historical revenue CAGR, capped at -5% to 20%, or None without enough data.

    The cap keeps one unusual stretch (a pandemic dip, a huge acquisition) from
    turning into a five-year forecast.
    """
    revenues = _row_values(income, "Revenue").dropna()
    if len(revenues) < 2:
        return None
    newest = _number(revenues.iloc[0])
    oldest = _number(revenues.iloc[-1])
    if newest is None or oldest is None or newest <= 0 or oldest <= 0:
        return None

    newest_year = _fiscal_year(revenues.index[0])
    oldest_year = _fiscal_year(revenues.index[-1])
    periods = newest_year - oldest_year if newest_year and oldest_year else len(revenues) - 1
    if periods <= 0:
        return None

    cagr = (newest / oldest) ** (1 / periods) - 1
    return min(max(cagr, -0.05), 0.20)


def growth_profile_rates(starting_growth, terminal_growth, profile="fade"):
    """Five yearly revenue growth rates for a preset profile.

    "fade": start at the historical rate and step evenly down (or up) to the
        terminal growth rate by year 5, so the forecast hands off smoothly to
        the terminal value. Growth usually slows as companies mature.
    "flat": hold the historical rate for all five years.
    "custom" has no preset rates (the user sets each year), so it returns None.
    """
    if profile not in GROWTH_PROFILES:
        raise ValueError(f"Unknown growth profile: {profile!r}")
    if profile == "custom" or starting_growth is None:
        return None
    if profile == "flat":
        return [starting_growth] * FORECAST_YEARS
    last_year = FORECAST_YEARS - 1
    return [
        starting_growth + (terminal_growth - starting_growth) * year / last_year
        for year in range(FORECAST_YEARS)
    ]


def _revenue_growth_rates(income, terminal_growth):
    return growth_profile_rates(
        historical_starting_growth(income), terminal_growth, "fade"
    )


def _fiscal_year(label):
    match = re.search(r"(?:FY)?(\d{4})", str(label))
    return int(match.group(1)) if match else None


NWC_RATIO_RANGE = (-0.10, 0.20)
NWC_FALLBACK = 0.05


def _working_capital_ratio(income, cashflow):
    """Working-capital investment per dollar of revenue added.

    Uses totals over the latest three years rather than year-by-year ratios, so
    one odd year (a tiny revenue change, a one-off swing) doesn't throw out the
    history. The result is clamped to -10%..20%: negative values are allowed
    because some businesses (Costco, Walmart) are funded by their suppliers and
    release cash as they grow.
    """
    revenue = _row_values(income, "Revenue")
    working_capital = _row_values(cashflow, "Change in Working Capital")
    invested, added, years = 0.0, 0.0, 0
    for position, column in enumerate(income.columns[:-1]):
        current_revenue = _number(revenue.get(column))
        prior_revenue = _number(revenue.get(income.columns[position + 1]))
        cashflow_value = _number(working_capital.get(column))
        if (
            current_revenue is not None
            and prior_revenue is not None
            and cashflow_value is not None
        ):
            # Cash-flow statements report working-capital investment as a cash outflow.
            invested += -cashflow_value
            added += current_revenue - prior_revenue
            years += 1
        if years == 3:
            break

    if years == 0:
        return (
            NWC_FALLBACK,
            "Fallback 5%; historical working-capital investment could not be calculated.",
        )
    latest_revenue = _number(revenue.dropna().iloc[0]) if not revenue.dropna().empty else None
    if latest_revenue is None or added <= 0.01 * abs(latest_revenue):
        return (
            NWC_FALLBACK,
            "Fallback 5%; revenue shrank or barely grew over the period, so the "
            "historical ratio isn't meaningful.",
        )

    ratio = invested / added
    low, high = NWC_RATIO_RANGE
    clamped = min(max(ratio, low), high)
    source = (
        f"Total working-capital investment / total revenue added over the latest "
        f"{years} year{'s' if years != 1 else ''}"
    )
    if clamped != ratio:
        source += f" ({ratio:.0%}), clamped to {low:.0%} to {high:.0%}."
    else:
        source += f", within the {low:.0%} to {high:.0%} range."
    return clamped, source


def _risk_free_rate(market_data):
    quoted_yield = _number(market_data.get("risk_free_rate"))
    if quoted_yield is None:
        return 0.04, "Assumed 4.0%; current ^TNX Treasury yield was unavailable."
    # ^TNX is quoted in percent (4.25 means 4.25%), including below 1% as in 2020,
    # so it's always divided by 100. Very old data used ten times the yield
    # (42.5 for 4.25%); no real 10-year yield has been 20% or higher.
    rate = quoted_yield / 1000 if quoted_yield >= 20 else quoted_yield / 100
    return rate, "Current ^TNX 10-year Treasury yield, normalized to a decimal rate."


def default_assumptions(
    cleaned_income,
    cleaned_balance,
    cleaned_cashflow,
    market_data,
    include_securities=True,
):
    """Derive run_dcf inputs and short explanations from cleaned statements and quotes."""
    market_data = market_data or {}
    reported_cash = _recent_value(cleaned_balance, "Cash")
    combined_cash = _recent_value(
        cleaned_balance, "Cash and short-term investments total"
    )
    cash_uses_combined_total = reported_cash is None and combined_cash is not None
    if cash_uses_combined_total:
        reported_cash = combined_cash
    reported_cash = reported_cash if reported_cash is not None else 0.0
    short_term_securities = _recent_value(
        cleaned_balance, "Short-term marketable securities"
    )
    short_term_securities_derived = False
    if cash_uses_combined_total:
        short_term_securities = 0.0
    elif short_term_securities is None and combined_cash is not None:
        short_term_securities = max(combined_cash - reported_cash, 0.0)
        short_term_securities_derived = True
    short_term_securities = short_term_securities or 0.0
    long_term_securities = (
        _recent_value(cleaned_balance, "Long-term marketable securities") or 0.0
    )
    securities_included = bool(include_securities)
    included_securities = (
        short_term_securities + long_term_securities if securities_included else 0.0
    )
    assumptions = {
        "base_revenue": _recent_value(cleaned_income, "Revenue"),
        "growth_rates": None,
        "starting_growth": historical_starting_growth(cleaned_income),
        "growth_profile": "fade",
        "ebit_margin": None,
        "tax_rate": None,
        "da_pct_revenue": None,
        "capex_pct_revenue": None,
        "delta_nwc_pct_revenue_change": None,
        "wacc": None,
        "terminal_method": "gordon",
        "terminal_growth": 0.025,
        "exit_ebitda_multiple": 0.0,
        "debt": _recent_value(cleaned_balance, "Total Debt"),
        "cash": reported_cash + included_securities,
        "reported_cash": reported_cash,
        "short_term_securities": short_term_securities,
        "long_term_securities": long_term_securities,
        "include_securities": securities_included,
        "diluted_shares": None,
        "mid_year": False,
    }
    sources = {
        "base_revenue": "Most recent reported revenue, in USD millions.",
        "growth_rates": "Unavailable; fewer than two positive annual revenue values.",
        "ebit_margin": "Unavailable; no reported EBIT / revenue history.",
        "tax_rate": "",
        "da_pct_revenue": "",
        "capex_pct_revenue": "",
        "delta_nwc_pct_revenue_change": "",
        "wacc": "",
        "terminal_method": "Gordon growth terminal value selected by default.",
        "terminal_growth": "Assumed 2.5% long-term growth rate.",
        "exit_ebitda_multiple": "",
        "debt": "Most recent reported total debt, in USD millions.",
        "cash": "",
        "reported_cash": "",
        "short_term_securities": "",
        "long_term_securities": "",
        "include_securities": "Securities are included in cash by default.",
        "diluted_shares": "Unavailable; no positive yfinance share count was found.",
        "mid_year": "Year-end discounting; mid-year convention is off.",
    }
    errors = []

    if assumptions["base_revenue"] is None or assumptions["base_revenue"] <= 0:
        errors.append("A positive reported revenue base is required.")
        sources["base_revenue"] = "Unavailable; no positive reported revenue was found."

    assumptions["growth_rates"] = _revenue_growth_rates(
        cleaned_income, assumptions["terminal_growth"]
    )
    if assumptions["growth_rates"] is None:
        errors.append("At least two positive annual revenue values are required.")
    else:
        sources["growth_rates"] = (
            "Historical revenue CAGR, capped at -5% to 20%, fading to terminal growth."
        )

    ebit_margin = _recent_average_ratio(cleaned_income, cleaned_income, "EBIT")
    assumptions["ebit_margin"] = ebit_margin
    if ebit_margin is None:
        errors.append("No reported EBIT is available.")
    else:
        sources["ebit_margin"] = "Average EBIT / revenue for the latest available three years."

    da_ratio = _recent_average_ratio(
        cleaned_income, cleaned_cashflow, "D&A", absolute=True
    )
    assumptions["da_pct_revenue"] = da_ratio if da_ratio is not None else 0.0
    sources["da_pct_revenue"] = (
        "Average D&A / revenue for the latest available three years."
        if da_ratio is not None
        else "No D&A reported; assumed 0% of revenue."
    )

    capex_ratio = _recent_average_ratio(
        cleaned_income, cleaned_cashflow, "CapEx", absolute=True
    )
    assumptions["capex_pct_revenue"] = capex_ratio if capex_ratio is not None else 0.0
    sources["capex_pct_revenue"] = (
        "Average absolute CapEx / revenue for the latest available three years."
        if capex_ratio is not None
        else "No CapEx reported; assumed 0% of revenue."
    )

    tax_rate, sources["tax_rate"] = _tax_rate(cleaned_income)
    assumptions["tax_rate"] = tax_rate

    nwc_ratio, sources["delta_nwc_pct_revenue_change"] = _working_capital_ratio(
        cleaned_income, cleaned_cashflow
    )
    assumptions["delta_nwc_pct_revenue_change"] = nwc_ratio

    if assumptions["debt"] is None:
        assumptions["debt"] = 0.0
        sources["debt"] = "No reported debt; assumed $0 million."
    cash_source = (
        "Cash component uses the combined cash-and-short-term-investments total "
        "because a separate cash balance was unavailable."
        if cash_uses_combined_total
        else "Most recent reported cash and cash equivalents, in USD millions."
    )
    sources["reported_cash"] = cash_source
    sources["short_term_securities"] = (
        "Derived as the combined cash-and-short-term-investments total less cash, "
        "in USD millions."
        if short_term_securities_derived
        else (
            "Short-term investments included separately from cash, in USD millions."
            if not cash_uses_combined_total
            else "Included in the combined cash-and-short-term-investments total above."
        )
    )
    sources["long_term_securities"] = (
        "Most recent reported long-term investments / securities, in USD millions."
    )
    if not securities_included:
        sources["include_securities"] = (
            "False; short- and long-term securities are excluded from cash."
        )
    sources["cash"] = (
        f"Cash ${reported_cash:,.1f}M + short-term securities "
        f"${short_term_securities:,.1f}M + long-term securities "
        f"${long_term_securities:,.1f}M"
        + ("; securities included." if securities_included else "; securities excluded.")
    )

    raw_shares = _number(market_data.get("diluted_shares_outstanding"))
    if raw_shares is None or raw_shares <= 0:
        errors.append("Diluted shares are unavailable; per-share value cannot be calculated.")
    else:
        assumptions["diluted_shares"] = raw_shares / 1_000_000
        sources["diluted_shares"] = (
            f"{market_data.get('diluted_shares_source', 'Most recent yfinance share count')} "
            "converted from shares to millions."
        )

    current_price = _number(market_data.get("current_price"))
    if current_price is None or current_price <= 0 or raw_shares is None or raw_shares <= 0:
        market_cap = None
    else:
        market_cap = current_price * raw_shares / 1_000_000

    if market_cap is None:
        errors.append("Market capitalization is unavailable; WACC cannot be weighted.")

    risk_free_rate, risk_free_source = _risk_free_rate(market_data)
    raw_beta = _number(market_data.get("beta"))
    if raw_beta is None:
        beta = 1.0
        beta_source = "Assumed beta of 1.0; no beta was available."
    else:
        beta = adjusted_beta(raw_beta)
        raw_source = market_data.get("beta_source") or "Raw beta reported by Yahoo Finance."
        beta_source = (
            f"Adjusted beta: ⅔ × raw beta {raw_beta:.2f} + ⅓ × 1.0, the Blume "
            f"adjustment Bloomberg reports. {raw_source}"
        )
    equity_cost = cost_of_equity(risk_free_rate, beta, 0.05)

    interest_period = _row_values(cleaned_income, "Interest Expense")
    debt_values = _row_values(cleaned_balance, "Total Debt")
    debt_cost = None
    if cleaned_income is not None and cleaned_balance is not None:
        for period in cleaned_income.columns:
            interest = _number(interest_period.get(period))
            debt_value = _number(debt_values.get(period))
            if interest is not None and debt_value is not None and debt_value > 0:
                debt_cost = abs(interest) / debt_value
                break
    if debt_cost is None:
        debt_cost = risk_free_rate + 0.015
        debt_cost_source = "Fallback: risk-free rate plus 1.5% credit spread."
    else:
        debt_cost_source = "Most recent year with both interest expense and total debt."

    debt = assumptions["debt"]
    if market_cap is not None:
        try:
            assumptions["wacc"] = wacc(
                market_cap, debt, equity_cost, debt_cost, assumptions["tax_rate"]
            )
            total_capital = market_cap + debt
            assumptions["equity_weight"] = market_cap / total_capital
            assumptions["debt_weight"] = debt / total_capital
        except ValueError as error:
            errors.append(str(error))
            assumptions["equity_weight"] = None
            assumptions["debt_weight"] = None
    else:
        assumptions["equity_weight"] = None
        assumptions["debt_weight"] = None
    sources["wacc"] = (
        (
            f"CAPM equity cost (rf {risk_free_rate:.2%}, adjusted beta {beta:.2f}, "
            "ERP 5%) and "
            "market-cap / debt weights."
        )
        if market_cap is not None
        else "Unavailable; market capitalization is needed to weight equity and debt."
    )
    sources["equity_weight"] = (
        "Current share price × shares outstanding, with market cap converted to USD millions."
        if market_cap is not None
        else "Unavailable; current price and diluted shares are required."
    )
    sources["debt_weight"] = "Most recent reported total debt, in USD millions."
    assumptions["risk_free_rate"] = risk_free_rate
    assumptions["beta"] = beta
    assumptions["raw_beta"] = raw_beta
    assumptions["erp"] = 0.05
    assumptions["pre_tax_cost_of_debt"] = debt_cost
    sources["risk_free_rate"] = risk_free_source
    sources["beta"] = beta_source
    sources["erp"] = "Assumed equity risk premium of 5.0%."
    sources["pre_tax_cost_of_debt"] = debt_cost_source

    # EBITDA is operating income + D&A (see data.clean.use_operating_ebitda),
    # the same measure the exit-multiple terminal value is applied to.
    ebitda = _recent_value(cleaned_income, "EBITDA")
    if ebitda is None:
        ebit = _recent_value(cleaned_income, "EBIT")
        da = _recent_value(cleaned_cashflow, "D&A")
        if ebit is not None and da is not None:
            ebitda = ebit + abs(da)
    if market_cap is not None and ebitda is not None and ebitda > 0:
        enterprise_value = market_cap + debt - assumptions["cash"]
        assumptions["exit_ebitda_multiple"] = (
            enterprise_value / ebitda if enterprise_value > 0 else 0.0
        )
        sources["exit_ebitda_multiple"] = (
            "Current enterprise value (market cap + debt - cash) / latest EBITDA "
            "(operating income + D&A)."
        )
    else:
        sources["exit_ebitda_multiple"] = (
            "Unavailable because current enterprise value or positive EBITDA is missing; "
            "not used by the Gordon method."
        )

    if (
        assumptions["terminal_method"] == "gordon"
        and assumptions["wacc"] is not None
        and assumptions["terminal_growth"] >= assumptions["wacc"]
    ):
        errors.append("Terminal growth must be below WACC for a stable DCF.")

    assumptions["sources"] = sources
    assumptions["error"] = " ".join(errors) if errors else None
    return assumptions
