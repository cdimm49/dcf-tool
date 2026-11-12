import pandas as pd

# In the terminal year a company replaces its worn-out assets and adds 10% on top
# to support growth, instead of repeating today's (often much larger) capex forever.
# Applied to the D&A that capex actually pays for: the smaller of the D&A add-back
# and today's capex share. Capex fades to that level in a straight line over the
# forecast, so year 5 and the terminal year line up.
TERMINAL_CAPEX_TO_DA = 1.1


def run_dcf(
    base_revenue,
    growth_rates,
    ebit_margin,
    tax_rate,
    da_pct_revenue,
    capex_pct_revenue,
    delta_nwc_pct_revenue_change,
    wacc,
    terminal_method,
    terminal_growth,
    exit_ebitda_multiple,
    debt,
    cash,
    diluted_shares,
    mid_year=False,
    da_addback_pct_revenue=None,
):
    """Project free cash flow, discount it, and calculate equity value per share.

    da_pct_revenue is reported D&A (used for EBITDA). da_addback_pct_revenue is the
    part added back to cash flow; None means all of it.
    """
    if terminal_method not in {"gordon", "exit"}:
        raise ValueError("terminal_method must be 'gordon' or 'exit'.")
    if terminal_method == "gordon" and terminal_growth >= wacc:
        raise ValueError("Terminal growth must be less than WACC.")
    if diluted_shares <= 0:
        raise ValueError("Diluted shares must be greater than zero.")

    addback_pct = da_pct_revenue if da_addback_pct_revenue is None else da_addback_pct_revenue
    terminal_capex_pct = TERMINAL_CAPEX_TO_DA * min(addback_pct, capex_pct_revenue)
    n_years = len(growth_rates)

    projections = []
    previous_revenue = base_revenue
    for year, growth_rate in enumerate(growth_rates, start=1):
        revenue = previous_revenue * (1 + growth_rate)
        ebit = revenue * ebit_margin
        nopat = ebit * (1 - tax_rate)
        da = revenue * da_pct_revenue
        da_added_back = revenue * addback_pct
        capex_pct = capex_pct_revenue + (
            (terminal_capex_pct - capex_pct_revenue) * year / n_years
        )
        capex = revenue * capex_pct
        delta_nwc = (revenue - previous_revenue) * delta_nwc_pct_revenue_change
        fcff = nopat + da_added_back - capex - delta_nwc
        discount_period = year - 0.5 if mid_year else year
        discount_factor = 1 / (1 + wacc) ** discount_period

        projections.append(
            {
                "Year": year,
                "Revenue": revenue,
                "EBIT": ebit,
                "NOPAT": nopat,
                "D&A": da,
                "D&A add-back": da_added_back,
                "CapEx": capex,
                "ΔNWC": delta_nwc,
                "FCFF": fcff,
                "Discount Factor": discount_factor,
                "PV": fcff * discount_factor,
            }
        )
        previous_revenue = revenue

    forecast = pd.DataFrame(projections).set_index("Year")
    sum_pv_fcff = forecast["PV"].sum()
    final_year = forecast.iloc[-1]

    implied_exit_multiple = None
    if terminal_method == "gordon":
        # Normalized terminal-year cash flow: one more year of growth, with capex
        # set to a small premium over D&A and working capital sized to that growth.
        terminal_revenue = final_year["Revenue"] * (1 + terminal_growth)
        terminal_fcff = (
            terminal_revenue * ebit_margin * (1 - tax_rate)
            + terminal_revenue * addback_pct
            - terminal_revenue * terminal_capex_pct
            - (terminal_revenue - final_year["Revenue"])
            * delta_nwc_pct_revenue_change
        )
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        # Gordon TV sits at N - 0.5 under mid-year; move it to year-end N so it
        # is comparable with an exit multiple, which assumes a sale at year-end N.
        final_ebitda = final_year["EBIT"] + final_year["D&A"]
        if final_ebitda > 0:
            year_end_value = terminal_value * (
                (1 + wacc) ** 0.5 if mid_year else 1
            )
            # A negative multiple (negative terminal cash flow) is not meaningful.
            if year_end_value > 0:
                implied_exit_multiple = year_end_value / final_ebitda
    else:
        terminal_value = (
            final_year["EBIT"] + final_year["D&A"]
        ) * exit_ebitda_multiple

    # Gordon TV values the cash flows from year N+1 onward, which arrive spread
    # through each year, so it shares year N's mid-year factor (N - 0.5). An exit
    # multiple models a sale at the end of year N, so it is discounted a full N years.
    if terminal_method == "exit" and mid_year:
        pv_terminal_value = terminal_value / (1 + wacc) ** len(growth_rates)
    else:
        pv_terminal_value = terminal_value * final_year["Discount Factor"]
    enterprise_value = sum_pv_fcff + pv_terminal_value
    equity_value = enterprise_value - debt + cash

    return {
        "forecast": forecast,
        "sum_pv_fcff": sum_pv_fcff,
        "terminal_value": terminal_value,
        "implied_exit_multiple": implied_exit_multiple,
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": equity_value / diluted_shares,
        "terminal_value_pct_ev": (
            pv_terminal_value / enterprise_value if enterprise_value else None
        ),
    }
