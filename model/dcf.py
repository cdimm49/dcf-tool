import pandas as pd


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
):
    """Project free cash flow, discount it, and calculate equity value per share."""
    if terminal_method not in {"gordon", "exit"}:
        raise ValueError("terminal_method must be 'gordon' or 'exit'.")
    if terminal_method == "gordon" and terminal_growth >= wacc:
        raise ValueError("Terminal growth must be less than WACC.")
    if diluted_shares <= 0:
        raise ValueError("Diluted shares must be greater than zero.")

    projections = []
    previous_revenue = base_revenue
    for year, growth_rate in enumerate(growth_rates, start=1):
        revenue = previous_revenue * (1 + growth_rate)
        ebit = revenue * ebit_margin
        nopat = ebit * (1 - tax_rate)
        da = revenue * da_pct_revenue
        capex = revenue * capex_pct_revenue
        delta_nwc = (revenue - previous_revenue) * delta_nwc_pct_revenue_change
        fcff = nopat + da - capex - delta_nwc
        discount_period = year - 0.5 if mid_year else year
        discount_factor = 1 / (1 + wacc) ** discount_period

        projections.append(
            {
                "Year": year,
                "Revenue": revenue,
                "EBIT": ebit,
                "NOPAT": nopat,
                "D&A": da,
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

    if terminal_method == "gordon":
        terminal_value = (
            final_year["FCFF"]
            * (1 + terminal_growth)
            / (wacc - terminal_growth)
        )
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
        "pv_terminal_value": pv_terminal_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": equity_value / diluted_shares,
        "terminal_value_pct_ev": (
            pv_terminal_value / enterprise_value if enterprise_value else None
        ),
    }
