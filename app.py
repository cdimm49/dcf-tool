import pandas as pd
import streamlit as st
from data.clean import add_income_margins, clean_statement
from data.fetch import get_company_data
from model.assumptions import DCF_INPUTS, default_assumptions
from model.dcf import run_dcf
from ui.tables import display_statement


_PERCENT_INPUTS = {
    "ebit_margin",
    "tax_rate",
    "da_pct_revenue",
    "capex_pct_revenue",
    "delta_nwc_pct_revenue_change",
    "wacc",
    "terminal_growth",
    "risk_free_rate",
    "erp",
    "pre_tax_cost_of_debt",
    "equity_weight",
    "debt_weight",
}


def _format_assumption(name, value):
    if value is None:
        return "Unavailable"
    if name in _PERCENT_INPUTS:
        return f"{value:.2%}"
    if name == "growth_rates":
        return ", ".join(f"{rate:.1%}" for rate in value)
    if name == "exit_ebitda_multiple":
        return f"{value:.1f}x"
    if name == "beta":
        return f"{value:.2f}"
    if name == "diluted_shares":
        return f"{value:,.1f} million"
    if name in {"base_revenue", "debt", "cash"}:
        return f"${value:,.1f} million"
    return str(value)


def _display_assumptions(assumptions):
    labels = {
        "base_revenue": "Base revenue",
        "growth_rates": "Revenue growth by year",
        "ebit_margin": "EBIT margin",
        "tax_rate": "Tax rate",
        "da_pct_revenue": "D&A as % of revenue",
        "capex_pct_revenue": "CapEx as % of revenue",
        "delta_nwc_pct_revenue_change": "Change in working capital",
        "wacc": "WACC",
        "terminal_method": "Terminal value method",
        "terminal_growth": "Terminal growth",
        "exit_ebitda_multiple": "Exit EV / EBITDA",
        "debt": "Debt",
        "cash": "Cash",
        "diluted_shares": "Diluted shares",
        "mid_year": "Mid-year convention",
        "risk_free_rate": "Risk-free rate",
        "beta": "Beta",
        "erp": "Equity risk premium",
        "pre_tax_cost_of_debt": "Pre-tax cost of debt",
        "equity_weight": "Equity weight",
        "debt_weight": "Debt weight",
    }
    inputs = (
        *DCF_INPUTS,
        "risk_free_rate",
        "beta",
        "erp",
        "pre_tax_cost_of_debt",
        "equity_weight",
        "debt_weight",
    )
    assumptions_table = pd.DataFrame(
        [
            {
                "Input": labels[name],
                "Value": _format_assumption(name, assumptions[name]),
                "Source": assumptions["sources"][name],
            }
            for name in inputs
        ]
    )
    st.subheader("Assumptions used")
    st.dataframe(
        assumptions_table,
        hide_index=True,
        alt="DCF inputs with their derived values and sources",
    )


def _display_valuation(assumptions, company_data):
    if assumptions["error"]:
        st.warning(
            f"A DCF could not be calculated: {assumptions['error']} "
            "A DCF may not fit this company type or the available data."
        )
        _display_assumptions(assumptions)
        return

    result = run_dcf(
        **{name: assumptions[name] for name in DCF_INPUTS}
    )
    current_price = company_data["current_price"]
    upside = (result["value_per_share"] / current_price - 1) * 100

    st.subheader("What the business may be worth")
    st.caption(
        "This estimate projects future cash flow, discounts it to today, and compares "
        "the result with the current share price."
    )
    value_col, price_col, upside_col = st.columns(3)
    value_col.metric("Implied value per share", f"${result['value_per_share']:,.2f}")
    price_col.metric("Current share price", f"${current_price:,.2f}")
    upside_col.metric("Upside / downside", f"{upside:+.1f}%")

    st.subheader("Five-year cash flow forecast")
    st.caption("Amounts are in USD millions; each year's cash flow is discounted to today.")
    forecast = result["forecast"].style.format(
        {
            column: "${:,.1f}"
            for column in forecast_columns(result["forecast"])
        }
        | {"Discount Factor": "{:.3f}"}
    )
    st.dataframe(forecast, alt="Five-year DCF revenue and cash flow forecast")

    st.subheader("Enterprise value to equity value")
    bridge = pd.DataFrame(
        {
            "Step": [
                "Present value of forecast cash flows",
                "Present value of terminal value",
                "Enterprise value",
                "Less: debt",
                "Add: cash",
                "Equity value",
            ],
            "USD millions": [
                result["sum_pv_fcff"],
                result["pv_terminal_value"],
                result["enterprise_value"],
                -assumptions["debt"],
                assumptions["cash"],
                result["equity_value"],
            ],
        }
    ).set_index("Step")
    st.dataframe(
        bridge.style.format({"USD millions": "${:,.1f}"}),
        alt="Bridge from enterprise value to equity value",
    )

    terminal_share = result["terminal_value_pct_ev"]
    st.metric("Terminal value share of enterprise value", f"{terminal_share:.1%}")
    if terminal_share > 0.75:
        st.warning(
            "More than 75% of enterprise value comes from the terminal value, "
            "so the estimate is especially sensitive to long-term assumptions."
        )

    _display_assumptions(assumptions)


def forecast_columns(forecast):
    return [column for column in forecast.columns if column != "Discount Factor"]

st.set_page_config(layout="wide")
st.title("DCF Tool")

with st.form("ticker_form"):
    ticker = st.text_input("Ticker symbol", value="AAPL")
    submitted = st.form_submit_button("Run")

if submitted:
    company_data = get_company_data(ticker)

    if "error" in company_data:
        st.error(company_data["error"])
    else:
        st.subheader(company_data["company_name"])
        st.metric("Current price", f"${company_data['current_price']:,.2f}")
        if company_data["diluted_shares_outstanding"] is None:
            st.warning("Diluted shares outstanding are unavailable.")

        income_statement = add_income_margins(
            clean_statement(company_data["income_statement"], "income")
        )
        balance_sheet = clean_statement(
            company_data["balance_sheet"], "balance"
        )
        cash_flow_statement = clean_statement(
            company_data["cash_flow_statement"], "cashflow"
        )
        market_data = {
            "current_price": company_data["current_price"],
            "diluted_shares_outstanding": company_data[
                "diluted_shares_outstanding"
            ],
            "diluted_shares_source": company_data["diluted_shares_source"],
            "beta": company_data["beta"],
            "risk_free_rate": company_data["risk_free_rate"],
        }
        assumptions = default_assumptions(
            income_statement, balance_sheet, cash_flow_statement, market_data
        )

        valuation_tab, income_tab, balance_tab, cash_flow_tab = st.tabs(
            ["Valuation", "Income statement", "Balance sheet", "Cash flow statement"]
        )

        with valuation_tab:
            _display_valuation(assumptions, company_data)

        with income_tab:
            display_statement(income_statement)
            st.caption("USD millions, fiscal years")

        with balance_tab:
            display_statement(balance_sheet)
            st.caption("USD millions, fiscal years")

        with cash_flow_tab:
            display_statement(cash_flow_statement)
            st.caption("USD millions, fiscal years")