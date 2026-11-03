import streamlit as st
from data.clean import add_income_margins, clean_statement
from data.fetch import get_company_data
from ui.tables import display_statement

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

        income_tab, balance_tab, cash_flow_tab = st.tabs(
            ["Income statement", "Balance sheet", "Cash flow statement"]
        )

        with income_tab:
            income_statement = add_income_margins(
                clean_statement(company_data["income_statement"], "income")
            )
            display_statement(income_statement)
            st.caption("USD millions, fiscal years")

        with balance_tab:
            balance_sheet = clean_statement(
                company_data["balance_sheet"], "balance"
            )
            display_statement(balance_sheet)
            st.caption("USD millions, fiscal years")

        with cash_flow_tab:
            cash_flow_statement = clean_statement(
                company_data["cash_flow_statement"], "cashflow"
            )
            display_statement(cash_flow_statement)
            st.caption("USD millions, fiscal years")