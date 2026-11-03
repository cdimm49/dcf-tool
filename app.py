import streamlit as st
from data.fetch import get_company_data

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
            st.dataframe(
                company_data["income_statement"],
                alt="Raw income statement",
            )

        with balance_tab:
            st.dataframe(
                company_data["balance_sheet"],
                alt="Raw balance sheet",
            )

        with cash_flow_tab:
            st.dataframe(
                company_data["cash_flow_statement"],
                alt="Raw cash flow statement",
            )