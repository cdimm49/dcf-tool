import pandas as pd
import pytest

from data.clean import clean_statement
from model.assumptions import default_assumptions


@pytest.fixture
def company_statements():
    income = pd.DataFrame(
        {
            "FY2024": [1000, 200, 100, 50, 200],
            "FY2023": [800, 160, 80, 40, 160],
            "FY2022": [600, 120, 60, 30, 120],
        },
        index=["Revenue", "EBIT", "Pre-Tax Income", "Tax Provision", "EBITDA"],
    )
    balance = pd.DataFrame(
        {"FY2024": [100, 50], "FY2023": [90, 45], "FY2022": [80, 40]},
        index=["Total Debt", "Cash"],
    )
    cashflow = pd.DataFrame(
        {
            "FY2024": [20, -40, -80],
            "FY2023": [16, -40, -64],
            "FY2022": [12, -40, -48],
        },
        index=["D&A", "Change in Working Capital", "CapEx"],
    )
    return income, balance, cashflow


def test_assumptions_convert_share_units_and_clamp_estimates(company_statements):
    income, balance, cashflow = company_statements

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
            "diluted_shares_source": "Most recent diluted average shares",
            "risk_free_rate": 42.5,
            "beta": 1.2,
        },
    )

    assert assumptions["error"] is None
    assert assumptions["base_revenue"] == 1000
    assert assumptions["growth_rates"] == pytest.approx(
        [0.20, 0.15625, 0.1125, 0.06875, 0.025]
    )
    assert assumptions["ebit_margin"] == pytest.approx(0.20)
    assert assumptions["da_pct_revenue"] == pytest.approx(0.02)
    assert assumptions["capex_pct_revenue"] == pytest.approx(0.08)
    assert assumptions["delta_nwc_pct_revenue_change"] == pytest.approx(0.20)
    assert assumptions["tax_rate"] == pytest.approx(0.30)
    assert assumptions["diluted_shares"] == pytest.approx(1000)
    assert assumptions["debt"] == 100
    assert assumptions["cash"] == 50
    assert assumptions["risk_free_rate"] == pytest.approx(0.0425)
    assert assumptions["equity_weight"] == pytest.approx(50_000 / 50_100)
    assert assumptions["sources"]["diluted_shares"].endswith(
        "converted from shares to millions."
    )
    assert assumptions["sources"]["risk_free_rate"].startswith("Current ^TNX")


def test_missing_ebit_returns_warning_and_uses_market_data_fallbacks(
    company_statements,
):
    income, balance, cashflow = company_statements
    income = income.drop(index="EBIT")

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
        },
    )

    assert "No reported EBIT is available" in assumptions["error"]
    assert assumptions["ebit_margin"] is None
    assert assumptions["risk_free_rate"] == pytest.approx(0.04)
    assert assumptions["beta"] == pytest.approx(1.0)
    assert assumptions["pre_tax_cost_of_debt"] == pytest.approx(0.055)


def test_small_revenue_change_falls_back_to_five_percent(company_statements):
    income, balance, cashflow = company_statements
    income.loc["Revenue"] = [1000, 995, 800]
    cashflow.loc["Change in Working Capital"] = [-50, -20, -10]

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
        },
    )

    assert assumptions["delta_nwc_pct_revenue_change"] == pytest.approx(0.05)
    assert "revenue change was negative or less than 1%" in assumptions["sources"][
        "delta_nwc_pct_revenue_change"
    ]


def test_historical_working_capital_ratio_outside_range_falls_back(
    company_statements,
):
    income, balance, cashflow = company_statements
    cashflow.loc["Change in Working Capital"] = [-80, -64, -48]

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
        },
    )

    assert assumptions["delta_nwc_pct_revenue_change"] == pytest.approx(0.05)
    assert "historical working-capital ratio was outside 0%-20%" in assumptions[
        "sources"
    ]["delta_nwc_pct_revenue_change"]


def test_securities_are_normalized_and_included_in_cash_by_default(company_statements):
    raw_balance = pd.DataFrame(
        {
            "2024-12-31": [
                10_000_000,
                3_000_000,
                7_000_000,
                20_000_000,
                30_000_000,
                40_000_000,
            ],
        },
        index=[
            "Cash And Cash Equivalents",
            "Other Short Term Investments",
            "Investments And Advances",
            "Total Debt",
            "Total Current Assets",
            "Total Assets",
        ],
    )
    balance = clean_statement(raw_balance, "balance")
    income, _, cashflow = company_statements
    market_data = {
        "current_price": 50,
        "diluted_shares_outstanding": 1_000_000_000,
    }

    assumptions = default_assumptions(income, balance, cashflow, market_data)
    excluded = default_assumptions(
        income, balance, cashflow, market_data, include_securities=False
    )

    assert assumptions["reported_cash"] == pytest.approx(10)
    assert assumptions["short_term_securities"] == pytest.approx(3)
    assert assumptions["long_term_securities"] == pytest.approx(7)
    assert assumptions["cash"] == pytest.approx(20)
    assert assumptions["include_securities"] is True
    assert "short-term securities $3.0M" in assumptions["sources"]["cash"]
    assert "long-term securities $7.0M" in assumptions["sources"]["cash"]
    assert excluded["cash"] == pytest.approx(10)
    assert excluded["include_securities"] is False


def test_combined_cash_and_investments_total_does_not_double_count_securities():
    raw_balance = pd.DataFrame(
        {
            "2024-12-31": [
                15_000_000,
                5_000_000,
                20_000_000,
                30_000_000,
                40_000_000,
            ],
        },
        index=[
            "Cash Cash Equivalents And Short Term Investments",
            "Other Short Term Investments",
            "Total Debt",
            "Total Current Assets",
            "Total Assets",
        ],
    )
    balance = clean_statement(raw_balance, "balance")
    income, _, cashflow = _sample_company_statements()

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
        },
    )

    assert assumptions["cash"] == pytest.approx(15)
    assert assumptions["short_term_securities"] == pytest.approx(0)
    assert "combined cash-and-short-term-investments total" in assumptions["sources"][
        "reported_cash"
    ]


def test_short_term_securities_can_be_derived_from_combined_cash_total():
    raw_balance = pd.DataFrame(
        {
            "2024-12-31": [
                10_000_000,
                15_000_000,
                7_000_000,
                20_000_000,
                30_000_000,
                40_000_000,
            ],
        },
        index=[
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Investments And Advances",
            "Total Debt",
            "Total Current Assets",
            "Total Assets",
        ],
    )
    balance = clean_statement(raw_balance, "balance")
    income, _, cashflow = _sample_company_statements()

    assumptions = default_assumptions(
        income,
        balance,
        cashflow,
        {
            "current_price": 50,
            "diluted_shares_outstanding": 1_000_000_000,
        },
    )

    assert assumptions["short_term_securities"] == pytest.approx(5)
    assert assumptions["cash"] == pytest.approx(22)
    assert "Derived as the combined cash-and-short-term-investments total" in (
        assumptions["sources"]["short_term_securities"]
    )


def test_bridge_currency_formatter_uses_accounting_negatives():
    from app import _format_currency

    assert _format_currency(-98657) == "($98,657.0)"


def _sample_company_statements():
    income = pd.DataFrame(
        {"FY2024": [1000, 200], "FY2023": [900, 180], "FY2022": [800, 160]},
        index=["Revenue", "EBIT"],
    )
    cashflow = pd.DataFrame(index=["Change in Working Capital"])
    return income, None, cashflow
