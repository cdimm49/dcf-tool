import pandas as pd
import pytest

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
            "FY2024": [20, -80, -80],
            "FY2023": [16, -60, -64],
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
