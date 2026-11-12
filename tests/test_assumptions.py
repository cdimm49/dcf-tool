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


MARKET = {"current_price": 50, "diluted_shares_outstanding": 1_000_000_000}


def _nwc(income, balance, cashflow):
    assumptions = default_assumptions(income, balance, cashflow, MARKET)
    return (
        assumptions["delta_nwc_pct_revenue_change"],
        assumptions["sources"]["delta_nwc_pct_revenue_change"],
    )


def test_one_tiny_revenue_change_no_longer_discards_the_history(company_statements):
    income, balance, cashflow = company_statements
    income.loc["Revenue"] = [1000, 995, 800]
    cashflow.loc["Change in Working Capital"] = [-1, -19, -10]

    ratio, source = _nwc(income, balance, cashflow)

    # (1 + 19) invested over (5 + 195) of revenue added = 10%.
    assert ratio == pytest.approx(0.10)
    assert "within the -10% to 20% range" in source


def test_working_capital_ratio_above_range_is_clamped(company_statements):
    income, balance, cashflow = company_statements
    cashflow.loc["Change in Working Capital"] = [-80, -64, -48]

    ratio, source = _nwc(income, balance, cashflow)

    # 144 invested over 400 added is 36%, clamped to 20%.
    assert ratio == pytest.approx(0.20)
    assert "(36%), clamped to -10% to 20%" in source


def test_supplier_funded_working_capital_can_be_negative(company_statements):
    income, balance, cashflow = company_statements
    cashflow.loc["Change in Working Capital"] = [10, 8, 6]  # cash released as sales grow

    ratio, _source = _nwc(income, balance, cashflow)

    assert ratio == pytest.approx(-18 / 400)


def test_shrinking_revenue_falls_back_to_five_percent(company_statements):
    income, balance, cashflow = company_statements
    income.loc["Revenue"] = [800, 900, 1000]

    ratio, source = _nwc(income, balance, cashflow)

    assert ratio == pytest.approx(0.05)
    assert "revenue shrank or barely grew" in source


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


def test_beta_is_blume_adjusted_and_labeled(company_statements):
    income, balance, cashflow = company_statements

    assumptions = default_assumptions(
        income, balance, cashflow, {**MARKET, "beta": 0.15}
    )

    assert assumptions["raw_beta"] == pytest.approx(0.15)
    assert assumptions["beta"] == pytest.approx(2 / 3 * 0.15 + 1 / 3)
    assert assumptions["sources"]["beta"].startswith("Adjusted beta")


@pytest.mark.parametrize(
    ("quoted", "expected"),
    [(4.95, 0.0495), (0.9, 0.009), (0.55, 0.0055), (42.5, 0.0425)],
)
def test_treasury_yield_is_read_as_percent_even_below_one(
    company_statements, quoted, expected
):
    income, balance, cashflow = company_statements

    assumptions = default_assumptions(
        income, balance, cashflow, {**MARKET, "risk_free_rate": quoted}
    )

    assert assumptions["risk_free_rate"] == pytest.approx(expected)


def _assumptions_with(company_statements, da_values, capex_values):
    income, balance, cashflow = company_statements
    cashflow = cashflow.copy()
    cashflow.loc["D&A"] = da_values
    cashflow.loc["CapEx"] = capex_values
    return default_assumptions(
        income, balance, cashflow, {"current_price": 50, "diluted_shares_outstanding": 1e9}
    )


def test_da_add_back_is_limited_to_capex_when_da_is_far_above_it(company_statements):
    # Revenue is 1000/800/... so D&A of 400 is ~40-50% of revenue vs capex ~8%.
    assumptions = _assumptions_with(
        company_statements, [400, 400, 400], [-80, -64, -48]
    )

    assert assumptions["da_capped"] is True
    assert assumptions["da_reported_pct_revenue"] > 0.3
    assert assumptions["da_pct_revenue"] == pytest.approx(assumptions["capex_pct_revenue"])
    assert "limited to capex" in assumptions["sources"]["da_pct_revenue"]


def test_da_add_back_untouched_when_below_capex(company_statements):
    assumptions = _assumptions_with(
        company_statements, [20, 16, 12], [-80, -64, -48]
    )

    assert assumptions["da_capped"] is False
    assert assumptions["da_pct_revenue"] == pytest.approx(
        assumptions["da_reported_pct_revenue"]
    )
    assert assumptions["da_pct_revenue"] == pytest.approx(0.02)


def test_da_note_ignores_gaps_within_half_a_point_of_revenue(company_statements):
    # Revenue 1000/800/... so D&A 3 vs capex 0 on 1000 is 0.3 points; the cap
    # still applies, but no note is flagged.
    small = _assumptions_with(company_statements, [83, 66, 49], [-80, -64, -48])
    assert small["da_capped"] is False
    assert small["da_pct_revenue"] == pytest.approx(small["capex_pct_revenue"])

    large = _assumptions_with(company_statements, [90, 72, 54], [-80, -64, -48])
    assert large["da_capped"] is True
