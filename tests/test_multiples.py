import math

import pandas as pd
import pytest

from model.multiples import calculate_multiples


@pytest.fixture
def statements():
    income = pd.DataFrame(
        {"FY2024": [2000, 400, 5], "FY2023": [1800, 350, 4]},
        index=["Revenue", "EBITDA", "Diluted EPS"],
    )
    balance = pd.DataFrame(
        {
            "FY2024": [300, 80, 20, 50, 100, 500],
            "FY2023": [280, 70, 10, 40, 80, 450],
        },
        index=[
            "Total Debt",
            "Cash",
            "Short-term marketable securities",
            "Long-term marketable securities",
            "Cash and short-term investments total",
            "Shareholders’ Equity",
        ],
    )
    cash_flow = pd.DataFrame(
        {"FY2024": [100], "FY2023": [80]}, index=["Free Cash Flow"]
    )
    market_data = {
        "current_price": 100,
        "diluted_shares_outstanding": 10_000_000,
    }
    return income, balance, cash_flow, market_data


def test_calculates_all_multiples_in_usd_millions(statements):
    income, balance, cash_flow, market_data = statements

    multiples = calculate_multiples(income, balance, cash_flow, market_data)

    # Market cap is $1,000M; EV is $1,000M + $300M - ($80M + $20M + $50M).
    assert multiples["P/E"] == pytest.approx(20)
    assert multiples["EV/EBITDA"] == pytest.approx(2.875)
    assert multiples["EV/Sales"] == pytest.approx(0.575)
    assert multiples["P/B"] == pytest.approx(2)
    assert multiples["FCF yield"] == pytest.approx(0.10)


def test_excluding_securities_matches_the_dcf_cash_option(statements):
    income, balance, cash_flow, market_data = statements

    multiples = calculate_multiples(
        income, balance, cash_flow, market_data, include_securities=False
    )

    assert multiples["EV/EBITDA"] == pytest.approx(3.05)


@pytest.mark.parametrize(
    ("statement", "row", "value", "metric"),
    [
        ("income", "Diluted EPS", -1, "P/E"),
        ("income", "EBITDA", -1, "EV/EBITDA"),
        ("income", "Revenue", -1, "EV/Sales"),
        ("balance", "Shareholders’ Equity", -1, "P/B"),
        ("cashflow", "Free Cash Flow", -1, "FCF yield"),
    ],
)
def test_negative_inputs_return_nan(statements, statement, row, value, metric):
    income, balance, cash_flow, market_data = statements
    selected = {
        "income": income,
        "balance": balance,
        "cashflow": cash_flow,
    }[statement]
    selected.loc[row, "FY2024"] = value

    multiples = calculate_multiples(income, balance, cash_flow, market_data)

    assert math.isnan(multiples[metric])


@pytest.mark.parametrize("row", ["Total Debt", "Cash"])
def test_missing_enterprise_value_inputs_return_nan(statements, row):
    income, balance, cash_flow, market_data = statements
    if row == "Cash":
        balance = balance.drop(
            index=["Cash", "Cash and short-term investments total"]
        )
    else:
        balance = balance.drop(index=row)

    multiples = calculate_multiples(income, balance, cash_flow, market_data)

    assert math.isnan(multiples["EV/EBITDA"])
    assert math.isnan(multiples["EV/Sales"])


@pytest.mark.parametrize("row", ["Total Debt", "Cash"])
def test_negative_enterprise_value_inputs_return_nan(statements, row):
    income, balance, cash_flow, market_data = statements
    balance.loc[row, "FY2024"] = -1

    multiples = calculate_multiples(income, balance, cash_flow, market_data)

    assert math.isnan(multiples["EV/EBITDA"])
    assert math.isnan(multiples["EV/Sales"])


def test_missing_diluted_shares_only_disables_market_cap_multiples(statements):
    income, balance, cash_flow, market_data = statements
    market_data.pop("diluted_shares_outstanding")

    multiples = calculate_multiples(income, balance, cash_flow, market_data)

    assert multiples["P/E"] == pytest.approx(20)
    assert math.isnan(multiples["EV/EBITDA"])
    assert math.isnan(multiples["EV/Sales"])
    assert math.isnan(multiples["P/B"])
    assert math.isnan(multiples["FCF yield"])
