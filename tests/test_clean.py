import pandas as pd
import pytest

from data.clean import use_operating_ebitda


def _statements():
    income = pd.DataFrame(
        {"FY2025": [100.0, 140.0], "FY2024": [90.0, 120.0], "FY2023": [80.0, 95.0]},
        index=["EBIT", "EBITDA"],  # reported EBITDA includes non-operating income
    )
    cashflow = pd.DataFrame(
        {"FY2025": [20.0], "FY2024": [-18.0], "FY2023": [float("nan")]},
        index=["D&A"],
    )
    return income, cashflow


def test_ebitda_becomes_operating_income_plus_da():
    income, cashflow = _statements()

    result = use_operating_ebitda(income, cashflow)

    assert result.loc["EBITDA", "FY2025"] == pytest.approx(120.0)
    assert result.loc["EBITDA", "FY2024"] == pytest.approx(108.0)  # D&A sign ignored
    assert result.loc["EBITDA", "FY2023"] == pytest.approx(95.0)  # no D&A: keep reported
    assert income.loc["EBITDA", "FY2025"] == 140.0  # input left unchanged


def test_ebitda_row_is_added_when_missing():
    income, cashflow = _statements()

    result = use_operating_ebitda(income.drop(index="EBITDA"), cashflow)

    assert result.loc["EBITDA", "FY2025"] == pytest.approx(120.0)
