import pytest

from model.dcf import run_dcf
from model.wacc import cost_of_equity, wacc


@pytest.fixture
def dcf_inputs():
    return {
        "base_revenue": 1000,
        "growth_rates": [0.10, 0.08, 0.06, 0.05, 0.04],
        "ebit_margin": 0.30,
        "tax_rate": 0.20,
        "da_pct_revenue": 0.05,
        "capex_pct_revenue": 0.06,
        "delta_nwc_pct_revenue_change": 0.10,
        "wacc": 0.088,
        "terminal_method": "gordon",
        "terminal_growth": 0.03,
        "exit_ebitda_multiple": 10,
        "debt": 200,
        "cash": 100,
        "diluted_shares": 100,
    }


def test_cost_of_equity_and_wacc():
    cost = cost_of_equity(0.04, 1.2, 0.05)
    assert cost == pytest.approx(0.10)
    assert wacc(800, 200, cost, 0.05, 0.20) == pytest.approx(0.088)


def test_gordon_dcf_matches_expected_values(dcf_inputs):
    result = run_dcf(**dcf_inputs)

    assert result["forecast"]["FCFF"].tolist() == pytest.approx(
        [243.00, 264.44, 282.51, 297.82, 310.99], abs=0.01
    )
    assert result["sum_pv_fcff"] == pytest.approx(1082.62, abs=0.01)
    assert result["terminal_value"] == pytest.approx(5522.79, abs=0.01)
    assert result["pv_terminal_value"] == pytest.approx(3622.54, abs=0.01)
    assert result["enterprise_value"] == pytest.approx(4705.16, abs=0.01)
    assert result["equity_value"] == pytest.approx(4605.16, abs=0.01)
    assert result["value_per_share"] == pytest.approx(46.05, abs=0.01)


def test_exit_multiple_dcf_matches_expected_values(dcf_inputs):
    dcf_inputs["terminal_method"] = "exit"
    result = run_dcf(**dcf_inputs)

    assert result["enterprise_value"] == pytest.approx(4239.57, abs=0.01)
    assert result["value_per_share"] == pytest.approx(41.40, abs=0.01)


def test_terminal_growth_must_be_less_than_wacc(dcf_inputs):
    dcf_inputs["terminal_growth"] = dcf_inputs["wacc"]

    with pytest.raises(ValueError, match="Terminal growth must be less than WACC"):
        run_dcf(**dcf_inputs)


def test_mid_year_convention_increases_value(dcf_inputs):
    year_end_value = run_dcf(**dcf_inputs)["value_per_share"]
    mid_year_value = run_dcf(**dcf_inputs, mid_year=True)["value_per_share"]

    assert mid_year_value > year_end_value
