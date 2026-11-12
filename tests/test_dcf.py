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
        [244.10, 266.82, 286.28, 303.11, 317.87], abs=0.01
    )  # capex fades from 6.0% of revenue to the terminal 5.5% (1.1 x 5% D&A)
    assert result["sum_pv_fcff"] == pytest.approx(1096.85, abs=0.01)
    # Terminal year: revenue 1375.13 x 1.03 = 1416.39; NOPAT 339.93; D&A 70.82
    # added back, capex 1.1 x D&A = 77.90 paid; NWC 4.13. FCFF 328.73 / (8.8% - 3%).
    assert result["terminal_value"] == pytest.approx(5667.68, abs=0.01)
    assert result["pv_terminal_value"] == pytest.approx(3717.59, abs=0.01)
    assert result["enterprise_value"] == pytest.approx(4814.44, abs=0.01)
    assert result["equity_value"] == pytest.approx(4714.44, abs=0.01)
    assert result["value_per_share"] == pytest.approx(47.14, abs=0.01)


def test_exit_multiple_dcf_matches_expected_values(dcf_inputs):
    dcf_inputs["terminal_method"] = "exit"
    result = run_dcf(**dcf_inputs)

    assert result["enterprise_value"] == pytest.approx(4253.81, abs=0.01)
    assert result["value_per_share"] == pytest.approx(41.54, abs=0.01)


def test_terminal_growth_must_be_less_than_wacc(dcf_inputs):
    dcf_inputs["terminal_growth"] = dcf_inputs["wacc"]

    with pytest.raises(ValueError, match="Terminal growth must be less than WACC"):
        run_dcf(**dcf_inputs)


def test_mid_year_convention_increases_value(dcf_inputs):
    year_end_value = run_dcf(**dcf_inputs)["value_per_share"]
    mid_year_value = run_dcf(**dcf_inputs, mid_year=True)["value_per_share"]

    assert mid_year_value > year_end_value


def test_exit_multiple_ignores_terminal_growth_above_wacc(dcf_inputs):
    dcf_inputs["terminal_method"] = "exit"
    dcf_inputs["terminal_growth"] = dcf_inputs["wacc"] + 0.01

    result = run_dcf(**dcf_inputs)

    assert result["enterprise_value"] == pytest.approx(4253.81, abs=0.01)


def test_mid_year_discounts_exit_value_full_years_and_gordon_mid_year(dcf_inputs):
    years = len(dcf_inputs["growth_rates"])
    wacc_rate = dcf_inputs["wacc"]

    gordon = run_dcf(**dcf_inputs, mid_year=True)
    assert gordon["pv_terminal_value"] == pytest.approx(
        gordon["terminal_value"] / (1 + wacc_rate) ** (years - 0.5)
    )

    dcf_inputs["terminal_method"] = "exit"
    exit_result = run_dcf(**dcf_inputs, mid_year=True)
    assert exit_result["pv_terminal_value"] == pytest.approx(
        exit_result["terminal_value"] / (1 + wacc_rate) ** years
    )


@pytest.mark.parametrize("shares", [0, -1])
def test_diluted_shares_must_be_positive(dcf_inputs, shares):
    dcf_inputs["diluted_shares"] = shares

    with pytest.raises(ValueError, match="Diluted shares must be greater than zero"):
        run_dcf(**dcf_inputs)


def test_terminal_value_pct_is_none_when_enterprise_value_is_zero(dcf_inputs):
    dcf_inputs["base_revenue"] = 0

    result = run_dcf(**dcf_inputs)

    assert result["enterprise_value"] == 0
    assert result["terminal_value_pct_ev"] is None


def test_negative_equity_value_is_not_floored(dcf_inputs):
    dcf_inputs["debt"] = 10_000

    result = run_dcf(**dcf_inputs)

    assert result["equity_value"] == pytest.approx(4814.44 - 10_000 + 100, abs=0.01)
    assert result["value_per_share"] < 0


def test_terminal_capex_is_a_premium_over_da_not_todays_capex(dcf_inputs):
    # Today's capex (6%) is above D&A (5%); the terminal year must not repeat that.
    heavy = run_dcf(**{**dcf_inputs, "capex_pct_revenue": 0.40})
    light = run_dcf(**{**dcf_inputs, "capex_pct_revenue": 0.06})
    assert heavy["terminal_value"] == pytest.approx(light["terminal_value"])
    assert heavy["enterprise_value"] < light["enterprise_value"]  # years 1-5 differ


def test_terminal_year_working_capital_uses_terminal_growth_not_year_five(dcf_inputs):
    dcf_inputs["growth_rates"] = [0.20] * 5  # "flat" profile far above terminal growth
    dcf_inputs["delta_nwc_pct_revenue_change"] = 0.0
    no_nwc = run_dcf(**dcf_inputs)["terminal_value"]
    dcf_inputs["delta_nwc_pct_revenue_change"] = 0.10
    result = run_dcf(**dcf_inputs)
    revenue = result["forecast"]["Revenue"].iloc[-1]
    expected_drop = revenue * 0.03 * 0.10 / (0.088 - 0.03)  # 3% growth, not 20%
    assert no_nwc - result["terminal_value"] == pytest.approx(expected_drop)


def test_implied_exit_multiple_is_terminal_value_over_final_year_ebitda(dcf_inputs):
    result = run_dcf(**dcf_inputs)
    last = result["forecast"].iloc[-1]
    assert result["implied_exit_multiple"] == pytest.approx(
        result["terminal_value"] / (last["EBIT"] + last["D&A"])
    )
    assert result["implied_exit_multiple"] == pytest.approx(11.776, abs=0.001)


def test_implied_exit_multiple_is_year_end_comparable_under_mid_year(dcf_inputs):
    year_end = run_dcf(**dcf_inputs)["implied_exit_multiple"]
    mid = run_dcf(**dcf_inputs, mid_year=True)["implied_exit_multiple"]
    assert mid == pytest.approx(year_end * 1.088**0.5)


def test_implied_exit_multiple_absent_for_exit_method_or_negative_ebitda(dcf_inputs):
    assert run_dcf(**{**dcf_inputs, "terminal_method": "exit"})["implied_exit_multiple"] is None
    losing = run_dcf(**{**dcf_inputs, "ebit_margin": -0.5, "da_pct_revenue": 0.05})
    assert losing["implied_exit_multiple"] is None


def test_capex_fades_in_a_straight_line_to_the_terminal_level(dcf_inputs):
    forecast = run_dcf(**dcf_inputs)["forecast"]
    capex_pct = (forecast["CapEx"] / forecast["Revenue"]).tolist()
    # Today 6.0%; terminal level 1.1 x min(5% add-back, 6% capex) = 5.5%.
    assert capex_pct == pytest.approx([0.059, 0.058, 0.057, 0.056, 0.055])


def test_year_five_and_terminal_year_cash_flow_line_up(dcf_inputs):
    dcf_inputs["growth_rates"] = [0.03] * 5  # same growth, so only capex could differ
    result = run_dcf(**dcf_inputs)
    last = result["forecast"].iloc[-1]
    terminal_fcff = result["terminal_value"] * (0.088 - 0.03)
    terminal_revenue = last["Revenue"] * 1.03
    assert terminal_fcff / terminal_revenue == pytest.approx(
        last["FCFF"] / last["Revenue"]
    )


def test_terminal_capex_uses_the_smaller_of_addback_and_todays_capex(dcf_inputs):
    # Capex (2%) below the D&A add-back (5%): terminal capex is 1.1 x 2%, not 1.1 x 5%.
    dcf_inputs["capex_pct_revenue"] = 0.02
    forecast = run_dcf(**dcf_inputs)["forecast"]
    assert forecast["CapEx"].iloc[-1] / forecast["Revenue"].iloc[-1] == pytest.approx(
        0.022
    )


def test_addback_defaults_to_reported_da(dcf_inputs):
    default = run_dcf(**dcf_inputs)
    explicit = run_dcf(**dcf_inputs, da_addback_pct_revenue=0.05)
    assert explicit["value_per_share"] == pytest.approx(default["value_per_share"])
    assert (default["forecast"]["D&A add-back"] == default["forecast"]["D&A"]).all()


def test_limiting_the_addback_lowers_cash_flow_but_not_ebitda_or_exit_value(dcf_inputs):
    dcf_inputs["terminal_method"] = "exit"
    full = run_dcf(**dcf_inputs)
    limited = run_dcf(**dcf_inputs, da_addback_pct_revenue=0.01)
    # The exit multiple is applied to reported EBITDA (EBIT + all D&A).
    assert limited["terminal_value"] == pytest.approx(full["terminal_value"])
    assert (limited["forecast"]["D&A"] == full["forecast"]["D&A"]).all()
    # Year 1 has less add-back to count on. Later years differ because terminal
    # capex also scales with the add-back (1.1 x 1% here), so only year 1 is compared.
    assert limited["forecast"]["FCFF"].iloc[0] < full["forecast"]["FCFF"].iloc[0]
    assert limited["value_per_share"] < full["value_per_share"]


def test_implied_multiple_uses_reported_ebitda_even_with_a_limited_addback(dcf_inputs):
    result = run_dcf(**dcf_inputs, da_addback_pct_revenue=0.01)
    last = result["forecast"].iloc[-1]
    assert result["implied_exit_multiple"] == pytest.approx(
        result["terminal_value"] / (last["EBIT"] + last["D&A"])
    )


def test_negative_terminal_cash_flow_gives_no_implied_multiple(dcf_inputs):
    dcf_inputs["ebit_margin"] = 0.0  # EBITDA stays positive (D&A), terminal FCFF < 0
    result = run_dcf(**dcf_inputs)
    assert result["terminal_value"] < 0
    assert result["implied_exit_multiple"] is None
