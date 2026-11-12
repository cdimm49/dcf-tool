import time

import numpy as np
import pytest

from model.dcf import run_dcf
from model.monte_carlo import run_monte_carlo_dcf

ZERO_STD = dict(
    growth_std=0,
    ebit_margin_std=0,
    wacc_std=0,
    terminal_growth_std=0,
    exit_multiple_std=0,
)


@pytest.fixture
def base_inputs():
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
        "mid_year": False,
    }


@pytest.mark.parametrize("distribution", ["normal", "triangular"])
@pytest.mark.parametrize("terminal_method", ["gordon", "exit"])
@pytest.mark.parametrize("mid_year", [False, True])
def test_zero_std_matches_run_dcf(base_inputs, terminal_method, mid_year, distribution):
    base_inputs.update(terminal_method=terminal_method, mid_year=mid_year)
    expected = run_dcf(**base_inputs)["value_per_share"]

    result = run_monte_carlo_dcf(
        base_inputs, n_simulations=500, distribution=distribution, **ZERO_STD
    )

    np.testing.assert_allclose(
        result["simulated_share_prices"], expected, rtol=0, atol=1e-9
    )
    assert result["summary_stats"]["pct_terminal_growth_clamped"] == 0.0


def test_terminal_growth_near_wacc_stays_finite(base_inputs):
    base_inputs.update(wacc=0.05, terminal_growth=0.049)

    result = run_monte_carlo_dcf(
        base_inputs, wacc_std=0.03, terminal_growth_std=0.03, growth_std=0.1
    )

    assert np.isfinite(result["simulated_share_prices"]).all()
    assert np.isfinite(result["simulated_ev"]).all()
    assert all(np.isfinite(v) for v in result["summary_stats"].values())
    assert result["summary_stats"]["pct_terminal_growth_clamped"] > 0.5


def test_seed_reproducibility(base_inputs):
    first = run_monte_carlo_dcf(base_inputs, seed=7)
    again = run_monte_carlo_dcf(base_inputs, seed=7)
    other = run_monte_carlo_dcf(base_inputs, seed=8)

    np.testing.assert_array_equal(
        first["simulated_share_prices"], again["simulated_share_prices"]
    )
    assert first["summary_stats"] == again["summary_stats"]
    assert not np.array_equal(
        first["simulated_share_prices"], other["simulated_share_prices"]
    )


def test_ten_thousand_simulations_under_two_seconds(base_inputs):
    start = time.perf_counter()
    result = run_monte_carlo_dcf(base_inputs, n_simulations=10_000, current_price=50)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0
    assert result["simulated_share_prices"].shape == (10_000,)
    assert 0.0 <= result["summary_stats"]["prob_above_market"] <= 1.0


@pytest.mark.parametrize("shares", [None, 0, -5])
def test_missing_or_nonpositive_shares_raise(base_inputs, shares):
    base_inputs["diluted_shares"] = shares
    with pytest.raises(ValueError):
        run_monte_carlo_dcf(base_inputs)


def test_zero_std_matches_run_dcf_with_heavy_capex_and_fast_final_year(base_inputs):
    # The terminal-year normalization is what differs from the old formula here.
    base_inputs.update(
        capex_pct_revenue=0.40, growth_rates=[0.2] * 5, delta_nwc_pct_revenue_change=0.15
    )
    expected = run_dcf(**base_inputs)["value_per_share"]
    result = run_monte_carlo_dcf(base_inputs, n_simulations=200, **ZERO_STD)
    np.testing.assert_allclose(result["simulated_share_prices"], expected, rtol=0, atol=1e-9)
