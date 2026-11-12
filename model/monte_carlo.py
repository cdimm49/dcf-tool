import numpy as np

from model.dcf import TERMINAL_CAPEX_TO_DA


def _unit_draws(rng, distribution, size):
    """Zero-mean, unit-std draws; callers scale by the std they want."""
    if distribution == "normal":
        return rng.standard_normal(size)
    if distribution == "triangular":
        # A symmetric triangular distribution with std s has half-width s * sqrt(6).
        half_width = np.sqrt(6.0)
        return rng.triangular(-half_width, 0.0, half_width, size)
    raise ValueError("distribution must be 'normal' or 'triangular'.")


def run_monte_carlo_dcf(
    base_inputs,
    n_simulations=10000,
    distribution="normal",
    growth_std=0.02,
    growth_persistence=0.7,
    ebit_margin_std=0.015,
    wacc_std=0.0075,
    terminal_growth_std=0.005,
    exit_multiple_std=1.5,
    current_price=None,
    seed=42,
):
    """Vectorized Monte Carlo over run_dcf's math; arrays are (n_simulations, n_years)."""
    shares = base_inputs.get("diluted_shares")
    if shares is None or shares <= 0:
        raise ValueError("Diluted shares must be greater than zero.")
    terminal_method = base_inputs["terminal_method"]
    if terminal_method not in {"gordon", "exit"}:
        raise ValueError("terminal_method must be 'gordon' or 'exit'.")
    if not 0.0 <= growth_persistence <= 1.0:
        raise ValueError("growth_persistence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    base_growth = np.asarray(base_inputs["growth_rates"], dtype=float)
    n, n_years = n_simulations, len(base_growth)

    # One persistent shock per simulation plus yearly noise; the weights'
    # squares sum to 1, so every year's growth std is exactly growth_std.
    persistent = _unit_draws(rng, distribution, (n, 1))
    yearly = _unit_draws(rng, distribution, (n, n_years))
    growth = base_growth + growth_std * (
        growth_persistence * persistent
        + np.sqrt(1.0 - growth_persistence**2) * yearly
    )
    margin = base_inputs["ebit_margin"] + ebit_margin_std * _unit_draws(
        rng, distribution, (n, 1)
    )
    wacc = np.maximum(
        base_inputs["wacc"] + wacc_std * _unit_draws(rng, distribution, n), 0.01
    )
    terminal_growth = base_inputs["terminal_growth"] + terminal_growth_std * _unit_draws(
        rng, distribution, n
    )
    exit_multiple = np.maximum(
        base_inputs["exit_ebitda_multiple"]
        + exit_multiple_std * _unit_draws(rng, distribution, n),
        1.0,
    )

    # Same operation order as run_dcf so zero-std runs match it to the last bits.
    revenue = np.empty((n, n_years))
    previous = np.full(n, float(base_inputs["base_revenue"]))
    for year in range(n_years):
        revenue[:, year] = previous * (1 + growth[:, year])
        previous = revenue[:, year]
    prior_revenue = np.column_stack(
        [np.full(n, float(base_inputs["base_revenue"])), revenue[:, :-1]]
    )
    ebit = revenue * margin
    nopat = ebit * (1 - base_inputs["tax_rate"])
    da = revenue * base_inputs["da_pct_revenue"]
    addback_pct = base_inputs.get("da_addback_pct_revenue")
    if addback_pct is None:
        addback_pct = base_inputs["da_pct_revenue"]
    capex_today = base_inputs["capex_pct_revenue"]
    terminal_capex_pct = TERMINAL_CAPEX_TO_DA * min(addback_pct, capex_today)
    # Capex fades in a straight line from today's share to the terminal level.
    capex_pct = capex_today + (terminal_capex_pct - capex_today) * (
        np.arange(1, n_years + 1) / n_years
    )
    capex = revenue * capex_pct
    delta_nwc = (revenue - prior_revenue) * base_inputs["delta_nwc_pct_revenue_change"]
    fcff = nopat + revenue * addback_pct - capex - delta_nwc

    years = np.arange(1, n_years + 1, dtype=float)
    mid_year = bool(base_inputs.get("mid_year", False))
    periods = years - 0.5 if mid_year else years
    discount_factor = 1 / (1 + wacc[:, None]) ** periods
    sum_pv_fcff = (fcff * discount_factor).sum(axis=1)

    if terminal_method == "gordon":
        cap = wacc - 0.005
        clamped = terminal_growth > cap
        terminal_growth = np.minimum(terminal_growth, cap)
        # Same normalized terminal year as run_dcf.
        terminal_revenue = revenue[:, -1] * (1 + terminal_growth)
        terminal_fcff = (
            terminal_revenue * margin[:, 0] * (1 - base_inputs["tax_rate"])
            + terminal_revenue * addback_pct
            - terminal_revenue * terminal_capex_pct
            - (terminal_revenue - revenue[:, -1])
            * base_inputs["delta_nwc_pct_revenue_change"]
        )
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        pv_terminal_value = terminal_value * discount_factor[:, -1]
        pct_clamped = float(clamped.mean())
    else:
        terminal_value = (ebit[:, -1] + da[:, -1]) * exit_multiple
        if mid_year:
            pv_terminal_value = terminal_value / (1 + wacc) ** n_years
        else:
            pv_terminal_value = terminal_value * discount_factor[:, -1]
        pct_clamped = 0.0

    enterprise_value = sum_pv_fcff + pv_terminal_value
    equity_value = enterprise_value - base_inputs["debt"] + base_inputs["cash"]
    share_prices = equity_value / shares

    percentiles = np.percentile(share_prices, [5, 10, 25, 50, 75, 90, 95])
    summary = {
        "mean": float(share_prices.mean()),
        "std": float(share_prices.std()),
        "min": float(share_prices.min()),
        "max": float(share_prices.max()),
        **{
            f"p{q}": float(v)
            for q, v in zip((5, 10, 25, 50, 75, 90, 95), percentiles)
        },
        "prob_negative_equity": float((equity_value < 0).mean()),
        "pct_terminal_growth_clamped": pct_clamped,
    }
    if current_price is not None:
        summary["prob_above_market"] = float((share_prices > current_price).mean())

    return {
        "simulated_share_prices": share_prices,
        "simulated_ev": enterprise_value,
        "summary_stats": summary,
    }
