import math

import streamlit as st

from model.assumptions import GROWTH_PROFILES, growth_profile_rates
from model.wacc import cost_of_equity, wacc

GROWTH_PROFILE_LABELS = {"fade": "Fade", "flat": "Flat", "custom": "Custom"}


def _initial_values(defaults):
    growth_rates = defaults.get("growth_rates") or [0.0] * 5
    growth_rates = list(growth_rates[:5])
    growth_rates.extend([0.0] * (5 - len(growth_rates)))

    return {
        **{
            f"dcf_growth_{year}": float(rate) * 100
            for year, rate in enumerate(growth_rates, start=1)
        },
        # Without a historical growth rate there is nothing to fade or hold flat.
        "dcf_growth_profile": (
            "fade" if defaults.get("starting_growth") is not None else "custom"
        ),
        "dcf_ebit_margin": float(defaults.get("ebit_margin") or 0.0) * 100,
        "dcf_tax_rate": float(defaults.get("tax_rate") or 0.0) * 100,
        "dcf_da_pct_revenue": float(defaults.get("da_pct_revenue") or 0.0) * 100,
        "dcf_capex_pct_revenue": float(defaults.get("capex_pct_revenue") or 0.0)
        * 100,
        "dcf_delta_nwc_pct_revenue_change": float(
            defaults.get("delta_nwc_pct_revenue_change") or 0.0
        )
        * 100,
        "dcf_risk_free_rate": float(defaults.get("risk_free_rate") or 0.0) * 100,
        "dcf_beta": float(defaults.get("beta") or 0.0),
        "dcf_erp": float(defaults.get("erp") or 0.0) * 100,
        "dcf_pre_tax_cost_of_debt": float(
            defaults.get("pre_tax_cost_of_debt") or 0.0
        )
        * 100,
        "dcf_override_debt_weight": False,
        "dcf_debt_weight_override": float(defaults.get("debt_weight") or 0.0) * 100,
        "dcf_terminal_method": defaults.get("terminal_method", "gordon"),
        "dcf_terminal_growth": float(defaults.get("terminal_growth") or 0.0) * 100,
        "dcf_exit_ebitda_multiple": max(
            0.0, float(defaults.get("exit_ebitda_multiple") or 0.0)
        ),
        "dcf_mid_year": bool(defaults.get("mid_year", False)),
        "dcf_include_securities": bool(defaults.get("include_securities", True)),
        "dcf_base_revenue": max(0.0, float(defaults.get("base_revenue") or 0.0)),
        "dcf_debt": max(0.0, float(defaults.get("debt") or 0.0)),
        "dcf_cash": max(0.0, float(defaults.get("cash") or 0.0)),
        "dcf_diluted_shares": max(
            0.0, float(defaults.get("diluted_shares") or 0.0)
        ),
    }


def _reset_controls(values):
    for key, value in values.items():
        st.session_state[key] = value


def _apply_growth_profile(starting_growth):
    """Refill the five growth sliders from the selected preset (Custom leaves them)."""
    rates = growth_profile_rates(
        starting_growth,
        st.session_state["dcf_terminal_growth"] / 100,
        st.session_state["dcf_growth_profile"],
    )
    if rates is None:
        return
    for year, rate in enumerate(rates, start=1):
        st.session_state[f"dcf_growth_{year}"] = rate * 100


def _growth_slider_moved():
    # A hand-set year no longer matches Fade or Flat, so label it honestly.
    st.session_state["dcf_growth_profile"] = "custom"


def _terminal_growth_moved(starting_growth):
    # Fade ends at terminal growth, so keep year 5 matched to the new rate.
    if st.session_state.get("dcf_growth_profile") == "fade":
        _apply_growth_profile(starting_growth)


def _growth_profile_caption(profile, starting_growth):
    if starting_growth is None:
        return "Historical revenue growth is unavailable, so set each year yourself."
    start = f"{starting_growth:.1%}"
    if profile == "fade":
        terminal = st.session_state.get("dcf_terminal_growth", 0.0) / 100
        if starting_growth > terminal:
            return (
                f"Starts at the historical growth rate ({start}) and steps down to "
                "the terminal growth rate by year 5, since growth usually slows as "
                "companies mature."
            )
        return (
            f"Starts at the historical growth rate ({start}) and steps up to the "
            "terminal growth rate by year 5, since recent growth has been below "
            "the long-term rate."
        )
    if profile == "flat":
        return f"Holds the historical growth rate ({start}) for all five years."
    return "Set each year yourself. Choose Fade or Flat to refill the sliders."


def _cash_for_securities_choice(included_cash, excluded_cash):
    if st.session_state["dcf_include_securities"]:
        st.session_state["dcf_cash"] = included_cash
    else:
        st.session_state["dcf_cash"] = excluded_cash


def _slider_bounds(default, minimum, maximum):
    return float(min(minimum, default)), float(max(maximum, default))


def _rate_slider(
    label, key, default, minimum, maximum, step=0.5, on_change=None, args=None
):
    default_percent = float(default or 0.0) * 100
    lower, upper = _slider_bounds(default_percent, minimum, maximum)
    return st.slider(
        label,
        min_value=lower,
        max_value=upper,
        step=step,
        format="%.1f%%",
        key=key,
        on_change=on_change,
        args=args,
        persist_state="session",
    ) / 100


def _different(value, default):
    if default is None:
        return value is not None and bool(value)
    if isinstance(value, (list, tuple)) and isinstance(default, (list, tuple)):
        return len(value) != len(default) or any(
            _different(item, original) for item, original in zip(value, default)
        )
    if isinstance(value, (int, float)) and isinstance(default, (int, float)):
        return not math.isclose(value, default, rel_tol=1e-9, abs_tol=1e-9)
    return value != default


def build_sidebar(ticker, defaults, company_data):
    """Build the assumption controls and return the live DCF inputs."""
    initial_values = _initial_values(defaults)
    if st.session_state.get("dcf_sidebar_ticker") != ticker:
        for key, value in initial_values.items():
            st.session_state[key] = value
        st.session_state["dcf_sidebar_ticker"] = ticker

    with st.sidebar:
        st.header("Valuation assumptions")
        st.caption("Adjust the starting estimates to see how value changes.")

        starting_growth = defaults.get("starting_growth")
        with st.expander("Growth", expanded=True):
            growth_profile = st.radio(
                "Growth profile",
                options=GROWTH_PROFILES,
                format_func=GROWTH_PROFILE_LABELS.get,
                horizontal=True,
                key="dcf_growth_profile",
                on_change=_apply_growth_profile,
                args=(starting_growth,),
                disabled=starting_growth is None,
                persist_state="session",
            )
            st.caption(_growth_profile_caption(growth_profile, starting_growth))
            growth_rates = [
                st.slider(
                    f"Year {year} revenue growth",
                    min_value=-10.0,
                    max_value=30.0,
                    step=0.5,
                    format="%.1f%%",
                    key=f"dcf_growth_{year}",
                    on_change=_growth_slider_moved,
                    persist_state="session",
                )
                / 100
                for year in range(1, 6)
            ]

        with st.expander("Profitability", expanded=True):
            ebit_margin = _rate_slider(
                "EBIT margin", "dcf_ebit_margin", defaults.get("ebit_margin"), -20, 60
            )
            tax_rate = _rate_slider(
                "Tax rate", "dcf_tax_rate", defaults.get("tax_rate"), 0, 50
            )

        with st.expander("Reinvestment", expanded=True):
            da_pct_revenue = _rate_slider(
                "D&A as a share of revenue",
                "dcf_da_pct_revenue",
                defaults.get("da_pct_revenue"),
                0,
                30,
            )
            capex_pct_revenue = _rate_slider(
                "CapEx as a share of revenue",
                "dcf_capex_pct_revenue",
                defaults.get("capex_pct_revenue"),
                0,
                50,
            )
            delta_nwc_pct_revenue_change = _rate_slider(
                "Working capital as a share of revenue change",
                "dcf_delta_nwc_pct_revenue_change",
                defaults.get("delta_nwc_pct_revenue_change"),
                -20,
                30,
            )

        with st.expander("Discount rate", expanded=True):
            risk_free_rate = _rate_slider(
                "Risk-free rate",
                "dcf_risk_free_rate",
                defaults.get("risk_free_rate"),
                0,
                15,
            )
            beta_default = float(defaults.get("beta") or 0.0)
            beta_min, beta_max = _slider_bounds(beta_default, 0.0, 4.0)
            beta = st.slider(
                "Beta",
                min_value=beta_min,
                max_value=beta_max,
                step=0.05,
                format="%.2f",
                key="dcf_beta",
                persist_state="session",
            )
            erp = _rate_slider(
                "Equity risk premium",
                "dcf_erp",
                defaults.get("erp"),
                0,
                15,
            )
            pre_tax_cost_of_debt = _rate_slider(
                "Pre-tax cost of debt",
                "dcf_pre_tax_cost_of_debt",
                defaults.get("pre_tax_cost_of_debt"),
                0,
                30,
            )
            override_debt_weight = st.checkbox(
                "Override debt weight",
                key="dcf_override_debt_weight",
                persist_state="session",
            )
            debt_weight_override = None
            if override_debt_weight:
                debt_weight_override = _rate_slider(
                    "Debt weight",
                    "dcf_debt_weight_override",
                    defaults.get("debt_weight"),
                    0,
                    100,
                )

            equity_cost = cost_of_equity(risk_free_rate, beta, erp)
            debt = st.session_state["dcf_debt"]
            diluted_shares = st.session_state["dcf_diluted_shares"]
            market_cap = company_data["current_price"] * diluted_shares
            total_capital = market_cap + debt

            if override_debt_weight:
                debt_weight = debt_weight_override
                equity_weight = 1 - debt_weight
                current_wacc = wacc(
                    equity_weight,
                    debt_weight,
                    equity_cost,
                    pre_tax_cost_of_debt,
                    tax_rate,
                )
            elif total_capital > 0:
                equity_weight = market_cap / total_capital
                debt_weight = debt / total_capital
                current_wacc = wacc(
                    market_cap,
                    debt,
                    equity_cost,
                    pre_tax_cost_of_debt,
                    tax_rate,
                )
            else:
                equity_weight = None
                debt_weight = None
                current_wacc = None

            st.metric("Cost of equity", f"{equity_cost:.2%}")
            st.metric(
                "WACC",
                f"{current_wacc:.2%}" if current_wacc is not None else "Unavailable",
            )
            st.caption(
                "Cost of equity uses the risk-free rate, beta, and equity risk premium."
            )
            st.caption(
                "WACC blends equity cost with after-tax debt cost using the selected "
                "weights."
            )

        with st.expander("Terminal value", expanded=True):
            terminal_method = st.radio(
                "Terminal value method",
                options=["gordon", "exit"],
                format_func=lambda method: (
                    "Gordon growth" if method == "gordon" else "Exit multiple"
                ),
                key="dcf_terminal_method",
                persist_state="session",
            )
            st.caption(
                "Gordon growth assumes steady long-term growth; an exit multiple "
                "uses final-year EBITDA."
            )
            if terminal_method == "gordon":
                _rate_slider(
                    "Terminal growth",
                    "dcf_terminal_growth",
                    defaults.get("terminal_growth"),
                    -5,
                    10,
                    on_change=_terminal_growth_moved,
                    args=(starting_growth,),
                )
            else:
                multiple_default = float(defaults.get("exit_ebitda_multiple") or 0.0)
                multiple_min, multiple_max = _slider_bounds(
                    multiple_default, 0.0, 30.0
                )
                exit_ebitda_multiple = st.slider(
                    "Exit EV / EBITDA",
                    min_value=multiple_min,
                    max_value=multiple_max,
                    step=0.5,
                    format="%.1fx",
                    key="dcf_exit_ebitda_multiple",
                    persist_state="session",
                )
            terminal_growth = st.session_state["dcf_terminal_growth"] / 100
            exit_ebitda_multiple = st.session_state["dcf_exit_ebitda_multiple"]

        with st.expander("Other", expanded=True):
            base_revenue = st.number_input(
                "Base revenue (USD millions)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="dcf_base_revenue",
                persist_state="session",
            )
            st.number_input(
                "Debt (USD millions)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="dcf_debt",
                persist_state="session",
            )
            cash_with_securities = float(defaults.get("reported_cash") or 0.0) + float(
                defaults.get("short_term_securities") or 0.0
            ) + float(defaults.get("long_term_securities") or 0.0)
            cash_without_securities = float(defaults.get("reported_cash") or 0.0)
            st.number_input(
                "Cash used (USD millions)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="dcf_cash",
                persist_state="session",
            )
            st.number_input(
                "Diluted shares (millions)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="dcf_diluted_shares",
                persist_state="session",
            )
            st.checkbox(
                "Include marketable securities in cash",
                key="dcf_include_securities",
                on_change=_cash_for_securities_choice,
                args=(cash_with_securities, cash_without_securities),
                persist_state="session",
            )
            mid_year = st.checkbox(
                "Use mid-year convention",
                key="dcf_mid_year",
                persist_state="session",
            )

        st.button(
            "Reset to defaults",
            key="dcf_reset_assumptions",
            on_click=_reset_controls,
            args=(initial_values,),
            width="stretch",
        )

    assumptions = {
        **defaults,
        "base_revenue": base_revenue,
        "growth_rates": growth_rates,
        "ebit_margin": ebit_margin,
        "tax_rate": tax_rate,
        "da_pct_revenue": da_pct_revenue,
        "capex_pct_revenue": capex_pct_revenue,
        "delta_nwc_pct_revenue_change": delta_nwc_pct_revenue_change,
        "wacc": current_wacc,
        "terminal_method": terminal_method,
        "terminal_growth": terminal_growth,
        "exit_ebitda_multiple": exit_ebitda_multiple,
        "debt": st.session_state["dcf_debt"],
        "cash": st.session_state["dcf_cash"],
        "diluted_shares": st.session_state["dcf_diluted_shares"],
        "mid_year": mid_year,
        "include_securities": st.session_state["dcf_include_securities"],
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "risk_free_rate": risk_free_rate,
        "beta": beta,
        "erp": erp,
        "pre_tax_cost_of_debt": pre_tax_cost_of_debt,
        "sources": dict(defaults["sources"]),
    }
    if override_debt_weight:
        assumptions["sources"]["debt_weight"] = "Debt weight overridden in the sidebar."
        assumptions["sources"]["equity_weight"] = (
            "Equity weight is the remainder after the debt weight override."
        )
    if current_wacc is not None:
        assumptions["sources"]["wacc"] = (
            "Live WACC from CAPM, the tax rate, and the selected debt/equity weights."
        )

    terminal_error = "Terminal growth must be below WACC for a stable DCF."
    assumptions["error"] = (defaults.get("error") or "").replace(
        terminal_error, ""
    ).strip(" .")
    assumptions["edited"] = {
        name
        for name in (
            "base_revenue",
            "growth_rates",
            "ebit_margin",
            "tax_rate",
            "da_pct_revenue",
            "capex_pct_revenue",
            "delta_nwc_pct_revenue_change",
            "wacc",
            "terminal_method",
            "terminal_growth",
            "exit_ebitda_multiple",
            "debt",
            "cash",
            "diluted_shares",
            "mid_year",
            "risk_free_rate",
            "beta",
            "erp",
            "pre_tax_cost_of_debt",
            "equity_weight",
            "debt_weight",
            "include_securities",
        )
        if _different(assumptions[name], defaults.get(name))
    }
    for name in assumptions["edited"]:
        assumptions["sources"][name] = (
            "Adjusted in the sidebar from the auto-derived starting value."
        )
    assumptions["growth_profile"] = growth_profile
    if "growth_rates" in assumptions["edited"]:
        if growth_profile == "flat":
            assumptions["sources"]["growth_rates"] = (
                "Flat profile: historical revenue CAGR, capped at -5% to 20%, held "
                "for all five years."
            )
        elif growth_profile == "fade":
            assumptions["sources"]["growth_rates"] = (
                "Fade profile: historical revenue CAGR, capped at -5% to 20%, fading "
                "to the terminal growth set in the sidebar."
            )

    if (
        terminal_method == "gordon"
        and current_wacc is not None
        and terminal_growth is not None
    ):
        if terminal_growth >= current_wacc:
            with st.sidebar:
                st.error(
                    "Terminal growth must be below WACC. Lower terminal growth or "
                    "increase the discount rate."
                )

    return assumptions
