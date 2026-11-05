from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from data.clean import add_income_margins, clean_statement
from data.fetch import SEC_SOURCE_NOTE, get_company_data
from data.news import get_recent_news
from data.snapshots import available_snapshots, load_saved_news
from model.assumptions import DCF_INPUTS, default_assumptions
from model.dcf import run_dcf
from model.monte_carlo import run_monte_carlo_dcf
from model.multiples import calculate_multiples
from ui.sidebar import build_sidebar
from ui.tables import display_statement


_PERCENT_INPUTS = {
    "ebit_margin",
    "tax_rate",
    "da_pct_revenue",
    "capex_pct_revenue",
    "delta_nwc_pct_revenue_change",
    "wacc",
    "terminal_growth",
    "risk_free_rate",
    "erp",
    "pre_tax_cost_of_debt",
    "equity_weight",
    "debt_weight",
}


def _format_assumption(name, value):
    if value is None:
        return "Unavailable"
    if name in _PERCENT_INPUTS:
        return f"{value:.2%}"
    if name == "growth_rates":
        return ", ".join(f"{rate:.1%}" for rate in value)
    if name == "exit_ebitda_multiple":
        return f"{value:.1f}x"
    if name == "beta":
        return f"{value:.2f}"
    if name == "diluted_shares":
        return f"{value:,.1f} million"
    if name in {
        "base_revenue",
        "debt",
        "cash",
        "reported_cash",
        "short_term_securities",
        "long_term_securities",
    }:
        return f"${value:,.1f} million"
    return str(value)


def _format_currency(value):
    return f"(${abs(value):,.1f})" if value < 0 else f"${value:,.1f}"


def _format_multiple(metric, value):
    if pd.isna(value):
        return "—"
    if metric == "FCF yield":
        return f"{value:.1%}"
    return f"{value:.1f}x"


def _is_financial_company(company_data):
    sector = (company_data.get("sector") or "").casefold()
    industry = (company_data.get("industry") or "").casefold()
    return (
        "financial" in sector
        or "bank" in industry
        or "insurance" in industry
        or "insurer" in industry
    )


def _friendly_ticker_error(fetched_data, symbol):
    details = str(fetched_data.get("error", "")).casefold()
    network_terms = (
        "currently down",
        "network",
        "connection",
        "timed out",
        "timeout",
        "rate limit",
        "too many requests",
        "curl",
        "service unavailable",
    )
    saved = available_snapshots()
    saved_note = f" Saved data is available for: {', '.join(saved)}." if saved else ""
    if "isn't in the sec's list" in details:
        return (
            f"We couldn't find {symbol}. Check the symbol. This site reads "
            "statements from SEC filings, so it supports US-listed companies."
            + saved_note
        )
    if any(term in details for term in network_terms):
        return "Yahoo Finance is temporarily unavailable. Please try again later."

    message = (
        f"We couldn't load data for {symbol}. Check the symbol, or Yahoo Finance "
        "may be limiting this server right now."
    )
    return message + saved_note


def _relative_time(published):
    if published is None:
        return "Publication time unavailable"

    now = datetime.now(timezone.utc)
    seconds = int((now - published).total_seconds())
    if abs(seconds) < 60:
        return "just now" if seconds >= 0 else "in less than a minute"

    is_future = seconds < 0
    seconds = abs(seconds)
    if seconds < 3600:
        amount, unit = seconds // 60, "minute"
    elif seconds < 86400:
        amount, unit = seconds // 3600, "hour"
    else:
        amount, unit = seconds // 86400, "day"
    label = f"{amount} {unit}{'' if amount == 1 else 's'}"
    return f"in {label}" if is_future else f"{label} ago"


def _display_news(ticker):
    st.caption("Recent headlines for the selected company.")
    articles, live_error = [], None
    try:
        with st.spinner("Fetching recent headlines..."):
            articles = get_recent_news(ticker)
    except Exception as error:
        live_error = error

    if not articles:
        saved_articles, saved_date = load_saved_news(ticker)
        if saved_articles:
            st.caption(
                "Live headlines aren't available right now. Showing headlines "
                f"saved on {saved_date}."
            )
            articles = saved_articles
        elif live_error is not None:
            st.warning(
                "We couldn't fetch headlines right now. Please try again later."
            )
            with st.expander("Details"):
                st.code(str(live_error))
            return
        else:
            st.info("No recent headlines were found for this ticker.")
            return

    for article in articles[:15]:
        st.link_button(article["title"], article["url"])
        st.caption(
            f"{article['source']} · {_relative_time(article['published'])}"
        )
        if article["summary"]:
            summary = article["summary"]
            if len(summary) > 280:
                summary = f"{summary[:277].rstrip()}..."
            st.write(summary)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_company_multiples(ticker, include_securities):
    """Fetch and calculate a company's multiples once per ticker and cash setting."""
    company_data = get_company_data(ticker)
    if "error" in company_data:
        return {"error": company_data["error"]}

    try:
        income_statement = clean_statement(
            company_data["income_statement"], "income"
        )
        balance_sheet = clean_statement(company_data["balance_sheet"], "balance")
        cash_flow_statement = clean_statement(
            company_data["cash_flow_statement"], "cashflow"
        )
        multiples = calculate_multiples(
            income_statement,
            balance_sheet,
            cash_flow_statement,
            {
                "current_price": company_data["current_price"],
                "diluted_shares_outstanding": company_data[
                    "diluted_shares_outstanding"
                ],
            },
            include_securities=include_securities,
        )
    except Exception as error:
        return {"error": f"Could not calculate multiples for {ticker}: {error}"}

    return {
        "company_name": company_data["company_name"],
        "current_price": company_data["current_price"],
        "multiples": multiples,
    }


def _display_assumptions(assumptions):
    labels = {
        "base_revenue": "Base revenue",
        "growth_rates": "Revenue growth by year",
        "ebit_margin": "EBIT margin",
        "tax_rate": "Tax rate",
        "da_pct_revenue": "D&A as % of revenue",
        "capex_pct_revenue": "CapEx as % of revenue",
        "delta_nwc_pct_revenue_change": "Change in working capital",
        "wacc": "WACC",
        "terminal_method": "Terminal value method",
        "terminal_growth": "Terminal growth",
        "exit_ebitda_multiple": "Exit EV / EBITDA",
        "debt": "Debt",
        "cash": "Cash",
        "reported_cash": "Reported cash and cash equivalents",
        "short_term_securities": "Short-term marketable securities",
        "long_term_securities": "Long-term marketable securities",
        "include_securities": "Include securities in cash",
        "diluted_shares": "Diluted shares",
        "mid_year": "Mid-year convention",
        "risk_free_rate": "Risk-free rate",
        "beta": "Beta",
        "erp": "Equity risk premium",
        "pre_tax_cost_of_debt": "Pre-tax cost of debt",
        "equity_weight": "Equity weight",
        "debt_weight": "Debt weight",
    }
    inputs = (
        *DCF_INPUTS,
        "risk_free_rate",
        "beta",
        "erp",
        "pre_tax_cost_of_debt",
        "equity_weight",
        "debt_weight",
        "reported_cash",
        "short_term_securities",
        "long_term_securities",
        "include_securities",
    )
    assumptions_table = pd.DataFrame(
        [
            {
                "Input": (
                    f"{labels[name]} (edited)"
                    if name in assumptions.get("edited", set())
                    else labels[name]
                ),
                "Value": _format_assumption(name, assumptions[name]),
                "Source": assumptions["sources"][name],
            }
            for name in inputs
        ]
    )
    st.subheader("Assumptions used")
    st.dataframe(
        assumptions_table,
        hide_index=True,
        alt="DCF inputs with their derived values and sources",
    )


def _display_valuation(assumptions, company_data):
    if (
        assumptions["terminal_method"] == "gordon"
        and assumptions["wacc"] is not None
        and assumptions["terminal_growth"] is not None
        and assumptions["terminal_growth"] >= assumptions["wacc"]
    ):
        st.error(
            "Terminal growth must be below WACC. Lower terminal growth or increase "
            "the discount rate."
        )
        _display_assumptions(assumptions)
        return None

    if assumptions["error"]:
        st.warning(
            f"A DCF could not be calculated: {assumptions['error']} "
            "A DCF may not fit this company type or the available data."
        )
        _display_assumptions(assumptions)
        return None

    if assumptions["diluted_shares"] <= 0:
        st.warning(
            "Diluted shares must be greater than zero to calculate value per share. "
            "Enter a share count in the sidebar."
        )
        _display_assumptions(assumptions)
        return None

    result = run_dcf(
        **{name: assumptions[name] for name in DCF_INPUTS}
    )
    current_price = company_data["current_price"]
    upside = (result["value_per_share"] / current_price - 1) * 100

    st.subheader("What the business may be worth")
    st.caption(
        "This estimate projects future cash flow, discounts it to today, and compares "
        "the result with the current share price."
    )
    value_col, price_col, upside_col = st.columns(3)
    value_col.metric("Implied value per share", f"${result['value_per_share']:,.2f}")
    price_col.metric("Current share price", f"${current_price:,.2f}")
    upside_col.metric("Upside / downside", f"{upside:+.1f}%")

    if result["equity_value"] < 0:
        st.warning(
            "Equity value is negative: net debt exceeds the business's estimated "
            "enterprise value, so shareholders' claim is worth less than nothing "
            "on these assumptions."
        )

    _display_sensitivity(assumptions)

    st.subheader("Five-year cash flow forecast")
    st.caption("Amounts are in USD millions; each year's cash flow is discounted to today.")
    forecast = result["forecast"].style.format(
        {
            column: "${:,.1f}"
            for column in forecast_columns(result["forecast"])
        }
        | {"Discount Factor": "{:.3f}"}
    )
    st.dataframe(forecast, alt="Five-year DCF revenue and cash flow forecast")

    st.subheader("Enterprise value to equity value")
    bridge = pd.DataFrame(
        {
            "Step": [
                "Present value of forecast cash flows",
                "Present value of terminal value",
                "Enterprise value",
                "Less: debt",
                "Add: cash",
                "Equity value",
            ],
            "USD millions": [
                result["sum_pv_fcff"],
                result["pv_terminal_value"],
                result["enterprise_value"],
                -assumptions["debt"],
                assumptions["cash"],
                result["equity_value"],
            ],
        }
    ).set_index("Step")
    st.dataframe(
        bridge.style.format({"USD millions": _format_currency}),
        alt="Bridge from enterprise value to equity value",
    )

    terminal_share = result["terminal_value_pct_ev"]
    if terminal_share is not None:
        st.metric("Terminal value share of enterprise value", f"{terminal_share:.1%}")
    if terminal_share is not None and terminal_share > 0.75:
        st.warning(
            "More than 75% of enterprise value comes from the terminal value, "
            "so the estimate is especially sensitive to long-term assumptions."
        )

    _display_assumptions(assumptions)
    return result


def _display_multiples(
    ticker, company_data, assumptions, valuation_result
):
    st.caption(
        "Multiples compare a company's price or total business value with its "
        "reported earnings, sales, assets, or cash flow."
    )
    include_securities = assumptions["include_securities"]
    selected = _cached_company_multiples(ticker, include_securities)
    if "error" in selected:
        st.error(selected["error"])
        return

    descriptions = {
        "P/E": "Share price for each dollar of the latest diluted EPS.",
        "EV/EBITDA": "Business value for each dollar of the latest EBITDA.",
        "EV/Sales": "Business value for each dollar of the latest sales.",
        "P/B": "Market value for each dollar of shareholders’ equity.",
        "FCF yield": "Free cash flow as a share of market value.",
    }
    st.subheader("Selected company")
    selected_table = pd.DataFrame(
        [
            {
                "Multiple": metric,
                "Value": _format_multiple(metric, selected["multiples"][metric]),
                "What it means": descriptions[metric],
            }
            for metric in descriptions
        ]
    )
    st.dataframe(
        selected_table,
        hide_index=True,
        alt="Selected company valuation multiples and plain-language descriptions",
    )

    peer_median_pe = float("nan")
    with st.form("multiples_peer_form"):
        peer_input = st.text_input(
            "Peer tickers",
            placeholder="MSFT, GOOGL, AMZN",
            help="Enter 3 to 5 comma-separated ticker symbols.",
        )
        compare_peers = st.form_submit_button("Compare peers")

    if compare_peers:
        peer_tickers = [
            peer.strip().upper() for peer in peer_input.split(",") if peer.strip()
        ]
        if not 3 <= len(peer_tickers) <= 5:
            st.session_state["multiples_peer_error"] = (
                "Enter 3 to 5 comma-separated peer tickers."
            )
            st.session_state.pop("multiples_peer_tickers", None)
        elif len(set(peer_tickers)) != len(peer_tickers):
            st.session_state["multiples_peer_error"] = (
                "Each peer ticker must be unique."
            )
            st.session_state.pop("multiples_peer_tickers", None)
        elif ticker.upper() in peer_tickers:
            st.session_state["multiples_peer_error"] = (
                "The selected company is already shown; enter different peer tickers."
            )
            st.session_state.pop("multiples_peer_tickers", None)
        else:
            st.session_state["multiples_peer_error"] = None
            st.session_state["multiples_peer_tickers"] = peer_tickers

    if st.session_state.get("multiples_peer_error"):
        st.warning(st.session_state["multiples_peer_error"])

    peer_tickers = st.session_state.get("multiples_peer_tickers", [])
    peer_values = []
    if peer_tickers:
        rows = [
            {
                "Company": f"{selected['company_name']} ({ticker.upper()})",
                **selected["multiples"],
            }
        ]
        for peer_ticker in peer_tickers:
            peer = _cached_company_multiples(peer_ticker, include_securities)
            if "error" in peer:
                st.warning(f"{peer_ticker}: {peer['error']}")
                peer_values.append({metric: float("nan") for metric in descriptions})
                rows.append(
                    {
                        "Company": peer_ticker,
                        **peer_values[-1],
                    }
                )
            else:
                peer_values.append(peer["multiples"])
                rows.append(
                    {
                        "Company": f"{peer['company_name']} ({peer_ticker})",
                        **peer["multiples"],
                    }
                )

        peer_frame = pd.DataFrame(peer_values)
        peer_median = peer_frame.median(numeric_only=True)
        peer_median_pe = peer_median.get("P/E", float("nan"))
        rows.append(
            {
                "Company": "Median (peers)",
                **{
                    metric: peer_median.get(metric, float("nan"))
                    for metric in descriptions
                },
            }
        )
        formatted_rows = [
            {
                "Company": row["Company"],
                **{
                    metric: _format_multiple(metric, row[metric])
                    for metric in descriptions
                },
            }
            for row in rows
        ]
        st.subheader("Peer comparison")
        st.dataframe(
            pd.DataFrame(formatted_rows),
            hide_index=True,
            alt="Selected company and peer valuation multiples, including peer medians",
        )

    implied_pe = float("nan")
    if (
        valuation_result is not None
        and valuation_result["value_per_share"] > 0
        and company_data["current_price"] > 0
        and not pd.isna(selected["multiples"]["P/E"])
    ):
        implied_pe = (
            selected["multiples"]["P/E"]
            * valuation_result["value_per_share"]
            / company_data["current_price"]
        )

    with st.container(border=True):
        st.markdown("**DCF cross-check**")
        price_col, value_col = st.columns(2)
        price_col.metric("Current price", f"${company_data['current_price']:,.2f}")
        if valuation_result is None:
            value_col.metric("DCF-implied value per share", "—")
        else:
            value_col.metric(
                "DCF-implied value per share",
                f"${valuation_result['value_per_share']:,.2f}",
            )
        st.write(f"Implied P/E at the DCF value: {_format_multiple('P/E', implied_pe)}")
        if pd.isna(implied_pe) or pd.isna(peer_median_pe):
            st.caption(
                "A comparison is unavailable until both the DCF-implied P/E and "
                "the peer median P/E are available."
            )
        else:
            st.caption(f"Peer median P/E: {_format_multiple('P/E', peer_median_pe)}")
            if implied_pe > peer_median_pe:
                st.caption("The DCF-implied P/E is above the peer median P/E.")
            elif implied_pe < peer_median_pe:
                st.caption("The DCF-implied P/E is below the peer median P/E.")
            else:
                st.caption("The DCF-implied P/E is equal to the peer median P/E.")


def _display_sensitivity(assumptions):
    st.subheader("Sensitivity: value per share")
    st.caption(
        "Each cell shows how value changes when the discount rate and terminal "
        "assumption move together."
    )
    wacc_values = [
        assumptions["wacc"] + (step - 4) * 0.005 for step in range(9)
    ]
    base_inputs = {name: assumptions[name] for name in DCF_INPUTS}

    if assumptions["terminal_method"] == "gordon":
        terminal_values = [0.015 + step * 0.005 for step in range(5)]
        column_labels = [f"{value:.1%}" for value in terminal_values]
        row_labels = [f"{value:.2%}" for value in wacc_values]
        rows = []
        for discount_rate in wacc_values:
            row = []
            for terminal_growth in terminal_values:
                if terminal_growth >= discount_rate:
                    row.append(None)
                    continue
                inputs = {
                    **base_inputs,
                    "wacc": discount_rate,
                    "terminal_growth": terminal_growth,
                }
                row.append(run_dcf(**inputs)["value_per_share"])
            rows.append(row)
        table = pd.DataFrame(rows, index=row_labels, columns=column_labels)
        table.index.name = "WACC"
        table.columns.name = "Terminal growth"
        closest_column = min(
            range(len(terminal_values)),
            key=lambda index: abs(
                terminal_values[index] - assumptions["terminal_growth"]
            ),
        )
    else:
        terminal_values = [
            round(assumptions["exit_ebitda_multiple"] + offset, 8)
            for offset in (-10, -5, 0, 5, 10)
        ]
        column_labels = [f"{value:.1f}x" for value in terminal_values]
        row_labels = [f"{value:.2%}" for value in wacc_values]
        rows = []
        for discount_rate in wacc_values:
            row = []
            for multiple in terminal_values:
                inputs = {
                    **base_inputs,
                    "wacc": discount_rate,
                    "exit_ebitda_multiple": multiple,
                }
                row.append(run_dcf(**inputs)["value_per_share"])
            rows.append(row)
        table = pd.DataFrame(rows, index=row_labels, columns=column_labels)
        table.index.name = "WACC"
        table.columns.name = "Exit EV / EBITDA"
        closest_column = min(
            range(len(terminal_values)),
            key=lambda index: abs(
                terminal_values[index] - assumptions["exit_ebitda_multiple"]
            ),
        )

    closest_row = min(
        range(len(wacc_values)),
        key=lambda index: abs(wacc_values[index] - assumptions["wacc"]),
    )

    def highlight_closest_cell(frame):
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        styles.iloc[closest_row, closest_column] = (
            "background-color: #1f6feb; color: white; font-weight: bold;"
        )
        return styles

    st.dataframe(
        table.style.format(
            lambda value: (
                "—" if pd.isna(value) else f"${value:,.2f}"
            )
        ).apply(highlight_closest_cell, axis=None),
        alt="Value per share across discount-rate and terminal-value assumptions",
    )


def _display_monte_carlo(assumptions, company_data, valuation_result):
    st.subheader("Monte Carlo: a range of values instead of one number")
    st.caption(
        "The DCF above gives one answer for one set of assumptions. This runs the "
        "same model thousands of times, each time nudging revenue growth, EBIT "
        "margin, WACC, and the terminal assumption at random around the sidebar "
        "values, to show how wide the plausible range is."
    )
    if valuation_result is None:
        st.info("Fix the issue shown on the Valuation tab to run the simulation.")
        return

    gordon = assumptions["terminal_method"] == "gordon"
    sims_col, dist_col, seed_col = st.columns(3)
    n_simulations = sims_col.selectbox(
        "Simulations", [1_000, 5_000, 10_000, 25_000], index=2, format_func="{:,}".format
    )
    distribution = dist_col.radio("Distribution", ["Normal", "Triangular"], horizontal=True)
    lock_seed = seed_col.checkbox(
        "Lock random seed",
        value=True,
        help="Keep the same random draws between reruns so results only change "
        "when you change an input.",
    )

    growth_col, margin_col, wacc_col, terminal_col = st.columns(4)
    growth_std = growth_col.slider(
        "Revenue growth std (pts)", 0.0, 10.0, 2.0, 0.25, format="%.2f%%"
    )
    margin_std = margin_col.slider(
        "EBIT margin std (pts)", 0.0, 5.0, 1.5, 0.25, format="%.2f%%"
    )
    wacc_std = wacc_col.slider("WACC std (pts)", 0.0, 3.0, 0.75, 0.25, format="%.2f%%")
    if gordon:
        terminal_growth_std = terminal_col.slider(
            "Terminal growth std (pts)", 0.0, 2.0, 0.5, 0.1, format="%.1f%%"
        )
        exit_multiple_std = 0.0
    else:
        exit_multiple_std = terminal_col.slider(
            "Exit multiple std (x)", 0.0, 5.0, 1.5, 0.25, format="%.2fx"
        )
        terminal_growth_std = 0.0

    current_price = company_data["current_price"]
    simulation = run_monte_carlo_dcf(
        {name: assumptions[name] for name in DCF_INPUTS},
        n_simulations=n_simulations,
        distribution=distribution.lower(),
        growth_std=growth_std / 100,
        ebit_margin_std=margin_std / 100,
        wacc_std=wacc_std / 100,
        terminal_growth_std=terminal_growth_std / 100,
        exit_multiple_std=exit_multiple_std,
        current_price=current_price,
        seed=42 if lock_seed else None,
    )
    prices = simulation["simulated_share_prices"]
    stats = simulation["summary_stats"]

    median_col, mean_col, range_col, above_col = st.columns(4)
    median_col.metric("Median value (P50)", f"${stats['p50']:,.2f}")
    mean_col.metric("Mean value", f"${stats['mean']:,.2f}")
    range_col.metric("80% range (P10–P90)", f"${stats['p10']:,.0f} – ${stats['p90']:,.0f}")
    above_col.metric("Chance above current price", f"{stats['prob_above_market']:.0%}")
    st.caption(
        "Read the 80% range as: in 8 of 10 simulated outcomes, the value per share "
        "landed between these two numbers. The median is the middle outcome; the "
        "mean is pulled toward the long tail of high values."
    )

    # Bin with numpy so the chart stays small at any simulation count. The
    # extreme 0.5% tails are left out of the bars so a few runaway terminal
    # values don't squash the rest of the histogram.
    low, high = np.percentile(prices, [0.5, 99.5])
    counts, edges = np.histogram(prices, bins=60, range=(low, high))
    bars = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "share": counts / len(prices)})
    markers = pd.DataFrame(
        {
            "Marker": ["Current price", "Base-case DCF", "P10", "P50", "P90"],
            "Value": [
                current_price,
                valuation_result["value_per_share"],
                stats["p10"],
                stats["p50"],
                stats["p90"],
            ],
        }
    )
    # Zoom the x-axis to the bars plus any marker within one chart-width of them.
    # A marker much further away (a price far above every simulated value, say)
    # would squash the histogram into hairlines, so it goes in a caption instead.
    span = max(high - low, 1e-9)
    nearby = markers["Value"].between(low - span, high + span)
    shown = markers[nearby]
    view_low = min(low, shown["Value"].min())
    view_high = max(high, shown["Value"].max())
    padding = (view_high - view_low) * 0.03
    x_scale = alt.Scale(
        domain=[view_low - padding, view_high + padding], zero=False, nice=False
    )
    histogram = alt.Chart(bars).mark_bar(opacity=0.75).encode(
        x=alt.X(
            "start:Q",
            title="Value per share (USD)",
            scale=x_scale,
            axis=alt.Axis(format="$,.0f"),
        ),
        x2="end:Q",
        y=alt.Y("share:Q", title="Share of simulations", axis=alt.Axis(format=".0%")),
        y2=alt.datum(0),  # anchor the bars at zero so they draw as columns
        tooltip=[
            alt.Tooltip("start:Q", title="From", format="$,.2f"),
            alt.Tooltip("end:Q", title="To", format="$,.2f"),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
        ],
    )
    rules = alt.Chart(shown).mark_rule(strokeWidth=2).encode(
        x=alt.X("Value:Q", scale=x_scale),
        color=alt.Color(
            "Marker:N",
            sort=list(markers["Marker"]),
            scale=alt.Scale(domain=list(shown["Marker"])),
            legend=alt.Legend(orient="bottom"),
        ),
        strokeDash=alt.condition(
            alt.FieldOneOfPredicate("Marker", ["P10", "P50", "P90"]),
            alt.value([4, 3]),
            alt.value([1, 0]),
        ),
        tooltip=["Marker:N", alt.Tooltip("Value:Q", format="$,.2f")],
    )
    st.altair_chart(
        histogram + rules,
        width="stretch",
        alt="Histogram of simulated values per share with price and percentile markers",
    )
    off_chart = markers[~nearby]
    if not off_chart.empty:
        side = lambda value: "right" if value > high else "left"
        st.caption(
            "Off the chart to keep the bars readable: "
            + ", ".join(
                f"{row.Marker} (${row.Value:,.2f}, to the {side(row.Value)})"
                for row in off_chart.itertuples()
            )
            + "."
        )
    st.caption(
        "Each bar is the share of simulations that landed in that price band. "
        "The more of the histogram that sits right of the current-price line, the "
        "more often the model says the stock is undervalued. The most extreme 1% "
        "of outcomes are left off the chart but included in every statistic."
    )

    labels = ["P5", "P10", "P25", "P50", "P75", "P90", "P95"]
    values = [stats[label.lower()] for label in labels]
    percentile_table = pd.DataFrame(
        {
            "Value per share": values,
            "Upside / downside": [(value / current_price - 1) * 100 for value in values],
        },
        index=pd.Index(labels, name="Percentile"),
    )
    st.dataframe(
        percentile_table.style.format(
            {"Value per share": "${:,.2f}", "Upside / downside": "{:+.1f}%"}
        ),
        alt="Simulated value per share at each percentile versus the current price",
    )
    st.caption(
        "P25 means 25% of simulations came in at or below that value. Upside / "
        "downside compares each value with the current share price."
    )

    if stats["pct_terminal_growth_clamped"] > 0.05:
        st.caption(
            f"{stats['pct_terminal_growth_clamped']:.0%} of draws had terminal growth "
            "at or above WACC − 0.5%, so it was capped there to keep the Gordon "
            "formula finite. Those draws sit in the high tail; consider a lower "
            "terminal growth std."
        )
    if stats["prob_negative_equity"] > 0:
        st.warning(
            f"{stats['prob_negative_equity']:.1%} of simulations produced negative "
            "equity value: in those outcomes net debt exceeds the business's "
            "estimated enterprise value."
        )


def forecast_columns(forecast):
    return [column for column in forecast.columns if column != "Discount Factor"]

st.set_page_config(page_title="DCF Valuation Tool", page_icon="📈", layout="wide")
st.title("DCF Tool")

with st.form("ticker_form"):
    ticker = st.text_input("Ticker symbol", value="AAPL", key="ticker_input")
    submitted = st.form_submit_button("Run")

if submitted:
    requested_ticker = ticker.strip().upper()
    if not requested_ticker:
        st.session_state["ticker_error"] = "Enter a ticker symbol to get started."
        st.session_state.pop("loaded_ticker", None)
        st.session_state.pop("loaded_company_data", None)
    else:
        fetched_data = get_company_data(requested_ticker)
        if "error" in fetched_data:
            st.session_state["ticker_error"] = _friendly_ticker_error(
                fetched_data, requested_ticker
            )
            st.session_state.pop("loaded_ticker", None)
            st.session_state.pop("loaded_company_data", None)
        else:
            if st.session_state.get("loaded_ticker") != requested_ticker:
                st.session_state.pop("multiples_peer_tickers", None)
                st.session_state.pop("multiples_peer_error", None)
            st.session_state["loaded_ticker"] = requested_ticker
            st.session_state["loaded_company_data"] = fetched_data
            st.session_state["ticker_error"] = None

if st.session_state.get("ticker_error"):
    st.error(st.session_state["ticker_error"])

company_data = st.session_state.get("loaded_company_data")
loaded_ticker = st.session_state.get("loaded_ticker")
if company_data:
    st.subheader(company_data["company_name"])
    st.metric("Current price", f"${company_data['current_price']:,.2f}")
    if company_data.get("data_source") == "sec":
        st.caption(SEC_SOURCE_NOTE)
    if company_data.get("snapshot_date"):
        st.info(
            f"Showing data saved on {company_data['snapshot_date']}. Live Yahoo "
            "Finance data isn't available from this server right now, so the "
            "price and statements may be out of date. Run the app locally for "
            "live data."
        )
    if company_data["diluted_shares_outstanding"] is None:
        st.warning("Diluted shares outstanding are unavailable.")
    if _is_financial_company(company_data):
        st.warning(
            "A standard DCF may not fit banks or insurers because debt and "
            "working capital are part of their operations."
        )

    income_statement = add_income_margins(
        clean_statement(company_data["income_statement"], "income")
    )
    balance_sheet = clean_statement(company_data["balance_sheet"], "balance")
    cash_flow_statement = clean_statement(
        company_data["cash_flow_statement"], "cashflow"
    )
    market_data = {
        "current_price": company_data["current_price"],
        "diluted_shares_outstanding": company_data["diluted_shares_outstanding"],
        "diluted_shares_source": company_data["diluted_shares_source"],
        "beta": company_data["beta"],
        "beta_source": company_data.get("beta_source"),
        "risk_free_rate": company_data["risk_free_rate"],
    }
    defaults = default_assumptions(
        income_statement, balance_sheet, cash_flow_statement, market_data
    )
    assumptions = build_sidebar(loaded_ticker, defaults, company_data)

    (
        valuation_tab,
        monte_carlo_tab,
        multiples_tab,
        income_tab,
        balance_tab,
        cash_flow_tab,
        news_tab,
    ) = st.tabs(
        [
            "Valuation",
            "Monte Carlo",
            "Multiples",
            "Income statement",
            "Balance sheet",
            "Cash flow statement",
            "News",
        ],
        on_change="rerun",
    )

    with valuation_tab:
        valuation_result = _display_valuation(assumptions, company_data)

    with monte_carlo_tab:
        _display_monte_carlo(assumptions, company_data, valuation_result)

    with multiples_tab:
        _display_multiples(
            loaded_ticker, company_data, assumptions, valuation_result
        )

    with income_tab:
        display_statement(income_statement)
        st.caption("USD millions, fiscal years")

    with balance_tab:
        display_statement(balance_sheet)
        st.caption("USD millions, fiscal years")

    with cash_flow_tab:
        display_statement(cash_flow_statement)
        st.caption("USD millions, fiscal years")

    if news_tab.open:
        with news_tab:
            _display_news(loaded_ticker)
elif not st.session_state.get("ticker_error"):
    st.info("Enter a ticker and select Run to begin a valuation.")

st.caption(
    "Educational tool, not investment advice. Data from Yahoo Finance via "
    "yfinance and may be delayed or incomplete."
)
