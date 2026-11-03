import pandas as pd
import streamlit as st


_PERCENT_ROWS = {
    "Gross Margin %",
    "EBIT Margin %",
    "Net Margin %",
    "Revenue Growth %",
}


def _format_number(value, decimals=0):
    if pd.isna(value):
        return "—"
    formatted = f"{abs(value):,.{decimals}f}"
    return f"({formatted})" if value < 0 else formatted


def display_statement(df):
    """Display numeric statement values with presentation-only formatting."""
    # Use a numeric display copy so None and NaN receive the same formatting.
    display_df = df.astype("float64")
    styler = display_df.style.format(
        lambda value: _format_number(value),
        subset=pd.IndexSlice[
            [
                row
                for row in display_df.index
                if row not in _PERCENT_ROWS | {"Diluted EPS"}
            ],
            :,
        ],
        na_rep="—",
    )
    percentage_rows = [row for row in display_df.index if row in _PERCENT_ROWS]
    if percentage_rows:
        styler = styler.format(
            lambda value: (
                "—"
                if pd.isna(value)
                else (f"({abs(value):.1%})" if value < 0 else f"{value:.1%}")
            ),
            subset=pd.IndexSlice[percentage_rows, :],
            na_rep="—",
        )
    if "Diluted EPS" in display_df.index:
        styler = styler.format(
            lambda value: _format_number(value, decimals=2),
            subset=pd.IndexSlice[["Diluted EPS"], :],
            na_rep="—",
        )

    st.dataframe(styler, alt="Cleaned financial statement")
