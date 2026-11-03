import re

import pandas as pd


_ROWS = {
    "income": {
        "Revenue": ("Total Revenue", "Operating Revenue", "Revenue"),
        "Cost of Revenue": (
            "Cost Of Revenue",
            "Cost Of Goods Sold",
            "Reconciled Cost Of Revenue",
        ),
        "Gross Profit": ("Gross Profit",),
        "Operating Expenses": (
            "Operating Expense",
            "Operating Expenses",
            "Total Operating Expenses",
        ),
        "EBIT": ("EBIT", "Operating Income"),
        "EBITDA": ("EBITDA",),
        "Interest Expense": (
            "Interest Expense",
            "Interest Expense Non Operating",
        ),
        "Pre-Tax Income": ("Pretax Income", "Income Before Tax"),
        "Tax Provision": ("Tax Provision",),
        "Net Income": (
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ),
        "Diluted EPS": ("Diluted EPS",),
    },
    "balance": {
        "Cash": (
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash",
        ),
        "Total Current Assets": ("Total Current Assets", "Current Assets"),
        "Total Assets": ("Total Assets",),
        "Total Current Liabilities": (
            "Total Current Liabilities",
            "Current Liabilities",
        ),
        "Total Debt": ("Total Debt",),
        "Total Liabilities": (
            "Total Liabilities Net Minority Interest",
            "Total Liabilities",
        ),
        "Shareholders’ Equity": (
            "Stockholders Equity",
            "Common Stock Equity",
            "Shareholders Equity",
        ),
    },
    "cashflow": {
        "Operating Cash Flow": (
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        ),
        "D&A": (
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
        ),
        "CapEx": ("Capital Expenditure", "Capital Expenditures"),
        "Change in Working Capital": ("Change In Working Capital",),
        "Free Cash Flow": ("Free Cash Flow",),
    },
}


def _normalize_label(label):
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def clean_statement(df, kind):
    """Return standard statement rows in USD millions with newest fiscal year first."""
    if kind not in _ROWS:
        raise ValueError("kind must be 'income', 'balance', or 'cashflow'")

    standard_rows = list(_ROWS[kind])
    if df is None or len(df.columns) == 0:
        return pd.DataFrame(index=standard_rows, dtype="float64")

    # Convert date columns to fiscal-year labels and order them newest first.
    dated_columns = []
    for position, column in enumerate(df.columns):
        date_value = column[0] if isinstance(column, tuple) else column
        date = pd.to_datetime(date_value, errors="coerce")
        if not pd.isna(date):
            dated_columns.append((position, date, f"FY{date.year}"))
    dated_columns.sort(key=lambda item: item[1], reverse=True)

    result = pd.DataFrame(
        index=standard_rows,
        columns=[item[2] for item in dated_columns],
        dtype="float64",
    )
    source_rows = {_normalize_label(row): row for row in df.index}
    for standard_name, aliases in _ROWS[kind].items():
        source_row = next(
            (
                source_rows[_normalize_label(alias)]
                for alias in aliases
                if _normalize_label(alias) in source_rows
            ),
            None,
        )
        if source_row is None:
            continue

        values = pd.to_numeric(df.loc[source_row], errors="coerce")
        for output_column, (position, _date, _label) in zip(
            result.columns, dated_columns
        ):
            value = values.iloc[position]
            # EPS is dollars per share; the other statement values are USD.
            result.loc[standard_name, output_column] = (
                value if standard_name == "Diluted EPS" else value / 1_000_000
            )

    # Ignore fiscal years where most standard rows have no reported value.
    result = result.loc[:, result.isna().sum() <= len(standard_rows) / 2]
    return result


def add_income_margins(df):
    """Append income margins and year-over-year revenue growth as numeric ratios."""
    result = df.copy()
    required_rows = (
        "Revenue",
        "Gross Profit",
        "EBIT",
        "Net Income",
        "Gross Margin %",
        "EBIT Margin %",
        "Net Margin %",
        "Revenue Growth %",
    )
    for row in required_rows:
        if row not in result.index:
            result.loc[row] = float("nan")

    revenue = result.loc["Revenue"]

    for name, numerator in (
        ("Gross Margin %", "Gross Profit"),
        ("EBIT Margin %", "EBIT"),
        ("Net Margin %", "Net Income"),
    ):
        result.loc[name] = result.loc[numerator].div(revenue.where(revenue != 0))

    # Calculate growth in chronological order, then restore newest-first columns.
    result.loc["Revenue Growth %"] = (
        revenue.iloc[::-1].pct_change(fill_method=None).iloc[::-1]
    )
    return result
