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
        # Prefer operating income: Yahoo's "EBIT" row is pretax income plus interest,
        # which includes non-operating income (interest on cash, equity stakes)
        # that the equity bridge already counts by adding cash and investments.
        "EBIT": ("Operating Income", "EBIT"),
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
            "Cash Equivalents",
            "Cash Financial",
            "Cash",
        ),
        "Cash and short-term investments total": (
            "Cash Cash Equivalents And Short Term Investments",
            "Cash Cash Equivalents And Federal Funds Sold",
        ),
        "Short-term marketable securities": (
            "Other Short Term Investments",
            "Short Term Investments",
            "Marketable Securities Current",
        ),
        "Long-term marketable securities": (
            "Investments And Advances",
            "Other Investments",
            "Held To Maturity Securities",
            "Trading Securities",
            "Available For Sale Securities",
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


def use_operating_ebitda(income, cashflow):
    """Return the income statement with EBITDA defined as EBIT + D&A.

    EBIT here is operating income, so this is the EBITDA the exit-multiple terminal
    value is applied to. Yahoo's own EBITDA row starts from pretax income plus
    interest, which adds non-operating income and can differ by 20% or more; using
    it to set the exit multiple would measure the multiple on one number and apply
    it to another. SEC data is built this way already. A year without D&A keeps
    the reported figure.
    """
    result = income.copy()
    if "EBIT" not in result.index or cashflow is None or "D&A" not in cashflow.index:
        return result
    if "EBITDA" not in result.index:
        result.loc["EBITDA"] = float("nan")
    for column in result.columns:
        ebit = result.at["EBIT", column]
        da = cashflow.at["D&A", column] if column in cashflow.columns else float("nan")
        if not (pd.isna(ebit) or pd.isna(da)):
            result.at["EBITDA", column] = ebit + abs(da)
    return result


def clean_company_statements(company_data):
    """Clean all three statements and apply the operating EBITDA definition."""
    income = clean_statement(company_data["income_statement"], "income")
    balance = clean_statement(company_data["balance_sheet"], "balance")
    cashflow = clean_statement(company_data["cash_flow_statement"], "cashflow")
    return use_operating_ebitda(income, cashflow), balance, cashflow


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
