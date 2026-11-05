"""Compare SEC-based statements with Yahoo's for a few tickers.

The deployed app reads statements from SEC filings because Yahoo blocks cloud
servers. Run this on your own computer, where both sources work, to check that
the SEC numbers line up:

    $env:SEC_USER_AGENT="DCF Tool your.email@example.com"   # PowerShell
    python scripts/check_sec.py                # default tickers
    python scripts/check_sec.py AAPL KO        # specific tickers
    python scripts/check_sec.py --debug XOM    # also show which SEC tags were used

Small differences are normal (Yahoo includes leases in debt, for example).
Large ones on a main line item are worth a look.
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows consoles and redirected output default to a code page that can't print
# every character; force UTF-8 so the report never crashes mid-run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from data.clean import clean_company_statements  # noqa: E402
from data.fetch import fetch_live, fetch_sec_hybrid  # noqa: E402
from model.assumptions import DCF_INPUTS, default_assumptions  # noqa: E402
from model.dcf import run_dcf  # noqa: E402
from data import sec  # noqa: E402

warnings.simplefilter("ignore")  # yfinance's pandas deprecation warnings

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "KO", "XOM", "JPM"]
CHECKS = [
    ("income", "Revenue"),
    ("income", "EBIT"),
    ("income", "Net Income"),
    ("income", "Diluted EPS"),
    ("balance", "Cash"),
    ("balance", "Short-term marketable securities"),
    ("balance", "Long-term marketable securities"),
    ("balance", "Total Debt"),
    ("balance", "Shareholders’ Equity"),
    ("cashflow", "Operating Cash Flow"),
    ("cashflow", "D&A"),
    ("cashflow", "CapEx"),
    ("cashflow", "Change in Working Capital"),
]


def _cleaned(data):
    income, balance, cashflow = clean_company_statements(data)
    return {"income": income, "balance": balance, "cashflow": cashflow}


def _value_per_share(data, cleaned):
    market = {
        key: data.get(key)
        for key in (
            "current_price", "diluted_shares_outstanding", "diluted_shares_source",
            "beta", "beta_source", "risk_free_rate",
        )
    }
    assumptions = default_assumptions(cleaned["income"], cleaned["balance"], cleaned["cashflow"], market)
    if assumptions.get("error"):
        return None, assumptions["error"], assumptions
    try:
        value = run_dcf(**{name: assumptions[name] for name in DCF_INPUTS})["value_per_share"]
        return value, None, assumptions
    except Exception as error:
        return None, str(error), assumptions


ASSUMPTION_ROWS = [
    ("Revenue growth, year 1", lambda a: (a.get("growth_rates") or [None])[0], "pct"),
    ("Revenue growth, year 5", lambda a: (a.get("growth_rates") or [None])[-1], "pct"),
    ("EBIT margin", lambda a: a.get("ebit_margin"), "pct"),
    ("Tax rate", lambda a: a.get("tax_rate"), "pct"),
    ("D&A % revenue", lambda a: a.get("da_pct_revenue"), "pct"),
    ("CapEx % revenue", lambda a: a.get("capex_pct_revenue"), "pct"),
    ("Change in NWC % revenue change", lambda a: a.get("delta_nwc_pct_revenue_change"), "pct"),
    ("Pre-tax cost of debt", lambda a: a.get("pre_tax_cost_of_debt"), "pct"),
    ("WACC", lambda a: a.get("wacc"), "pct"),
    ("Cash in bridge ($M)", lambda a: a.get("cash"), "num"),
    ("Debt in bridge ($M)", lambda a: a.get("debt"), "num"),
    ("Diluted shares (M)", lambda a: a.get("diluted_shares"), "num"),
]


def _fmt_assumption(value, kind):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:.2%}" if kind == "pct" else _fmt(value)


DEBUG_KEYWORDS = (
    "Revenue", "NetIncomeLoss", "OperatingIncome", "Depreciation", "Amortization",
    "Debt", "Borrowing", "CommercialPaper", "LeaseLiability", "MarketableSecurities",
    "ShortTermInvestments", "LongTermInvestments", "EquityMethodInvestments",
    "IncreaseDecreaseInOperatingCapital", "AssetsCurrent", "LiabilitiesCurrent",
)


def debug_tags(symbol, user_agent):
    """Show the latest annual 10-K value of every SEC tag that looks relevant."""
    cik, name = sec.lookup_company(symbol, user_agent)
    fact_sets = [sec._get_json(sec.FACTS_URL.format(cik=cik), user_agent)]
    for predecessor in sec.PREDECESSOR_CIKS.get(cik, ()):
        fact_sets.append(sec._get_json(sec.FACTS_URL.format(cik=predecessor), user_agent))
    facts = sec.merge_facts(fact_sets) if len(fact_sets) > 1 else fact_sets[0]
    taxonomies = facts.get("facts", {})
    ciks = ", ".join(str(c) for c in (cik, *sec.PREDECESSOR_CIKS.get(cik, ())))
    print(f"\n--- SEC tags for {symbol}: CIK {ciks}, {fact_sets[0].get('entityName', name)} ---")
    print(f"Taxonomies: {', '.join(f'{k} ({len(v)} tags)' for k, v in taxonomies.items())}")
    for taxonomy, tags in taxonomies.items():
        for tag in sorted(tags):
            if not any(keyword in tag for keyword in DEBUG_KEYWORDS):
                continue
            for unit, entries in tags[tag].get("units", {}).items():
                annual = [e for e in entries if str(e.get("form", "")).startswith("10-K")]
                if not annual:
                    continue
                latest = max(annual, key=lambda e: (e.get("end", ""), e.get("filed", "")))
                span = f"{latest.get('start', '')}..{latest['end']}" if latest.get("start") else latest["end"]
                print(
                    f"  {taxonomy}:{tag} [{unit}] {latest['val'] / 1e6:,.0f}M  "
                    f"({span}, {latest.get('form')}, {len(annual)} annual facts)"
                )


def _fmt(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"


def compare(symbol, user_agent):
    print(f"\n=== {symbol} ===")
    try:
        yahoo = fetch_live(symbol)
    except Exception as error:
        print(f"Yahoo failed: {str(error)[:120]}")
        return None
    try:
        sec_data = fetch_sec_hybrid(symbol, user_agent)
    except Exception as error:
        print(f"SEC failed: {str(error)[:120]}")
        if DEBUG:
            debug_tags(symbol, user_agent)
        return None

    y, s = _cleaned(yahoo), _cleaned(sec_data)
    year = next((c for c in y["income"].columns if c in s["income"].columns), None)
    if year is None:
        print(f"No common fiscal year. Yahoo {list(y['income'].columns)}, SEC {list(s['income'].columns)}")
        return None

    print(f"Latest common fiscal year: {year}   (USD millions, EPS in dollars)")
    print(f"{'Line item':36}{'Yahoo':>14}{'SEC':>14}{'Diff':>9}")
    flagged = 0
    for kind, label in CHECKS:
        yv = y[kind].loc[label, year] if label in y[kind].index else None
        sv = s[kind].loc[label, year] if year in s[kind].columns else None
        diff = ""
        if yv is not None and sv is not None and not pd.isna(yv) and not pd.isna(sv) and yv:
            pct = (sv - yv) / abs(yv) * 100
            diff = f"{pct:+.0f}%"
            if abs(pct) > 10:
                diff += " !"
                flagged += 1
        elif (yv is None or pd.isna(yv)) != (sv is None or pd.isna(sv)):
            diff = "missing !"
            flagged += 1
        print(f"{label:36}{_fmt(yv):>14}{_fmt(sv):>14}{diff:>9}")

    print(f"{'Beta':36}{_fmt(yahoo.get('beta')):>14}{_fmt(sec_data.get('beta')):>14}")
    y_value, y_err, y_assumptions = _value_per_share(yahoo, y)
    s_value, s_err, s_assumptions = _value_per_share(sec_data, s)
    print("\nDefault assumptions")
    for label, getter, kind in ASSUMPTION_ROWS:
        print(
            f"  {label:34}{_fmt_assumption(getter(y_assumptions), kind):>14}"
            f"{_fmt_assumption(getter(s_assumptions), kind):>14}"
        )
    print(f"{'Implied value per share ($)':36}{_fmt(y_value):>14}{_fmt(s_value):>14}")
    print(f"  Fiscal years  Yahoo: {', '.join(y['income'].columns)}   SEC: {', '.join(s['income'].columns)}")
    for name, err in (("Yahoo", y_err), ("SEC", s_err)):
        if err:
            print(f"  {name} DCF note: {err[:110]}")
    if DEBUG:
        debug_tags(symbol, user_agent)
    return flagged


DEBUG = False


def main(tickers):
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        print('Set SEC_USER_AGENT first, e.g.  $env:SEC_USER_AGENT="DCF Tool you@example.com"')
        return 1
    results = {}
    for symbol in tickers:
        results[symbol.upper()] = compare(symbol.upper(), user_agent)
        time.sleep(1)
    print("\nSummary (lines more than 10% apart or missing):")
    for symbol, flagged in results.items():
        print(f"  {symbol}: {'failed' if flagged is None else flagged}")
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--debug" in arguments:
        DEBUG = True
        arguments.remove("--debug")
    sys.exit(main(arguments or DEFAULT_TICKERS))
