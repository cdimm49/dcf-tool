"""Annual financial statements from SEC EDGAR, the SEC's free public data API.

Yahoo Finance blocks statement requests from shared cloud servers, but the SEC
serves every US-listed company's 10-K data to anyone who identifies themselves
with a User-Agent that includes a contact email. This module turns the SEC's
XBRL "company facts" into DataFrames shaped like yfinance's statements (same
row labels, fiscal-year-end date columns, raw USD), so data/clean.py and the
model work on them unchanged.

Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

import time

import pandas as pd
import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

YEARS = 4  # fiscal years of statements to return, like yfinance
TICKER_MAP_TTL = 24 * 3600


class SecDataError(Exception):
    """SEC data was unavailable or incomplete for a ticker."""


# Each row lists XBRL tags in priority order. For each fiscal year the first tag
# with a value wins, because companies switch tags over time (for example
# SalesRevenueNet became RevenueFromContractWithCustomerExcludingAssessedTax).
INCOME_ROWS = {
    "Total Revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ),
    "Cost Of Revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ),
    "Gross Profit": ("GrossProfit",),
    "Operating Expense": ("OperatingExpenses",),
    "Operating Income": ("OperatingIncomeLoss",),
    "Interest Expense": (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ),
    "Pretax Income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "Tax Provision": ("IncomeTaxExpenseBenefit",),
    "Net Income": ("NetIncomeLoss", "ProfitLoss"),
}
PER_SHARE_ROWS = {"Diluted EPS": ("EarningsPerShareDiluted",)}
SHARE_ROWS = {
    "Diluted Average Shares": (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",  # XOM
    )
}
BALANCE_ROWS = {
    "Cash And Cash Equivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndDueFromBanks",
        "Cash",
    ),
    "Cash Cash Equivalents And Short Term Investments": (
        "CashCashEquivalentsAndShortTermInvestments",
    ),
    "Investments And Advances": (
        "MarketableSecuritiesNoncurrent",
        "LongTermInvestments",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
        "EquityMethodInvestments",
    ),
    "Total Current Assets": ("AssetsCurrent",),
    "Total Assets": ("Assets",),
    "Total Current Liabilities": ("LiabilitiesCurrent",),
    "Total Liabilities Net Minority Interest": ("Liabilities",),
    "Stockholders Equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
}
# Some companies split short-term investments into two lines (KO reports both
# "short-term investments" and "marketable securities"), so these are added.
SHORT_TERM_SECURITY_TAGS = (
    (
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "MarketableSecurities",  # KO tags its current securities without "Current"
    ),
    ("ShortTermInvestments", "OtherShortTermInvestments"),
)
# Used only when a year has no short-term securities line at all. NVDA moved its
# balance-sheet line to a company-specific tag the API doesn't include; its
# available-for-sale debt securities are the closest standard figure.
SHORT_TERM_SECURITY_LAST_RESORT = ("AvailableForSaleSecuritiesDebtSecurities",)
DEBT_TAGS = {
    "total_long_term": ("LongTermDebt",),
    "long_term_noncurrent": ("LongTermDebtNoncurrent",),
    "long_term_current": ("LongTermDebtCurrent",),
    # These "...AndCapitalLeaseObligations" tags already include finance leases.
    "with_leases_noncurrent": ("LongTermDebtAndCapitalLeaseObligations",),
    "with_leases_current": ("LongTermDebtAndCapitalLeaseObligationsCurrent",),
    "short_term": ("ShortTermBorrowings", "CommercialPaper"),
    "debt_current": ("DebtCurrent",),
    "finance_lease_total": ("FinanceLeaseLiability",),
    "finance_lease_noncurrent": ("FinanceLeaseLiabilityNoncurrent",),
    "finance_lease_current": ("FinanceLeaseLiabilityCurrent",),
}
CASHFLOW_ROWS = {
    "Operating Cash Flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "Depreciation And Amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ),
}
CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
# Working capital: the company's own total when reported (an increase in
# operating capital uses cash), otherwise the change in balance-sheet NWC.
WORKING_CAPITAL_TOTAL = ("IncreaseDecreaseInOperatingCapital",)


# ---------------------------------------------------------------- network


def _get_json(url, user_agent, timeout=20):
    if not user_agent:
        raise SecDataError(
            "SEC_USER_AGENT isn't set, so SEC data can't be requested. "
            "See the README for setup."
        )
    response = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    if response.status_code == 404:
        raise SecDataError("The SEC has no filing data for this company.")
    response.raise_for_status()
    return response.json()


_ticker_cache = {"loaded_at": 0.0, "map": {}}


def parse_ticker_map(payload):
    """{TICKER: (cik, company name)} from the SEC's company_tickers.json."""
    rows = payload.values() if isinstance(payload, dict) else payload
    mapping = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        cik = row.get("cik_str")
        if ticker and cik:
            mapping.setdefault(ticker, (int(cik), row.get("title") or ticker))
    return mapping


# When a company reorganizes under a new holding company, the ticker moves to a
# new SEC filer with little history. Map the new CIK to the old one so both
# filers' facts are combined. (CIK 2115436 is ExxonMobil Holdings Corporation;
# 34088 is Exxon Mobil Corp.)
PREDECESSOR_CIKS = {2115436: (34088,)}


def lookup_company(symbol, user_agent, ticker_map=None):
    """Return (cik, name) for a ticker, accepting BRK.B and BRK-B styles."""
    if ticker_map is None:
        now = time.time()
        if not _ticker_cache["map"] or now - _ticker_cache["loaded_at"] > TICKER_MAP_TTL:
            _ticker_cache["map"] = parse_ticker_map(_get_json(TICKERS_URL, user_agent))
            _ticker_cache["loaded_at"] = now
        ticker_map = _ticker_cache["map"]

    for candidate in (symbol, symbol.replace(".", "-"), symbol.replace("-", ".")):
        if candidate in ticker_map:
            return ticker_map[candidate]
    raise SecDataError(f"{symbol} isn't in the SEC's list of US-listed companies.")


# ---------------------------------------------------------------- parsing


def _annual_points(facts, tag, unit, duration):
    """{period end: value} from 10-K filings, keeping the latest-filed value."""
    entries = (
        facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, [])
    )
    best = {}
    for entry in entries:
        if not str(entry.get("form", "")).startswith("10-K"):
            continue
        value, end = entry.get("val"), entry.get("end")
        if value is None or not end:
            continue
        end = pd.Timestamp(end)
        start = entry.get("start")
        if duration:
            # Full fiscal years only (52/53-week years run 364 or 371 days).
            if not start or not 340 <= (end - pd.Timestamp(start)).days <= 380:
                continue
        elif start:
            continue
        filed = str(entry.get("filed", ""))
        if end not in best or filed > best[end][0]:
            best[end] = (filed, float(value))
    return {end: value for end, (_filed, value) in best.items()}


def _merged(facts, tags, unit="USD", duration=True):
    merged = {}
    for tag in tags:
        for end, value in _annual_points(facts, tag, unit, duration).items():
            merged.setdefault(end, value)
    return merged


def _align(points, fiscal_ends, tolerance_days=7):
    """Match point dates to fiscal year ends, allowing a few days of drift."""
    aligned = []
    for fiscal_end in fiscal_ends:
        nearby = [
            (abs((end - fiscal_end).days), value)
            for end, value in points.items()
            if abs((end - fiscal_end).days) <= tolerance_days
        ]
        aligned.append(min(nearby)[1] if nearby else float("nan"))
    return aligned


def _fiscal_year_ends(facts, years=YEARS):
    ends = set()
    for tags in (INCOME_ROWS["Total Revenue"], INCOME_ROWS["Net Income"]):
        ends.update(_merged(facts, tags))
    if not ends:
        raise SecDataError("No annual revenue or net income was found in SEC filings.")
    return sorted(ends, reverse=True)[:years]


def _frame(rows, fiscal_ends):
    frame = pd.DataFrame(
        {label: values for label, values in rows.items()},
        index=pd.DatetimeIndex(fiscal_ends),
        dtype="float64",
    ).T
    # Leave out rows with no data, as yfinance does, so the cleaner's alias
    # lookup falls through to the next row name (e.g. EBIT after Operating Income).
    return frame.dropna(how="all")


def _sum_available(*values):
    present = [value for value in values if not pd.isna(value)]
    return sum(present) if present else float("nan")


def _instant(facts, tags, fiscal_ends):
    return _align(_merged(facts, tags, duration=False), fiscal_ends)


def _short_term_securities(facts, fiscal_ends):
    parts = [_instant(facts, tags, fiscal_ends) for tags in SHORT_TERM_SECURITY_TAGS]
    last_resort = _instant(facts, SHORT_TERM_SECURITY_LAST_RESORT, fiscal_ends)
    totals = []
    for i, values in enumerate(zip(*parts)):
        present = [v for v in values if not pd.isna(v)]
        # Identical values mean one line was tagged twice, not two holdings.
        totals.append(sum(set(present)) if present else last_resort[i])
    return totals


def _debt_parts(facts, fiscal_ends):
    parts = {name: _instant(facts, tags, fiscal_ends) for name, tags in DEBT_TAGS.items()}
    long_term, current, short_term, leases = [], [], [], []
    for i in range(len(fiscal_ends)):
        # Noncurrent long-term debt, and whether that figure already includes
        # finance leases (XOM switched to the "...AndCapitalLeaseObligations" tag).
        noncurrent = parts["long_term_noncurrent"][i]
        leases_included = False
        if pd.isna(noncurrent):
            noncurrent = parts["with_leases_noncurrent"][i]
            leases_included = not pd.isna(noncurrent)
        current_lt = parts["long_term_current"][i]
        if pd.isna(current_lt):
            current_lt = parts["with_leases_current"][i]

        split_total = _sum_available(noncurrent, current_lt)
        reported_total = parts["total_long_term"][i]
        # Some filers use LongTermDebt for a single note in the footnotes,
        # so take the larger of it and the balance-sheet lines.
        candidates = [v for v in (reported_total, split_total) if not pd.isna(v)]
        lt = max(candidates) if candidates else float("nan")

        short = parts["short_term"][i]
        short_only = False
        if pd.isna(short) and pd.isna(current_lt) and not pd.isna(parts["debt_current"][i]):
            # DebtCurrent covers short-term borrowings and the current part of
            # long-term debt; use it only when neither is tagged separately.
            short = parts["debt_current"][i]
            short_only = True

        lease_total = parts["finance_lease_total"][i]
        if pd.isna(lease_total):
            lease_total = _sum_available(
                parts["finance_lease_noncurrent"][i], parts["finance_lease_current"][i]
            )
        long_term.append(lt)
        current.append(parts["debt_current"][i] if short_only else current_lt)
        short_term.append(short)
        leases.append(float("nan") if leases_included else lease_total)
    return long_term, current, short_term, leases, parts["finance_lease_current"]


def _total_debt(facts, fiscal_ends):
    """Borrowings plus finance leases. Operating leases are left out: their cost
    is already an operating expense, so counting them as debt too double counts."""
    long_term, _current, short_term, leases, _ = _debt_parts(facts, fiscal_ends)
    return [_sum_available(*values) for values in zip(long_term, short_term, leases)]


def _working_capital_change(facts, fiscal_ends, prior_end):
    """Cash-flow sign: negative when working capital grew (cash was used)."""
    reported = _align(_merged(facts, WORKING_CAPITAL_TOTAL), fiscal_ends)

    ends = list(fiscal_ends) + ([prior_end] if prior_end is not None else [])
    current_assets = _instant(facts, ("AssetsCurrent",), ends)
    current_liabilities = _instant(facts, ("LiabilitiesCurrent",), ends)
    cash = _instant(facts, BALANCE_ROWS["Cash And Cash Equivalents"], ends)
    securities = _short_term_securities(facts, ends)
    _lt, current_debt, short_debt, _leases, lease_current = _debt_parts(facts, ends)

    def zero(value):
        return 0.0 if pd.isna(value) else value

    nwc = []
    for i in range(len(ends)):
        if pd.isna(current_assets[i]) or pd.isna(current_liabilities[i]):
            nwc.append(float("nan"))
            continue
        operating_assets = current_assets[i] - zero(cash[i]) - zero(securities[i])
        operating_liabilities = (
            current_liabilities[i]
            - zero(current_debt[i])
            - zero(short_debt[i])
            - zero(lease_current[i])
        )
        nwc.append(operating_assets - operating_liabilities)

    changes = []
    for i in range(len(fiscal_ends)):
        if not pd.isna(reported[i]):
            changes.append(-reported[i])
        elif i + 1 < len(nwc) and not (pd.isna(nwc[i]) or pd.isna(nwc[i + 1])):
            changes.append(-(nwc[i] - nwc[i + 1]))
        else:
            changes.append(float("nan"))
    return changes


def statements_from_facts(facts):
    """Build yfinance-shaped (income, balance, cash flow) frames from company facts."""
    all_ends = _fiscal_year_ends(facts, YEARS + 1)
    fiscal_ends = all_ends[:YEARS]
    prior_end = all_ends[YEARS] if len(all_ends) > YEARS else None

    income = {
        label: _align(_merged(facts, tags), fiscal_ends)
        for label, tags in INCOME_ROWS.items()
    }
    for label, tags in PER_SHARE_ROWS.items():
        income[label] = _align(_merged(facts, tags, unit="USD/shares"), fiscal_ends)
    for label, tags in SHARE_ROWS.items():
        income[label] = _align(_merged(facts, tags, unit="shares"), fiscal_ends)

    cashflow = {
        label: _align(_merged(facts, tags), fiscal_ends)
        for label, tags in CASHFLOW_ROWS.items()
    }
    capex = _align(_merged(facts, CAPEX_TAGS), fiscal_ends)
    cashflow["Capital Expenditure"] = [-abs(v) if not pd.isna(v) else v for v in capex]
    cashflow["Change In Working Capital"] = _working_capital_change(
        facts, fiscal_ends, prior_end
    )
    cashflow["Free Cash Flow"] = [
        ocf + cap if not (pd.isna(ocf) or pd.isna(cap)) else float("nan")
        for ocf, cap in zip(cashflow["Operating Cash Flow"], cashflow["Capital Expenditure"])
    ]

    # Fill gross profit and EBITDA the way yfinance reports them.
    income["Gross Profit"] = [
        gp if not pd.isna(gp) else rev - cost
        for gp, rev, cost in zip(
            income["Gross Profit"], income["Total Revenue"], income["Cost Of Revenue"]
        )
    ]
    # Oil majors and some others don't report operating income. Fall back to
    # pretax income plus interest expense, which is how Yahoo defines EBIT.
    income["EBIT"] = [
        float("nan")
        if not pd.isna(operating) or pd.isna(pretax)
        else pretax + (0.0 if pd.isna(interest) else interest)
        for operating, pretax, interest in zip(
            income["Operating Income"], income["Pretax Income"], income["Interest Expense"]
        )
    ]
    income["EBITDA"] = [
        (operating if not pd.isna(operating) else ebit) + da
        if not (pd.isna(da) or (pd.isna(operating) and pd.isna(ebit)))
        else float("nan")
        for operating, ebit, da in zip(
            income["Operating Income"], income["EBIT"], cashflow["Depreciation And Amortization"]
        )
    ]

    balance = {
        label: _align(_merged(facts, tags, duration=False), fiscal_ends)
        for label, tags in BALANCE_ROWS.items()
    }
    balance["Other Short Term Investments"] = _short_term_securities(facts, fiscal_ends)
    # Long-term investments are spread over overlapping tags (MSFT's total
    # includes its equity-method stakes; XOM's LongTermInvestments is a small
    # side line), so take the largest rather than the first.
    candidates = [
        _instant(facts, (tag,), fiscal_ends) for tag in BALANCE_ROWS["Investments And Advances"]
    ]
    balance["Investments And Advances"] = [
        max((v for v in values if not pd.isna(v)), default=float("nan"))
        for values in zip(*candidates)
    ]
    balance["Total Debt"] = _total_debt(facts, fiscal_ends)

    return _frame(income, fiscal_ends), _frame(balance, fiscal_ends), _frame(cashflow, fiscal_ends)


_FINANCIAL_SIC = ((6000, 6499), (6700, 6799))


def classify_sic(sic, description):
    """Return (sector, industry) from an SEC SIC code. Only financials get a sector,
    which is what the app uses it for (the bank/insurer DCF warning)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None, description or None
    sector = (
        "Financial Services"
        if any(low <= code <= high for low, high in _FINANCIAL_SIC)
        else None
    )
    return sector, description or None


def merge_facts(fact_sets):
    """Combine company facts from several filers. Duplicate periods are fine:
    _annual_points keeps the latest-filed value for each period."""
    merged = {"facts": {}}
    for facts in fact_sets:
        for taxonomy, tags in facts.get("facts", {}).items():
            merged_tags = merged["facts"].setdefault(taxonomy, {})
            for tag, body in tags.items():
                target = merged_tags.setdefault(tag, {"units": {}})
                for unit, entries in body.get("units", {}).items():
                    target["units"].setdefault(unit, []).extend(entries)
    return merged


def latest_shares_outstanding(facts):
    """Shares outstanding from the latest filing's cover page (dei taxonomy)."""
    entries = (
        facts.get("facts", {})
        .get("dei", {})
        .get("EntityCommonStockSharesOutstanding", {})
        .get("units", {})
        .get("shares", [])
    )
    if not entries:
        return None
    latest_end = max(entry.get("end", "") for entry in entries)
    # Companies with several share classes report one fact per class on the
    # same date; add them up, keeping the latest filing's figure for each.
    by_filing = {}
    for entry in entries:
        if entry.get("end") == latest_end and entry.get("val"):
            by_filing.setdefault(entry.get("accn"), []).append(float(entry["val"]))
    if not by_filing:
        return None
    return sum(max(by_filing.items(), key=lambda item: len(item[1]))[1])


def display_name(name):
    """SEC names are often all caps ("MICROSOFT CORP"); make them readable."""
    if isinstance(name, str) and name.isupper():
        return name.title()
    return name


def fetch_company(symbol, user_agent):
    """Download a company's SEC data and return name, sector and statements."""
    cik, name = lookup_company(symbol, user_agent)
    fact_sets = [_get_json(FACTS_URL.format(cik=cik), user_agent)]
    for predecessor in PREDECESSOR_CIKS.get(cik, ()):
        try:
            fact_sets.append(_get_json(FACTS_URL.format(cik=predecessor), user_agent))
        except Exception:
            pass
    facts = merge_facts(fact_sets) if len(fact_sets) > 1 else fact_sets[0]
    income, balance, cashflow = statements_from_facts(facts)

    sector, industry = None, None
    try:
        submissions = _get_json(SUBMISSIONS_URL.format(cik=cik), user_agent)
        name = submissions.get("name") or name
        sector, industry = classify_sic(
            submissions.get("sic"), submissions.get("sicDescription")
        )
    except Exception:
        pass  # name from the ticker list is fine; sector only drives a warning

    return {
        "company_name": display_name(name),
        "shares_outstanding": latest_shares_outstanding(facts),
        "sector": sector,
        "industry": industry,
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow_statement": cashflow,
    }
