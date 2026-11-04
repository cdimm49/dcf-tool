import math

import pandas as pd
import pytest

from data import sec
from data.clean import clean_statement


def fy(start, end, val, filed, form="10-K"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": "FY"}


def inst(end, val, filed, form="10-K"):
    return {"end": end, "val": val, "filed": filed, "form": form, "fp": "FY"}


YEARS = [  # (start, end, filed) — Apple-style 52/53-week fiscal years
    ("2021-09-26", "2022-09-24", "2022-10-28"),
    ("2022-09-25", "2023-09-30", "2023-11-03"),
    ("2023-10-01", "2024-09-28", "2024-11-01"),
    ("2024-09-29", "2025-09-27", "2025-10-31"),
]
B = 1e9


def duration_series(values):
    return [fy(s, e, v * B, f) for (s, e, f), v in zip(YEARS, values) if v is not None]


def instant_series(values):
    return [inst(e, v * B, f) for (_s, e, f), v in zip(YEARS, values) if v is not None]


def usd(entries):
    return {"units": {"USD": entries}}


@pytest.fixture
def facts():
    gaap = {
        # Older years use SalesRevenueNet, newer ones the post-2018 tag.
        "SalesRevenueNet": usd(duration_series([380, None, None, None])),
        "RevenueFromContractWithCustomerExcludingAssessedTax": usd(
            duration_series([None, 383, 391, 416])
            # A later filing restated FY2023 revenue: the latest filing wins.
            + [fy("2022-09-25", "2023-09-30", 383.5 * B, "2024-11-01")]
            # A 10-Q and a 3-month Q4 figure inside a 10-K must be ignored.
            + [fy("2025-06-29", "2025-09-27", 102 * B, "2025-10-31")]
            + [fy("2024-09-29", "2025-06-28", 313 * B, "2025-08-01", form="10-Q")]
        ),
        "CostOfGoodsAndServicesSold": usd(duration_series([223, 214, 210, 221])),
        "GrossProfit": usd(duration_series([170, 169, None, 195])),
        "OperatingExpenses": usd(duration_series([51, 55, 57, 62])),
        "OperatingIncomeLoss": usd(duration_series([119, 114, 123, 133])),
        "InterestExpense": usd(duration_series([2.9, 3.9, None, None])),
        "InterestExpenseNonoperating": usd(duration_series([None, None, 3.5, 3.4])),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": usd(
            duration_series([119, 114, 124, 132])
        ),
        "IncomeTaxExpenseBenefit": usd(duration_series([19, 17, 29, 20])),
        "NetIncomeLoss": usd(duration_series([100, 97, 94, 112])),
        "EarningsPerShareDiluted": {
            "units": {"USD/shares": [fy(s, e, v, f) for (s, e, f), v in zip(YEARS, [6.11, 6.13, 6.08, 7.46])]}
        },
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {"shares": [fy(s, e, v * B, f) for (s, e, f), v in zip(YEARS, [16.3, 15.8, 15.4, 15.0])]}
        },
        "CashAndCashEquivalentsAtCarryingValue": usd(instant_series([23.6, 30.0, 29.9, 35.9])),
        "MarketableSecuritiesCurrent": usd(instant_series([24.7, 31.6, 35.2, 18.8])),
        "MarketableSecuritiesNoncurrent": usd(instant_series([120.8, 100.5, 91.5, 77.7])),
        "AssetsCurrent": usd(instant_series([135, 143, 153, 148])),
        "Assets": usd(instant_series([353, 353, 365, 359])),
        "LiabilitiesCurrent": usd(instant_series([154, 145, 177, 165])),
        "Liabilities": usd(instant_series([302, 290, 308, 285])),
        "StockholdersEquity": usd(instant_series([51, 62, 57, 74])),
        "LongTermDebtNoncurrent": usd(instant_series([99, 95.3, 86, 78])),
        "LongTermDebtCurrent": usd(instant_series([11, 9.8, 10.9, 12])),
        "CommercialPaper": usd(instant_series([10, 6, 10, 8])),
        "NetCashProvidedByUsedInOperatingActivities": usd(duration_series([122, 110, 118, 111])),
        # The SEC reports capex as a positive payment.
        "PaymentsToAcquirePropertyPlantAndEquipment": usd(duration_series([10.7, 11, 9.4, 12.7])),
        "DepreciationDepletionAndAmortization": usd(duration_series([11.1, 11.5, 11.4, 11.7])),
        # Only one year reports the total; others need the components.
        "IncreaseDecreaseInOperatingCapital": usd(duration_series([None, None, None, 4])),
        "IncreaseDecreaseInAccountsReceivable": usd(duration_series([1.8, -1.7, 3.8, None])),
        "IncreaseDecreaseInInventories": usd(duration_series([-1.5, -1.6, 1.0, None])),
        "IncreaseDecreaseInAccountsPayable": usd(duration_series([9.4, -1.9, 6.0, None])),
    }
    return {"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": gaap}}


@pytest.fixture
def statements(facts):
    return sec.statements_from_facts(facts)


def row(frame, label):
    return [None if pd.isna(v) else round(v / B, 4) for v in frame.loc[label]]


def test_columns_are_the_four_latest_fiscal_year_ends_newest_first(statements):
    income, balance, cashflow = statements
    expected = list(pd.to_datetime(["2025-09-27", "2024-09-28", "2023-09-30", "2022-09-24"]))
    for frame in statements:
        assert list(frame.columns) == expected


def test_revenue_merges_tags_takes_restatements_and_skips_partial_periods(statements):
    income = statements[0]
    assert row(income, "Total Revenue") == [416, 391, 383.5, 380]


def test_missing_gross_profit_is_derived_and_interest_tags_are_merged(statements):
    income = statements[0]
    assert row(income, "Gross Profit") == [195, 181, 169, 170]
    assert row(income, "Interest Expense") == [3.4, 3.5, 3.9, 2.9]


def test_ebitda_is_operating_income_plus_depreciation(statements):
    assert row(statements[0], "EBITDA") == [144.7, 134.4, 125.5, 130.1]


def test_eps_and_share_counts_keep_their_units(statements):
    income = statements[0]
    assert list(income.loc["Diluted EPS"]) == [7.46, 6.08, 6.13, 6.11]
    assert row(income, "Diluted Average Shares") == [15.0, 15.4, 15.8, 16.3]


def test_total_debt_adds_long_term_current_and_commercial_paper(statements):
    assert row(statements[1], "Total Debt") == [98, 106.9, 111.1, 120]


def test_capex_is_negative_and_free_cash_flow_subtracts_it(statements):
    cashflow = statements[2]
    assert row(cashflow, "Capital Expenditure") == [-12.7, -9.4, -11, -10.7]
    assert row(cashflow, "Free Cash Flow") == [98.3, 108.6, 99, 111.3]


def test_working_capital_uses_reported_total_or_balance_sheet_change(statements):
    # FY2025 reports the total: an increase in operating capital of 4 uses 4 of cash.
    # Other years: NWC = (current assets - cash - securities)
    #                    - (current liabilities - current debt - commercial paper)
    # FY2024 -68.2, FY2023 -47.8, FY2022 -46.3; there's no FY2021 balance sheet.
    assert row(statements[2], "Change In Working Capital") == [-4, 20.4, 1.5, None]


def test_prior_year_balance_sheet_fills_the_oldest_change(facts):
    gaap = facts["facts"]["us-gaap"]
    prior = ("2020-09-27", "2021-09-25", "2021-10-29")
    gaap["NetIncomeLoss"]["units"]["USD"].append(fy(*prior[:2], 95 * B, prior[2]))
    gaap["AssetsCurrent"]["units"]["USD"].append(inst(prior[1], 134 * B, prior[2]))
    gaap["LiabilitiesCurrent"]["units"]["USD"].append(inst(prior[1], 125 * B, prior[2]))

    cashflow = sec.statements_from_facts(facts)[2]
    assert len(cashflow.columns) == 4  # the extra year only feeds the change
    # FY2021 NWC = 134 - 125 = 9 (no cash or debt lines that year); FY2022 = -46.3.
    assert row(cashflow, "Change In Working Capital")[3] == 55.3


def test_short_term_securities_add_separate_lines_but_not_duplicates(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["ShortTermInvestments"] = usd(instant_series([None, 31.6, 2.0, None]))  # oldest first
    balance = sec.statements_from_facts(facts)[1]
    # FY2025: only one line. FY2024: 35.2 + 2.0. FY2023: same value twice = one line.
    assert row(balance, "Other Short Term Investments") == [18.8, 37.2, 31.6, 24.7]


def test_finance_leases_count_as_debt_but_operating_leases_do_not(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["FinanceLeaseLiability"] = usd(instant_series([1, 1, 1, 2]))
    gaap["OperatingLeaseLiability"] = usd(instant_series([50, 50, 50, 50]))
    balance = sec.statements_from_facts(facts)[1]
    # Series run oldest to newest: leases are 2 in FY2025, 1 before.
    assert row(balance, "Total Debt") == [100, 107.9, 112.1, 121]


def test_debt_tags_that_already_include_leases_are_not_doubled():
    gaap = {
        "NetIncomeLoss": usd(duration_series([1, 1, 1, 1])),
        "LongTermDebtAndCapitalLeaseObligations": usd(instant_series([10, 10, 10, 10])),
        "LongTermDebtAndCapitalLeaseObligationsCurrent": usd(instant_series([2, 2, 2, 2])),
        "FinanceLeaseLiability": usd(instant_series([3, 3, 3, 3])),
    }
    balance = sec.statements_from_facts({"facts": {"us-gaap": gaap}})[1]
    assert row(balance, "Total Debt") == [12, 12, 12, 12]


def test_combined_cash_and_investments_total_is_passed_through(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["CashCashEquivalentsAndShortTermInvestments"] = usd(instant_series([1, 2, 3, 54.7]))
    cleaned = clean_statement(sec.statements_from_facts(facts)[1], "balance")
    assert cleaned.loc["Cash and short-term investments total", "FY2025"] == pytest.approx(54_700)


def test_frames_pass_through_the_existing_cleaner(statements):
    income, balance, cashflow = statements
    cleaned_income = clean_statement(income, "income")
    cleaned_balance = clean_statement(balance, "balance")
    cleaned_cashflow = clean_statement(cashflow, "cashflow")

    assert list(cleaned_income.columns) == ["FY2025", "FY2024", "FY2023", "FY2022"]
    assert cleaned_income.loc["Revenue", "FY2025"] == pytest.approx(416_000)
    assert cleaned_income.loc["EBIT", "FY2025"] == pytest.approx(133_000)
    assert cleaned_income.loc["Diluted EPS", "FY2025"] == pytest.approx(7.46)
    assert cleaned_balance.loc["Cash", "FY2025"] == pytest.approx(35_900)
    assert cleaned_balance.loc["Short-term marketable securities", "FY2025"] == pytest.approx(18_800)
    assert cleaned_balance.loc["Long-term marketable securities", "FY2025"] == pytest.approx(77_700)
    assert cleaned_balance.loc["Total Debt", "FY2025"] == pytest.approx(98_000)
    assert cleaned_cashflow.loc["D&A", "FY2025"] == pytest.approx(11_700)
    assert cleaned_cashflow.loc["CapEx", "FY2025"] == pytest.approx(-12_700)


def test_missing_revenue_and_income_raises(facts):
    with pytest.raises(sec.SecDataError):
        sec.statements_from_facts({"facts": {"us-gaap": {}}})


def test_ticker_map_and_lookup_accept_dot_and_dash_styles():
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    }
    mapping = sec.parse_ticker_map(payload)
    assert sec.lookup_company("AAPL", "ua", mapping) == (320193, "Apple Inc.")
    assert sec.lookup_company("BRK.B", "ua", mapping)[0] == 1067983
    with pytest.raises(sec.SecDataError, match="isn't in the SEC's list"):
        sec.lookup_company("ZZZZ", "ua", mapping)


def test_missing_user_agent_raises_before_any_request():
    with pytest.raises(sec.SecDataError, match="SEC_USER_AGENT"):
        sec._get_json("https://data.sec.gov/x", None)


@pytest.mark.parametrize(
    "sic, sector",
    [("6021", "Financial Services"), ("6331", "Financial Services"), ("3571", None), (None, None)],
)
def test_sic_codes_flag_banks_and_insurers(sic, sector):
    assert sec.classify_sic(sic, "Desc")[0] == sector


def test_untagged_current_marketable_securities_are_added_like_ko(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["MarketableSecurities"] = gaap.pop("MarketableSecuritiesCurrent")
    gaap["OtherShortTermInvestments"] = usd(instant_series([1, 1, 1, 1]))
    balance = sec.statements_from_facts(facts)[1]
    assert row(balance, "Other Short Term Investments") == [19.8, 36.2, 32.6, 25.7]


def test_available_for_sale_total_is_used_only_when_nothing_else_exists(facts):
    gaap = facts["facts"]["us-gaap"]
    # Like NVDA: the current-securities tag stops after FY2024.
    entries = gaap["MarketableSecuritiesCurrent"]["units"]["USD"]
    gaap["MarketableSecuritiesCurrent"]["units"]["USD"] = entries[:3]
    gaap["AvailableForSaleSecuritiesDebtSecurities"] = usd(instant_series([90, 90, 90, 40]))
    balance = sec.statements_from_facts(facts)[1]
    assert row(balance, "Other Short Term Investments") == [40, 35.2, 31.6, 24.7]


def test_predecessor_facts_are_merged_with_the_new_filer(facts):
    new_filer = {"facts": {"us-gaap": {"Assets": usd([inst("2026-09-26", 400 * B, "2026-10-30")])}}}
    merged = sec.merge_facts([new_filer, facts])
    income, balance, _ = sec.statements_from_facts(merged)
    assert row(income, "Total Revenue")[0] == 416  # history came from the old filer
    assert len(merged["facts"]["us-gaap"]["Assets"]["units"]["USD"]) == 5


def test_fetch_company_combines_predecessor_filings(facts, monkeypatch):
    calls = []
    empty = {"facts": {"us-gaap": {}}}

    def fake_get_json(url, user_agent, timeout=20):
        calls.append(url)
        if "companyfacts" in url:
            return empty if "CIK0002115436" in url else facts
        return {"name": "ExxonMobil Holdings Corporation", "sic": "2911"}

    monkeypatch.setattr(sec, "_get_json", fake_get_json)
    monkeypatch.setattr(sec, "lookup_company", lambda symbol, ua: (2115436, "EXXON"))
    company = sec.fetch_company("XOM", "ua")
    assert any("CIK0000034088" in url for url in calls)
    assert row(company["income_statement"], "Total Revenue")[0] == 416


def test_all_caps_names_are_made_readable():
    assert sec.display_name("MICROSOFT CORP") == "Microsoft Corp"
    assert sec.display_name("Apple Inc.") == "Apple Inc."


def test_ebit_falls_back_to_pretax_plus_interest_without_operating_income(facts):
    gaap = facts["facts"]["us-gaap"]
    del gaap["OperatingIncomeLoss"]
    income = sec.statements_from_facts(facts)[0]
    assert row(income, "EBIT") == [135.4, 127.5, 117.9, 121.9]
    assert row(income, "EBITDA") == [147.1, 138.9, 129.4, 133]
    cleaned = clean_statement(income, "income")
    assert cleaned.loc["EBIT", "FY2025"] == pytest.approx(135_400)


def test_operating_income_is_preferred_when_reported(statements):
    cleaned = clean_statement(statements[0], "income")
    assert cleaned.loc["EBIT", "FY2025"] == pytest.approx(133_000)


def test_basic_and_diluted_share_tag_is_a_fallback(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["WeightedAverageNumberOfShareOutstandingBasicAndDiluted"] = gaap.pop(
        "WeightedAverageNumberOfDilutedSharesOutstanding"
    )
    income = sec.statements_from_facts(facts)[0]
    assert row(income, "Diluted Average Shares") == [15.0, 15.4, 15.8, 16.3]


def test_partial_long_term_debt_tag_does_not_hide_balance_sheet_debt(facts):
    gaap = facts["facts"]["us-gaap"]
    gaap["LongTermDebt"] = usd(instant_series([5, 5, 5, 5]))  # a single footnote item
    balance = sec.statements_from_facts(facts)[1]
    assert row(balance, "Total Debt") == [98, 106.9, 111.1, 120]


def test_debt_current_is_used_when_nothing_more_specific_exists():
    gaap = {
        "NetIncomeLoss": usd(duration_series([1, 1, 1, 1])),
        "LongTermDebtNoncurrent": usd(instant_series([30, 30, 30, 30])),
        "DebtCurrent": usd(instant_series([4, 4, 4, 4])),
    }
    balance = sec.statements_from_facts({"facts": {"us-gaap": gaap}})[1]
    assert row(balance, "Total Debt") == [34, 34, 34, 34]


def test_long_term_investments_take_the_largest_overlapping_tag(facts):
    gaap = facts["facts"]["us-gaap"]
    del gaap["MarketableSecuritiesNoncurrent"]
    gaap["LongTermInvestments"] = usd(instant_series([0.3, 0.3, 0.3, 0.3]))
    gaap["EquityMethodInvestments"] = usd(instant_series([30, 31, 32, 33]))
    balance = sec.statements_from_facts(facts)[1]
    assert row(balance, "Investments And Advances") == [33, 32, 31, 30]


def test_cover_page_shares_add_share_classes_from_the_latest_filing():
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2025-10-15", "val": 4.0e9, "accn": "old"},
        {"end": "2026-10-20", "val": 3.0e9, "accn": "new"},
        {"end": "2026-10-20", "val": 1.0e9, "accn": "new"},
    ]}}}}}
    assert sec.latest_shares_outstanding(facts) == 4.0e9
    assert sec.latest_shares_outstanding({"facts": {}}) is None


def test_merge_keeps_cover_page_data(facts):
    cover = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-10-20", "val": 4.3e9, "accn": "a"}]}}}}}
    merged = sec.merge_facts([cover, facts])
    assert sec.latest_shares_outstanding(merged) == 4.3e9
    assert "us-gaap" in merged["facts"]


def test_current_portion_alone_does_not_stand_in_for_long_term_debt_like_xom():
    gaap = {
        "NetIncomeLoss": usd(duration_series([1, 1, 1, 1])),
        "LongTermDebtCurrent": usd(instant_series([6.2, 6.2, 6.2, 6.2])),
        "LongTermDebtAndCapitalLeaseObligations": usd(instant_series([34.2, 34.2, 34.2, 34.2])),
        "CommercialPaper": usd(instant_series([3.1, 3.1, 3.1, 3.1])),
        "FinanceLeaseLiability": usd(instant_series([2.7, 2.7, 2.7, 2.7])),  # already included
    }
    balance = sec.statements_from_facts({"facts": {"us-gaap": gaap}})[1]
    assert row(balance, "Total Debt") == [43.5, 43.5, 43.5, 43.5]
