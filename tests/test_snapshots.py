from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from data import snapshots


@pytest.fixture
def company_data():
    income = pd.DataFrame(
        [[400_000e6, 390_000e6, 380_000e6], [15_000e6, 15_500e6, np.nan]],
        index=["Total Revenue", "Diluted Average Shares"],
        columns=pd.to_datetime(["2025-09-30", "2024-09-30", "2023-09-30"]),
    )
    balance = pd.DataFrame(
        [[100e6, 90e6, 80e6]],
        index=["Total Debt"],
        columns=income.columns,
    )
    cash_flow = pd.DataFrame(
        [[50e6, 45e6, 40e6]],
        index=["Capital Expenditure"],
        columns=income.columns,
    )
    return {
        "company_name": "Example Corp",
        "current_price": 123.45,
        "diluted_shares_outstanding": 15_000e6,
        "diluted_shares_source": "Most recent diluted average shares",
        "beta": 1.1,
        "risk_free_rate": 4.1,
        "sector": "Technology",
        "industry": "Software",
        "income_statement": income,
        "balance_sheet": balance,
        "cash_flow_statement": cash_flow,
    }


@pytest.fixture(autouse=True)
def snapshot_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", tmp_path)
    return tmp_path


def test_frame_round_trip_keeps_values_nans_and_date_columns(company_data):
    original = company_data["income_statement"]
    restored = snapshots.dict_to_frame(snapshots.frame_to_dict(original))

    pd.testing.assert_frame_equal(restored, original, check_freq=False, check_column_type=False)
    assert isinstance(restored.columns, pd.DatetimeIndex)
    assert np.isnan(restored.loc["Diluted Average Shares"].iloc[2])


def test_save_and_load_snapshot_round_trip(company_data):
    snapshots.save_snapshot("exmp", company_data, as_of="2026-11-04")
    loaded = snapshots.load_snapshot("EXMP")

    assert loaded["snapshot_date"] == "2026-11-04"
    assert loaded["company_name"] == "Example Corp"
    assert loaded["current_price"] == 123.45
    assert loaded["beta"] == 1.1
    for key in ("income_statement", "balance_sheet", "cash_flow_statement"):
        pd.testing.assert_frame_equal(
            loaded[key], company_data[key], check_freq=False, check_column_type=False
        )


def test_snapshot_has_the_same_keys_as_live_data(company_data):
    snapshots.save_snapshot("EXMP", company_data)
    loaded = snapshots.load_snapshot("EXMP")

    assert set(company_data) <= set(loaded)


def test_missing_corrupt_or_incomplete_snapshots_return_none(snapshot_dir, company_data):
    assert snapshots.load_snapshot("NOPE") is None

    (snapshot_dir / "BAD.json").write_text("{not json", encoding="utf-8")
    assert snapshots.load_snapshot("BAD") is None

    (snapshot_dir / "EMPTY.json").write_text('{"as_of": "2026-01-01"}', encoding="utf-8")
    assert snapshots.load_snapshot("EMPTY") is None


def test_symbols_that_are_not_plain_tickers_are_rejected(company_data):
    for bad in ("../secrets", "a/b", "", "AAPL.json/..", "X" * 30):
        assert snapshots.load_snapshot(bad) is None
        with pytest.raises(ValueError):
            snapshots.save_snapshot(bad, company_data)


def test_dotted_and_dashed_tickers_are_allowed(company_data):
    snapshots.save_snapshot("BRK-B", company_data)
    assert snapshots.load_snapshot("brk-b")["company_name"] == "Example Corp"


def test_available_snapshots_are_sorted(company_data):
    assert snapshots.available_snapshots() == []
    for symbol in ("MSFT", "AAPL"):
        snapshots.save_snapshot(symbol, company_data)

    assert snapshots.available_snapshots() == ["AAPL", "MSFT"]


def test_saved_news_round_trip_and_none_when_absent(company_data):
    published = datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
    articles = [
        {
            "title": "Example beats estimates",
            "source": "Wire",
            "published": published,
            "summary": "Short summary.",
            "url": "https://example.com/a",
        }
    ]
    snapshots.save_snapshot("EXMP", company_data, news=articles, as_of="2026-11-04")
    saved, saved_date = snapshots.load_saved_news("EXMP")

    assert saved_date == "2026-11-04"
    assert saved == articles

    snapshots.save_snapshot("NONEWS", company_data, news=[])
    assert snapshots.load_saved_news("NONEWS") == (None, None)
    assert snapshots.load_saved_news("MISSING") == (None, None)
