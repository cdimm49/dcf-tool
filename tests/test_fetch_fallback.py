import pandas as pd
import pytest

from data import fetch, snapshots


def _frame(label):
    return pd.DataFrame(
        [[1.0, 2.0]], index=[label], columns=pd.to_datetime(["2025-12-31", "2024-12-31"])
    )


LIVE = {"company_name": "Live Corp", "current_price": 10.0}


@pytest.fixture
def saved_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshots, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(fetch, "_live_blocked_until", 0.0)
    monkeypatch.delenv("DCF_FORCE_SNAPSHOT", raising=False)
    snapshots.save_snapshot(
        "AAPL",
        {
            "company_name": "Saved Apple",
            "current_price": 200.0,
            "income_statement": _frame("Total Revenue"),
            "balance_sheet": _frame("Total Debt"),
            "cash_flow_statement": _frame("Capital Expenditure"),
        },
        as_of="2026-11-04",
    )


class _LiveStub:
    """Stands in for the cached live fetch and counts calls."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def __call__(self, symbol):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_live_data_wins_when_it_works(saved_snapshot, monkeypatch):
    monkeypatch.setattr(fetch, "_cached_live", _LiveStub(result=LIVE))

    assert fetch.get_company_data("aapl") == LIVE


def test_falls_back_to_snapshot_when_live_fails(saved_snapshot, monkeypatch):
    stub = _LiveStub(error=fetch.LiveDataError("Missing: income statement"))
    monkeypatch.setattr(fetch, "_cached_live", stub)

    data = fetch.get_company_data("AAPL")

    assert data["company_name"] == "Saved Apple"
    assert data["snapshot_date"] == "2026-11-04"
    assert "error" not in data


def test_after_a_failure_live_is_skipped_until_the_block_expires(saved_snapshot, monkeypatch):
    stub = _LiveStub(error=RuntimeError("429 Too Many Requests"))
    monkeypatch.setattr(fetch, "_cached_live", stub)

    fetch.get_company_data("AAPL")
    fetch.get_company_data("AAPL")
    assert stub.calls == 1

    monkeypatch.setattr(fetch, "_live_blocked_until", 0.0)
    fetch.get_company_data("AAPL")
    assert stub.calls == 2


def test_unknown_ticker_without_snapshot_returns_error_and_does_not_block(saved_snapshot, monkeypatch):
    stub = _LiveStub(error=fetch.LiveDataError("Could not fetch complete data for ZZZZ."))
    monkeypatch.setattr(fetch, "_cached_live", stub)

    assert "error" in fetch.get_company_data("ZZZZ")
    assert "error" in fetch.get_company_data("ZZZZ")
    assert stub.calls == 2  # a bad symbol must not switch live data off for everyone


def test_force_snapshot_env_var_skips_live(saved_snapshot, monkeypatch):
    stub = _LiveStub(result=LIVE)
    monkeypatch.setattr(fetch, "_cached_live", stub)
    monkeypatch.setenv("DCF_FORCE_SNAPSHOT", "1")

    assert fetch.get_company_data("AAPL")["snapshot_date"] == "2026-11-04"
    assert stub.calls == 0


def test_blank_ticker_returns_error(saved_snapshot):
    assert "error" in fetch.get_company_data("   ")
    assert "error" in fetch.get_company_data(None)
