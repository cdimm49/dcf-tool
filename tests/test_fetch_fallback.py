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
    monkeypatch.delenv("DCF_SKIP_YAHOO", raising=False)
    # No SEC access unless a test turns it on (and never a real request).
    monkeypatch.setattr(fetch, "_sec_user_agent", lambda: None)
    monkeypatch.setattr(fetch, "_cached_sec", _LiveStub(error=AssertionError("SEC called")))
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


SEC_DATA = {"company_name": "SEC Corp", "current_price": 50.0, "data_source": "sec"}


def test_sec_is_used_when_yahoo_fails_and_then_yahoo_is_skipped(saved_snapshot, monkeypatch):
    yahoo = _LiveStub(error=fetch.LiveDataError("Invalid Crumb"))
    sec_stub = _LiveStub(result=SEC_DATA)
    monkeypatch.setattr(fetch, "_cached_live", yahoo)
    monkeypatch.setattr(fetch, "_cached_sec", sec_stub)
    monkeypatch.setattr(fetch, "_sec_user_agent", lambda: "DCF Tool test@example.com")

    assert fetch.get_company_data("MSFT") == SEC_DATA  # no snapshot needed
    assert fetch.get_company_data("NVDA") == SEC_DATA
    assert yahoo.calls == 1  # blocked after the first confirmed failure
    assert sec_stub.calls == 2


def test_sec_failure_falls_through_to_snapshot(saved_snapshot, monkeypatch):
    monkeypatch.setattr(fetch, "_cached_live", _LiveStub(error=RuntimeError("401")))
    monkeypatch.setattr(fetch, "_cached_sec", _LiveStub(error=RuntimeError("SEC down")))
    monkeypatch.setattr(fetch, "_sec_user_agent", lambda: "DCF Tool test@example.com")

    assert fetch.get_company_data("AAPL")["company_name"] == "Saved Apple"


def test_error_mentions_sec_when_ticker_is_unknown_everywhere(saved_snapshot, monkeypatch):
    from data import sec

    monkeypatch.setattr(fetch, "_cached_live", _LiveStub(error=fetch.LiveDataError("Missing: all")))
    monkeypatch.setattr(
        fetch, "_cached_sec", _LiveStub(error=sec.SecDataError("ZZZZ isn't in the SEC's list"))
    )
    monkeypatch.setattr(fetch, "_sec_user_agent", lambda: "DCF Tool test@example.com")

    result = fetch.get_company_data("ZZZZ")
    assert "isn't in the SEC's list" in result["error"]
    assert fetch._live_blocked_until == 0.0  # an unknown symbol doesn't block Yahoo


def test_skip_yahoo_switch_goes_straight_to_sec(saved_snapshot, monkeypatch):
    yahoo = _LiveStub(result=LIVE)
    monkeypatch.setattr(fetch, "_cached_live", yahoo)
    monkeypatch.setattr(fetch, "_cached_sec", _LiveStub(result=SEC_DATA))
    monkeypatch.setattr(fetch, "_sec_user_agent", lambda: "DCF Tool test@example.com")
    monkeypatch.setenv("DCF_SKIP_YAHOO", "1")

    assert fetch.get_company_data("MSFT") == SEC_DATA
    assert yahoo.calls == 0


def test_estimate_beta_recovers_a_known_beta():
    import numpy as np

    rng = np.random.default_rng(0)
    market_returns = rng.normal(0.01, 0.04, 60)
    stock_returns = 1.5 * market_returns  # beta exactly 1.5
    months = pd.period_range("2021-01", periods=61, freq="M")
    market = pd.Series(100 * np.cumprod(np.r_[1, 1 + market_returns]), index=months)
    stock = pd.Series(50 * np.cumprod(np.r_[1, 1 + stock_returns]), index=months)

    assert fetch.estimate_beta(stock, market) == 1.5
    assert fetch.estimate_beta(stock[:10], market[:10]) is None  # too little history


def test_sec_price_comes_from_the_short_quote_cache(monkeypatch):
    statements = _LiveStub(
        result={
            "company_name": "SEC Corp",
            "shares_outstanding": 1_000.0,
            "sector": None,
            "industry": None,
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow_statement": pd.DataFrame(),
            "beta": 1.1,
        }
    )
    quote = _LiveStub(result={"current_price": 42.0, "risk_free_rate": 4.2})
    monkeypatch.setattr(fetch, "_cached_sec_statements", statements)
    monkeypatch.setattr(fetch, "_cached_quote", quote)

    data = fetch._cached_sec("ABC")

    assert data["current_price"] == 42.0
    assert data["risk_free_rate"] == 4.2
    assert data["data_source"] == "sec"
    assert statements.calls == quote.calls == 1
    # Statements are kept for hours; the price for minutes.
    assert fetch.QUOTE_TTL_SECONDS <= 15 * 60


def test_unknown_sec_ticker_fails_before_any_price_request(monkeypatch):
    from data import sec

    quote = _LiveStub(result={"current_price": 1.0, "risk_free_rate": 4.0})
    monkeypatch.setattr(
        fetch, "_cached_sec_statements", _LiveStub(error=sec.SecDataError("unknown"))
    )
    monkeypatch.setattr(fetch, "_cached_quote", quote)

    with pytest.raises(sec.SecDataError):
        fetch._cached_sec("ZZZZ")
    assert quote.calls == 0
