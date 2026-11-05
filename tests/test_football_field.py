import pandas as pd
import pytest

from model.assumptions import DCF_INPUTS
from model.dcf import run_dcf
from model.football_field import football_field
from model.monte_carlo import run_monte_carlo_dcf

ASSUMPTIONS = {
    "base_revenue": 1000.0,
    "growth_rates": [0.08, 0.07, 0.06, 0.05, 0.04],
    "ebit_margin": 0.20,
    "tax_rate": 0.21,
    "da_pct_revenue": 0.03,
    "capex_pct_revenue": 0.04,
    "delta_nwc_pct_revenue_change": 0.10,
    "wacc": 0.103,
    "terminal_method": "gordon",
    "terminal_growth": 0.025,
    "exit_ebitda_multiple": 12.0,
    "debt": 50.0,
    "cash": 30.0,
    "diluted_shares": 10.0,
    "mid_year": False,
}
INCOME = pd.DataFrame(
    {"FY2025": [100.0, 2.0], "FY2024": [90.0, 1.8]}, index=["EBITDA", "Diluted EPS"]
)
PEERS = [
    {"P/E": 10.0, "EV/EBITDA": 8.0},
    {"P/E": 20.0, "EV/EBITDA": 12.0},
    {"P/E": 15.0, "EV/EBITDA": 10.0},
]


def _by_method(ranges):
    return {r["method"]: r for r in ranges}


def _ordered(r):
    return r["low"] <= r["mid"] <= r["high"] if r["mid"] is not None else r["low"] <= r["high"]


def test_dcf_ranges_center_on_run_dcf():
    ranges = _by_method(football_field(ASSUMPTIONS))

    gordon = ranges["DCF: Gordon growth"]
    expected = run_dcf(**{name: ASSUMPTIONS[name] for name in DCF_INPUTS})
    assert gordon["mid"] == expected["value_per_share"]
    assert gordon["note"] == "WACC 9.3%–11.3%, terminal growth 2.0%–3.0%"
    assert gordon["low"] < gordon["mid"] < gordon["high"]

    exit_range = ranges["DCF: Exit multiple"]
    assert exit_range["mid"] == run_dcf(
        **{**{name: ASSUMPTIONS[name] for name in DCF_INPUTS}, "terminal_method": "exit"}
    )["value_per_share"]
    assert exit_range["low"] < exit_range["mid"] < exit_range["high"]
    assert "exit multiple 9.6x–14.4x" in exit_range["note"]


def test_gordon_corners_with_growth_above_wacc_are_skipped():
    # WACC 3%: the 2% WACC x 2.5% growth corner is invalid; the other three run.
    ranges = _by_method(football_field({**ASSUMPTIONS, "wacc": 0.03, "terminal_growth": 0.02}))

    gordon = ranges["DCF: Gordon growth"]
    assert _ordered(gordon)
    assert "1 invalid corner skipped" in gordon["note"]


def test_gordon_skipped_when_base_case_invalid():
    ranges = _by_method(football_field({**ASSUMPTIONS, "terminal_growth": 0.103}))

    assert "DCF: Gordon growth" not in ranges
    assert "DCF: Exit multiple" in ranges


@pytest.mark.parametrize("multiple", [0.0, -5.0])
def test_non_positive_exit_multiple_skips_exit_method(multiple):
    ranges = _by_method(football_field({**ASSUMPTIONS, "exit_ebitda_multiple": multiple}))

    assert "DCF: Exit multiple" not in ranges
    assert "DCF: Gordon growth" in ranges


def test_peer_ranges_hand_calculated():
    ranges = _by_method(football_field(ASSUMPTIONS, INCOME, PEERS))

    pe = ranges["Peer P/E"]
    assert (pe["low"], pe["mid"], pe["high"]) == pytest.approx((20.0, 30.0, 40.0))
    assert pe["category"] == "relative"

    # (100 x multiple - 50 debt + 30 cash) / 10 shares
    ev = ranges["Peer EV/EBITDA"]
    assert (ev["low"], ev["mid"], ev["high"]) == pytest.approx((78.0, 98.0, 118.0))


def test_peer_ranges_need_two_valid_peers():
    peers = [{"P/E": 15.0, "EV/EBITDA": 10.0}, {"P/E": float("nan"), "EV/EBITDA": -3.0}]

    ranges = _by_method(football_field(ASSUMPTIONS, INCOME, peers))

    assert "Peer P/E" not in ranges
    assert "Peer EV/EBITDA" not in ranges


def test_peer_ranges_skip_non_positive_eps_and_ebitda():
    income = pd.DataFrame({"FY2025": [-10.0, 0.0]}, index=["EBITDA", "Diluted EPS"])

    ranges = _by_method(football_field(ASSUMPTIONS, income, PEERS))

    assert "Peer P/E" not in ranges
    assert "Peer EV/EBITDA" not in ranges


def test_monte_carlo_range_uses_passed_percentiles():
    result = run_monte_carlo_dcf(ASSUMPTIONS, n_simulations=500)
    stats = result["summary_stats"]

    mc = _by_method(football_field(ASSUMPTIONS, monte_carlo=result))["Monte Carlo (P10–P90)"]

    assert (mc["low"], mc["mid"], mc["high"]) == (stats["p10"], stats["p50"], stats["p90"])
    assert mc["category"] == "intrinsic"


def test_week52_range_has_no_mid():
    ranges = _by_method(football_field(ASSUMPTIONS, week52_low=80.0, week52_high=120.0))

    week52 = ranges["52-week range"]
    assert (week52["low"], week52["mid"], week52["high"]) == (80.0, None, 120.0)
    assert week52["category"] == "market"


def test_missing_inputs_return_only_what_is_available():
    assert football_field({}) == []
    assert [r["method"] for r in football_field({}, week52_low=1.0, week52_high=2.0)] == [
        "52-week range"
    ]


def test_negative_values_are_kept():
    deep_debt = {**ASSUMPTIONS, "debt": 100_000.0}

    ranges = football_field(deep_debt, INCOME, PEERS)

    by_method = _by_method(ranges)
    assert by_method["DCF: Gordon growth"]["mid"] < 0
    assert by_method["DCF: Exit multiple"]["high"] < 0
    assert by_method["Peer EV/EBITDA"]["low"] < 0
    assert all(_ordered(r) for r in ranges)
