"""Per-share value ranges for a football field chart, one per valuation method.

Each range is {"method", "category", "low", "mid", "high", "note"} with
low <= mid <= high (mid may be None). A method whose inputs are missing or
invalid is left out rather than returned empty.
"""

from statistics import median

from model.assumptions import DCF_INPUTS
from model.dcf import run_dcf
from model.multiples import _latest_value, _number


def _range(method, category, low, mid, high, note):
    return {
        "method": method,
        "category": category,
        "low": float(low),
        "mid": None if mid is None else float(mid),
        "high": float(high),
        "note": note,
    }


def _value(inputs):
    """run_dcf's value per share, or None when the inputs are invalid."""
    try:
        return _number(run_dcf(**inputs)["value_per_share"])
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _dcf_range(assumptions, terminal_method):
    if any(assumptions.get(name) is None for name in DCF_INPUTS):
        return None
    base = {name: assumptions[name] for name in DCF_INPUTS}
    base["terminal_method"] = terminal_method
    wacc = base["wacc"]

    if terminal_method == "gordon":
        method, driver, step = "DCF: Gordon growth", "terminal_growth", 0.005
        center = base["terminal_growth"]
        others = (center - step, center + step)
        label = f"terminal growth {others[0]:.1%}–{others[1]:.1%}"
    else:
        method, driver = "DCF: Exit multiple", "exit_ebitda_multiple"
        center = base["exit_ebitda_multiple"]
        if center <= 0:
            return None
        others = (center * 0.8, center * 1.2)
        label = f"exit multiple {others[0]:.1f}x–{others[1]:.1f}x"

    mid = _value(base)
    if mid is None:
        return None

    corners = [
        _value({**base, "wacc": corner_wacc, driver: other})
        for corner_wacc in (wacc - 0.01, wacc + 0.01)
        for other in others
    ]
    valid = [value for value in corners if value is not None]
    note = f"WACC {wacc - 0.01:.1%}–{wacc + 0.01:.1%}, {label}"
    if len(valid) < len(corners):
        skipped = len(corners) - len(valid)
        note += f" ({skipped} invalid corner{'s' if skipped != 1 else ''} skipped)"
    # Including mid keeps low <= mid <= high even when corners were skipped.
    return _range(method, "intrinsic", min(valid + [mid]), mid, max(valid + [mid]), note)


def _monte_carlo_range(monte_carlo):
    stats = (monte_carlo or {}).get("summary_stats") or {}
    p10, p50, p90 = (_number(stats.get(key)) for key in ("p10", "p50", "p90"))
    if None in (p10, p50, p90):
        return None
    runs = len(monte_carlo.get("simulated_share_prices", ()))
    note = f"P10–P50–P90 of {runs:,} simulations" if runs else "P10–P50–P90"
    return _range("Monte Carlo (P10–P90)", "intrinsic", p10, p50, p90, note)


def _peer_multiples(peer_multiples, metric):
    values = [_number((peer or {}).get(metric)) for peer in peer_multiples or ()]
    return [value for value in values if value is not None and value > 0]


def _peer_pe_range(income_statement, peer_multiples):
    eps = _latest_value(income_statement, "Diluted EPS")
    multiples = _peer_multiples(peer_multiples, "P/E")
    if eps is None or eps <= 0 or len(multiples) < 2:
        return None
    prices = [eps * multiple for multiple in multiples]
    note = (
        f"{len(multiples)} peers, P/E {min(multiples):.1f}x–{max(multiples):.1f}x "
        f"× diluted EPS ${eps:,.2f}"
    )
    return _range("Peer P/E", "relative", min(prices), median(prices), max(prices), note)


def _peer_ev_ebitda_range(income_statement, assumptions, peer_multiples):
    ebitda = _latest_value(income_statement, "EBITDA")
    debt, cash, shares = (
        _number(assumptions.get(name)) for name in ("debt", "cash", "diluted_shares")
    )
    multiples = _peer_multiples(peer_multiples, "EV/EBITDA")
    if (
        ebitda is None
        or ebitda <= 0
        or None in (debt, cash, shares)
        or shares <= 0
        or len(multiples) < 2
    ):
        return None
    prices = [(ebitda * multiple - debt + cash) / shares for multiple in multiples]
    note = (
        f"{len(multiples)} peers, EV/EBITDA {min(multiples):.1f}x–{max(multiples):.1f}x "
        f"× EBITDA ${ebitda:,.0f}M"
    )
    return _range(
        "Peer EV/EBITDA", "relative", min(prices), median(prices), max(prices), note
    )


def _week52_range(low, high):
    low, high = _number(low), _number(high)
    if low is None or high is None or low > high:
        return None
    return _range("52-week range", "market", low, None, high, "52-week low/high close")


def football_field(
    assumptions,
    income_statement=None,
    peer_multiples=None,
    monte_carlo=None,
    week52_low=None,
    week52_high=None,
):
    """Return the value ranges every method with valid inputs can produce.

    assumptions: the DCF inputs (as passed to run_dcf); debt, cash and diluted
        shares are also used for the peer EV/EBITDA bridge. USD millions.
    income_statement: cleaned statement (data.clean.clean_company_statements),
        for the latest diluted EPS and operating EBITDA.
    peer_multiples: one model.multiples.calculate_multiples dict per peer.
    monte_carlo: a run_monte_carlo_dcf result; no new simulation is run.
    """
    assumptions = assumptions or {}
    ranges = [
        _dcf_range(assumptions, "gordon"),
        _dcf_range(assumptions, "exit"),
        _monte_carlo_range(monte_carlo),
        _peer_pe_range(income_statement, peer_multiples),
        _peer_ev_ebitda_range(income_statement, assumptions, peer_multiples),
        _week52_range(week52_low, week52_high),
    ]
    return [r for r in ranges if r is not None]
