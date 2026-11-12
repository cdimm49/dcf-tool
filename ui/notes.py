"""Plain-English notes about the cash-flow rules. No Streamlit, so they can be tested."""

from model.assumptions import DA_CAP_GAP_THRESHOLD
from model.dcf import TERMINAL_CAPEX_TO_DA

DA_WARNING_RATIO = 2  # D&A this many times capex counts as "far above"
_TRIMMED = 0.0005  # half a tenth of a point: below this the add-back equals D&A


def _live(assumptions):
    da = assumptions.get("da_pct_revenue") or 0.0
    add = assumptions.get("da_addback_pct_revenue")
    return da, (da if add is None else add), assumptions.get("capex_pct_revenue") or 0.0


def terminal_note():
    return (
        f"Terminal year and fade: capex moves in a straight line from today's share of "
        f"revenue to {TERMINAL_CAPEX_TO_DA:.1f}× the D&A that capex pays for (the smaller "
        "of the D&A add-back and today's capex), reaching it in year 5 and staying there "
        "in the terminal year. In words: the company replaces its worn-out assets and "
        "adds 10% on top to grow, instead of spending today's share forever. The fade "
        "shapes the year 1–5 cash flows under both terminal methods."
    )


def da_note(assumptions):
    """What is added back to cash flow and why, from the live (sidebar) values."""
    status = assumptions.get("da_cap_status")
    da, add, capex = _live(assumptions)
    gap = assumptions.get("da_cash_gap")
    if status == "capex_missing" and capex == 0:
        return (
            "D&A add-back: no capex was reported, so the cap could not be checked. All "
            "D&A is added back and capex is assumed to be 0%, which probably overstates "
            "cash flow. Enter a capex share in the sidebar."
        )
    if da - add > _TRIMMED:
        if status == "capped" and abs(add - min(da, capex)) < _TRIMMED and gap is not None:
            return (
                f"D&A add-back: reported D&A is {da:.1%} of revenue, but only {add:.1%} "
                f"(capex) is added back. The model's cash flow was {gap:.0%} of revenue "
                f"above reported free cash flow (limit {DA_CAP_GAP_THRESHOLD:.0%}), so part "
                "of the amortization is probably paid for outside capex, like content. "
                "Reported D&A still feeds EBITDA and the multiples."
            )
        return (
            f"D&A add-back: reported D&A is {da:.1%} of revenue, but you set the add-back "
            f"to cash flow at {add:.1%}."
        )
    if status == "capped":
        return (
            f"D&A add-back: you set the full {da:.1%} of D&A back, overriding the cap the "
            f"check suggested (model cash flow was {gap:.0%} of revenue above reported "
            "free cash flow)."
        )
    if status == "fcf_missing":
        return (
            "D&A add-back: all D&A is added back. No reported free cash flow was "
            "available to check it against, so no cap could be tested."
        )
    return (
        f"D&A add-back: all D&A is added back, because the model's cash flow is within "
        f"{DA_CAP_GAP_THRESHOLD:.0%} of revenue of reported free cash flow."
    )


def ebitda_caveat(assumptions):
    """Extra note for companies whose D&A add-back was capped (Netflix-like)."""
    if assumptions.get("da_cap_status") != "capped":
        return None
    return (
        "EBITDA multiples are unreliable here because reported EBITDA includes content "
        "amortization; rely on the Gordon growth value."
    )


def da_warning(assumptions):
    """(level, text) when D&A is far above capex, else None. Level: warning or info."""
    status = assumptions.get("da_cap_status")
    da, add, capex = _live(assumptions)
    if status == "capex_missing" and capex == 0:
        return None  # nothing to compare with; da_note explains
    if capex <= 0 or da <= DA_WARNING_RATIO * capex:
        return None
    head = f"D&A ({da:.1%} of revenue) is far above capex ({capex:.1%}). "
    if da - add > _TRIMMED:
        return "warning", head + (
            "The difference is often amortization of content or acquired assets. "
            "Content is paid for outside capex, so adding it all back would overstate "
            f"cash flow; only {add:.1%} is added back."
        )
    if status == "capped":
        return "warning", head + (
            "The check against reported free cash flow says part of this is not covered "
            "by capex, yet all of it is added back, so cash flow is probably overstated."
        )
    return "info", head + (
        "Reported free cash flow matches the model, so the gap is probably non-cash "
        "amortization of acquisitions, and all of it is added back."
    )


def implied_multiple(implied, current):
    """(value text, caption) for the implied exit EV/EBITDA from Gordon growth."""
    basics = (
        "The terminal value expressed as a multiple of final-year EBITDA, a sanity "
        "check rather than a target."
    )
    if implied is None:
        return "Not meaningful", (
            basics + " Here the terminal cash flow or EBITDA is zero or negative, so "
            "it cannot be written as a multiple."
        )
    text = f"{implied:.1f}x"
    if not current or current <= 0:
        return text, basics
    ratio = implied / current
    if ratio > 1.25:
        verdict = (
            f"It is above the current {current:.1f}x: growth or margin assumptions may "
            "be too generous, or the market is more cautious than the model."
        )
    elif ratio < 0.75:
        verdict = (
            f"It is below the current {current:.1f}x: either the market expects more "
            "growth or profit than this model does, or the model's assumptions (fading "
            "growth, discount rate, margins) are conservative."
        )
    else:
        verdict = f"It is close to the current {current:.1f}x."
    return text, f"{basics} {verdict}"


LIMITS = (
    "Limit 1, perpetual add-back: when D&A is above capex (companies that grew by "
    "acquisition, such as AMD, Adobe, J&J), the terminal year adds back more D&A than it "
    "spends on capex for ever. Acquisition amortization eventually ends, so terminal "
    "value is slightly overstated.",
    f"Limit 2, a judgement threshold: the D&A cap applies only when model cash flow beats "
    f"reported free cash flow by more than {DA_CAP_GAP_THRESHOLD:.0%} of revenue. A company "
    "near that line (Coca-Cola is close) can flip on one unusual year of cash flow.",
    "Limit 3, fade ignores growth: capex falls on schedule whatever revenue growth is, "
    "which matters most for a company in the middle of a build-out (Oracle).",
)
