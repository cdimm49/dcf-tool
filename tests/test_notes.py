import pytest

from ui import notes


def _a(status="no_cap", da=0.05, add=None, capex=0.06, gap=0.02):
    return {
        "da_cap_status": status,
        "da_pct_revenue": da,
        "da_addback_pct_revenue": da if add is None else add,
        "capex_pct_revenue": capex,
        "da_cash_gap": gap,
    }


def test_note_for_a_capped_company_explains_the_gap_and_keeps_ebitda_reported():
    text = notes.da_note(_a("capped", da=0.40, add=0.01, capex=0.01, gap=0.41))
    assert "40.0%" in text and "1.0%" in text
    assert "41% of revenue above reported free cash flow" in text
    assert "EBITDA" in text


def test_note_follows_a_sidebar_edit_instead_of_staying_stale():
    capped = _a("capped", da=0.40, add=0.20, capex=0.01, gap=0.41)  # add-back raised
    assert "you set the add-back" in notes.da_note(capped)
    assert "above reported free cash flow" not in notes.da_note(capped)
    full = _a("capped", da=0.40, add=0.40, capex=0.01, gap=0.41)  # cap fully overridden
    assert "overriding the cap" in notes.da_note(full)


def test_note_for_an_uncapped_company_is_true_even_when_d_and_a_barely_exceeds_capex():
    # Apple-like: D&A 2.9% vs capex 2.8%, nothing trimmed, so "all D&A is added back".
    text = notes.da_note(_a("no_cap", da=0.029, capex=0.028, gap=-0.003))
    assert "all D&A is added back" in text
    assert "15%" in text


def test_note_when_capex_is_missing_says_so_and_warns_about_overstatement():
    text = notes.da_note(_a("capex_missing", da=0.40, capex=0.0, gap=None))
    assert "no capex was reported" in text
    assert "overstates" in text
    assert notes.da_warning(_a("capex_missing", da=0.40, capex=0.0, gap=None)) is None


def test_note_when_reported_free_cash_flow_is_missing():
    assert "No reported free cash flow" in notes.da_note(_a("fcf_missing", gap=None))


def test_warning_levels():
    capped = notes.da_warning(_a("capped", da=0.40, add=0.01, capex=0.01))
    assert capped[0] == "warning" and "only 1.0% is added back" in capped[1]
    overridden = notes.da_warning(_a("capped", da=0.40, add=0.40, capex=0.01))
    assert overridden[0] == "warning" and "probably overstated" in overridden[1]
    acquirer = notes.da_warning(_a("no_cap", da=0.12, capex=0.03))
    assert acquirer[0] == "info" and "matches the model" in acquirer[1]
    assert notes.da_warning(_a("no_cap", da=0.05, capex=0.04)) is None  # under 2x


def test_implied_multiple_not_meaningful_when_negative_or_missing():
    text, caption = notes.implied_multiple(None, 20.0)
    assert text == "Not meaningful"
    assert "zero or negative" in caption


def test_implied_multiple_caption_covers_above_below_and_close():
    assert "too generous" in notes.implied_multiple(30.0, 20.0)[1]
    below = notes.implied_multiple(10.0, 20.0)[1]
    assert "below the current 20.0x" in below and "conservative" in below
    assert "close to the current" in notes.implied_multiple(21.0, 20.0)[1]
    assert notes.implied_multiple(12.3, 0)[0] == "12.3x"


def test_three_known_limits_are_listed():
    assert len(notes.LIMITS) == 3
    assert all(t.startswith(f"Limit {i}") for i, t in enumerate(notes.LIMITS, 1))
    assert "15%" in notes.LIMITS[1]


def test_terminal_note_describes_the_fade_and_the_one_point_one_rule():
    text = notes.terminal_note()
    assert "1.1×" in text and "straight line" in text and "year 5" in text


def test_ebitda_caveat_only_for_capped_companies():
    text = notes.ebitda_caveat(_a("capped", da=0.40, add=0.01, capex=0.01))
    assert text == (
        "EBITDA multiples are unreliable here because reported EBITDA includes content "
        "amortization; rely on the Gordon growth value."
    )
    assert notes.ebitda_caveat(_a("no_cap")) is None
    assert notes.ebitda_caveat(_a("capex_missing")) is None
