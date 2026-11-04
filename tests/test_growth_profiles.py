import pandas as pd
import pytest

from model.assumptions import growth_profile_rates, historical_starting_growth


def _income(revenues):
    years = [f"FY{2024 - offset}" for offset in range(len(revenues))]
    return pd.DataFrame([revenues], index=["Revenue"], columns=years)


def test_fade_steps_evenly_from_start_to_terminal():
    rates = growth_profile_rates(0.10, 0.02, "fade")

    assert rates == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02])


def test_fade_rises_when_history_is_below_terminal_growth():
    rates = growth_profile_rates(-0.02, 0.02, "fade")

    assert rates == pytest.approx([-0.02, -0.01, 0.0, 0.01, 0.02])


def test_flat_holds_the_starting_rate():
    assert growth_profile_rates(0.10, 0.02, "flat") == pytest.approx([0.10] * 5)


def test_custom_and_missing_history_have_no_preset():
    assert growth_profile_rates(0.10, 0.02, "custom") is None
    assert growth_profile_rates(None, 0.02, "fade") is None
    assert growth_profile_rates(None, 0.02, "flat") is None


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        growth_profile_rates(0.10, 0.02, "bullish")


def test_starting_growth_is_capped_cagr():
    # 1000 -> 1210 over two years is a 10% CAGR.
    assert historical_starting_growth(_income([1210, 1100, 1000])) == pytest.approx(0.10)
    # Doubling every year is capped at 20%; a collapse is floored at -5%.
    assert historical_starting_growth(_income([400, 200, 100])) == pytest.approx(0.20)
    assert historical_starting_growth(_income([50, 100])) == pytest.approx(-0.05)


def test_starting_growth_needs_two_positive_revenues():
    assert historical_starting_growth(_income([1000])) is None
    assert historical_starting_growth(_income([1000, -5])) is None
