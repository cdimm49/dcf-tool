"""Runs the whole Streamlit app on saved snapshots, the way a visitor would.

Unit tests check the math; these catch the app itself breaking: a widget
error, a crash for an unusual company, or a tab that stops rendering.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(monkeypatch):
    # Saved data only: no network, and the same numbers every run.
    monkeypatch.setenv("DCF_FORCE_SNAPSHOT", "1")
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    return app


def _load(app, ticker):
    app.text_input(key="ticker_input").set_value(ticker)
    next(button for button in app.button if button.label == "Run").click().run()
    assert not app.exception, [error.value for error in app.exception]


def _metric(app, label):
    return next((m.value for m in app.metric if m.label == label), None)


def _warnings(app):
    return [warning.value for warning in app.warning]


@pytest.mark.parametrize("ticker", ["AAPL", "BA", "NFLX", "XOM", "V"])
def test_valuation_and_monte_carlo_render_for_both_terminal_methods(app, ticker):
    _load(app, ticker)
    assert _metric(app, "Implied value per share") is not None
    assert _metric(app, "Median value (P50)") is not None

    app.radio(key="dcf_terminal_method").set_value("exit").run()
    assert not app.exception
    grid = next(
        frame.value
        for frame in app.dataframe
        if frame.value.columns.name == "Exit EV / EBITDA"
    )
    multiples = [float(label.rstrip("x")) for label in grid.columns]
    assert all(multiple > 0 for multiple in multiples)


def test_switching_tickers_in_one_session(app):
    for ticker in ["AAPL", "JPM", "MSFT"]:
        _load(app, ticker)


def test_bank_gets_one_clear_message(app):
    _load(app, "JPM")
    messages = [w for w in _warnings(app) if "could not be calculated" in w]
    assert len(messages) == 1
    assert messages[0].count("may not fit") == 1
    assert any("banks, insurers" in w for w in _warnings(app))


def test_payment_network_is_not_treated_as_a_bank(app):
    _load(app, "V")
    assert not any("banks, insurers" in w for w in _warnings(app))


@pytest.mark.parametrize("ticker", ["AAPL", "BA"])
def test_football_field_renders_without_52_week_request(app, monkeypatch, ticker):
    import data.fetch

    requested = []
    monkeypatch.setattr(
        data.fetch, "_cached_52_week_range", lambda symbol: requested.append(symbol)
    )
    _load(app, ticker)

    assert any(s.value.startswith("Football field") for s in app.subheader)
    shown = next(m for m in app.multiselect if m.label == "Methods shown")
    assert "DCF: Gordon growth" in shown.value
    assert "Monte Carlo (P10–P90)" in shown.value
    assert requested == []


def _captions(app):
    return [caption.value for caption in app.caption]


def _infos(app):
    return [info.value for info in app.info]


def test_netflix_is_capped_with_warning_note_limits_and_implied_multiple(app):
    _load(app, "NFLX")
    assert any("far above capex" in w for w in _warnings(app))
    captions = " ".join(_captions(app))
    assert "above reported free cash flow" in captions
    assert "straight line" in captions
    for number in (1, 2, 3):
        assert f"Limit {number}" in captions
    assert _metric(app, "Implied exit EV / EBITDA (from Gordon growth)") is not None


def test_amd_is_not_capped_and_gets_an_info_box_not_a_warning(app):
    _load(app, "AMD")  # D&A 4x capex, but reported cash flow agrees with the model
    assert not any("far above capex" in w for w in _warnings(app))
    assert any("far above capex" in i for i in _infos(app))
    assert "all D&A is added back" in " ".join(_captions(app))


@pytest.mark.parametrize("ticker", ["AAPL", "KO", "JNJ", "ADBE"])
def test_other_companies_are_not_capped(app, ticker):
    _load(app, ticker)
    captions = " ".join(_captions(app))
    assert "all D&A is added back" in captions
    assert "above reported free cash flow" not in captions


def test_editing_the_addback_replaces_the_cap_note(app):
    _load(app, "NFLX")
    assert "above reported free cash flow" in " ".join(_captions(app))
    app.slider(key="dcf_da_addback_pct_revenue").set_value(20.0).run()
    captions = " ".join(_captions(app))
    assert "you set the add-back" in captions
    assert "above reported free cash flow" not in captions


def test_editing_da_moves_the_addback_unless_capped(app):
    _load(app, "KO")
    app.slider(key="dcf_da_pct_revenue").set_value(3.0).run()
    assert app.slider(key="dcf_da_addback_pct_revenue").value == 3.0
    _load(app, "NFLX")
    app.slider(key="dcf_da_pct_revenue").set_value(35.0).run()
    assert app.slider(key="dcf_da_addback_pct_revenue").value == pytest.approx(1.2, abs=0.1)


def test_ebitda_caveat_shows_for_netflix_only(app):
    _load(app, "NFLX")
    assert "rely on the Gordon growth value" in " ".join(_captions(app))
    _load(app, "AMD")
    assert "rely on the Gordon growth value" not in " ".join(_captions(app))
