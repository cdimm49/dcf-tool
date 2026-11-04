# DCF Tool

DCF Tool is an educational Streamlit app for exploring a company's estimated
value using discounted cash flow (DCF). It loads financial statements and
market data from Yahoo Finance through yfinance, then lets you adjust the
assumptions and compare the result with a market price.

## Tabs

The app has the five original valuation and financial-statement tabs, plus the
new **News** tab:

- **Valuation** — estimated value per share, cash flow forecast, and sensitivity
  to the discount rate and terminal assumption.
- **Multiples** — selected valuation multiples and an optional comparison with
  three to five peer companies.
- **Income statement** — cleaned historical income statement and margins.
- **Balance sheet** — cleaned historical balance sheet.
- **Cash flow statement** — cleaned historical cash flow statement.
- **News** — up to 15 recent company headlines with their source, publication
  time, and summary when available.

## Install and run

Use Python 3.10 or newer. From the project folder, create and activate a virtual
environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install the packages and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests

With the virtual environment active, run:

```bash
pytest
```

## Main assumptions and limitations

- The model projects five years of free cash flow to the firm.
- The terminal value uses either the Gordon growth method or an exit EV/EBITDA
  multiple.
- The discount rate and operating assumptions are estimates. The sidebar lets
  you review and adjust them; the assumptions table shows their sources.
- Financial statements and market data come from Yahoo Finance via yfinance.
  They can be delayed, incomplete, or unavailable, and the news feed may have
  no recent articles.
- A standard DCF is generally a poor fit for banks and insurers because debt and
  working capital are part of their operations.
- This is an educational tool, not investment advice. It does not account for
  every company-specific risk or replace independent research.

## Screenshots

_Screenshot placeholders: add app screenshots here._
