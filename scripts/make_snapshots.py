"""Save live Yahoo Finance data to data/snapshots/ for the app's offline fallback.

Run this on your own computer (Yahoo usually doesn't block home connections),
then commit the new files in data/snapshots/ and push:

    python scripts/make_snapshots.py              # the default ticker list
    python scripts/make_snapshots.py AAPL NVDA    # only these tickers

Re-run it every month or so to keep the saved data reasonably fresh.
"""

import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Streamlit's cache decorators warn when used outside the app; the script doesn't need them.
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from data import snapshots  # noqa: E402
from data.fetch import fetch_live  # noqa: E402
from data.news import fetch_news  # noqa: E402

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "ORCL",
    "ADBE", "INTC", "JPM", "V", "KO", "PEP", "WMT", "COST", "HD", "NKE",
    "DIS", "JNJ", "PG", "XOM", "BA",
]
PAUSE_SECONDS = 3  # be polite to Yahoo between tickers


def main(tickers):
    saved, failed = [], []
    for position, symbol in enumerate(tickers):
        symbol = symbol.strip().upper()
        print(f"[{position + 1}/{len(tickers)}] {symbol} ... ", end="", flush=True)
        try:
            company_data = fetch_live(symbol)
        except Exception as error:
            print(f"FAILED ({str(error)[:90]})")
            failed.append(symbol)
            continue

        try:
            news = fetch_news(symbol)
        except Exception:
            news = []  # a snapshot without headlines is still useful

        path = snapshots.save_snapshot(symbol, company_data, news=news)
        print(f"saved {path.name} ({path.stat().st_size // 1024} KB, {len(news)} headlines)")
        saved.append(symbol)
        time.sleep(PAUSE_SECONDS)

    print(f"\nSaved {len(saved)}: {', '.join(saved) or '-'}")
    if failed:
        print(f"Failed {len(failed)}: {', '.join(failed)}  (wait a few minutes and re-run for just these)")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_TICKERS))
