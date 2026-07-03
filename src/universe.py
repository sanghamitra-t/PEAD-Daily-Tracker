"""Fetches current constituent lists for the US (S&P 500) and India (Nifty 500) universes."""

import time
from io import StringIO

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NSE_FALLBACK_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
]


def get_sp500_tickers():
    """Returns a sorted list of current S&P 500 ticker symbols."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def get_sp500_table():
    """Returns a {ticker: company_name} dict for current S&P 500 constituents."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0][["Symbol", "Security"]].copy()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(df["Symbol"], df["Security"]))


def get_nifty500_tickers():
    """Returns a sorted list of current Nifty 500 ticker symbols (with .NS suffix).

    Falls back to a small hardcoded list if the NSE archive fetch fails --
    NSE occasionally changes its endpoint path/format or blocks non-browser
    requests, so this should not be treated as guaranteed-reliable.
    """
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        session.get("https://www.nseindia.com", timeout=15)  # warms up cookies
        time.sleep(1)
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        tickers = (df["Symbol"].astype(str).str.strip() + ".NS").tolist()
        if len(tickers) < 100:
            raise ValueError("Unexpectedly short Nifty 500 list, falling back.")
        return sorted(set(tickers))
    except Exception as e:
        print(f"[warn] Nifty 500 fetch failed ({e}); using fallback list.")
        return NSE_FALLBACK_TICKERS
