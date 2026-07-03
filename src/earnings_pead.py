"""Earnings lookup and PEAD (post-earnings-announcement-drift) classification.

Coverage note: yfinance's EPS-estimate/SUE data is decent for US large-caps
but thin for NSE tickers. For India names, expect "SUE unavailable" often --
treat days-since-earnings as more reliable than the SUE sign there, and
verify manually (Screener.in / Trendlyne / exchange filings) before acting.
"""

import numpy as np
import pandas as pd
import yfinance as yf

PEAD_WINDOW_TRADING_DAYS = 90  # typical drift window used in the literature
SUE_AMBIGUOUS_THRESHOLD = 0.1  # |SUE| below this is treated as noise, not signal


def get_earnings_info(ticker):
    """Pulls last reported earnings + SUE for a ticker. Returns None if
    no usable earnings/estimate data is available."""
    try:
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=12)
        if edf is None or edf.empty:
            return None
        edf = edf.dropna(subset=["Reported EPS", "EPS Estimate"])
        if edf.empty:
            return None
        edf = edf.sort_index()
        surprises = edf["Reported EPS"] - edf["EPS Estimate"]
        sue_std = surprises.std()
        last_row = edf.iloc[-1]
        sue = (
            (last_row["Reported EPS"] - last_row["EPS Estimate"]) / sue_std
            if sue_std and sue_std > 0 else np.nan
        )
        return {
            "last_earnings_date": edf.index[-1],
            "reported_eps": last_row["Reported EPS"],
            "eps_estimate": last_row["EPS Estimate"],
            "surprise_pct": last_row.get("Surprise(%)", np.nan),
            "sue": sue,
        }
    except Exception:
        return None


def classify_pead(trading_days_since, sue, move_direction):
    if trading_days_since is None:
        return "No recent earnings on record"
    if trading_days_since <= 0:
        return "Day 0 earnings reaction (not drift yet)"
    if trading_days_since > PEAD_WINDOW_TRADING_DAYS:
        return f"Outside typical PEAD window (Day {trading_days_since})"
    if pd.isna(sue):
        return f"In drift window (Day {trading_days_since}) - SUE unavailable, verify manually"
    if abs(sue) < SUE_AMBIGUOUS_THRESHOLD:
        return f"In drift window (Day {trading_days_since}) - SUE near zero, ambiguous signal"
    expected_direction = 1 if sue > 0 else -1
    if expected_direction == move_direction:
        return f"Consistent with PEAD continuation (Day {trading_days_since}, SUE={sue:.2f})"
    return f"Contradicts SUE sign (Day {trading_days_since}, SUE={sue:.2f}) - likely fresh catalyst"
