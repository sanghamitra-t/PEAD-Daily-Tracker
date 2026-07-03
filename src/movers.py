"""Ranks a ticker universe by latest daily % return and returns the top N."""

import numpy as np
import pandas as pd
import yfinance as yf


def get_top_movers(tickers, market_label, n=10, lookback_days=10):
    """Downloads recent price history and ranks by latest daily % return.

    Returns a DataFrame with columns: ticker, market, last_close, pct_change,
    volume_vs_avg, as_of. Empty DataFrame if nothing could be computed.
    """
    data = yf.download(
        tickers, period=f"{lookback_days}d", group_by="ticker",
        threads=True, progress=False, auto_adjust=True,
    )

    rows = []
    for ticker in tickers:
        try:
            close = data[ticker]["Close"].dropna()
            volume = data[ticker]["Volume"].dropna()
            if len(close) < 2:
                continue
            pct_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            vol_ratio = (
                volume.iloc[-1] / volume.iloc[:-1].mean()
                if len(volume) > 1 and volume.iloc[:-1].mean() > 0 else np.nan
            )
            rows.append({
                "ticker": ticker,
                "market": market_label,
                "last_close": close.iloc[-1],
                "pct_change": pct_change,
                "volume_vs_avg": vol_ratio,
                "as_of": close.index[-1].date(),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("pct_change", ascending=False).head(n).reset_index(drop=True)
