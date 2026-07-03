"""Run this periodically (e.g. weekly, once the log has a few weeks of
history) to check whether stocks flagged 'Consistent with PEAD continuation'
actually kept drifting -- i.e. to validate your own flags against what
really happened afterward.

Usage:
    python src/evaluate_drift.py
"""

import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_PATH = os.path.join(DATA_DIR, "daily_top_performers_log.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "drift_validation_results.csv")
HORIZONS = (5, 10, 20, 60)


def evaluate_forward_returns(log_path=LOG_PATH, horizons=HORIZONS):
    if not os.path.exists(log_path):
        print("No log file yet - run run_daily.py for a few weeks first.")
        return None

    log = pd.read_csv(log_path, parse_dates=["as_of"])
    flagged = log[log["pead_flag"].str.contains("Consistent with PEAD", na=False)]

    results = []
    for _, row in flagged.iterrows():
        ticker = row["ticker"]
        start_date = pd.to_datetime(row["as_of"])
        try:
            hist = yf.download(
                ticker, start=start_date, end=start_date + pd.Timedelta(days=120),
                progress=False, auto_adjust=True,
            )["Close"]
            if hist.empty:
                continue
            base_price = hist.iloc[0]
            fwd_returns = {}
            for h in horizons:
                if len(hist) > h:
                    fwd_returns[f"fwd_{h}d_return_pct"] = round(
                        (hist.iloc[h] / base_price - 1) * 100, 2
                    )
            results.append({"ticker": ticker, "flag_date": row["as_of"], **fwd_returns})
        except Exception:
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = evaluate_forward_returns()
    if results is not None and not results.empty:
        results.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved {len(results)} validation rows to {OUTPUT_PATH}")
        print(results.to_string(index=False))
    else:
        print("No PEAD-flagged rows found yet to validate.")
