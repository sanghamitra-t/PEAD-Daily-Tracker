"""Main entry point: run this once per day (manually, or via the included
GitHub Action) to find today's top 10 movers in US + India, tag a likely
reason, classify PEAD status, and append everything to the running log.

Usage:
    python src/run_daily.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from universe import get_sp500_tickers, get_nifty500_tickers
from movers import get_top_movers
from news_tagging import get_news_headlines, tag_news_reason
from earnings_pead import get_earnings_info, classify_pead

warnings.filterwarnings("ignore")

TOP_N = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_PATH = os.path.join(DATA_DIR, "daily_top_performers_log.csv")


def build_report(movers_df):
    results = []
    for _, row in movers_df.iterrows():
        ticker = row["ticker"]
        headlines = get_news_headlines(ticker)
        tags = tag_news_reason(headlines)
        earn_info = get_earnings_info(ticker)

        trading_days_since = None
        sue = np.nan
        sue_persistence = None
        sue_persistence_detail = None
        if earn_info is not None:
            calendar_days = (pd.Timestamp.today(tz=None).normalize()
                              - earn_info["last_earnings_date"].tz_localize(None)).days
            trading_days_since = max(int(round(calendar_days * 5 / 7)), 0)
            sue = earn_info["sue"]
            sue_persistence = earn_info["sue_persistence"]
            sue_persistence_detail = earn_info["sue_persistence_detail"]

        move_dir = 1 if row["pct_change"] > 0 else -1
        pead_flag = classify_pead(trading_days_since, sue, move_dir, sue_persistence)

        data_check = None
        vol_ratio = row["volume_vs_avg"]
        if pd.notna(vol_ratio) and vol_ratio > 20:
            data_check = f"Volume {vol_ratio:.1f}x average - verify (corporate action / thin base volume?)"

        results.append({
            "ticker": ticker,
            "market": row["market"],
            "pct_change": round(row["pct_change"], 2),
            "volume_vs_avg": round(row["volume_vs_avg"], 2) if pd.notna(row["volume_vs_avg"]) else None,
            "news_tags": ", ".join(tags),
            "top_headline": headlines[0] if headlines else "(no headline found)",
            "days_since_earnings": trading_days_since,
            "sue": round(sue, 2) if pd.notna(sue) else None,
            "sue_persistence": sue_persistence,
            "sue_persistence_detail": sue_persistence_detail,
            "pead_flag": pead_flag,
            "data_check": data_check,
            "as_of": row["as_of"],
        })
    return pd.DataFrame(results)


def append_to_log(report_df, path=LOG_PATH):
    report_df = report_df.copy()
    report_df["logged_at"] = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        report_df.to_csv(path, mode="a", header=False, index=False)
    else:
        report_df.to_csv(path, index=False)


def run_daily_screen():
    print("Fetching universes...")
    us_tickers = get_sp500_tickers()
    india_tickers = get_nifty500_tickers()
    print(f"US universe: {len(us_tickers)} tickers | India universe: {len(india_tickers)} tickers")

    print("Finding top movers...")
    us_movers = get_top_movers(us_tickers, "US", n=TOP_N)
    india_movers = get_top_movers(india_tickers, "India", n=TOP_N)

    all_movers = pd.concat([us_movers, india_movers], ignore_index=True)
    if all_movers.empty:
        print("No movers found - check data connectivity.")
        return None

    print("Pulling news + earnings context for each mover...")
    report = build_report(all_movers)
    append_to_log(report)
    return report


if __name__ == "__main__":
    report = run_daily_screen()
    if report is not None:
        pd.set_option("display.max_colwidth", 60)
        print("\n=== TODAY'S TOP MOVERS + PEAD ASSESSMENT ===")
        print(report.to_string(index=False))
        print(f"\nAppended to log: {LOG_PATH}")
