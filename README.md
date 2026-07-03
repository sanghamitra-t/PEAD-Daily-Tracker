# PEAD Daily Tracker

A live, self-updating repository that finds the top 10 daily gaining stocks
in the US (S&P 500) and India (Nifty 500), tags a likely reason from recent
news, checks whether the move coincides with a recent earnings report, and
classifies each as consistent with post-earnings-announcement drift (PEAD),
a reversal, a fresh unrelated catalyst, or not earnings-related at all.

Every run appends to `data/daily_top_performers_log.csv` — over time this
becomes your own empirical dataset for checking whether PEAD flags actually
played out.

## Structure

```
pead-daily-tracker/
├── src/
│   ├── universe.py        # S&P 500 / Nifty 500 ticker list fetching
│   ├── movers.py           # ranks tickers by daily % return, returns top N
│   ├── news_tagging.py      # pulls headlines, tags a likely reason
│   ├── earnings_pead.py     # earnings lookup + SUE + PEAD classification
│   ├── run_daily.py         # main script -- run this once per trading day
│   └── evaluate_drift.py    # run periodically to validate past PEAD flags
├── data/
│   └── daily_top_performers_log.csv   # grows by one batch each run
├── .github/workflows/daily_run.yml    # optional full automation
└── requirements.txt
```

## Running it manually (Colab or local)

```bash
pip install -r requirements.txt
python src/run_daily.py
```

Run `src/evaluate_drift.py` every week or two once the log has enough
history to check forward returns against your PEAD flags.

## Running it automatically (GitHub Actions)

1. Create a new GitHub repository and push this folder to it.
2. That's it — `.github/workflows/daily_run.yml` is already wired up to run
   on weekdays and commit the updated log back to the repo automatically.
   No external server or paid service needed; GitHub's free tier covers
   this comfortably.
3. Adjust the `cron` schedule in that file if you want a different run
   time (it's in UTC).
4. You can also trigger a run manually anytime from the repo's
   **Actions** tab (`workflow_dispatch`).

## Known limitations (read before trusting the output)

- **India earnings data is thin.** yfinance's EPS-estimate/SUE coverage is
  solid for US large-caps but sparse for NSE tickers — expect "SUE
  unavailable" often, and treat days-since-earnings as more reliable than
  the SUE sign for Indian names. Cross-check manually (Screener.in,
  Trendlyne, exchange filings) before acting on an India PEAD flag.
- **Stale earnings-date snapshots.** If a company reports again after a
  flagged move, some data sources (this applies more to Market Chameleon
  than yfinance, but worth remembering generally) will show the newer
  report instead of the one you're tracking. Always sanity-check the
  reported date against what you expect.
- **News tagging is a blunt keyword match.** It's meant to save you time
  scanning, not replace reading the actual top headline before trusting a
  tag — and it can't tell a *symptom* headline ("Why X Stock Jumped Today")
  from an actual *cause*.
- **±1 trading day earnings-match window** (used in the historical tagging
  script, not this live one — this one uses same-day earnings only via
  `days_since_earnings == 0`) is a judgment call balancing BMO/AMC timing
  against false positives.
- **Universe snapshot bias.** Current S&P 500 / Nifty 500 constituents are
  used going forward — fine for a live tracker, since membership only
  changes occasionally.
