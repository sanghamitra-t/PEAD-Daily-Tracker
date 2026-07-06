"""Earnings lookup and PEAD (post-earnings-announcement-drift) classification.

REDESIGNED based on two findings from the literature (see conversation for
full paper discussion):

(a) SUE PERSISTENCE (Kettell, McInnis & Zhao 2022): a single quarter's SUE
    sign is a weak signal on its own. What predicts real drift is whether a
    stock's SUE has been PERSISTENT -- i.e., it tends to stay in the same
    extreme (good news keeps beating, bad news keeps missing) across
    consecutive quarters ("stayers"), rather than bouncing around
    ("movers"). Kettell et al. found stayers drove essentially all of the
    measured drift; movers showed no significant drift at all.

(b) IMMEDIATE VS. LATER WINDOW (same paper): the decline in PEAD over time
    is concentrated in the LATER part of the drift window (roughly day 7
    onward). The IMMEDIATE window (roughly days 1-6) has, if anything,
    gotten STRONGER over time. So a "Day 3" flag and a "Day 45" flag should
    not carry the same confidence -- they're different phenomena.

Coverage note unchanged from before: yfinance's EPS-estimate/SUE data is
decent for US large-caps but thin for NSE tickers.
"""

import numpy as np
import pandas as pd
import yfinance as yf

PEAD_WINDOW_TRADING_DAYS = 90        # outer bound of the "later" drift window
IMMEDIATE_WINDOW_END_DAYS = 6         # per Kettell et al.'s [+2,+6] immediate window
SUE_AMBIGUOUS_THRESHOLD = 0.1         # |SUE| below this is treated as noise, not signal
MIN_QUARTERS_FOR_PERSISTENCE = 2      # need at least this many prior quarters to judge persistence


def get_earnings_history(ticker, n_quarters=6):
    """Pulls up to n_quarters of reported earnings + SUE for a ticker, sorted
    oldest to newest. Returns None if no usable data is available.

    Returns a DataFrame with columns: reported_eps, eps_estimate, sue,
    indexed by earnings date.
    """
    try:
        t = yf.Ticker(ticker)
        edf = t.get_earnings_dates(limit=max(n_quarters + 4, 12))  # pad for dropna losses
        if edf is None or edf.empty:
            return None
        edf = edf.dropna(subset=["Reported EPS", "EPS Estimate"])
        if edf.empty:
            return None
        edf = edf.sort_index()
        surprises = edf["Reported EPS"] - edf["EPS Estimate"]
        sue_std = surprises.std()
        if not sue_std or sue_std == 0:
            return None
        edf["SUE"] = surprises / sue_std
        edf = edf.tail(n_quarters)
        return edf[["Reported EPS", "EPS Estimate", "SUE"]]
    except Exception:
        return None


def get_earnings_info(ticker):
    """Pulls last reported earnings + SUE + persistence classification for a
    ticker. Returns None if no usable earnings/estimate data is available.
    """
    history = get_earnings_history(ticker)
    if history is None or history.empty:
        return None

    last_row = history.iloc[-1]
    current_sue = last_row["SUE"]
    prior_sues = history["SUE"].iloc[:-1]  # everything before the most recent quarter

    persistence_label, persistence_detail = classify_sue_persistence(current_sue, prior_sues)

    return {
        "last_earnings_date": history.index[-1],
        "reported_eps": last_row["Reported EPS"],
        "eps_estimate": last_row["EPS Estimate"],
        "sue": current_sue,
        "sue_persistence": persistence_label,
        "sue_persistence_detail": persistence_detail,
        "n_prior_quarters": len(prior_sues),
    }


def classify_sue_persistence(current_sue, prior_sues):
    """Classifies whether a stock's SUE has been a 'stayer' (persistent,
    same-direction surprises) or a 'mover' (inconsistent), per Kettell et
    al.'s stayers-vs-movers framework.

    A 'stayer' is a firm whose current SUE and MOST prior quarters' SUEs
    share the same sign -- i.e., persistent good news or persistent bad
    news. A 'mover' has a mixed history. This is a simplified, single-stock
    analogue of Kettell et al.'s original cross-sectional decile-based
    measure (which requires ranking against the whole universe each
    quarter -- not practical for a lightweight daily screener).
    """
    if pd.isna(current_sue) or len(prior_sues) < MIN_QUARTERS_FOR_PERSISTENCE:
        return "Insufficient history", "Fewer than 2 prior quarters with usable SUE data"

    current_sign = np.sign(current_sue)
    prior_sues_clean = prior_sues.dropna()
    if len(prior_sues_clean) < MIN_QUARTERS_FOR_PERSISTENCE:
        return "Insufficient history", "Fewer than 2 prior quarters with usable SUE data"

    same_sign_count = (np.sign(prior_sues_clean) == current_sign).sum()
    total = len(prior_sues_clean)
    match_ratio = same_sign_count / total

    detail = f"{same_sign_count}/{total} prior quarters same sign as current"

    if match_ratio > 0.5:
        return "Stayer (persistent)", detail
    elif match_ratio < 0.5:
        return "Mover (inconsistent)", detail
    else:
        return "Mixed", detail


def classify_pead(trading_days_since, sue, move_direction, sue_persistence=None):
    """Classifies a move relative to earnings, now window-aware and
    persistence-aware.

    sue_persistence: one of "Stayer (persistent)", "Mover (inconsistent)",
    "Mixed", "Insufficient history", or None (if earnings info unavailable).
    """
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

    window_label = "immediate window" if trading_days_since <= IMMEDIATE_WINDOW_END_DAYS else "later window"
    expected_direction = 1 if sue > 0 else -1
    direction_matches = (expected_direction == move_direction)

    if not direction_matches:
        return (f"Contradicts SUE sign (Day {trading_days_since}, {window_label}, "
                f"SUE={sue:.2f}) - likely fresh catalyst")

    # Direction matches -- now factor in persistence and window per Kettell et al.
    if sue_persistence == "Stayer (persistent)":
        confidence = "high confidence" if window_label == "immediate window" else "moderate confidence"
        return (f"Consistent with PEAD continuation, {confidence} (Day {trading_days_since}, "
                f"{window_label}, SUE={sue:.2f}, persistent SUE history)")
    elif sue_persistence == "Mover (inconsistent)":
        return (f"Directionally consistent but LOW confidence (Day {trading_days_since}, "
                f"{window_label}, SUE={sue:.2f}) - this stock's SUE history is inconsistent "
                f"(mover, not stayer) -- per Kettell et al. 2022, movers show ~no significant drift")
    elif sue_persistence == "Insufficient history":
        confidence = "moderate confidence" if window_label == "immediate window" else "low confidence"
        return (f"Consistent with PEAD continuation, {confidence} (Day {trading_days_since}, "
                f"{window_label}, SUE={sue:.2f}, insufficient history to assess persistence)")
    else:  # Mixed or None
        confidence = "moderate confidence" if window_label == "immediate window" else "low confidence"
        return (f"Consistent with PEAD continuation, {confidence} (Day {trading_days_since}, "
                f"{window_label}, SUE={sue:.2f}, mixed SUE history)")
