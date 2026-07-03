"""Pulls recent news headlines for a ticker and tags a likely move reason
using a simple keyword match. This is meant to save you time scanning,
not to replace actually reading the top headline before trusting a flag."""

import yfinance as yf

KEYWORD_TAGS = {
    "earnings": ["earnings", "eps", "quarterly results", "profit", "revenue",
                 "guidance", "q1", "q2", "q3", "q4"],
    "M&A": ["acquire", "acquisition", "merger", "buyout", "stake sale", "takeover"],
    "upgrade/downgrade": ["upgrade", "downgrade", "price target", "rating", "initiates"],
    "regulatory/legal": ["lawsuit", "investigation", "fine", "sec probe", "ban", "penalty"],
    "macro/sector": ["fed", "rbi", "rate hike", "rate cut", "inflation", "crude", "tariff", "gdp"],
    "product/contract": ["launch", "contract", "order win", "deal", "partnership", "unveils"],
}


def get_news_headlines(ticker, n=5):
    """Handles both the legacy yfinance news schema (top-level 'title')
    and the newer schema (title nested under 'content')."""
    try:
        t = yf.Ticker(ticker)
        news = t.news[:n]
        headlines = []
        for item in news:
            title = item.get("title")
            if not title and isinstance(item.get("content"), dict):
                title = item["content"].get("title")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def tag_news_reason(headlines):
    text = " ".join(headlines).lower()
    tags = [tag for tag, kws in KEYWORD_TAGS.items() if any(kw in text for kw in kws)]
    return tags if tags else ["unclassified - check headlines manually"]
