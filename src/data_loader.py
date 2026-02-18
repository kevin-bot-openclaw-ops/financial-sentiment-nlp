"""
Financial news data loader.

Supports:
- Static bundled sample dataset (no network required for demo)
- RSS feed ingestion from Reuters / FT / Bloomberg public feeds
- Manual headline input

The Financial PhraseBank dataset categories map to:
  positive  → bullish signal
  negative  → bearish signal
  neutral   → no signal / monitor
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import datetime

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False


SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample" / "headlines.json"

# Curated sample headlines covering real financial language patterns
BUILTIN_HEADLINES = [
    # Positive signals
    {"text": "Goldman Sachs beats Q3 earnings expectations by 15%, raises full-year guidance", "source": "Reuters", "date": "2024-10-15"},
    {"text": "ECB signals end of rate hiking cycle as inflation approaches 2% target", "source": "FT", "date": "2024-10-14"},
    {"text": "JPMorgan reports record investment banking fees on resurgent M&A activity", "source": "Bloomberg", "date": "2024-10-13"},
    {"text": "Visa card payment volumes surge 12% YoY driven by travel and e-commerce recovery", "source": "Reuters", "date": "2024-10-12"},
    {"text": "European banks capital ratios strengthen as loan loss provisions normalize", "source": "FT", "date": "2024-10-11"},
    {"text": "BlackRock AUM hits $10 trillion milestone as investors return to equity markets", "source": "Bloomberg", "date": "2024-10-10"},
    {"text": "Santander consumer credit portfolio quality improves across all European markets", "source": "Reuters", "date": "2024-10-09"},
    # Negative signals
    {"text": "Credit Suisse faces potential $2bn writedown on leveraged loan exposure", "source": "Bloomberg", "date": "2024-10-08"},
    {"text": "Deutsche Bank warns of rising NPL ratios as commercial real estate defaults mount", "source": "FT", "date": "2024-10-07"},
    {"text": "Fed signals rates higher for longer, triggering selloff in rate-sensitive financials", "source": "Reuters", "date": "2024-10-06"},
    {"text": "Regional US banks report surge in deposit outflows amid confidence crisis", "source": "Bloomberg", "date": "2024-10-05"},
    {"text": "HSBC profit warning issued as Asia operations face headwinds from China slowdown", "source": "FT", "date": "2024-10-04"},
    {"text": "BNP Paribas trading revenues collapse 23% on adverse fixed income conditions", "source": "Reuters", "date": "2024-10-03"},
    {"text": "Moody's downgrades 10 US regional banks citing commercial real estate concentration risk", "source": "Bloomberg", "date": "2024-10-02"},
    # Neutral / mixed
    {"text": "Bank of England holds rates steady, maintains data-dependent forward guidance", "source": "Reuters", "date": "2024-10-01"},
    {"text": "Basel IV implementation timeline extended to 2026 pending final calibration", "source": "FT", "date": "2024-09-30"},
    {"text": "ING Group announces strategic review of retail banking operations in Germany", "source": "Bloomberg", "date": "2024-09-29"},
    {"text": "Citigroup restructuring enters final phase with 7,000 roles eliminated to date", "source": "Reuters", "date": "2024-09-28"},
    {"text": "SWIFT announces expansion of ISO 20022 migration deadline to March 2025", "source": "FT", "date": "2024-09-27"},
    {"text": "European Central Bank maintains asset purchase programme at current pace", "source": "Bloomberg", "date": "2024-09-26"},
]

# Real Financial PhraseBank examples by category (Malo et al., 2014)
# Used to illustrate training data quality and model calibration
FINANCIAL_PHRASEBANK_EXAMPLES = [
    # positive
    {"text": "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the year-earlier period.", "label": "positive"},
    {"text": "The company's order backlog is at a record high level.", "label": "positive"},
    {"text": "Net sales of the Paper segment decreased by 18.5 % to EUR 3.1 bn.", "label": "negative"},
    {"text": "Finnish electronics maker Nokia cut its profit forecast for the rest of the year.", "label": "negative"},
    {"text": "Citycon is developing its shopping centres in Sweden and Estonia.", "label": "neutral"},
    {"text": "Diluted EPS before exceptional items was EUR 0.32.", "label": "neutral"},
]


@dataclass
class Headline:
    """Represents a single financial news headline."""
    text: str
    source: str = "unknown"
    date: str = field(default_factory=lambda: datetime.date.today().isoformat())
    label: Optional[str] = None  # ground truth if available

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "date": self.date,
            "label": self.label,
        }


def load_sample_headlines() -> List[Headline]:
    """Load bundled sample headlines. No network required."""
    if SAMPLE_DATA_PATH.exists():
        with open(SAMPLE_DATA_PATH) as f:
            data = json.load(f)
        return [Headline(**h) for h in data]
    return [Headline(**h) for h in BUILTIN_HEADLINES]


def load_rss_headlines(feed_urls: Optional[List[str]] = None, max_per_feed: int = 10) -> List[Headline]:
    """
    Fetch headlines from RSS feeds.

    Default feeds are public financial RSS endpoints. Falls back to sample
    data if feedparser is not installed or feeds are unreachable.
    """
    if not _HAS_FEEDPARSER:
        print("feedparser not installed — falling back to sample data")
        return load_sample_headlines()

    default_feeds = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.ft.com/?format=rss",
    ]
    urls = feed_urls or default_feeds
    headlines = []

    for url in urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                text = entry.get("title", "").strip()
                if text:
                    headlines.append(Headline(
                        text=text,
                        source=feed.feed.get("title", url),
                        date=datetime.date.today().isoformat(),
                    ))
        except Exception as e:
            print(f"[data_loader] Failed to fetch {url}: {e}")

    if not headlines:
        print("[data_loader] No RSS headlines fetched — falling back to sample data")
        return load_sample_headlines()

    return headlines


def load_custom_headlines(texts: List[str]) -> List[Headline]:
    """Wrap raw strings as Headline objects for ad-hoc analysis."""
    return [Headline(text=t, source="user_input") for t in texts]
