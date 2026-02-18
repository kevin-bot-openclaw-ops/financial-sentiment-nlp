"""
Financial Entity Extractor.

Extracts structured information from financial news headlines:
  - Company / institution names
  - Financial metrics (EPS, revenue, profit, etc.)
  - Directional signals (up/down, beats/misses)
  - Numeric values with units (%, $, €, bn)

Uses regex patterns for efficiency — no model download required.
In production, you'd layer in a NER model (e.g., dslim/bert-base-NER)
for higher recall on company names. This is a valid trade-off discussion
point in senior ML interviews.

Interview talking point: Named Entity Recognition in finance has a different
precision/recall trade-off than general NLP. False positives (wrong company
name) cause compliance risk; false negatives (missed entity) lose signal.
Most production systems use ensemble: rules for known entities + ML for novel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# Known financial institution patterns (partial — production would use full registry)
KNOWN_INSTITUTIONS = [
    "Goldman Sachs", "JPMorgan", "Citigroup", "Citi", "Morgan Stanley",
    "Bank of America", "Wells Fargo", "Deutsche Bank", "HSBC", "Barclays",
    "BNP Paribas", "Santander", "ING", "Commerzbank", "Credit Suisse",
    "UBS", "ABN AMRO", "Lloyds", "NatWest", "Standard Chartered",
    "BlackRock", "Vanguard", "Fidelity", "PIMCO", "Bridgewater",
    "ECB", "Federal Reserve", "Fed", "Bank of England", "BoE",
    "Moody's", "S&P", "Fitch", "Visa", "Mastercard", "PayPal",
    "SWIFT", "DTCC", "Euroclear", "Clearstream",
]

# Regex for financial metrics
_METRIC_PATTERNS = [
    (r'\bEPS\b', "earnings_per_share"),
    (r'\bEBITDA\b', "ebitda"),
    (r'\bEBIT\b', "ebit"),
    (r'\bROE\b', "return_on_equity"),
    (r'\bROA\b', "return_on_assets"),
    (r'\bNPL\b', "non_performing_loans"),
    (r'\bNIM\b', "net_interest_margin"),
    (r'\bCET1\b', "cet1_capital_ratio"),
    (r'\bAUM\b', "assets_under_management"),
    (r'\b(?:net\s+)?(?:profit|income)\b', "profit"),
    (r'\brevenue\b', "revenue"),
    (r'\boperating\s+(?:profit|income)\b', "operating_profit"),
    (r'\bdividend\b', "dividend"),
    (r'\bbuyback\b', "share_buyback"),
    (r'\bwrite(?:-?down|-?off)\b', "write_down"),
    (r'\bimpairment\b', "impairment"),
    (r'\bprovision\b', "loan_loss_provision"),
    (r'\bloan\s+loss\b', "loan_loss"),
]

# Directional signal words
_BULLISH_SIGNALS = [
    "beats", "beat", "surges", "surge", "rises", "rise", "grows", "growth",
    "record", "raises", "strong", "above", "outperforms", "upgrade",
    "accretive", "expansion", "improves", "improvement", "recovery",
]
_BEARISH_SIGNALS = [
    "misses", "miss", "falls", "decline", "drops", "warning", "cut",
    "below", "write-down", "writedown", "default", "breach", "fine",
    "collapse", "downgrade", "layoffs", "loss", "surge outflows",
]

# Numeric extraction: $1.2bn, €500m, 15%, -3.2%
_NUMERIC_RE = re.compile(
    r'(?:[\$€£¥])?\s*'
    r'(-?\d+(?:\.\d+)?)'
    r'\s*(?:bn|mn|m|k|trillion|billion|million|thousand)?'
    r'\s*(?:%|percent|bps|bp)?',
    re.IGNORECASE,
)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str  # institution | metric | numeric | signal


@dataclass
class ExtractionResult:
    text: str
    institutions: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    numerics: List[str] = field(default_factory=list)
    directional: str = "neutral"  # bullish | bearish | neutral

    def to_dict(self) -> dict:
        return {
            "institutions": self.institutions,
            "metrics": self.metrics,
            "numerics": self.numerics,
            "directional": self.directional,
        }


class EntityExtractor:
    """Rule-based financial entity extractor."""

    def __init__(self):
        # Compile institution regex once
        self._inst_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(inst) for inst in KNOWN_INSTITUTIONS) + r')\b',
            re.IGNORECASE,
        )
        # Compile metric patterns
        self._metric_patterns = [(re.compile(p, re.IGNORECASE), label) for p, label in _METRIC_PATTERNS]

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from a single headline."""
        result = ExtractionResult(text=text)

        # Institutions
        result.institutions = list({m.group(1) for m in self._inst_pattern.finditer(text)})

        # Financial metrics
        metrics_found = set()
        for pattern, label in self._metric_patterns:
            if pattern.search(text):
                metrics_found.add(label)
        result.metrics = sorted(metrics_found)

        # Numeric values
        result.numerics = [m.group(0).strip() for m in _NUMERIC_RE.finditer(text)
                           if m.group(1) and float(m.group(1)) != 0][:5]  # cap at 5

        # Directional signal
        lower = text.lower()
        bull = sum(1 for s in _BULLISH_SIGNALS if s in lower)
        bear = sum(1 for s in _BEARISH_SIGNALS if s in lower)
        if bull > bear:
            result.directional = "bullish"
        elif bear > bull:
            result.directional = "bearish"
        else:
            result.directional = "neutral"

        return result

    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        return [self.extract(t) for t in texts]
