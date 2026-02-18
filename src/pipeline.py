"""
End-to-end Financial NLP Pipeline.

Orchestrates: data loading → sentiment → entity extraction → risk aggregation → report

Usage:
    from src.pipeline import FinancialNLPPipeline

    pipeline = FinancialNLPPipeline()
    results = pipeline.run_on_samples()
    pipeline.print_report(results)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.data_loader import Headline, load_sample_headlines, load_custom_headlines
from src.sentiment import SentimentClassifier, SentimentResult
from src.entity_extractor import EntityExtractor, ExtractionResult
from src.risk_aggregator import RiskAggregator, RiskSignal


@dataclass
class AnalysisResult:
    """Full analysis result for a single headline."""
    headline: Headline
    sentiment: SentimentResult
    entities: ExtractionResult
    risk: RiskSignal

    def to_dict(self) -> dict:
        return {
            "headline": self.headline.to_dict(),
            "sentiment": self.sentiment.to_dict(),
            "entities": self.entities.to_dict(),
            "risk": self.risk.to_dict(),
        }


class FinancialNLPPipeline:
    """
    Three-stage pipeline: Sentiment → Entity Extraction → Risk Aggregation.

    Design decisions:
    - Sentiment runs first (most expensive, drives downstream)
    - Entity extraction is rule-based (fast, auditable, no GPU required)
    - Risk aggregation is deterministic given the two inputs (auditable)

    Scaling note: In production, sentiment would run as a batched microservice
    (e.g., Triton Inference Server + ONNX-optimized model). Entity extraction
    could be a separate K8s sidecar. This monolith is appropriate for portfolio
    demo and interview discussion.
    """

    def __init__(self, prefer_finbert: bool = True):
        self.sentiment_classifier = SentimentClassifier(prefer_finbert=prefer_finbert)
        self.entity_extractor = EntityExtractor()
        self.risk_aggregator = RiskAggregator()

    def analyze(self, headlines: List[Headline]) -> List[AnalysisResult]:
        """Run full pipeline on a list of headlines."""
        if not headlines:
            return []

        texts = [h.text for h in headlines]

        # Stage 1: Sentiment classification (batched)
        sentiments = self.sentiment_classifier.analyze(texts)

        # Stage 2: Entity extraction (rule-based, fast)
        entities = self.entity_extractor.extract_batch(texts)

        # Stage 3: Risk aggregation
        risks = self.risk_aggregator.aggregate_batch(sentiments, entities)

        return [
            AnalysisResult(headline=h, sentiment=s, entities=e, risk=r)
            for h, s, e, r in zip(headlines, sentiments, entities, risks)
        ]

    def analyze_texts(self, texts: List[str]) -> List[AnalysisResult]:
        """Convenience wrapper for raw text input."""
        return self.analyze(load_custom_headlines(texts))

    def analyze_one(self, text: str) -> AnalysisResult:
        """Analyze a single headline."""
        return self.analyze_texts([text])[0]

    def run_on_samples(self) -> List[AnalysisResult]:
        """Run pipeline on bundled sample data."""
        return self.analyze(load_sample_headlines())

    @staticmethod
    def print_report(results: List[AnalysisResult], top_risks: int = 5) -> None:
        """Pretty-print analysis results to stdout."""
        print("\n" + "=" * 80)
        print("FINANCIAL SENTIMENT & RISK ANALYSIS REPORT")
        print("=" * 80)

        # Summary stats
        total = len(results)
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        risk_counts = {"low": 0, "medium": 0, "elevated": 0, "high": 0}
        for r in results:
            sentiment_counts[r.sentiment.label] += 1
            risk_counts[r.risk.risk_level] += 1

        print(f"\nHeadlines analyzed: {total}")
        print(f"Sentiment: {sentiment_counts['positive']} positive | "
              f"{sentiment_counts['neutral']} neutral | "
              f"{sentiment_counts['negative']} negative")
        print(f"Risk distribution: "
              f"{risk_counts['low']} low | {risk_counts['medium']} medium | "
              f"{risk_counts['elevated']} elevated | {risk_counts['high']} high")

        # Top risk signals
        sorted_by_risk = sorted(results, key=lambda r: r.risk.risk_score, reverse=True)
        print(f"\n{'─' * 80}")
        print(f"TOP {top_risks} RISK SIGNALS")
        print(f"{'─' * 80}")
        for i, r in enumerate(sorted_by_risk[:top_risks], 1):
            print(f"\n{i}. [{r.risk.risk_level.upper():8s}] Score: {r.risk.risk_score:.3f}")
            print(f"   {r.headline.text[:78]}")
            print(f"   Sentiment: {r.sentiment.label} ({r.sentiment.confidence:.0%} confidence) | "
                  f"Directional: {r.risk.directional}")
            if r.risk.institutions:
                print(f"   Institutions: {', '.join(r.risk.institutions[:3])}")
            if r.risk.metrics:
                print(f"   Metrics: {', '.join(r.risk.metrics[:4])}")
            print(f"   → {r.risk.recommendation}")

        # Positive signals
        positive = [r for r in results if r.sentiment.label == "positive"]
        if positive:
            print(f"\n{'─' * 80}")
            print(f"POSITIVE MARKET SIGNALS ({len(positive)})")
            print(f"{'─' * 80}")
            for r in positive[:3]:
                print(f"  ✓ {r.headline.text[:75]}")
                print(f"    Confidence: {r.sentiment.confidence:.0%} | {r.risk.recommendation}")

        # Model info
        print(f"\n{'─' * 80}")
        print(f"Model: {results[0].sentiment.model if results else 'n/a'}")
        if results:
            latencies = [r.sentiment.latency_ms for r in results]
            print(f"Avg latency: {sum(latencies)/len(latencies):.1f}ms per headline")
        print("=" * 80 + "\n")
