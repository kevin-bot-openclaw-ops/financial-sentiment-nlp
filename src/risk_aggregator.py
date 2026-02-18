"""
Risk Signal Aggregator.

Combines sentiment + entity extraction into a composite risk score.

Risk score formula:
  base_score = sentiment_confidence × sentiment_direction
  entity_multiplier = 1 + (0.1 × known_institutions) + (0.05 × financial_metrics)
  directional_alignment_bonus = +0.1 if sentiment and entity directional match
  final_score = base_score × entity_multiplier + alignment_bonus
  clamped to [0.0, 1.0]

Interpretation:
  0.0–0.3  → Low risk / positive market signal
  0.3–0.6  → Neutral / monitoring required
  0.6–0.8  → Elevated risk / recommend analyst review
  0.8–1.0  → High risk / escalate immediately

Interview talking point: Risk scoring systems in banking must be auditable.
Every score component must be traceable to source data (explainability).
Black-box scores fail regulatory review (SR 11-7, EBA ML guidelines).
This decomposed approach supports model risk management requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.sentiment import SentimentResult
from src.entity_extractor import ExtractionResult


@dataclass
class RiskSignal:
    """Composite risk signal for a single financial headline."""
    text: str
    risk_score: float           # 0.0–1.0
    risk_level: str             # low | medium | elevated | high
    sentiment_label: str
    sentiment_confidence: float
    directional: str
    institutions: List[str]
    metrics: List[str]
    score_components: dict      # breakdown for auditability
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "sentiment": {
                "label": self.sentiment_label,
                "confidence": round(self.sentiment_confidence, 4),
            },
            "directional": self.directional,
            "institutions": self.institutions,
            "metrics": self.metrics,
            "score_components": self.score_components,
            "recommendation": self.recommendation,
        }


def _risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.8:
        return "elevated"
    else:
        return "high"


def _recommendation(risk_level: str, sentiment_label: str) -> str:
    recs = {
        "low": "Monitor — positive signal, continue standard monitoring",
        "medium": "Watch — neutral or mixed signals, check next 24h",
        "elevated": "Review — negative signal with entity context, analyst attention required",
        "high": "Escalate — high-confidence negative signal involving known institution",
    }
    if sentiment_label == "positive" and risk_level in ("low", "medium"):
        return "Opportunity — positive signal, consider for investment committee briefing"
    return recs.get(risk_level, "Monitor")


class RiskAggregator:
    """Combines sentiment and entity results into a composite risk score."""

    def aggregate(
        self,
        sentiment: SentimentResult,
        entities: ExtractionResult,
    ) -> RiskSignal:
        # 1. Sentiment direction score (negative = high risk)
        if sentiment.label == "negative":
            sentiment_direction = sentiment.confidence
        elif sentiment.label == "positive":
            sentiment_direction = 1.0 - sentiment.confidence  # low risk
        else:
            sentiment_direction = 0.4  # neutral → moderate baseline

        # 2. Entity multiplier — named institutions increase signal importance
        inst_factor = min(len(entities.institutions) * 0.15, 0.45)
        metric_factor = min(len(entities.metrics) * 0.05, 0.20)
        entity_multiplier = 1.0 + inst_factor + metric_factor

        # 3. Directional alignment — sentiment and rule-based agree → higher confidence
        alignment_bonus = 0.0
        if sentiment.label == "negative" and entities.directional == "bearish":
            alignment_bonus = 0.10
        elif sentiment.label == "positive" and entities.directional == "bullish":
            alignment_bonus = -0.05  # reduces risk score

        # 4. Final score
        raw_score = (sentiment_direction * entity_multiplier) + alignment_bonus
        final_score = max(0.0, min(1.0, raw_score))

        score_components = {
            "sentiment_direction": round(sentiment_direction, 4),
            "entity_multiplier": round(entity_multiplier, 4),
            "alignment_bonus": round(alignment_bonus, 4),
            "raw_score": round(raw_score, 4),
        }

        level = _risk_level(final_score)
        return RiskSignal(
            text=sentiment.text,
            risk_score=final_score,
            risk_level=level,
            sentiment_label=sentiment.label,
            sentiment_confidence=sentiment.confidence,
            directional=entities.directional,
            institutions=entities.institutions,
            metrics=entities.metrics,
            score_components=score_components,
            recommendation=_recommendation(level, sentiment.label),
        )

    def aggregate_batch(
        self,
        sentiments: List[SentimentResult],
        entities: List[ExtractionResult],
    ) -> List[RiskSignal]:
        return [self.aggregate(s, e) for s, e in zip(sentiments, entities)]
