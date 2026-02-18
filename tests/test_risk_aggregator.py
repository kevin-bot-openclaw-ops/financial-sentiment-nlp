"""Tests for risk aggregation module."""

import pytest
from src.sentiment import SentimentResult
from src.entity_extractor import ExtractionResult
from src.risk_aggregator import RiskAggregator, RiskSignal, _risk_level


def make_sentiment(label="negative", confidence=0.9, text="Test"):
    return SentimentResult(
        text=text,
        label=label,
        confidence=confidence,
        scores={label: confidence, "neutral": 0.05, "other": 0.05},
        model="test",
        latency_ms=5.0,
    )


def make_entities(institutions=None, metrics=None, directional="neutral"):
    return ExtractionResult(
        text="Test",
        institutions=institutions or [],
        metrics=metrics or [],
        directional=directional,
    )


class TestRiskLevel:
    def test_low(self):
        assert _risk_level(0.1) == "low"
        assert _risk_level(0.29) == "low"

    def test_medium(self):
        assert _risk_level(0.3) == "medium"
        assert _risk_level(0.59) == "medium"

    def test_elevated(self):
        assert _risk_level(0.6) == "elevated"
        assert _risk_level(0.79) == "elevated"

    def test_high(self):
        assert _risk_level(0.8) == "high"
        assert _risk_level(1.0) == "high"


class TestRiskAggregator:
    def setup_method(self):
        self.agg = RiskAggregator()

    def test_returns_risk_signal(self):
        s = make_sentiment()
        e = make_entities()
        result = self.agg.aggregate(s, e)
        assert isinstance(result, RiskSignal)

    def test_risk_score_between_0_and_1(self):
        s = make_sentiment(label="negative", confidence=0.95)
        e = make_entities(institutions=["Goldman Sachs", "ECB"], metrics=["profit", "npl"])
        result = self.agg.aggregate(s, e)
        assert 0.0 <= result.risk_score <= 1.0

    def test_negative_high_confidence_yields_high_risk(self):
        s = make_sentiment(label="negative", confidence=0.95)
        e = make_entities(institutions=["Deutsche Bank"], metrics=["non_performing_loans"], directional="bearish")
        result = self.agg.aggregate(s, e)
        assert result.risk_score > 0.5

    def test_positive_high_confidence_yields_low_risk(self):
        s = make_sentiment(label="positive", confidence=0.95)
        e = make_entities(institutions=["Goldman Sachs"], directional="bullish")
        result = self.agg.aggregate(s, e)
        assert result.risk_score < 0.5

    def test_neutral_sentiment_gives_medium_risk(self):
        s = make_sentiment(label="neutral", confidence=0.8)
        e = make_entities()
        result = self.agg.aggregate(s, e)
        assert result.risk_level in ("low", "medium", "elevated")

    def test_institutions_increase_risk_score(self):
        s = make_sentiment(label="negative", confidence=0.7)
        no_entity = make_entities()
        with_entity = make_entities(institutions=["Goldman Sachs", "JPMorgan", "Deutsche Bank"])
        r1 = self.agg.aggregate(s, no_entity)
        r2 = self.agg.aggregate(s, with_entity)
        assert r2.risk_score > r1.risk_score

    def test_directional_alignment_boosts_risk(self):
        s = make_sentiment(label="negative", confidence=0.7)
        no_align = make_entities(directional="neutral")
        aligned = make_entities(directional="bearish")
        r1 = self.agg.aggregate(s, no_align)
        r2 = self.agg.aggregate(s, aligned)
        assert r2.risk_score >= r1.risk_score

    def test_to_dict_has_all_keys(self):
        s = make_sentiment()
        e = make_entities()
        result = self.agg.aggregate(s, e)
        d = result.to_dict()
        required_keys = {"text", "risk_score", "risk_level", "sentiment", "directional",
                         "institutions", "metrics", "score_components", "recommendation"}
        assert required_keys.issubset(set(d.keys()))

    def test_score_components_traceable(self):
        s = make_sentiment()
        e = make_entities()
        result = self.agg.aggregate(s, e)
        components = result.risk.score_components if hasattr(result, 'risk') else result.score_components
        assert "sentiment_direction" in result.score_components
        assert "entity_multiplier" in result.score_components
        assert "alignment_bonus" in result.score_components

    def test_recommendation_not_empty(self):
        s = make_sentiment()
        e = make_entities()
        result = self.agg.aggregate(s, e)
        assert len(result.recommendation) > 0

    def test_batch_aggregate(self):
        sentiments = [make_sentiment("positive"), make_sentiment("negative"), make_sentiment("neutral")]
        entities = [make_entities(), make_entities(institutions=["ECB"]), make_entities()]
        results = self.agg.aggregate_batch(sentiments, entities)
        assert len(results) == 3

    def test_positive_gets_opportunity_recommendation(self):
        s = make_sentiment(label="positive", confidence=0.9)
        e = make_entities()
        result = self.agg.aggregate(s, e)
        assert "Opportunity" in result.recommendation or "Monitor" in result.recommendation
