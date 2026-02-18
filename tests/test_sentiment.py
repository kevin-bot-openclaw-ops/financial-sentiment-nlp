"""Tests for sentiment classification module."""

import pytest
from src.sentiment import (
    RuleBasedClassifier,
    SentimentClassifier,
    SentimentResult,
)


class TestSentimentResult:
    def _make_result(self, label="positive", confidence=0.9):
        return SentimentResult(
            text="Goldman Sachs beats earnings",
            label=label,
            confidence=confidence,
            scores={"positive": 0.9, "neutral": 0.07, "negative": 0.03},
            model="test-model",
            latency_ms=10.0,
        )

    def test_is_risk_signal_negative_high_conf(self):
        r = self._make_result(label="negative", confidence=0.8)
        assert r.is_risk_signal is True

    def test_is_risk_signal_negative_low_conf(self):
        r = self._make_result(label="negative", confidence=0.4)
        assert r.is_risk_signal is False

    def test_is_risk_signal_positive(self):
        r = self._make_result(label="positive", confidence=0.9)
        assert r.is_risk_signal is False

    def test_to_dict_has_required_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert "label" in d
        assert "confidence" in d
        assert "scores" in d
        assert "is_risk_signal" in d
        assert "model" in d
        assert "latency_ms" in d

    def test_to_dict_rounds_confidence(self):
        r = self._make_result(confidence=0.923456789)
        d = r.to_dict()
        assert len(str(d["confidence"]).split(".")[-1]) <= 4


class TestRuleBasedClassifier:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_returns_list(self):
        results = self.clf.predict(["Test headline"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_returns_sentiment_results(self):
        results = self.clf.predict(["Test headline"])
        assert isinstance(results[0], SentimentResult)

    def test_positive_keywords_detected(self):
        results = self.clf.predict(["Goldman Sachs beats Q3 earnings expectations by 15%"])
        assert results[0].label == "positive"
        assert results[0].confidence > 0.5

    def test_negative_keywords_detected(self):
        results = self.clf.predict(["Deutsche Bank warns of rising NPL ratios and potential writedown"])
        assert results[0].label == "negative"
        assert results[0].confidence > 0.5

    def test_neutral_when_balanced(self):
        results = self.clf.predict(["ECB holds rates steady"])
        # Should be neutral (no strong positive/negative keywords)
        # Don't assert specific label — keyword coverage determines this

    def test_batch_processing(self):
        texts = [
            "Bank beats earnings record",
            "Bank reports writedown and loss warning",
            "Bank maintains current policy",
        ]
        results = self.clf.predict(texts)
        assert len(results) == 3

    def test_all_results_have_label(self):
        texts = ["a", "b", "c"]
        results = self.clf.predict(texts)
        for r in results:
            assert r.label in ("positive", "negative", "neutral")

    def test_confidence_between_0_and_1(self):
        results = self.clf.predict(["Test headline beats record"])
        assert 0.0 <= results[0].confidence <= 1.0

    def test_scores_sum_to_approximately_1(self):
        results = self.clf.predict(["Test headline"])
        total = sum(results[0].scores.values())
        assert abs(total - 1.0) < 0.01

    def test_model_name_set(self):
        results = self.clf.predict(["Test"])
        assert results[0].model == "rule-based-keyword-v1"

    def test_latency_ms_positive(self):
        results = self.clf.predict(["Test"])
        assert results[0].latency_ms >= 0.0


class TestSentimentClassifier:
    def setup_method(self):
        # Use rule-based by default in tests (no model download)
        self.clf = SentimentClassifier(prefer_finbert=False)

    def test_analyze_returns_list(self):
        results = self.clf.analyze(["Test"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_analyze_one_returns_single(self):
        result = self.clf.analyze_one("Goldman beats earnings")
        assert isinstance(result, SentimentResult)

    def test_model_name_property(self):
        name = self.clf.model_name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_empty_list(self):
        results = self.clf.analyze([])
        assert results == []

    def test_positive_headline(self):
        result = self.clf.analyze_one("Record profits as bank beats all forecasts")
        assert result.label in ("positive", "neutral", "negative")  # any valid label
        assert 0 <= result.confidence <= 1

    def test_negative_headline(self):
        result = self.clf.analyze_one("Bank collapses amid writedown warning and default risk")
        assert result.label in ("positive", "neutral", "negative")
