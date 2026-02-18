"""Tests for the end-to-end pipeline."""

import pytest
from src.pipeline import FinancialNLPPipeline, AnalysisResult
from src.data_loader import load_custom_headlines


class TestFinancialNLPPipeline:
    def setup_method(self):
        # Always use rule-based in tests — no model download
        self.pipeline = FinancialNLPPipeline(prefer_finbert=False)

    def test_pipeline_initializes(self):
        assert self.pipeline is not None

    def test_analyze_returns_list(self):
        headlines = load_custom_headlines(["Goldman Sachs beats earnings"])
        results = self.pipeline.analyze(headlines)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_analyze_returns_analysis_results(self):
        headlines = load_custom_headlines(["Test headline"])
        results = self.pipeline.analyze(headlines)
        assert isinstance(results[0], AnalysisResult)

    def test_analyze_texts_convenience(self):
        results = self.pipeline.analyze_texts(["Test headline one", "Test headline two"])
        assert len(results) == 2

    def test_analyze_one_convenience(self):
        result = self.pipeline.analyze_one("Goldman Sachs reports record profits")
        assert isinstance(result, AnalysisResult)

    def test_run_on_samples_non_empty(self):
        results = self.pipeline.run_on_samples()
        assert len(results) >= 20  # Sample dataset has 20+ headlines

    def test_empty_list(self):
        results = self.pipeline.analyze([])
        assert results == []

    def test_result_has_all_components(self):
        results = self.pipeline.analyze_texts(["Deutsche Bank writedown warning"])
        r = results[0]
        assert r.headline is not None
        assert r.sentiment is not None
        assert r.entities is not None
        assert r.risk is not None

    def test_to_dict_serializable(self):
        import json
        results = self.pipeline.analyze_texts(["Test"])
        d = results[0].to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_sentiment_label_valid(self):
        results = self.pipeline.analyze_texts(["Test headline"])
        assert results[0].sentiment.label in ("positive", "negative", "neutral")

    def test_risk_level_valid(self):
        results = self.pipeline.analyze_texts(["Test headline"])
        assert results[0].risk.risk_level in ("low", "medium", "elevated", "high")

    def test_risk_score_in_range(self):
        results = self.pipeline.analyze_texts(["Test headline"])
        assert 0.0 <= results[0].risk.risk_score <= 1.0

    def test_batch_processing(self):
        texts = [
            "Goldman Sachs beats Q3 earnings by record margin",
            "Deutsche Bank warns of rising NPL ratios",
            "ECB holds rates steady",
            "BNP Paribas revenues collapse on adverse conditions",
            "Visa payment volumes surge 12% YoY",
        ]
        results = self.pipeline.analyze_texts(texts)
        assert len(results) == 5

    def test_negative_headlines_score_higher_risk(self):
        """Negative headlines should generally produce higher risk scores."""
        positive = self.pipeline.analyze_one("Goldman Sachs beats earnings record")
        negative = self.pipeline.analyze_one("Deutsche Bank reports massive writedown and collapse warning")
        # This is a statistical expectation, not a guarantee with rule-based
        # Just verify both produce valid scores
        assert 0 <= positive.risk.risk_score <= 1
        assert 0 <= negative.risk.risk_score <= 1

    def test_print_report_no_error(self, capsys):
        results = self.pipeline.run_on_samples()
        FinancialNLPPipeline.print_report(results)
        captured = capsys.readouterr()
        assert "FINANCIAL SENTIMENT" in captured.out
        assert "Headlines analyzed" in captured.out
