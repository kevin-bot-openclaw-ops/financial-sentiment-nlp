"""Tests for data loading module."""

import pytest
from src.data_loader import (
    load_sample_headlines,
    load_custom_headlines,
    Headline,
    BUILTIN_HEADLINES,
)


class TestHeadline:
    def test_headline_creation(self):
        h = Headline(text="Test headline", source="Reuters")
        assert h.text == "Test headline"
        assert h.source == "Reuters"
        assert h.label is None

    def test_headline_to_dict(self):
        h = Headline(text="Test", source="FT", date="2024-01-01")
        d = h.to_dict()
        assert d["text"] == "Test"
        assert d["source"] == "FT"
        assert d["date"] == "2024-01-01"
        assert "label" in d

    def test_headline_with_label(self):
        h = Headline(text="Test", label="positive")
        assert h.label == "positive"


class TestLoadSampleHeadlines:
    def test_returns_list(self):
        headlines = load_sample_headlines()
        assert isinstance(headlines, list)

    def test_non_empty(self):
        headlines = load_sample_headlines()
        assert len(headlines) > 0

    def test_all_headline_objects(self):
        headlines = load_sample_headlines()
        for h in headlines:
            assert isinstance(h, Headline)

    def test_all_have_text(self):
        headlines = load_sample_headlines()
        for h in headlines:
            assert h.text
            assert len(h.text) > 10

    def test_minimum_20_headlines(self):
        """Sample dataset should have at least 20 headlines for meaningful stats."""
        headlines = load_sample_headlines()
        assert len(headlines) >= 20


class TestLoadCustomHeadlines:
    def test_wraps_strings(self):
        texts = ["Headline one", "Headline two"]
        headlines = load_custom_headlines(texts)
        assert len(headlines) == 2
        assert headlines[0].text == "Headline one"
        assert headlines[1].text == "Headline two"

    def test_source_is_user_input(self):
        headlines = load_custom_headlines(["Test"])
        assert headlines[0].source == "user_input"

    def test_empty_list(self):
        assert load_custom_headlines([]) == []

    def test_single_item(self):
        headlines = load_custom_headlines(["Single headline"])
        assert len(headlines) == 1


class TestBuiltinHeadlines:
    def test_builtin_coverage(self):
        """Builtin headlines should cover positive, negative, and neutral signals."""
        texts_lower = " ".join(h["text"].lower() for h in BUILTIN_HEADLINES)
        # Check for representative financial terms
        assert any(kw in texts_lower for kw in ["beats", "record", "surge"])  # positive
        assert any(kw in texts_lower for kw in ["warning", "writedown", "default", "collapse"])  # negative
