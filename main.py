#!/usr/bin/env python3
"""
Financial Sentiment NLP Pipeline — CLI entry point.

Usage:
    python main.py                    # Run on bundled sample data
    python main.py --quick            # Run with 5 headlines (fast demo)
    python main.py --rss              # Fetch live headlines from RSS feeds
    python main.py --text "headline"  # Analyze a single headline

API server:
    uvicorn src.api:app --reload --port 8080
    curl http://localhost:8080/sample
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import FinancialNLPPipeline
from src.data_loader import load_sample_headlines, load_rss_headlines, load_custom_headlines, BUILTIN_HEADLINES


def main():
    parser = argparse.ArgumentParser(
        description="Financial Sentiment NLP Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--quick", action="store_true", help="Run on first 5 sample headlines")
    parser.add_argument("--rss", action="store_true", help="Fetch live headlines from RSS feeds")
    parser.add_argument("--text", type=str, help="Analyze a single custom headline")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report")
    parser.add_argument("--no-finbert", action="store_true", help="Use rule-based fallback (no model download)")
    args = parser.parse_args()

    print("Financial NLP Pipeline — Initializing...")
    print("(First run downloads FinBERT model ~440MB; subsequent runs use cache)")
    print()

    pipeline = FinancialNLPPipeline(prefer_finbert=not args.no_finbert)

    if args.text:
        print(f"Analyzing: {args.text!r}\n")
        headlines = load_custom_headlines([args.text])
    elif args.rss:
        print("Fetching live headlines from RSS feeds...")
        headlines = load_rss_headlines()
    elif args.quick:
        print("Quick mode: 5 sample headlines")
        headlines = load_sample_headlines()[:5]
    else:
        headlines = load_sample_headlines()

    results = pipeline.analyze(headlines)

    if args.json:
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
    else:
        pipeline.print_report(results)

    # Exit summary
    high_risk = sum(1 for r in results if r.risk.risk_level in ("elevated", "high"))
    if high_risk > 0:
        print(f"⚠️  {high_risk} elevated/high risk signal(s) detected")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
