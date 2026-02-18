"""
FastAPI REST service for Financial NLP.

Endpoints:
  POST /analyze         Analyze 1–50 headlines
  POST /analyze/batch   Alias for /analyze
  GET  /health          Health check + model info
  GET  /sample          Run on bundled sample data

Run with:
  uvicorn src.api:app --reload --port 8080

Production considerations (interview talking points):
  - Add auth: OAuth2 client credentials for B2B banking APIs
  - Rate limiting: 100 req/min per client (prevent abuse)
  - Async inference: queue heavy FinBERT calls, return job_id for polling
  - Caching: Redis TTL=300s for duplicate headlines (high-volume news feeds)
  - Observability: Prometheus metrics (request rate, latency p50/p95, model errors)
"""

from __future__ import annotations

from typing import List, Optional
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.pipeline import FinancialNLPPipeline


app = FastAPI(
    title="Financial NLP API",
    description="Sentiment analysis and risk scoring for financial news headlines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Singleton pipeline — model loaded on first request (lazy)
_pipeline: Optional[FinancialNLPPipeline] = None


def get_pipeline() -> FinancialNLPPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FinancialNLPPipeline(prefer_finbert=True)
    return _pipeline


# ─── Request / Response Models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50, description="Financial headlines to analyze")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(text) > 1000:
                raise ValueError(f"texts[{i}] exceeds 1000 characters")
        return [t.strip() for t in v]


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    scores: dict
    is_risk_signal: bool
    model: str
    latency_ms: float


class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    sentiment_label: str
    sentiment_confidence: float
    directional: str
    institutions: List[str]
    metrics: List[str]
    score_components: dict
    recommendation: str


class HeadlineAnalysis(BaseModel):
    text: str
    sentiment: SentimentResponse
    entities: dict
    risk: RiskResponse


class AnalyzeResponse(BaseModel):
    count: int
    results: List[HeadlineAnalysis]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Health check. Returns model info without loading the model."""
    return HealthResponse(
        status="ok",
        model="yiyanghkust/finbert-tone (lazy-loaded)",
        version="1.0.0",
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Analyze financial news headlines for sentiment and risk.

    Returns per-headline sentiment (label, confidence, probabilities),
    entity extraction (institutions, metrics), and composite risk score.
    """
    t0 = time.perf_counter()
    pipeline = get_pipeline()

    try:
        results = pipeline.analyze_texts(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    processing_ms = (time.perf_counter() - t0) * 1000

    return AnalyzeResponse(
        count=len(results),
        processing_time_ms=round(processing_ms, 1),
        results=[
            HeadlineAnalysis(
                text=r.headline.text,
                sentiment=SentimentResponse(**r.sentiment.to_dict()),
                entities=r.entities.to_dict(),
                risk=RiskResponse(**r.risk.to_dict()),
            )
            for r in results
        ],
    )


@app.post("/analyze/batch", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze_batch(request: AnalyzeRequest):
    """Alias for /analyze — batch endpoint for clarity in client code."""
    return await analyze(request)


@app.get("/sample", response_model=AnalyzeResponse, tags=["demo"])
async def sample():
    """Run analysis on bundled sample financial headlines (no input required)."""
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    results = pipeline.run_on_samples()
    processing_ms = (time.perf_counter() - t0) * 1000

    return AnalyzeResponse(
        count=len(results),
        processing_time_ms=round(processing_ms, 1),
        results=[
            HeadlineAnalysis(
                text=r.headline.text,
                sentiment=SentimentResponse(**r.sentiment.to_dict()),
                entities=r.entities.to_dict(),
                risk=RiskResponse(**r.risk.to_dict()),
            )
            for r in results
        ],
    )
