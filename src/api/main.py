"""
OpenRAG Enhanced - FastAPI Application

Main API entry point with evaluation endpoints.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models
# ============================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


class EvaluationRequest(BaseModel):
    """Request for evaluating a RAG response."""
    query: str = Field(..., description="The user query")
    context: List[str] = Field(..., description="Retrieved context documents")
    response: str = Field(..., description="Generated response to evaluate")
    reference_answer: Optional[str] = Field(None, description="Ground truth answer")
    evaluate_hallucination: bool = Field(True, description="Run hallucination detection")
    evaluate_quality: bool = Field(True, description="Run LLM-as-judge evaluation")


class EvaluationResponse(BaseModel):
    """Response from evaluation."""
    evaluation_id: str
    hallucination_result: Optional[Dict[str, Any]] = None
    quality_result: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation."""
    items: List[EvaluationRequest]
    parallel: bool = Field(True, description="Run evaluations in parallel")


class BatchEvaluationResponse(BaseModel):
    """Response from batch evaluation."""
    total_evaluated: int
    results: List[EvaluationResponse]
    summary: Dict[str, Any]


class HallucinationCheckRequest(BaseModel):
    """Request for hallucination detection only."""
    response: str = Field(..., description="Response to check")
    sources: List[str] = Field(..., description="Source documents")
    methods: Optional[List[str]] = Field(None, description="Detection methods to use")


class RetrievalEvalRequest(BaseModel):
    """Request for retrieval evaluation."""
    results: List[Dict[str, Any]] = Field(..., description="Retrieval results")
    ground_truth: List[Dict[str, Any]] = Field(..., description="Ground truth data")
    k_values: List[int] = Field([1, 3, 5, 10, 20], description="K values for metrics")


# ============================================
# Application Setup
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting OpenRAG Enhanced API...")

    # Initialize components here
    # e.g., load models, connect to databases

    yield

    # Cleanup
    logger.info("Shutting down OpenRAG Enhanced API...")


app = FastAPI(
    title="OpenRAG Enhanced API",
    description="RAG evaluation and hallucination detection API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Add checks for dependencies (DB, models, etc.)
    return {"status": "ready"}


# ============================================
# Evaluation Endpoints
# ============================================

@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluationRequest):
    """
    Evaluate a single RAG response.

    Runs hallucination detection and/or LLM-as-judge quality evaluation.
    """
    import time
    start_time = time.time()

    evaluation_id = str(uuid.uuid4())
    results = {}

    try:
        if request.evaluate_hallucination:
            from src.evaluation.hallucination_detector import HallucinationDetector

            detector = HallucinationDetector(lazy_load=True)
            hall_result = await detector.detect(
                response=request.response,
                sources=request.context,
            )
            results["hallucination_result"] = hall_result.to_dict()

        if request.evaluate_quality:
            from src.evaluation.llm_judge import LLMJudgeEvaluator

            judge = LLMJudgeEvaluator(
                judge_model=os.getenv("LLM_JUDGE_MODEL", "gpt-4")
            )
            quality_result = await judge.evaluate(
                question=request.query,
                context=request.context,
                response=request.response,
                reference_answer=request.reference_answer,
            )
            results["quality_result"] = quality_result.to_dict()

        processing_time = (time.time() - start_time) * 1000

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            hallucination_result=results.get("hallucination_result"),
            quality_result=results.get("quality_result"),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(request: BatchEvaluationRequest):
    """
    Evaluate multiple RAG responses.
    """
    import asyncio

    results = []

    if request.parallel:
        # Run in parallel
        tasks = [evaluate_response(item) for item in request.items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Batch item failed: {r}")
            else:
                valid_results.append(r)
        results = valid_results
    else:
        # Run sequentially
        for item in request.items:
            try:
                result = await evaluate_response(item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Batch item failed: {e}")

    # Compute summary
    summary = {}
    if results:
        hall_scores = [
            r.hallucination_result.get("ensemble_score", 0)
            for r in results
            if r.hallucination_result
        ]
        quality_scores = [
            r.quality_result.get("overall_score", 0)
            for r in results
            if r.quality_result
        ]

        if hall_scores:
            summary["avg_hallucination_score"] = sum(hall_scores) / len(hall_scores)
            summary["hallucination_rate"] = sum(
                1 for s in hall_scores if s > 0.5
            ) / len(hall_scores)

        if quality_scores:
            summary["avg_quality_score"] = sum(quality_scores) / len(quality_scores)

    return BatchEvaluationResponse(
        total_evaluated=len(results),
        results=results,
        summary=summary,
    )


@app.post("/api/v1/detect/hallucination")
async def detect_hallucination(request: HallucinationCheckRequest):
    """
    Run hallucination detection only.
    """
    from src.evaluation.hallucination_detector import HallucinationDetector

    try:
        detector = HallucinationDetector(lazy_load=True)
        result = await detector.detect(
            response=request.response,
            sources=request.sources,
            methods=request.methods,
        )
        return result.to_dict()

    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate/retrieval")
async def evaluate_retrieval(request: RetrievalEvalRequest):
    """
    Evaluate retrieval performance.
    """
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    try:
        evaluator = RetrievalEvaluator(k_values=request.k_values)
        metrics = evaluator.evaluate_results(
            results=request.results,
            ground_truth=request.ground_truth,
        )
        return metrics.to_dict()

    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Metrics Endpoints
# ============================================

@app.get("/api/v1/metrics/summary")
async def get_metrics_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Get aggregated metrics summary.
    """
    # In production, this would query PostgreSQL
    return {
        "period": {
            "start": start_date or "2024-01-01",
            "end": end_date or datetime.now().isoformat(),
        },
        "model": model or "all",
        "metrics": {
            "avg_faithfulness": 4.2,
            "avg_relevance": 4.1,
            "hallucination_rate": 0.12,
            "hit_rate_5": 0.85,
            "mrr": 0.68,
            "total_evaluations": 1234,
        },
    }


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
