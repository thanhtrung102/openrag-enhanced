"""
OpenRAG Hallucination Detection API Router

Provides REST endpoints for hallucination detection in RAG responses.
Integrates with the existing OpenRAG API structure.

Endpoints:
    POST /api/v1/hallucination/detect - Detect hallucinations in a response
    POST /api/v1/hallucination/batch - Batch detection
    GET /api/v1/hallucination/health - Health check
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from openrag.components.hallucination import HallucinationDetector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hallucination", tags=["hallucination"])

# Initialize detector (lazy loading)
_detector = None


def get_detector() -> HallucinationDetector:
    """Get or create hallucination detector instance."""
    global _detector
    if _detector is None:
        _detector = HallucinationDetector(lazy_load=True)
    return _detector


# ============================================
# Request/Response Models
# ============================================

class DetectionRequest(BaseModel):
    """Request for hallucination detection."""
    response: str = Field(..., description="The generated response to check")
    sources: List[str] = Field(..., description="Source documents used for generation")
    methods: Optional[List[str]] = Field(
        None,
        description="Detection methods to use (default: ['nli', 'alignment'])"
    )
    threshold: Optional[float] = Field(
        0.5,
        description="Hallucination threshold (0-1)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Python was created by Guido van Rossum in 1991.",
                "sources": [
                    "Python is a programming language created by Guido van Rossum.",
                    "Python was first released in 1991."
                ],
                "methods": ["nli", "alignment"],
                "threshold": 0.5
            }
        }


class DetectionResponse(BaseModel):
    """Response from hallucination detection."""
    ensemble_score: float = Field(..., description="Overall hallucination score (0-1)")
    is_hallucinated: bool = Field(..., description="Whether response is flagged as hallucinated")
    confidence: float = Field(..., description="Confidence in the detection")
    nli_score: Optional[float] = Field(None, description="NLI method score")
    alignment_score: Optional[float] = Field(None, description="Alignment method score")
    flagged_claims: List[str] = Field(default_factory=list, description="Specific claims flagged")
    methods_used: List[str] = Field(default_factory=list, description="Methods that were used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchDetectionRequest(BaseModel):
    """Request for batch hallucination detection."""
    items: List[DetectionRequest] = Field(..., description="List of items to check")
    parallel: bool = Field(True, description="Run detections in parallel")


class BatchDetectionResponse(BaseModel):
    """Response from batch detection."""
    total: int
    results: List[DetectionResponse]
    summary: dict


# ============================================
# Endpoints
# ============================================

@router.get("/health")
async def health_check():
    """Health check for hallucination detection service."""
    return {
        "status": "healthy",
        "service": "hallucination-detection",
        "detector_loaded": _detector is not None,
    }


@router.post("/detect", response_model=DetectionResponse)
async def detect_hallucination(request: DetectionRequest):
    """
    Detect hallucinations in a RAG response.

    Uses multiple methods (NLI, alignment) to identify potentially
    fabricated content not supported by the source documents.

    Returns:
        DetectionResponse with scores and flagged claims
    """
    try:
        detector = get_detector()

        # Override threshold if provided
        if request.threshold:
            detector.threshold = request.threshold

        result = await detector.detect(
            response=request.response,
            sources=request.sources,
            methods=request.methods,
        )

        return DetectionResponse(
            ensemble_score=result.ensemble_score,
            is_hallucinated=result.is_hallucinated,
            confidence=result.confidence,
            nli_score=result.nli_score,
            alignment_score=result.alignment_score,
            flagged_claims=result.flagged_claims,
            methods_used=result.methods_used,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(request: BatchDetectionRequest):
    """
    Batch hallucination detection.

    Efficiently processes multiple responses in parallel or sequentially.
    """
    try:
        detector = get_detector()

        items = [
            {"response": item.response, "sources": item.sources}
            for item in request.items
        ]

        results = await detector.detect_batch(
            items=items,
            parallel=request.parallel,
        )

        responses = [
            DetectionResponse(
                ensemble_score=r.ensemble_score,
                is_hallucinated=r.is_hallucinated,
                confidence=r.confidence,
                nli_score=r.nli_score,
                alignment_score=r.alignment_score,
                flagged_claims=r.flagged_claims,
                methods_used=r.methods_used,
                processing_time_ms=r.processing_time_ms,
            )
            for r in results
        ]

        # Calculate summary
        hallucinated_count = sum(1 for r in results if r.is_hallucinated)
        avg_score = sum(r.ensemble_score for r in results) / len(results) if results else 0

        return BatchDetectionResponse(
            total=len(results),
            results=responses,
            summary={
                "hallucination_rate": hallucinated_count / len(results) if results else 0,
                "avg_score": avg_score,
                "hallucinated_count": hallucinated_count,
            }
        )

    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Integration helper
# ============================================

def add_hallucination_routes(app):
    """
    Add hallucination detection routes to an existing FastAPI app.

    Usage in OpenRAG:
        from openrag.routers.hallucination import add_hallucination_routes
        add_hallucination_routes(app)
    """
    app.include_router(router)
    logger.info("Hallucination detection routes added")
