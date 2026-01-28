"""
OpenRAG Hallucination Detector

Main interface for hallucination detection in RAG responses.
Combines multiple detection methods for robust results.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from .nli_checker import NLIChecker
from .alignment_checker import AlignmentChecker

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Complete hallucination detection result."""
    # Overall scores
    ensemble_score: float
    is_hallucinated: bool
    confidence: float

    # Per-method scores
    nli_score: Optional[float] = None
    alignment_score: Optional[float] = None

    # Detailed results
    flagged_claims: List[str] = field(default_factory=list)

    # Metadata
    methods_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "ensemble_score": self.ensemble_score,
            "is_hallucinated": self.is_hallucinated,
            "confidence": self.confidence,
            "nli_score": self.nli_score,
            "alignment_score": self.alignment_score,
            "flagged_claims": self.flagged_claims,
            "methods_used": self.methods_used,
            "processing_time_ms": self.processing_time_ms,
        }


class HallucinationDetector:
    """
    Multi-method hallucination detector for OpenRAG.

    Combines NLI-based entailment checking and semantic alignment
    to detect fabricated content in RAG responses.

    Example:
        detector = HallucinationDetector()
        result = await detector.detect(
            response="Python was created by Guido in 1991.",
            sources=["Python is a programming language created by Guido van Rossum..."]
        )

        if result.is_hallucinated:
            print(f"Warning: Potential hallucination detected!")
            print(f"Flagged: {result.flagged_claims}")

    Integration with OpenRAG pipeline:
        # In openrag/components/pipeline.py
        from openrag.components.hallucination import HallucinationDetector

        detector = HallucinationDetector()

        async def generate_with_detection(query, context):
            response = await llm.generate(query, context)
            detection = await detector.detect(response, context)

            return {
                "response": response,
                "hallucination_score": detection.ensemble_score,
                "flagged_claims": detection.flagged_claims,
            }
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        embedding_model: str = "intfloat/multilingual-e5-large-instruct",
        threshold: float = 0.5,
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        """
        Initialize the hallucination detector.

        Args:
            nli_model: Model for NLI-based detection
            embedding_model: Model for semantic alignment
            threshold: Score above which response is flagged as hallucinated
            device: Device for model inference ('cuda' or 'cpu')
            lazy_load: If True, models are loaded on first use
        """
        self.threshold = threshold
        self.device = device

        # Initialize checkers
        self.nli_checker = NLIChecker(
            model_name=nli_model,
            device=device,
            lazy_load=lazy_load,
        )
        self.alignment_checker = AlignmentChecker(
            model_name=embedding_model,
            device=device,
            lazy_load=lazy_load,
        )

    async def detect(
        self,
        response: str,
        sources: List[str],
        methods: Optional[List[str]] = None,
    ) -> DetectionResult:
        """
        Detect hallucinations in a RAG response.

        Args:
            response: The generated response to check
            sources: List of source documents used for generation
            methods: Detection methods to use (default: ["nli", "alignment"])

        Returns:
            DetectionResult with scores and flagged claims
        """
        start_time = time.time()

        if methods is None:
            methods = ["nli", "alignment"]

        # Extract claims from response
        claims = self._extract_claims(response)

        results = {}

        # Run detection methods
        if "nli" in methods and claims:
            results["nli"] = self.nli_checker.check(claims, sources)

        if "alignment" in methods:
            results["alignment"] = self.alignment_checker.check(
                response, sources, sentences=claims
            )

        # Compute ensemble score
        scores = [r.get("score", 0) for r in results.values()]
        ensemble_score = float(np.mean(scores)) if scores else 0.0

        # Collect flagged claims
        flagged_claims = []
        for method_result in results.values():
            flagged_claims.extend(method_result.get("flagged_claims", []))
        flagged_claims = list(set(flagged_claims))  # Deduplicate

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            ensemble_score=ensemble_score,
            is_hallucinated=ensemble_score > self.threshold,
            confidence=1.0 - abs(ensemble_score - 0.5) * 2,
            nli_score=results.get("nli", {}).get("score"),
            alignment_score=results.get("alignment", {}).get("score"),
            flagged_claims=flagged_claims,
            methods_used=methods,
            processing_time_ms=processing_time,
        )

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract individual claims/sentences from text.

        Filters out questions and very short sentences.
        """
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            sentences = text.split(". ")

        claims = []
        for sent in sentences:
            sent = sent.strip()

            # Filter criteria
            if len(sent) < 10:  # Too short
                continue
            if sent.endswith("?"):  # Questions
                continue
            if len(sent.split()) < 4:  # Too few words
                continue

            claims.append(sent)

        return claims

    async def detect_batch(
        self,
        items: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> List[DetectionResult]:
        """
        Detect hallucinations in multiple responses.

        Args:
            items: List of dicts with 'response' and 'sources'
            parallel: Whether to run in parallel

        Returns:
            List of DetectionResult
        """
        if parallel:
            tasks = [
                self.detect(item["response"], item["sources"])
                for item in items
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for item in items:
                result = await self.detect(item["response"], item["sources"])
                results.append(result)
            return results
