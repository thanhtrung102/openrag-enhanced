"""
NLI-based Hallucination Checker

Uses Natural Language Inference to check if response claims
are entailed by the source documents.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NLIResult:
    """Result from NLI checking."""
    score: float  # 0 = supported, 1 = hallucinated
    label: str  # 'entailment', 'contradiction', 'neutral'
    confidence: float
    claim: str
    evidence: Optional[str] = None


class NLIChecker:
    """
    NLI-based hallucination checker.

    Uses a pre-trained NLI model to check if each claim in the response
    is entailed by the source documents.

    Example:
        checker = NLIChecker()
        results = checker.check(
            claims=["Python was created in 1991."],
            sources=["Python was created by Guido van Rossum in 1991."]
        )
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        """
        Initialize NLI checker.

        Args:
            model_name: HuggingFace model for NLI
            device: Device for inference ('cuda' or 'cpu')
            lazy_load: If True, load model on first use
        """
        self.model_name = model_name
        self.device = device
        self._model = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the NLI model."""
        try:
            from transformers import pipeline

            logger.info(f"Loading NLI model: {self.model_name}")
            self._model = pipeline(
                "text-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("NLI model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def check(
        self,
        claims: List[str],
        sources: List[str],
        max_source_length: int = 1000,
    ) -> Dict[str, Any]:
        """
        Check claims against sources using NLI.

        Args:
            claims: List of claims extracted from response
            sources: List of source documents
            max_source_length: Maximum source text length

        Returns:
            Dictionary with score, flagged claims, and detailed results
        """
        if not claims:
            return {
                "score": 0.0,
                "flagged_claims": [],
                "results": [],
                "method": "nli",
            }

        # Combine sources
        source_text = " ".join(sources)
        if len(source_text) > max_source_length:
            source_text = source_text[:max_source_length] + "..."

        results = []
        flagged_claims = []
        scores = []

        for claim in claims:
            try:
                result = self._check_single_claim(claim, source_text)
                results.append(result)
                scores.append(result.score)

                if result.label in ["contradiction", "neutral"] and result.confidence > 0.7:
                    flagged_claims.append(claim)

            except Exception as e:
                logger.warning(f"NLI check failed for claim: {e}")
                continue

        return {
            "score": float(np.mean(scores)) if scores else 0.0,
            "flagged_claims": flagged_claims,
            "results": [self._result_to_dict(r) for r in results],
            "method": "nli",
        }

    def _check_single_claim(self, claim: str, source_text: str) -> NLIResult:
        """Check a single claim against source text."""
        # NLI format: premise [SEP] hypothesis
        nli_input = f"{source_text}</s></s>{claim}"

        output = self.model(nli_input)[0]
        label = output["label"].lower()
        confidence = output["score"]

        # Map to hallucination score
        if label == "entailment":
            score = 0.0  # Supported
        elif label == "contradiction":
            score = 1.0  # Definitely hallucinated
        else:  # neutral
            score = 0.5  # Uncertain

        return NLIResult(
            score=score,
            label=label,
            confidence=confidence,
            claim=claim,
        )

    def _result_to_dict(self, result: NLIResult) -> Dict[str, Any]:
        """Convert NLIResult to dictionary."""
        return {
            "claim": result.claim,
            "score": result.score,
            "label": result.label,
            "confidence": result.confidence,
        }
