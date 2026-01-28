"""
Semantic Alignment Hallucination Checker

Measures semantic similarity between response and source documents
to detect potential hallucinations.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result from alignment checking."""
    score: float  # 0 = aligned, 1 = misaligned (hallucinated)
    similarity: float
    claim: str


class AlignmentChecker:
    """
    Semantic alignment hallucination checker.

    Uses embedding models to measure how well the response
    aligns with the source documents semantically.

    Example:
        checker = AlignmentChecker()
        results = checker.check(
            response="Python is great for web development.",
            sources=["Python is used for web development, data science..."]
        )
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        device: str = "cuda",
        threshold: float = 0.5,
        lazy_load: bool = True,
    ):
        """
        Initialize alignment checker.

        Args:
            model_name: Sentence transformer model
            device: Device for inference
            threshold: Similarity threshold below which content is flagged
            lazy_load: If True, load model on first use
        """
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def check(
        self,
        response: str,
        sources: List[str],
        sentences: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Check response alignment with sources.

        Args:
            response: Generated response text
            sources: List of source documents
            sentences: Optional pre-extracted sentences from response

        Returns:
            Dictionary with score, flagged content, and details
        """
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            # Get embeddings
            response_embedding = self.model.encode(
                response, convert_to_numpy=True
            ).reshape(1, -1)

            source_embeddings = self.model.encode(
                sources, convert_to_numpy=True
            )

            # Overall similarity
            similarities = cosine_similarity(response_embedding, source_embeddings)[0]
            max_similarity = float(np.max(similarities))
            avg_similarity = float(np.mean(similarities))

            # Hallucination score (inverse of similarity)
            overall_score = 1.0 - max_similarity

            # Per-sentence analysis
            flagged_claims = []
            results = []

            if sentences:
                for sent in sentences:
                    sent_embedding = self.model.encode(
                        sent, convert_to_numpy=True
                    ).reshape(1, -1)

                    sent_sims = cosine_similarity(sent_embedding, source_embeddings)[0]
                    max_sent_sim = float(np.max(sent_sims))

                    result = AlignmentResult(
                        score=1.0 - max_sent_sim,
                        similarity=max_sent_sim,
                        claim=sent,
                    )
                    results.append(result)

                    if max_sent_sim < self.threshold:
                        flagged_claims.append(sent)

            return {
                "score": overall_score,
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "flagged_claims": flagged_claims,
                "results": [self._result_to_dict(r) for r in results],
                "method": "alignment",
            }

        except Exception as e:
            logger.error(f"Alignment check failed: {e}")
            return {
                "score": 0.5,  # Uncertain
                "flagged_claims": [],
                "results": [],
                "method": "alignment",
                "error": str(e),
            }

    def _result_to_dict(self, result: AlignmentResult) -> Dict[str, Any]:
        """Convert AlignmentResult to dictionary."""
        return {
            "claim": result.claim,
            "score": result.score,
            "similarity": result.similarity,
        }
