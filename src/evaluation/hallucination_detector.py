"""
Hallucination Detection Module for OpenRAG

Implements multiple methods for detecting hallucinations in RAG responses:
1. NLI-based entailment checking
2. Source-answer semantic alignment
3. Self-consistency verification
4. Ensemble scoring

Usage:
    detector = HallucinationDetector()
    result = await detector.detect(
        response="The model generated this response...",
        sources=["Source document 1...", "Source document 2..."]
    )
    print(f"Hallucination score: {result['ensemble_score']}")
    print(f"Flagged claims: {result['flagged_claims']}")
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Available hallucination detection methods."""
    NLI = "nli"
    ALIGNMENT = "alignment"
    CONSISTENCY = "consistency"
    ENSEMBLE = "ensemble"


@dataclass
class ClaimResult:
    """Result for a single claim evaluation."""
    claim: str
    score: float  # 0 = supported, 1 = hallucinated
    label: str  # 'supported', 'contradicted', 'neutral'
    confidence: float
    source_evidence: Optional[str] = None


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
    consistency_score: Optional[float] = None

    # Detailed results
    flagged_claims: List[str] = field(default_factory=list)
    claim_results: List[ClaimResult] = field(default_factory=list)

    # Metadata
    methods_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "ensemble_score": self.ensemble_score,
            "is_hallucinated": self.is_hallucinated,
            "confidence": self.confidence,
            "nli_score": self.nli_score,
            "alignment_score": self.alignment_score,
            "consistency_score": self.consistency_score,
            "flagged_claims": self.flagged_claims,
            "claim_results": [
                {
                    "claim": cr.claim,
                    "score": cr.score,
                    "label": cr.label,
                    "confidence": cr.confidence,
                }
                for cr in self.claim_results
            ],
            "methods_used": self.methods_used,
            "processing_time_ms": self.processing_time_ms,
        }


class HallucinationDetector:
    """
    Multi-method hallucination detector for RAG systems.

    Combines multiple detection strategies:
    - NLI: Uses natural language inference to check if claims are entailed
    - Alignment: Measures semantic similarity between response and sources
    - Consistency: Checks if multiple generations are consistent

    Example:
        detector = HallucinationDetector(
            nli_model="cross-encoder/nli-deberta-v3-large",
            embedding_model="intfloat/multilingual-e5-large-instruct"
        )

        result = await detector.detect(
            response="Python was created by Guido van Rossum in 1991.",
            sources=["Python is a programming language created by Guido van Rossum..."]
        )
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-large",
        embedding_model: str = "intfloat/multilingual-e5-large-instruct",
        hallucination_threshold: float = 0.5,
        device: str = "cuda",
        lazy_load: bool = True,
    ):
        """
        Initialize the hallucination detector.

        Args:
            nli_model: Model for NLI-based detection
            embedding_model: Model for semantic similarity
            hallucination_threshold: Score above which response is flagged
            device: Device for model inference ('cuda' or 'cpu')
            lazy_load: If True, models are loaded on first use
        """
        self.nli_model_name = nli_model
        self.embedding_model_name = embedding_model
        self.threshold = hallucination_threshold
        self.device = device
        self.lazy_load = lazy_load

        # Models (lazy loaded)
        self._nli_model = None
        self._embedder = None

        if not lazy_load:
            self._load_models()

    def _load_models(self):
        """Load ML models for detection."""
        try:
            from transformers import pipeline
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading NLI model: {self.nli_model_name}")
            self._nli_model = pipeline(
                "text-classification",
                model=self.nli_model_name,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    @property
    def nli_model(self):
        """Lazy-load NLI model."""
        if self._nli_model is None:
            self._load_models()
        return self._nli_model

    @property
    def embedder(self):
        """Lazy-load embedding model."""
        if self._embedder is None:
            self._load_models()
        return self._embedder

    async def detect(
        self,
        response: str,
        sources: List[str],
        methods: Optional[List[str]] = None,
        llm_client: Optional[Any] = None,
    ) -> DetectionResult:
        """
        Detect hallucinations in a RAG response.

        Args:
            response: The generated response to check
            sources: List of source documents used for generation
            methods: Detection methods to use (default: all)
            llm_client: Optional LLM client for consistency checking

        Returns:
            DetectionResult with scores and flagged claims
        """
        import time
        start_time = time.time()

        if methods is None:
            methods = ["nli", "alignment"]
            if llm_client is not None:
                methods.append("consistency")

        results = {}

        # Run detection methods
        if "nli" in methods:
            results["nli"] = await self._nli_check(response, sources)

        if "alignment" in methods:
            results["alignment"] = await self._alignment_check(response, sources)

        if "consistency" in methods and llm_client is not None:
            results["consistency"] = await self._consistency_check(
                response, sources, llm_client
            )

        # Compute ensemble score
        scores = [r["score"] for r in results.values()]
        ensemble_score = float(np.mean(scores)) if scores else 0.0

        # Collect flagged claims
        all_flagged = []
        all_claim_results = []

        for method_name, method_result in results.items():
            all_flagged.extend(method_result.get("flagged_claims", []))
            all_claim_results.extend(method_result.get("claim_results", []))

        # Deduplicate flagged claims
        flagged_claims = list(set(all_flagged))

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            ensemble_score=ensemble_score,
            is_hallucinated=ensemble_score > self.threshold,
            confidence=1.0 - abs(ensemble_score - 0.5) * 2,  # Highest at extremes
            nli_score=results.get("nli", {}).get("score"),
            alignment_score=results.get("alignment", {}).get("score"),
            consistency_score=results.get("consistency", {}).get("score"),
            flagged_claims=flagged_claims,
            claim_results=all_claim_results,
            methods_used=methods,
            processing_time_ms=processing_time,
        )

    async def _nli_check(
        self,
        response: str,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Check hallucination using Natural Language Inference.

        For each claim in the response, check if it's entailed by the sources.
        """
        # Extract claims from response
        claims = self._extract_claims(response)

        if not claims:
            return {
                "score": 0.0,
                "flagged_claims": [],
                "claim_results": [],
                "method": "nli",
            }

        # Combine sources into premise
        source_text = " ".join(sources)

        # Limit source text length for model
        max_source_len = 1000
        if len(source_text) > max_source_len:
            source_text = source_text[:max_source_len] + "..."

        hallucination_scores = []
        flagged_claims = []
        claim_results = []

        for claim in claims:
            try:
                # NLI format: premise [SEP] hypothesis
                nli_input = f"{source_text}</s></s>{claim}"

                result = self.nli_model(nli_input)[0]
                label = result["label"].upper()
                confidence = result["score"]

                if label == "ENTAILMENT":
                    score = 0.0  # Supported
                    claim_label = "supported"
                elif label == "CONTRADICTION":
                    score = 1.0  # Definitely hallucinated
                    claim_label = "contradicted"
                    flagged_claims.append(claim)
                else:  # NEUTRAL
                    score = 0.5  # Uncertain
                    claim_label = "neutral"
                    if confidence > 0.8:
                        flagged_claims.append(claim)

                hallucination_scores.append(score)
                claim_results.append(ClaimResult(
                    claim=claim,
                    score=score,
                    label=claim_label,
                    confidence=confidence,
                ))

            except Exception as e:
                logger.warning(f"NLI check failed for claim: {e}")
                continue

        avg_score = float(np.mean(hallucination_scores)) if hallucination_scores else 0.0

        return {
            "score": avg_score,
            "flagged_claims": flagged_claims,
            "claim_results": claim_results,
            "method": "nli",
        }

    async def _alignment_check(
        self,
        response: str,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Check hallucination using semantic alignment.

        Measures how well the response aligns semantically with the sources.
        Low alignment suggests potential hallucination.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            # Get embeddings
            response_embedding = self.embedder.encode(
                response,
                convert_to_numpy=True,
            ).reshape(1, -1)

            source_embeddings = self.embedder.encode(
                sources,
                convert_to_numpy=True,
            )

            # Compute similarities
            similarities = cosine_similarity(response_embedding, source_embeddings)[0]
            max_similarity = float(np.max(similarities))
            avg_similarity = float(np.mean(similarities))

            # Low similarity = high hallucination score
            hallucination_score = 1.0 - max_similarity

            # Check individual sentences for more granular analysis
            sentences = self._extract_claims(response)
            flagged_claims = []
            claim_results = []

            for sent in sentences:
                sent_embedding = self.embedder.encode(
                    sent,
                    convert_to_numpy=True,
                ).reshape(1, -1)

                sent_sims = cosine_similarity(sent_embedding, source_embeddings)[0]
                max_sent_sim = float(np.max(sent_sims))

                if max_sent_sim < 0.5:  # Low alignment threshold
                    flagged_claims.append(sent)

                claim_results.append(ClaimResult(
                    claim=sent,
                    score=1.0 - max_sent_sim,
                    label="low_alignment" if max_sent_sim < 0.5 else "aligned",
                    confidence=max_sent_sim,
                ))

            return {
                "score": hallucination_score,
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "flagged_claims": flagged_claims,
                "claim_results": claim_results,
                "method": "alignment",
            }

        except Exception as e:
            logger.error(f"Alignment check failed: {e}")
            return {
                "score": 0.5,  # Uncertain
                "flagged_claims": [],
                "claim_results": [],
                "method": "alignment",
                "error": str(e),
            }

    async def _consistency_check(
        self,
        response: str,
        sources: List[str],
        llm_client: Any,
        n_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Check hallucination using self-consistency.

        Generate multiple responses and check for contradictions.
        Inconsistent facts across generations suggest hallucination.
        """
        try:
            # Generate additional responses
            context = "\n".join(sources)

            # Extract the likely original question (heuristic)
            # In production, this would be passed explicitly
            prompt = f"""Based on the following context, generate a response:

Context:
{context[:2000]}

Generate a factual response based only on the context above."""

            responses = [response]  # Include original

            for _ in range(n_samples):
                # Generate with temperature for variation
                new_response = await llm_client.generate(
                    prompt,
                    temperature=0.7,
                    max_tokens=500,
                )
                responses.append(new_response)

            # Extract facts from each response
            all_facts = []
            for resp in responses:
                facts = self._extract_claims(resp)
                all_facts.append(set(facts))

            # Find facts that appear in only one response
            from collections import Counter
            fact_counts = Counter()
            for facts in all_facts:
                for fact in facts:
                    # Normalize fact for comparison
                    normalized = fact.lower().strip()
                    fact_counts[normalized] += 1

            # Facts appearing in only one response are suspicious
            unique_facts = [f for f, c in fact_counts.items() if c == 1]

            # Calculate consistency score
            if len(fact_counts) == 0:
                consistency_score = 0.0
            else:
                unique_ratio = len(unique_facts) / len(fact_counts)
                consistency_score = unique_ratio  # Higher = more inconsistent = more hallucination

            return {
                "score": consistency_score,
                "unique_facts": unique_facts[:5],  # Top 5
                "total_facts": len(fact_counts),
                "flagged_claims": unique_facts[:5],
                "claim_results": [],
                "method": "consistency",
            }

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {
                "score": 0.5,
                "flagged_claims": [],
                "claim_results": [],
                "method": "consistency",
                "error": str(e),
            }

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract individual claims/sentences from text.

        Filters out questions and very short sentences.
        """
        sentences = sent_tokenize(text)

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

    async def evaluate_benchmark(
        self,
        benchmark_data: List[Dict],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate detection performance on a benchmark dataset.

        Args:
            benchmark_data: List of dicts with 'response', 'sources', 'label'
            methods: Detection methods to use

        Returns:
            Evaluation metrics (precision, recall, F1, etc.)
        """
        predictions = []
        labels = []

        for item in benchmark_data:
            result = await self.detect(
                response=item["response"],
                sources=item["sources"],
                methods=methods,
            )

            predictions.append(1 if result.is_hallucinated else 0)
            labels.append(1 if item["label"] == "hallucinated" else 0)

        # Calculate metrics
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
            roc_auc_score,
        )

        predictions = np.array(predictions)
        labels = np.array(labels)

        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions)),
            "recall": float(recall_score(labels, predictions)),
            "f1_score": float(f1_score(labels, predictions)),
            "total_samples": len(labels),
            "hallucinated_count": int(np.sum(labels)),
            "detected_count": int(np.sum(predictions)),
        }


# Convenience function for quick detection
async def detect_hallucination(
    response: str,
    sources: List[str],
    threshold: float = 0.5,
) -> DetectionResult:
    """
    Quick hallucination detection with default settings.

    Args:
        response: Generated response to check
        sources: Source documents
        threshold: Hallucination threshold

    Returns:
        DetectionResult
    """
    detector = HallucinationDetector(
        hallucination_threshold=threshold,
        lazy_load=True,
    )
    return await detector.detect(response, sources)
