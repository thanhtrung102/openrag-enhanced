"""
Unit tests for Hallucination Detector
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np


class TestHallucinationDetector:
    """Tests for HallucinationDetector class."""

    def test_extract_claims_filters_short_sentences(self):
        """Test that short sentences are filtered out."""
        from src.evaluation.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector(lazy_load=True)

        text = "This is a valid claim about something important. Hi. What? This is another valid claim."
        claims = detector._extract_claims(text)

        assert len(claims) == 2
        assert "Hi." not in claims
        assert "What?" not in claims

    def test_extract_claims_filters_questions(self):
        """Test that questions are filtered out."""
        from src.evaluation.hallucination_detector import HallucinationDetector

        detector = HallucinationDetector(lazy_load=True)

        text = "Python is a programming language. Is Python good? It was created in 1991."
        claims = detector._extract_claims(text)

        assert "Is Python good?" not in claims
        assert len(claims) == 2

    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization."""
        from src.evaluation.hallucination_detector import DetectionResult, ClaimResult

        result = DetectionResult(
            ensemble_score=0.3,
            is_hallucinated=False,
            confidence=0.8,
            nli_score=0.2,
            alignment_score=0.4,
            flagged_claims=["suspicious claim"],
            claim_results=[
                ClaimResult(
                    claim="test claim",
                    score=0.3,
                    label="neutral",
                    confidence=0.7,
                )
            ],
            methods_used=["nli", "alignment"],
            processing_time_ms=150.0,
        )

        result_dict = result.to_dict()

        assert result_dict["ensemble_score"] == 0.3
        assert result_dict["is_hallucinated"] is False
        assert len(result_dict["flagged_claims"]) == 1
        assert len(result_dict["claim_results"]) == 1

    @pytest.mark.asyncio
    async def test_detect_returns_result(self):
        """Test that detect returns a DetectionResult."""
        from src.evaluation.hallucination_detector import HallucinationDetector

        # Mock the models
        with patch.object(HallucinationDetector, '_load_models'):
            detector = HallucinationDetector(lazy_load=True)

            # Mock the internal methods
            detector._nli_check = AsyncMock(return_value={
                "score": 0.3,
                "flagged_claims": [],
                "claim_results": [],
                "method": "nli",
            })
            detector._alignment_check = AsyncMock(return_value={
                "score": 0.4,
                "flagged_claims": [],
                "claim_results": [],
                "method": "alignment",
            })

            result = await detector.detect(
                response="Python is a great programming language.",
                sources=["Python is a programming language created by Guido."],
                methods=["nli", "alignment"],
            )

            assert result.ensemble_score == 0.35  # Average of 0.3 and 0.4
            assert result.is_hallucinated is False  # Below 0.5 threshold


class TestClaimResult:
    """Tests for ClaimResult dataclass."""

    def test_claim_result_creation(self):
        """Test ClaimResult instantiation."""
        from src.evaluation.hallucination_detector import ClaimResult

        result = ClaimResult(
            claim="Test claim",
            score=0.5,
            label="neutral",
            confidence=0.8,
            source_evidence="Some evidence",
        )

        assert result.claim == "Test claim"
        assert result.score == 0.5
        assert result.label == "neutral"
        assert result.confidence == 0.8
        assert result.source_evidence == "Some evidence"


class TestDetectionMethods:
    """Tests for individual detection methods."""

    @pytest.mark.asyncio
    async def test_alignment_check_low_similarity(self):
        """Test alignment check with low similarity content."""
        from src.evaluation.hallucination_detector import HallucinationDetector

        # This would require mocking the embedding model
        # Placeholder for actual implementation
        pass

    @pytest.mark.asyncio
    async def test_nli_check_entailment(self):
        """Test NLI check with entailed content."""
        # This would require mocking the NLI model
        # Placeholder for actual implementation
        pass
