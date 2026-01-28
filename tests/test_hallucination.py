"""
Unit Tests for OpenRAG Hallucination Detection Module

Tests the hallucination detection components:
- HallucinationDetector (main interface)
- NLIChecker (NLI-based detection)
- AlignmentChecker (semantic alignment)

Run with: pytest tests/test_hallucination.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import modules to test
from openrag.components.hallucination import (
    HallucinationDetector,
    NLIChecker,
    AlignmentChecker,
)
from openrag.components.hallucination.detector import DetectionResult


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def sample_response():
    """Sample RAG response for testing."""
    return "Python was created by Guido van Rossum in 1991. It is the most popular programming language."


@pytest.fixture
def sample_sources():
    """Sample source documents."""
    return [
        "Python is a high-level programming language created by Guido van Rossum.",
        "Python was first released in 1991 and has grown to be widely used.",
        "Python supports multiple programming paradigms including procedural and object-oriented.",
    ]


@pytest.fixture
def hallucinated_response():
    """Response with obvious hallucination."""
    return "Python was created by Bill Gates in 2005. It can only be used for web development."


@pytest.fixture
def mock_nli_model():
    """Mock NLI model for testing without GPU."""
    mock = MagicMock()
    mock.return_value = [{"label": "ENTAILMENT", "score": 0.95}]
    return mock


@pytest.fixture
def mock_embedder():
    """Mock sentence transformer for testing."""
    mock = MagicMock()
    # Return random embeddings
    mock.encode.return_value = np.random.rand(768)
    return mock


# ============================================
# NLIChecker Tests
# ============================================

class TestNLIChecker:
    """Tests for NLI-based hallucination checking."""

    def test_init_lazy_load(self):
        """Test lazy loading initialization."""
        checker = NLIChecker(lazy_load=True)
        assert checker._model is None
        assert checker.model_name == "cross-encoder/nli-deberta-v3-large"

    def test_check_empty_claims(self):
        """Test with no claims."""
        checker = NLIChecker(lazy_load=True)
        result = checker.check(claims=[], sources=["some source"])

        assert result["score"] == 0.0
        assert result["flagged_claims"] == []
        assert result["method"] == "nli"

    @patch.object(NLIChecker, 'model', new_callable=lambda: MagicMock())
    def test_check_entailed_claim(self, mock_model):
        """Test claim that is entailed by source."""
        mock_model.return_value = [{"label": "ENTAILMENT", "score": 0.95}]

        checker = NLIChecker(lazy_load=True)
        checker._model = mock_model

        result = checker.check(
            claims=["Python was created by Guido."],
            sources=["Python was created by Guido van Rossum."]
        )

        assert result["score"] == 0.0  # Entailed = not hallucinated
        assert len(result["flagged_claims"]) == 0

    @patch.object(NLIChecker, 'model', new_callable=lambda: MagicMock())
    def test_check_contradicted_claim(self, mock_model):
        """Test claim that contradicts source."""
        mock_model.return_value = [{"label": "CONTRADICTION", "score": 0.90}]

        checker = NLIChecker(lazy_load=True)
        checker._model = mock_model

        result = checker.check(
            claims=["Python was created by Bill Gates."],
            sources=["Python was created by Guido van Rossum."]
        )

        assert result["score"] == 1.0  # Contradiction = hallucinated
        assert "Python was created by Bill Gates." in result["flagged_claims"]

    @patch.object(NLIChecker, 'model', new_callable=lambda: MagicMock())
    def test_check_neutral_claim(self, mock_model):
        """Test claim that is neutral (uncertain)."""
        mock_model.return_value = [{"label": "NEUTRAL", "score": 0.85}]

        checker = NLIChecker(lazy_load=True)
        checker._model = mock_model

        result = checker.check(
            claims=["Python is great for beginners."],
            sources=["Python is a programming language."]
        )

        assert result["score"] == 0.5  # Neutral = uncertain
        # High confidence neutral claims are flagged
        assert "Python is great for beginners." in result["flagged_claims"]


# ============================================
# AlignmentChecker Tests
# ============================================

class TestAlignmentChecker:
    """Tests for semantic alignment checking."""

    def test_init_lazy_load(self):
        """Test lazy loading initialization."""
        checker = AlignmentChecker(lazy_load=True)
        assert checker._model is None
        assert checker.threshold == 0.5

    @patch.object(AlignmentChecker, 'model', new_callable=lambda: MagicMock())
    def test_check_high_alignment(self, mock_model):
        """Test response with high alignment to sources."""
        # Mock embeddings that are similar
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Response embedding
            np.array([[0.99, 0.01, 0.0]]),  # Source embeddings
        ]

        checker = AlignmentChecker(lazy_load=True)
        checker._model = mock_model

        result = checker.check(
            response="Python is a programming language.",
            sources=["Python is a high-level programming language."]
        )

        assert result["score"] < 0.5  # High alignment = low hallucination score
        assert result["method"] == "alignment"

    @patch.object(AlignmentChecker, 'model', new_callable=lambda: MagicMock())
    def test_check_low_alignment(self, mock_model):
        """Test response with low alignment to sources."""
        # Mock embeddings that are dissimilar
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),  # Response embedding
            np.array([[0.0, 1.0, 0.0]]),  # Source embeddings (orthogonal)
        ]

        checker = AlignmentChecker(lazy_load=True)
        checker._model = mock_model

        result = checker.check(
            response="JavaScript is used for web development.",
            sources=["Python is a programming language."]
        )

        assert result["score"] > 0.5  # Low alignment = high hallucination score


# ============================================
# HallucinationDetector Tests
# ============================================

class TestHallucinationDetector:
    """Tests for main HallucinationDetector class."""

    def test_init_default(self):
        """Test default initialization."""
        detector = HallucinationDetector(lazy_load=True)
        assert detector.threshold == 0.5
        assert detector.nli_checker is not None
        assert detector.alignment_checker is not None

    def test_init_custom_threshold(self):
        """Test custom threshold."""
        detector = HallucinationDetector(threshold=0.7, lazy_load=True)
        assert detector.threshold == 0.7

    def test_extract_claims(self):
        """Test claim extraction from response."""
        detector = HallucinationDetector(lazy_load=True)

        text = "Python is great. What do you think? This is a test sentence with more words."
        claims = detector._extract_claims(text)

        # Should filter out short sentences and questions
        assert "What do you think?" not in claims
        assert len(claims) >= 1

    def test_extract_claims_empty(self):
        """Test claim extraction from empty text."""
        detector = HallucinationDetector(lazy_load=True)
        claims = detector._extract_claims("")
        assert claims == []

    @pytest.mark.asyncio
    async def test_detect_returns_result(self):
        """Test that detect returns DetectionResult."""
        detector = HallucinationDetector(lazy_load=True)

        # Mock the checkers
        detector.nli_checker.check = Mock(return_value={
            "score": 0.3,
            "flagged_claims": [],
            "results": [],
            "method": "nli",
        })
        detector.alignment_checker.check = Mock(return_value={
            "score": 0.2,
            "flagged_claims": [],
            "results": [],
            "method": "alignment",
        })

        result = await detector.detect(
            response="Python is a programming language.",
            sources=["Python is a high-level language."]
        )

        assert isinstance(result, DetectionResult)
        assert result.ensemble_score == 0.25  # Average of 0.3 and 0.2
        assert result.is_hallucinated is False  # Below 0.5 threshold

    @pytest.mark.asyncio
    async def test_detect_hallucinated(self):
        """Test detection of hallucinated content."""
        detector = HallucinationDetector(lazy_load=True)

        # Mock high hallucination scores
        detector.nli_checker.check = Mock(return_value={
            "score": 0.8,
            "flagged_claims": ["Fake claim"],
            "results": [],
            "method": "nli",
        })
        detector.alignment_checker.check = Mock(return_value={
            "score": 0.7,
            "flagged_claims": ["Fake claim"],
            "results": [],
            "method": "alignment",
        })

        result = await detector.detect(
            response="This is completely made up.",
            sources=["Real source content."]
        )

        assert result.is_hallucinated is True
        assert result.ensemble_score > 0.5
        assert "Fake claim" in result.flagged_claims

    @pytest.mark.asyncio
    async def test_detect_specific_methods(self):
        """Test using specific detection methods."""
        detector = HallucinationDetector(lazy_load=True)

        detector.nli_checker.check = Mock(return_value={
            "score": 0.4,
            "flagged_claims": [],
            "results": [],
            "method": "nli",
        })

        result = await detector.detect(
            response="Test response.",
            sources=["Test source."],
            methods=["nli"]  # Only NLI
        )

        assert "nli" in result.methods_used
        assert "alignment" not in result.methods_used
        assert result.nli_score == 0.4
        assert result.alignment_score is None

    @pytest.mark.asyncio
    async def test_detect_batch(self):
        """Test batch detection."""
        detector = HallucinationDetector(lazy_load=True)

        # Mock checkers
        detector.nli_checker.check = Mock(return_value={
            "score": 0.3,
            "flagged_claims": [],
            "results": [],
            "method": "nli",
        })
        detector.alignment_checker.check = Mock(return_value={
            "score": 0.2,
            "flagged_claims": [],
            "results": [],
            "method": "alignment",
        })

        items = [
            {"response": "Response 1", "sources": ["Source 1"]},
            {"response": "Response 2", "sources": ["Source 2"]},
        ]

        results = await detector.detect_batch(items, parallel=True)

        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)


# ============================================
# DetectionResult Tests
# ============================================

class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DetectionResult(
            ensemble_score=0.3,
            is_hallucinated=False,
            confidence=0.8,
            nli_score=0.2,
            alignment_score=0.4,
            flagged_claims=["claim1"],
            methods_used=["nli", "alignment"],
            processing_time_ms=150.5,
        )

        d = result.to_dict()

        assert d["ensemble_score"] == 0.3
        assert d["is_hallucinated"] is False
        assert d["flagged_claims"] == ["claim1"]
        assert d["processing_time_ms"] == 150.5


# ============================================
# Integration Tests (require models)
# ============================================

@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires GPU and models")
class TestIntegration:
    """Integration tests that require actual models."""

    @pytest.mark.asyncio
    async def test_full_detection_pipeline(self, sample_response, sample_sources):
        """Test full detection with real models."""
        detector = HallucinationDetector(device="cpu", lazy_load=False)

        result = await detector.detect(
            response=sample_response,
            sources=sample_sources
        )

        assert isinstance(result, DetectionResult)
        assert 0 <= result.ensemble_score <= 1
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_obvious_hallucination(self, hallucinated_response, sample_sources):
        """Test detection of obvious hallucination."""
        detector = HallucinationDetector(device="cpu", lazy_load=False)

        result = await detector.detect(
            response=hallucinated_response,
            sources=sample_sources
        )

        # Should detect hallucination
        assert result.is_hallucinated is True
        assert len(result.flagged_claims) > 0


# ============================================
# Run tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
