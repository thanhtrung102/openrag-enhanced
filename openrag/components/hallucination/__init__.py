"""
OpenRAG Hallucination Detection Module

Provides multi-method hallucination detection for RAG responses:
- NLI-based entailment checking
- Source-answer semantic alignment
- Self-consistency verification

Usage:
    from openrag.components.hallucination import HallucinationDetector

    detector = HallucinationDetector()
    result = await detector.detect(response, sources)
"""

from .detector import HallucinationDetector
from .nli_checker import NLIChecker
from .alignment_checker import AlignmentChecker

__all__ = [
    "HallucinationDetector",
    "NLIChecker",
    "AlignmentChecker",
]
