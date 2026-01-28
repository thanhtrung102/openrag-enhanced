"""
OpenRAG Enhanced - Evaluation Framework

This module provides comprehensive evaluation capabilities for RAG systems:
- Hallucination detection (NLI, alignment, consistency methods)
- Retrieval evaluation (Hit Rate, MRR, NDCG)
- Generation evaluation (LLM-as-Judge)
- Ground truth generation
"""

from .hallucination_detector import HallucinationDetector
from .llm_judge import LLMJudgeEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .ground_truth_generator import GroundTruthGenerator

__all__ = [
    "HallucinationDetector",
    "LLMJudgeEvaluator",
    "RetrievalEvaluator",
    "GroundTruthGenerator",
]
