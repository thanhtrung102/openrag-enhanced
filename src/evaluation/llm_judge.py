"""
LLM-as-Judge Evaluator for RAG Systems

Implements automated evaluation of RAG responses using LLM judges.
Evaluates multiple dimensions:
- Faithfulness: Is the response grounded in the sources?
- Answer Relevance: Does it answer the question?
- Context Relevance: Was the retrieved context appropriate?
- Completeness: Does it fully address the question?
- Coherence: Is it well-structured and clear?

Usage:
    evaluator = LLMJudgeEvaluator(judge_model="gpt-4")
    result = await evaluator.evaluate(
        question="What is Python?",
        context=["Python is a programming language..."],
        response="Python is a high-level programming language...",
    )
    print(f"Overall score: {result['overall_score']}/5")
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class JudgeModel(Enum):
    """Supported judge models."""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"


@dataclass
class EvaluationResult:
    """Result from LLM judge evaluation."""
    # Scores (0-5 scale)
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    completeness: float
    coherence: float
    overall_score: float

    # Additional analysis
    hallucination_detected: bool
    hallucinated_claims: List[str]
    explanation: str

    # Metadata
    judge_model: str
    processing_time_ms: float
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "overall_score": self.overall_score,
            "hallucination_detected": self.hallucination_detected,
            "hallucinated_claims": self.hallucinated_claims,
            "explanation": self.explanation,
            "judge_model": self.judge_model,
            "processing_time_ms": self.processing_time_ms,
        }


class LLMJudgeEvaluator:
    """
    LLM-as-Judge evaluator for RAG responses.

    Uses a powerful LLM to evaluate the quality of RAG responses
    across multiple dimensions.

    Example:
        evaluator = LLMJudgeEvaluator(
            judge_model="gpt-4",
            api_key="sk-..."
        )

        result = await evaluator.evaluate(
            question="How does Python handle memory?",
            context=["Python uses automatic garbage collection..."],
            response="Python manages memory automatically using...",
            reference_answer="Python uses reference counting and GC..."
        )
    """

    EVALUATION_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.

Evaluate the following response on these criteria. Score each from 0-5:

## Criteria

1. **Faithfulness** (0-5): Does the response only contain information supported by the provided context?
   - 0: Completely fabricated, no connection to context
   - 1: Mostly fabricated with minor supported elements
   - 2: Mix of supported and unsupported claims
   - 3: Mostly supported with some unsupported additions
   - 4: Nearly all claims supported, minor extrapolations
   - 5: Every claim is directly supported by the context

2. **Answer Relevance** (0-5): Does the response directly answer the question asked?
   - 0: Completely irrelevant, doesn't address the question
   - 1: Tangentially related but misses the point
   - 2: Partially addresses the question
   - 3: Addresses the main question with gaps
   - 4: Fully addresses the question
   - 5: Perfectly addresses the question with appropriate depth

3. **Context Relevance** (0-5): Was the retrieved context appropriate for answering the question?
   - 0: Context is completely unrelated
   - 1: Context has minimal relevance
   - 2: Context is somewhat relevant
   - 3: Context is relevant but incomplete
   - 4: Context is highly relevant
   - 5: Context is perfectly suited to answer the question

4. **Completeness** (0-5): Does the response cover all important aspects of the question?
   - 0: Missing all key information
   - 1: Covers only a small fraction
   - 2: Covers some aspects, missing major points
   - 3: Covers main points, missing some details
   - 4: Comprehensive with minor omissions
   - 5: Thoroughly complete coverage

5. **Coherence** (0-5): Is the response well-structured, clear, and easy to understand?
   - 0: Incoherent, incomprehensible
   - 1: Very difficult to follow
   - 2: Somewhat disorganized
   - 3: Reasonably clear with some issues
   - 4: Well-organized and clear
   - 5: Excellently structured and crystal clear

## Input

**Question**: {question}

**Retrieved Context**:
{context}

**Response to Evaluate**:
{response}

**Reference Answer** (if available):
{reference}

## Output Format

Return a valid JSON object with this exact structure:
{{
    "faithfulness": <0-5>,
    "answer_relevance": <0-5>,
    "context_relevance": <0-5>,
    "completeness": <0-5>,
    "coherence": <0-5>,
    "hallucination_detected": <true/false>,
    "hallucinated_claims": ["list of specific claims not supported by context"],
    "explanation": "Brief 2-3 sentence justification of your scores"
}}

Respond ONLY with the JSON object, no additional text."""

    PAIRWISE_COMPARISON_PROMPT = """You are an expert evaluator comparing two RAG system responses.

Given the same question and context, determine which response is better.

## Question
{question}

## Context
{context}

## Response A
{response_a}

## Response B
{response_b}

## Evaluation Criteria
Consider: faithfulness to context, relevance to question, completeness, and clarity.

## Output Format
Return a valid JSON object:
{{
    "winner": "A" or "B" or "tie",
    "score_a": <1-10>,
    "score_b": <1-10>,
    "explanation": "Brief explanation of why one is better"
}}

Respond ONLY with the JSON object."""

    def __init__(
        self,
        judge_model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM judge evaluator.

        Args:
            judge_model: Model to use for evaluation
            api_key: API key (uses env var if not provided)
            temperature: Temperature for judge responses (0 for determinism)
            max_retries: Number of retries on failure
        """
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize client based on model
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        """Get or create the appropriate LLM client."""
        if self._client is not None:
            return self._client

        import os

        if "gpt" in self.judge_model.lower():
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key or os.getenv("OPENAI_API_KEY")
            )
            self._client_type = "openai"

        elif "claude" in self.judge_model.lower():
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=self._api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self._client_type = "anthropic"

        else:
            raise ValueError(f"Unsupported judge model: {self.judge_model}")

        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """Make LLM API call."""
        client = self._get_client()

        if self._client_type == "openai":
            response = await client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content

        elif self._client_type == "anthropic":
            response = await client.messages.create(
                model=self.judge_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    async def evaluate(
        self,
        question: str,
        context: Union[List[str], str],
        response: str,
        reference_answer: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a RAG response using LLM-as-judge.

        Args:
            question: The user's question
            context: Retrieved context (list of strings or single string)
            response: The generated response to evaluate
            reference_answer: Optional ground truth answer

        Returns:
            EvaluationResult with scores and analysis
        """
        start_time = time.time()

        # Format context
        if isinstance(context, list):
            context_str = "\n---\n".join(context)
        else:
            context_str = context

        # Truncate if too long
        max_context_len = 4000
        if len(context_str) > max_context_len:
            context_str = context_str[:max_context_len] + "\n[... truncated ...]"

        # Build prompt
        prompt = self.EVALUATION_PROMPT.format(
            question=question,
            context=context_str,
            response=response,
            reference=reference_answer or "Not provided",
        )

        # Call LLM with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_response = await self._call_llm(prompt)

                # Parse JSON response
                # Handle potential markdown code blocks
                if "```json" in raw_response:
                    raw_response = raw_response.split("```json")[1].split("```")[0]
                elif "```" in raw_response:
                    raw_response = raw_response.split("```")[1].split("```")[0]

                result_dict = json.loads(raw_response.strip())

                processing_time = (time.time() - start_time) * 1000

                # Calculate overall score
                scores = [
                    result_dict.get("faithfulness", 0),
                    result_dict.get("answer_relevance", 0),
                    result_dict.get("context_relevance", 0),
                    result_dict.get("completeness", 0),
                    result_dict.get("coherence", 0),
                ]
                overall = sum(scores) / len(scores)

                return EvaluationResult(
                    faithfulness=float(result_dict.get("faithfulness", 0)),
                    answer_relevance=float(result_dict.get("answer_relevance", 0)),
                    context_relevance=float(result_dict.get("context_relevance", 0)),
                    completeness=float(result_dict.get("completeness", 0)),
                    coherence=float(result_dict.get("coherence", 0)),
                    overall_score=overall,
                    hallucination_detected=result_dict.get("hallucination_detected", False),
                    hallucinated_claims=result_dict.get("hallucinated_claims", []),
                    explanation=result_dict.get("explanation", ""),
                    judge_model=self.judge_model,
                    processing_time_ms=processing_time,
                    raw_response=result_dict,
                )

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"Evaluation error on attempt {attempt + 1}: {e}")
                last_error = e

            if attempt < self.max_retries - 1:
                await asyncio.sleep(1)  # Brief delay before retry

        # Return default scores on failure
        logger.error(f"Evaluation failed after {self.max_retries} attempts: {last_error}")
        processing_time = (time.time() - start_time) * 1000

        return EvaluationResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_relevance=0.0,
            completeness=0.0,
            coherence=0.0,
            overall_score=0.0,
            hallucination_detected=False,
            hallucinated_claims=[],
            explanation=f"Evaluation failed: {last_error}",
            judge_model=self.judge_model,
            processing_time_ms=processing_time,
        )

    async def evaluate_batch(
        self,
        items: List[Dict[str, Any]],
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple items.

        Args:
            items: List of dicts with 'question', 'context', 'response', 'reference'
            parallel: Whether to run evaluations in parallel
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of EvaluationResult
        """
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def eval_with_semaphore(item):
                async with semaphore:
                    return await self.evaluate(
                        question=item["question"],
                        context=item["context"],
                        response=item["response"],
                        reference_answer=item.get("reference"),
                    )

            results = await asyncio.gather(
                *[eval_with_semaphore(item) for item in items]
            )
        else:
            results = []
            for item in items:
                result = await self.evaluate(
                    question=item["question"],
                    context=item["context"],
                    response=item["response"],
                    reference_answer=item.get("reference"),
                )
                results.append(result)

        return results

    async def compare_responses(
        self,
        question: str,
        context: Union[List[str], str],
        response_a: str,
        response_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two responses and determine which is better.

        Args:
            question: The user's question
            context: Retrieved context
            response_a: First response
            response_b: Second response

        Returns:
            Comparison result with winner and scores
        """
        if isinstance(context, list):
            context_str = "\n---\n".join(context)
        else:
            context_str = context

        prompt = self.PAIRWISE_COMPARISON_PROMPT.format(
            question=question,
            context=context_str[:3000],
            response_a=response_a,
            response_b=response_b,
        )

        try:
            raw_response = await self._call_llm(prompt)

            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0]
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0]

            return json.loads(raw_response.strip())

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {
                "winner": "tie",
                "score_a": 5,
                "score_b": 5,
                "explanation": f"Comparison failed: {e}",
            }

    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics from multiple evaluations.

        Args:
            results: List of EvaluationResult

        Returns:
            Dictionary of aggregate metrics
        """
        import numpy as np

        if not results:
            return {}

        faithfulness = [r.faithfulness for r in results]
        relevance = [r.answer_relevance for r in results]
        completeness = [r.completeness for r in results]
        coherence = [r.coherence for r in results]
        overall = [r.overall_score for r in results]
        hallucination_count = sum(1 for r in results if r.hallucination_detected)

        return {
            "avg_faithfulness": float(np.mean(faithfulness)),
            "std_faithfulness": float(np.std(faithfulness)),
            "avg_answer_relevance": float(np.mean(relevance)),
            "std_answer_relevance": float(np.std(relevance)),
            "avg_completeness": float(np.mean(completeness)),
            "std_completeness": float(np.std(completeness)),
            "avg_coherence": float(np.mean(coherence)),
            "std_coherence": float(np.std(coherence)),
            "avg_overall_score": float(np.mean(overall)),
            "std_overall_score": float(np.std(overall)),
            "hallucination_rate": hallucination_count / len(results),
            "total_evaluated": len(results),
        }
