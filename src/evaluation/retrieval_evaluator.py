"""
Retrieval Evaluation Module for OpenRAG

Implements comprehensive retrieval evaluation metrics:
- Hit Rate @ K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Precision @ K
- Recall @ K

Usage:
    evaluator = RetrievalEvaluator()
    results = evaluator.evaluate(
        retriever=my_retriever,
        ground_truth=ground_truth_data,
        k_values=[1, 3, 5, 10, 20]
    )
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    # Hit rates
    hit_rate: Dict[int, float] = field(default_factory=dict)

    # Ranking metrics
    mrr: float = 0.0
    ndcg: Dict[int, float] = field(default_factory=dict)

    # Precision/Recall
    precision: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)

    # Performance
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Metadata
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "precision": self.precision,
            "recall": self.recall,
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
            },
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
        }


@dataclass
class QueryResult:
    """Result for a single query evaluation."""
    query_id: str
    query_text: str
    retrieved_ids: List[str]
    retrieved_scores: List[float]
    relevant_ids: List[str]
    latency_ms: float
    hit_at_k: Dict[int, bool]
    reciprocal_rank: float
    ndcg_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]


class RetrievalEvaluator:
    """
    Comprehensive retrieval evaluator following BEIR/MTEB standards.

    Supports both synchronous and asynchronous retrievers.

    Example:
        evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10, 20])

        # With a retriever function
        results = await evaluator.evaluate(
            retriever=my_retriever,
            ground_truth=ground_truth_data
        )

        # Or evaluate pre-computed results
        results = evaluator.evaluate_results(
            results=retrieval_results,
            ground_truth=ground_truth_data
        )
    """

    def __init__(
        self,
        k_values: List[int] = None,
        relevance_threshold: float = 0.0,
    ):
        """
        Initialize evaluator.

        Args:
            k_values: List of K values for @K metrics
            relevance_threshold: Minimum score to consider a document relevant
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.relevance_threshold = relevance_threshold

    async def evaluate(
        self,
        retriever: Callable,
        ground_truth: List[Dict[str, Any]],
        max_k: Optional[int] = None,
        parallel: bool = True,
        max_concurrent: int = 10,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance on a ground truth dataset.

        Args:
            retriever: Async function that takes query and returns list of (doc_id, score)
            ground_truth: List of dicts with 'query', 'query_id', 'relevant_doc_ids'
            max_k: Maximum number of documents to retrieve
            parallel: Whether to run queries in parallel
            max_concurrent: Maximum concurrent queries

        Returns:
            RetrievalMetrics with all computed metrics
        """
        max_k = max_k or max(self.k_values)

        query_results = []
        latencies = []

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_query(item):
                async with semaphore:
                    return await self._evaluate_single_query(
                        retriever, item, max_k
                    )

            query_results = await asyncio.gather(
                *[run_query(item) for item in ground_truth],
                return_exceptions=True
            )

            # Filter out exceptions
            valid_results = []
            for result in query_results:
                if isinstance(result, Exception):
                    logger.warning(f"Query failed: {result}")
                else:
                    valid_results.append(result)
                    latencies.append(result.latency_ms)

            query_results = valid_results

        else:
            for item in ground_truth:
                try:
                    result = await self._evaluate_single_query(
                        retriever, item, max_k
                    )
                    query_results.append(result)
                    latencies.append(result.latency_ms)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")

        return self._aggregate_results(query_results, latencies, len(ground_truth))

    async def _evaluate_single_query(
        self,
        retriever: Callable,
        item: Dict[str, Any],
        max_k: int,
    ) -> QueryResult:
        """Evaluate a single query."""
        query = item["question"] if "question" in item else item["query"]
        query_id = item.get("query_id", item.get("id", str(hash(query))))
        relevant_ids = item.get("relevant_doc_ids", [item.get("document_id")])

        if isinstance(relevant_ids, str):
            relevant_ids = [relevant_ids]

        # Time the retrieval
        start_time = time.time()

        try:
            # Call retriever - handle both sync and async
            if asyncio.iscoroutinefunction(retriever):
                results = await retriever(query, k=max_k)
            else:
                results = retriever(query, k=max_k)

            latency_ms = (time.time() - start_time) * 1000

            # Extract IDs and scores
            if results and hasattr(results[0], 'id'):
                # Object with .id attribute
                retrieved_ids = [r.id for r in results]
                retrieved_scores = [getattr(r, 'score', 1.0) for r in results]
            elif results and isinstance(results[0], tuple):
                # (id, score) tuples
                retrieved_ids = [r[0] for r in results]
                retrieved_scores = [r[1] for r in results]
            elif results and isinstance(results[0], dict):
                # Dictionaries
                retrieved_ids = [r.get('id', r.get('doc_id')) for r in results]
                retrieved_scores = [r.get('score', 1.0) for r in results]
            else:
                # Just IDs
                retrieved_ids = list(results) if results else []
                retrieved_scores = [1.0] * len(retrieved_ids)

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
            raise

        # Compute metrics
        hit_at_k = {}
        for k in self.k_values:
            if k <= len(retrieved_ids):
                hit_at_k[k] = any(rid in relevant_ids for rid in retrieved_ids[:k])
            else:
                hit_at_k[k] = any(rid in relevant_ids for rid in retrieved_ids)

        # MRR
        reciprocal_rank = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                reciprocal_rank = 1.0 / (i + 1)
                break

        # NDCG
        ndcg_at_k = {}
        for k in self.k_values:
            ndcg_at_k[k] = self._calculate_ndcg(
                retrieved_ids[:k], relevant_ids, k
            )

        # Precision and Recall
        precision_at_k = {}
        recall_at_k = {}
        for k in self.k_values:
            retrieved_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_ids)

            relevant_retrieved = len(retrieved_k & relevant_set)

            precision_at_k[k] = relevant_retrieved / k if k > 0 else 0.0
            recall_at_k[k] = relevant_retrieved / len(relevant_set) if relevant_set else 0.0

        return QueryResult(
            query_id=query_id,
            query_text=query,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            relevant_ids=relevant_ids,
            latency_ms=latency_ms,
            hit_at_k=hit_at_k,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg_at_k,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
        )

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """Calculate NDCG@K."""
        # Binary relevance
        relevance = [1 if rid in relevant_ids else 0 for rid in retrieved_ids]

        # DCG
        dcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(relevance[:k])
        )

        # Ideal DCG (all relevant docs at top)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_relevance[:k])
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _aggregate_results(
        self,
        query_results: List[QueryResult],
        latencies: List[float],
        total_queries: int,
    ) -> RetrievalMetrics:
        """Aggregate individual query results into overall metrics."""
        if not query_results:
            return RetrievalMetrics(
                total_queries=total_queries,
                failed_queries=total_queries,
            )

        # Hit rates
        hit_rate = {}
        for k in self.k_values:
            hits = [qr.hit_at_k.get(k, False) for qr in query_results]
            hit_rate[k] = sum(hits) / len(hits)

        # MRR
        mrr = np.mean([qr.reciprocal_rank for qr in query_results])

        # NDCG
        ndcg = {}
        for k in self.k_values:
            ndcg_values = [qr.ndcg_at_k.get(k, 0) for qr in query_results]
            ndcg[k] = float(np.mean(ndcg_values))

        # Precision
        precision = {}
        for k in self.k_values:
            p_values = [qr.precision_at_k.get(k, 0) for qr in query_results]
            precision[k] = float(np.mean(p_values))

        # Recall
        recall = {}
        for k in self.k_values:
            r_values = [qr.recall_at_k.get(k, 0) for qr in query_results]
            recall[k] = float(np.mean(r_values))

        # Latency
        latency_array = np.array(latencies) if latencies else np.array([0])

        return RetrievalMetrics(
            hit_rate=hit_rate,
            mrr=float(mrr),
            ndcg=ndcg,
            precision=precision,
            recall=recall,
            latency_mean_ms=float(np.mean(latency_array)),
            latency_p50_ms=float(np.percentile(latency_array, 50)),
            latency_p95_ms=float(np.percentile(latency_array, 95)),
            latency_p99_ms=float(np.percentile(latency_array, 99)),
            total_queries=total_queries,
            successful_queries=len(query_results),
            failed_queries=total_queries - len(query_results),
        )

    def evaluate_results(
        self,
        results: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> RetrievalMetrics:
        """
        Evaluate pre-computed retrieval results.

        Args:
            results: List of dicts with 'query_id', 'retrieved_ids', 'retrieved_scores'
            ground_truth: List of dicts with 'query_id', 'relevant_doc_ids'

        Returns:
            RetrievalMetrics
        """
        # Build lookup for ground truth
        gt_lookup = {}
        for item in ground_truth:
            qid = item.get("query_id", item.get("id"))
            relevant = item.get("relevant_doc_ids", [item.get("document_id")])
            if isinstance(relevant, str):
                relevant = [relevant]
            gt_lookup[qid] = relevant

        # Evaluate each result
        query_results = []
        latencies = []

        for result in results:
            qid = result.get("query_id", result.get("id"))
            if qid not in gt_lookup:
                logger.warning(f"No ground truth for query_id: {qid}")
                continue

            relevant_ids = gt_lookup[qid]
            retrieved_ids = result.get("retrieved_ids", [])
            retrieved_scores = result.get("retrieved_scores", [1.0] * len(retrieved_ids))
            latency = result.get("latency_ms", 0)

            # Compute metrics
            hit_at_k = {}
            for k in self.k_values:
                hit_at_k[k] = any(rid in relevant_ids for rid in retrieved_ids[:k])

            reciprocal_rank = 0.0
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant_ids:
                    reciprocal_rank = 1.0 / (i + 1)
                    break

            ndcg_at_k = {}
            precision_at_k = {}
            recall_at_k = {}

            for k in self.k_values:
                ndcg_at_k[k] = self._calculate_ndcg(retrieved_ids[:k], relevant_ids, k)

                retrieved_k = set(retrieved_ids[:k])
                relevant_set = set(relevant_ids)
                relevant_retrieved = len(retrieved_k & relevant_set)

                precision_at_k[k] = relevant_retrieved / k if k > 0 else 0.0
                recall_at_k[k] = relevant_retrieved / len(relevant_set) if relevant_set else 0.0

            query_results.append(QueryResult(
                query_id=qid,
                query_text=result.get("query", ""),
                retrieved_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                relevant_ids=relevant_ids,
                latency_ms=latency,
                hit_at_k=hit_at_k,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg_at_k,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
            ))
            latencies.append(latency)

        return self._aggregate_results(query_results, latencies, len(results))

    def compare_strategies(
        self,
        results_by_strategy: Dict[str, RetrievalMetrics],
    ) -> Dict[str, Any]:
        """
        Compare multiple retrieval strategies.

        Args:
            results_by_strategy: Dict mapping strategy name to RetrievalMetrics

        Returns:
            Comparison report
        """
        comparison = {
            "strategies": list(results_by_strategy.keys()),
            "metrics": {},
            "ranking": {},
            "best_by_metric": {},
        }

        metrics_to_compare = ["mrr"]
        for k in self.k_values:
            metrics_to_compare.append(f"hit_rate@{k}")
            metrics_to_compare.append(f"ndcg@{k}")

        for metric in metrics_to_compare:
            comparison["metrics"][metric] = {}

            for strategy, results in results_by_strategy.items():
                if metric == "mrr":
                    value = results.mrr
                elif metric.startswith("hit_rate@"):
                    k = int(metric.split("@")[1])
                    value = results.hit_rate.get(k, 0)
                elif metric.startswith("ndcg@"):
                    k = int(metric.split("@")[1])
                    value = results.ndcg.get(k, 0)
                else:
                    continue

                comparison["metrics"][metric][strategy] = value

            # Find best strategy for this metric
            if comparison["metrics"][metric]:
                best = max(
                    comparison["metrics"][metric].items(),
                    key=lambda x: x[1]
                )
                comparison["best_by_metric"][metric] = best[0]

        # Overall ranking (by average of key metrics)
        avg_scores = {}
        for strategy, results in results_by_strategy.items():
            avg_scores[strategy] = np.mean([
                results.mrr,
                results.hit_rate.get(5, 0),
                results.ndcg.get(10, 0),
            ])

        comparison["ranking"] = dict(
            sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return comparison
