"""
OpenRAG Enhanced Benchmark

Extension of OpenRAG's automatic-evaluation-pipeline/benchmark.py
with hallucination detection and metrics persistence.

This file shows how to integrate the new features with OpenRAG's
existing evaluation pipeline.

Usage:
    python benchmark_enhanced.py --partition my_partition --persist --detect-hallucination
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# OpenRAG imports (existing)
# from openrag.components.retriever import Retriever
# from openrag.components.pipeline import RAGPipeline

# New imports for enhanced evaluation
from openrag.components.hallucination import HallucinationDetector
from openrag.storage import EvaluationStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBenchmark:
    """
    Enhanced benchmark runner with hallucination detection and persistence.

    Extends OpenRAG's existing benchmark with:
    1. Hallucination detection for each response
    2. Metrics persistence to PostgreSQL
    3. Enhanced reporting with hallucination rates

    Example:
        benchmark = EnhancedBenchmark(
            partition="my_docs",
            persist=True,
            detect_hallucination=True
        )
        results = await benchmark.run("dataset.json")
    """

    def __init__(
        self,
        partition: str,
        openrag_url: str = "http://localhost:8000",
        persist: bool = False,
        db_connection: Optional[str] = None,
        detect_hallucination: bool = True,
        hallucination_threshold: float = 0.5,
    ):
        """
        Initialize enhanced benchmark.

        Args:
            partition: OpenRAG partition to evaluate
            openrag_url: OpenRAG API URL
            persist: Whether to persist results to database
            db_connection: PostgreSQL connection string
            detect_hallucination: Whether to run hallucination detection
            hallucination_threshold: Threshold for flagging hallucinations
        """
        self.partition = partition
        self.openrag_url = openrag_url
        self.persist = persist
        self.detect_hallucination = detect_hallucination

        # Initialize hallucination detector
        if detect_hallucination:
            self.hallucination_detector = HallucinationDetector(
                threshold=hallucination_threshold,
                lazy_load=True,
            )
        else:
            self.hallucination_detector = None

        # Initialize persistence store
        if persist:
            self.store = EvaluationStore(
                connection_string=db_connection or os.getenv("DATABASE_URL")
            )
        else:
            self.store = None

    async def run(
        self,
        dataset_path: str,
        model_name: str = "default",
        retrieval_method: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Run enhanced benchmark on dataset.

        Args:
            dataset_path: Path to evaluation dataset JSON
            model_name: Name of model being evaluated
            retrieval_method: Retrieval method being used

        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting enhanced benchmark on partition: {self.partition}")

        # Load dataset
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # Connect to store if persisting
        if self.store:
            await self.store.connect()

        results = {
            "partition": self.partition,
            "model_name": model_name,
            "retrieval_method": retrieval_method,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(dataset),
            "queries": [],
            # Standard metrics (from original benchmark.py)
            "retrieval_metrics": {
                "hit_rate": [],
                "mrr": [],
                "ndcg": [],
                "recall": [],
            },
            "generation_metrics": {
                "completion_score": [],
                "precision_score": [],
            },
            # NEW: Hallucination metrics
            "hallucination_metrics": {
                "scores": [],
                "flagged_count": 0,
                "flagged_queries": [],
            },
        }

        # Process each query
        for i, item in enumerate(dataset):
            logger.info(f"Processing query {i+1}/{len(dataset)}")

            query_result = await self._evaluate_single_query(
                item=item,
                model_name=model_name,
                retrieval_method=retrieval_method,
            )

            results["queries"].append(query_result)

            # Aggregate metrics
            if query_result.get("retrieval"):
                results["retrieval_metrics"]["hit_rate"].append(
                    query_result["retrieval"].get("hit_rate", 0)
                )
                results["retrieval_metrics"]["mrr"].append(
                    query_result["retrieval"].get("mrr", 0)
                )
                results["retrieval_metrics"]["ndcg"].append(
                    query_result["retrieval"].get("ndcg", 0)
                )

            if query_result.get("generation"):
                results["generation_metrics"]["completion_score"].append(
                    query_result["generation"].get("completion_score", 0)
                )
                results["generation_metrics"]["precision_score"].append(
                    query_result["generation"].get("precision_score", 0)
                )

            if query_result.get("hallucination"):
                score = query_result["hallucination"]["ensemble_score"]
                results["hallucination_metrics"]["scores"].append(score)

                if query_result["hallucination"]["is_hallucinated"]:
                    results["hallucination_metrics"]["flagged_count"] += 1
                    results["hallucination_metrics"]["flagged_queries"].append({
                        "query": item["question"],
                        "score": score,
                        "flagged_claims": query_result["hallucination"]["flagged_claims"],
                    })

        # Calculate aggregates
        results["summary"] = self._calculate_summary(results)

        # Close store connection
        if self.store:
            await self.store.close()

        return results

    async def _evaluate_single_query(
        self,
        item: Dict[str, Any],
        model_name: str,
        retrieval_method: str,
    ) -> Dict[str, Any]:
        """Evaluate a single query with all metrics."""
        query = item["question"]
        ground_truth_chunks = item.get("ground_truth_chunks", [])
        reference_answer = item.get("reference_answer", "")

        result = {
            "query": query,
            "ground_truth_chunks": ground_truth_chunks,
        }

        try:
            # Step 1: Call OpenRAG API for retrieval and generation
            # (This would call the actual OpenRAG API)
            rag_response = await self._call_openrag(query)

            result["response"] = rag_response["response"]
            result["retrieved_chunks"] = rag_response["chunks"]

            # Step 2: Calculate retrieval metrics (existing logic)
            result["retrieval"] = self._calculate_retrieval_metrics(
                retrieved=rag_response["chunks"],
                relevant=ground_truth_chunks,
            )

            # Step 3: Calculate generation metrics (existing LLM-as-judge)
            result["generation"] = await self._evaluate_generation(
                query=query,
                response=rag_response["response"],
                context=rag_response["chunks"],
                reference=reference_answer,
            )

            # Step 4: NEW - Hallucination detection
            if self.hallucination_detector:
                chunk_texts = [c["text"] for c in rag_response["chunks"]]
                detection = await self.hallucination_detector.detect(
                    response=rag_response["response"],
                    sources=chunk_texts,
                )
                result["hallucination"] = detection.to_dict()

            # Step 5: NEW - Persist results
            if self.store:
                await self.store.save_evaluation(
                    query=query,
                    response=rag_response["response"],
                    sources=[c["text"] for c in rag_response["chunks"]],
                    metrics={
                        **result.get("retrieval", {}),
                        **result.get("generation", {}),
                        "hallucination_score": result.get("hallucination", {}).get("ensemble_score"),
                        "is_hallucinated": result.get("hallucination", {}).get("is_hallucinated"),
                        "flagged_claims": result.get("hallucination", {}).get("flagged_claims", []),
                    },
                    model_name=model_name,
                    retrieval_method=retrieval_method,
                )

        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            result["error"] = str(e)

        return result

    async def _call_openrag(self, query: str) -> Dict[str, Any]:
        """
        Call OpenRAG API for retrieval and generation.

        In production, this would call the actual OpenRAG API.
        """
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.openrag_url}/api/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": query}],
                    "partition": self.partition,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "response": data["choices"][0]["message"]["content"],
                "chunks": data.get("context", {}).get("chunks", []),
            }

    def _calculate_retrieval_metrics(
        self,
        retrieved: List[Dict],
        relevant: List[str],
    ) -> Dict[str, float]:
        """Calculate retrieval metrics (existing OpenRAG logic)."""
        import numpy as np

        retrieved_ids = [c.get("id", c.get("chunk_id")) for c in retrieved]
        relevant_set = set(relevant)

        # Hit rate
        hit = any(rid in relevant_set for rid in retrieved_ids[:5])

        # MRR
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        # NDCG@5
        relevance = [1 if rid in relevant_set else 0 for rid in retrieved_ids[:5]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_set), 5)))
        ndcg = dcg / idcg if idcg > 0 else 0

        return {
            "hit_rate": 1.0 if hit else 0.0,
            "mrr": mrr,
            "ndcg": ndcg,
        }

    async def _evaluate_generation(
        self,
        query: str,
        response: str,
        context: List[Dict],
        reference: str,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using LLM-as-judge.

        This mirrors OpenRAG's existing completion/precision scoring.
        """
        # In production, this would call the LLM judge
        # For now, return placeholder
        return {
            "completion_score": 0.0,  # Would be calculated by LLM
            "precision_score": 0.0,   # Would be calculated by LLM
        }

    def _calculate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        import numpy as np

        retrieval = results["retrieval_metrics"]
        generation = results["generation_metrics"]
        hallucination = results["hallucination_metrics"]

        summary = {
            # Retrieval
            "avg_hit_rate": float(np.mean(retrieval["hit_rate"])) if retrieval["hit_rate"] else 0,
            "avg_mrr": float(np.mean(retrieval["mrr"])) if retrieval["mrr"] else 0,
            "avg_ndcg": float(np.mean(retrieval["ndcg"])) if retrieval["ndcg"] else 0,
            # Generation
            "avg_completion_score": float(np.mean(generation["completion_score"])) if generation["completion_score"] else 0,
            "avg_precision_score": float(np.mean(generation["precision_score"])) if generation["precision_score"] else 0,
        }

        # NEW: Hallucination summary
        if hallucination["scores"]:
            summary["avg_hallucination_score"] = float(np.mean(hallucination["scores"]))
            summary["hallucination_rate"] = hallucination["flagged_count"] / results["total_queries"]
            summary["total_flagged"] = hallucination["flagged_count"]

        return summary


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="OpenRAG Enhanced Benchmark")
    parser.add_argument("--partition", required=True, help="Partition to evaluate")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset")
    parser.add_argument("--openrag-url", default="http://localhost:8000", help="OpenRAG API URL")
    parser.add_argument("--model-name", default="default", help="Model name")
    parser.add_argument("--retrieval-method", default="hybrid", help="Retrieval method")
    parser.add_argument("--persist", action="store_true", help="Persist results to database")
    parser.add_argument("--db-url", help="Database connection URL")
    parser.add_argument("--detect-hallucination", action="store_true", help="Run hallucination detection")
    parser.add_argument("--hallucination-threshold", type=float, default=0.5, help="Hallucination threshold")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    benchmark = EnhancedBenchmark(
        partition=args.partition,
        openrag_url=args.openrag_url,
        persist=args.persist,
        db_connection=args.db_url,
        detect_hallucination=args.detect_hallucination,
        hallucination_threshold=args.hallucination_threshold,
    )

    results = await benchmark.run(
        dataset_path=args.dataset,
        model_name=args.model_name,
        retrieval_method=args.retrieval_method,
    )

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nPartition: {results['partition']}")
    print(f"Total Queries: {results['total_queries']}")
    print(f"\nRetrieval Metrics:")
    print(f"  Hit Rate: {results['summary']['avg_hit_rate']:.2%}")
    print(f"  MRR: {results['summary']['avg_mrr']:.3f}")
    print(f"  NDCG: {results['summary']['avg_ndcg']:.3f}")

    if "avg_hallucination_score" in results["summary"]:
        print(f"\nHallucination Metrics:")
        print(f"  Avg Score: {results['summary']['avg_hallucination_score']:.3f}")
        print(f"  Hallucination Rate: {results['summary']['hallucination_rate']:.2%}")
        print(f"  Flagged Queries: {results['summary']['total_flagged']}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
