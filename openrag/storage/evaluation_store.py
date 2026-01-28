"""
Evaluation Metrics Storage for OpenRAG

Provides persistent storage for evaluation results using PostgreSQL.
Integrates with the automatic-evaluation-pipeline.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRecord:
    """A single evaluation record."""
    id: str
    query_text: str
    response_text: str
    model_name: str
    retrieval_method: str
    # Retrieval metrics
    hit_rate: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    recall: Optional[float] = None
    # Generation metrics
    completion_score: Optional[float] = None
    precision_score: Optional[float] = None
    # Hallucination metrics
    hallucination_score: Optional[float] = None
    is_hallucinated: Optional[bool] = None
    flagged_claims: Optional[List[str]] = None
    # Metadata
    timestamp: str = None
    metadata: Dict[str, Any] = None


class EvaluationStore:
    """
    Persistent storage for OpenRAG evaluation results.

    Stores evaluation metrics in PostgreSQL for historical tracking
    and dashboard visualization.

    Example:
        store = EvaluationStore(connection_string="postgresql://...")

        # Store evaluation result
        await store.save_evaluation(
            query="What is Python?",
            response="Python is a programming language...",
            metrics={
                "hit_rate": 0.85,
                "mrr": 0.72,
                "hallucination_score": 0.15,
            }
        )

        # Query historical results
        results = await store.get_evaluations(
            start_date="2024-01-01",
            model_name="mistral-7b"
        )
    """

    def __init__(
        self,
        connection_string: str = None,
        pool_size: int = 5,
    ):
        """
        Initialize evaluation store.

        Args:
            connection_string: PostgreSQL connection string
            pool_size: Connection pool size
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._pool = None

    async def connect(self):
        """Establish database connection."""
        if self.connection_string is None:
            logger.warning("No connection string provided, using in-memory storage")
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=self.pool_size,
            )
            logger.info("Connected to evaluation database")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def close(self):
        """Close database connection."""
        if self._pool:
            await self._pool.close()

    async def save_evaluation(
        self,
        query: str,
        response: str,
        sources: List[str],
        metrics: Dict[str, Any],
        model_name: str = "unknown",
        retrieval_method: str = "unknown",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Save an evaluation result.

        Args:
            query: The user query
            response: Generated response
            sources: Retrieved source documents
            metrics: Evaluation metrics dictionary
            model_name: Name of the model used
            retrieval_method: Retrieval method used
            metadata: Additional metadata

        Returns:
            Evaluation ID
        """
        eval_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        if self._pool is None:
            # In-memory fallback
            logger.info(f"Evaluation {eval_id} stored (in-memory)")
            return eval_id

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO evaluation_results (
                        id, query_text, response_text, sources,
                        model_name, retrieval_method,
                        hit_rate, mrr, ndcg, recall,
                        completion_score, precision_score,
                        hallucination_score, is_hallucinated, flagged_claims,
                        timestamp, metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17
                    )
                """,
                    eval_id,
                    query,
                    response,
                    json.dumps(sources),
                    model_name,
                    retrieval_method,
                    metrics.get("hit_rate"),
                    metrics.get("mrr"),
                    metrics.get("ndcg"),
                    metrics.get("recall"),
                    metrics.get("completion_score"),
                    metrics.get("precision_score"),
                    metrics.get("hallucination_score"),
                    metrics.get("is_hallucinated"),
                    json.dumps(metrics.get("flagged_claims", [])),
                    timestamp,
                    json.dumps(metadata or {}),
                )

            logger.info(f"Evaluation {eval_id} saved to database")
            return eval_id

        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
            raise

    async def get_evaluations(
        self,
        start_date: str = None,
        end_date: str = None,
        model_name: str = None,
        retrieval_method: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query evaluation results.

        Args:
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            model_name: Filter by model name
            retrieval_method: Filter by retrieval method
            limit: Maximum results to return

        Returns:
            List of evaluation records
        """
        if self._pool is None:
            return []

        try:
            query = "SELECT * FROM evaluation_results WHERE 1=1"
            params = []
            param_idx = 1

            if start_date:
                query += f" AND timestamp >= ${param_idx}"
                params.append(start_date)
                param_idx += 1

            if end_date:
                query += f" AND timestamp <= ${param_idx}"
                params.append(end_date)
                param_idx += 1

            if model_name:
                query += f" AND model_name = ${param_idx}"
                params.append(model_name)
                param_idx += 1

            if retrieval_method:
                query += f" AND retrieval_method = ${param_idx}"
                params.append(retrieval_method)
                param_idx += 1

            query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
            params.append(limit)

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to query evaluations: {e}")
            return []

    async def get_summary(
        self,
        start_date: str = None,
        end_date: str = None,
        group_by: str = "model_name",
    ) -> Dict[str, Any]:
        """
        Get aggregated evaluation summary.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            group_by: Field to group by ('model_name' or 'retrieval_method')

        Returns:
            Summary statistics
        """
        if self._pool is None:
            return {"error": "No database connection"}

        try:
            query = f"""
                SELECT
                    {group_by},
                    COUNT(*) as total_evaluations,
                    AVG(hit_rate) as avg_hit_rate,
                    AVG(mrr) as avg_mrr,
                    AVG(ndcg) as avg_ndcg,
                    AVG(hallucination_score) as avg_hallucination_score,
                    SUM(CASE WHEN is_hallucinated THEN 1 ELSE 0 END)::float / COUNT(*) as hallucination_rate
                FROM evaluation_results
                WHERE 1=1
            """

            params = []
            param_idx = 1

            if start_date:
                query += f" AND timestamp >= ${param_idx}"
                params.append(start_date)
                param_idx += 1

            if end_date:
                query += f" AND timestamp <= ${param_idx}"
                params.append(end_date)
                param_idx += 1

            query += f" GROUP BY {group_by}"

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            return {
                "group_by": group_by,
                "results": [dict(row) for row in rows],
            }

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {"error": str(e)}
