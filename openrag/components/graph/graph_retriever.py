"""
Graph-Enhanced Retriever for Graph RAG

Combines traditional vector retrieval with knowledge graph
traversal for enhanced context in RAG pipelines.

Supports:
- Entity-based graph expansion
- Multi-hop relationship traversal
- Hybrid vector + graph retrieval
- Context ranking and deduplication
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Retrieval strategy modes."""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"


@dataclass
class RetrievalResult:
    """Result from graph-enhanced retrieval."""
    content: str
    source: str  # "vector", "graph", "hybrid"
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class GraphRetrievalResult:
    """Combined retrieval results."""
    results: List[RetrievalResult]
    query_entities: List[str]
    graph_context: Optional[str] = None
    total_vector_results: int = 0
    total_graph_results: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "query_entities": self.query_entities,
            "graph_context": self.graph_context,
            "total_vector_results": self.total_vector_results,
            "total_graph_results": self.total_graph_results,
        }

    def get_combined_context(self, max_length: int = 4000) -> str:
        """Get combined context from all results."""
        context_parts = []
        current_length = 0

        # Add graph context first if available
        if self.graph_context:
            context_parts.append(f"Related Knowledge:\n{self.graph_context}")
            current_length += len(self.graph_context)

        # Add vector results
        for result in self.results:
            if current_length + len(result.content) > max_length:
                break
            context_parts.append(result.content)
            current_length += len(result.content)

        return "\n\n".join(context_parts)


class GraphRetriever:
    """
    Graph-enhanced retriever for RAG pipelines.

    Combines knowledge graph traversal with vector similarity
    search to provide richer context for LLM generation.

    Example:
        retriever = GraphRetriever(
            graph_store=graph_store,
            entity_extractor=extractor,
        )

        results = await retriever.retrieve(
            query="How does Python handle memory management?",
            mode=RetrievalMode.GRAPH_ENHANCED,
            top_k=5,
        )

        context = results.get_combined_context()

    Integration with OpenRAG:
        # In the RAG pipeline
        graph_retriever = GraphRetriever(
            graph_store=graph_store,
            entity_extractor=entity_extractor,
            vector_store=milvus_store,  # OpenRAG's vector store
        )

        # Replace or augment existing retrieval
        async def enhanced_retrieve(query: str):
            return await graph_retriever.retrieve(
                query=query,
                mode=RetrievalMode.HYBRID,
            )
    """

    def __init__(
        self,
        graph_store,
        entity_extractor,
        vector_store=None,
        default_mode: RetrievalMode = RetrievalMode.GRAPH_ENHANCED,
        graph_weight: float = 0.3,
        max_hops: int = 2,
    ):
        """
        Initialize graph retriever.

        Args:
            graph_store: GraphStore instance for graph queries
            entity_extractor: EntityExtractor for query analysis
            vector_store: Optional vector store for hybrid retrieval
            default_mode: Default retrieval mode
            graph_weight: Weight for graph results in hybrid mode (0-1)
            max_hops: Maximum graph traversal depth
        """
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor
        self.vector_store = vector_store
        self.default_mode = default_mode
        self.graph_weight = graph_weight
        self.max_hops = max_hops

    async def retrieve(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
        top_k: int = 5,
        entity_types: Optional[List[str]] = None,
        include_graph_context: bool = True,
    ) -> GraphRetrievalResult:
        """
        Retrieve relevant content using graph-enhanced search.

        Args:
            query: User query
            mode: Retrieval mode (defaults to instance default)
            top_k: Number of results to return
            entity_types: Filter graph results by entity types
            include_graph_context: Whether to include graph context string

        Returns:
            GraphRetrievalResult with combined results
        """
        mode = mode or self.default_mode

        # Extract entities from query
        extraction = self.entity_extractor.extract(query, extract_relationships=False)
        query_entities = [e.text for e in extraction.entities]

        logger.info(f"Extracted {len(query_entities)} entities from query: {query_entities}")

        results = []
        graph_context = None
        total_vector = 0
        total_graph = 0

        if mode == RetrievalMode.VECTOR_ONLY:
            results = await self._vector_retrieve(query, top_k)
            total_vector = len(results)

        elif mode == RetrievalMode.GRAPH_ONLY:
            results, graph_context = await self._graph_retrieve(
                query_entities, top_k, entity_types, include_graph_context
            )
            total_graph = len(results)

        elif mode == RetrievalMode.HYBRID:
            results, graph_context, total_vector, total_graph = await self._hybrid_retrieve(
                query, query_entities, top_k, entity_types, include_graph_context
            )

        elif mode == RetrievalMode.GRAPH_ENHANCED:
            results, graph_context, total_vector, total_graph = await self._graph_enhanced_retrieve(
                query, query_entities, top_k, entity_types, include_graph_context
            )

        return GraphRetrievalResult(
            results=results,
            query_entities=query_entities,
            graph_context=graph_context if include_graph_context else None,
            total_vector_results=total_vector,
            total_graph_results=total_graph,
        )

    async def _vector_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve using vector similarity only."""
        if not self.vector_store:
            logger.warning("No vector store configured, returning empty results")
            return []

        try:
            # Assume vector_store has a search method
            vector_results = await self.vector_store.search(query, top_k=top_k)

            return [
                RetrievalResult(
                    content=r.get("content", r.get("text", "")),
                    source="vector",
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in vector_results
            ]
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    async def _graph_retrieve(
        self,
        entities: List[str],
        top_k: int,
        entity_types: Optional[List[str]] = None,
        include_context: bool = True,
    ) -> Tuple[List[RetrievalResult], Optional[str]]:
        """Retrieve using graph traversal."""
        results = []
        all_subgraphs = []

        for entity_text in entities[:5]:  # Limit entities to process
            try:
                # Get subgraph around entity
                subgraph = await self.graph_store.get_entity_context(
                    entity_text=entity_text,
                    max_hops=self.max_hops,
                    max_nodes=20,
                )

                if subgraph.nodes:
                    all_subgraphs.append(subgraph)

                    # Convert graph nodes to results
                    for node in subgraph.nodes:
                        if entity_types and node.properties.get("type") not in entity_types:
                            continue

                        results.append(RetrievalResult(
                            content=self._node_to_content(node),
                            source="graph",
                            score=node.properties.get("confidence", 0.8),
                            metadata={
                                "entity_text": node.properties.get("text"),
                                "entity_type": node.properties.get("type"),
                                "source_entity": entity_text,
                            },
                        ))

            except Exception as e:
                logger.error(f"Graph retrieval failed for entity '{entity_text}': {e}")

        # Deduplicate and sort by score
        results = self._deduplicate_results(results)
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        # Build combined graph context
        graph_context = None
        if include_context and all_subgraphs:
            graph_context = self._build_graph_context(all_subgraphs)

        return results, graph_context

    async def _hybrid_retrieve(
        self,
        query: str,
        entities: List[str],
        top_k: int,
        entity_types: Optional[List[str]] = None,
        include_context: bool = True,
    ) -> Tuple[List[RetrievalResult], Optional[str], int, int]:
        """Combine vector and graph retrieval with weighted scoring."""
        # Get results from both sources
        vector_results = await self._vector_retrieve(query, top_k * 2)
        graph_results, graph_context = await self._graph_retrieve(
            entities, top_k * 2, entity_types, include_context
        )

        # Apply weights
        for r in vector_results:
            r.score *= (1 - self.graph_weight)

        for r in graph_results:
            r.score *= self.graph_weight

        # Combine and sort
        combined = vector_results + graph_results
        combined = self._deduplicate_results(combined)
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined[:top_k], graph_context, len(vector_results), len(graph_results)

    async def _graph_enhanced_retrieve(
        self,
        query: str,
        entities: List[str],
        top_k: int,
        entity_types: Optional[List[str]] = None,
        include_context: bool = True,
    ) -> Tuple[List[RetrievalResult], Optional[str], int, int]:
        """
        Use graph to enhance vector retrieval.

        Strategy:
        1. Get vector results
        2. Extract entities from vector results
        3. Expand with graph relationships
        4. Re-rank based on graph connectivity
        """
        # Get initial vector results
        vector_results = await self._vector_retrieve(query, top_k)

        if not vector_results:
            # Fall back to graph-only
            graph_results, graph_context = await self._graph_retrieve(
                entities, top_k, entity_types, include_context
            )
            return graph_results, graph_context, 0, len(graph_results)

        # Extract entities from vector results
        result_entities = set(entities)
        for result in vector_results:
            extraction = self.entity_extractor.extract(
                result.content, extract_relationships=False
            )
            result_entities.update(e.text for e in extraction.entities)

        # Get graph context for all relevant entities
        all_subgraphs = []
        entity_connections = {}

        for entity_text in list(result_entities)[:10]:
            try:
                subgraph = await self.graph_store.get_entity_context(
                    entity_text=entity_text,
                    max_hops=self.max_hops,
                    max_nodes=15,
                )

                if subgraph.nodes:
                    all_subgraphs.append(subgraph)
                    # Track connectivity
                    entity_connections[entity_text] = len(subgraph.edges)

            except Exception as e:
                logger.debug(f"Graph expansion failed for '{entity_text}': {e}")

        # Re-rank vector results based on graph connectivity
        enhanced_results = []
        for result in vector_results:
            # Calculate graph boost based on entity connectivity
            graph_boost = 0.0
            result_extraction = self.entity_extractor.extract(
                result.content, extract_relationships=False
            )

            for entity in result_extraction.entities:
                if entity.text in entity_connections:
                    graph_boost += entity_connections[entity.text] * 0.01

            enhanced_results.append(RetrievalResult(
                content=result.content,
                source="graph_enhanced",
                score=result.score + min(graph_boost, 0.2),  # Cap boost at 0.2
                metadata={**result.metadata, "graph_boost": graph_boost},
            ))

        enhanced_results.sort(key=lambda x: x.score, reverse=True)

        # Build graph context
        graph_context = None
        if include_context and all_subgraphs:
            graph_context = self._build_graph_context(all_subgraphs)

        return enhanced_results[:top_k], graph_context, len(vector_results), len(all_subgraphs)

    def _node_to_content(self, node) -> str:
        """Convert graph node to content string."""
        props = node.properties
        text = props.get("text", node.id)
        entity_type = props.get("type", "ENTITY")

        # Include document context if available
        doc_ids = props.get("document_ids", [])
        doc_info = f" (from {len(doc_ids)} documents)" if doc_ids else ""

        return f"{text} [{entity_type}]{doc_info}"

    def _build_graph_context(self, subgraphs: List) -> str:
        """Build natural language context from subgraphs."""
        seen_facts = set()
        facts = []

        for subgraph in subgraphs:
            # Add relationship facts
            node_map = {n.id: n for n in subgraph.nodes}

            for edge in subgraph.edges:
                source = node_map.get(edge.source_id)
                target = node_map.get(edge.target_id)

                if source and target:
                    source_text = source.properties.get("text", edge.source_id)
                    target_text = target.properties.get("text", edge.target_id)
                    relation = edge.relation_type.lower().replace("_", " ")

                    fact = f"{source_text} {relation} {target_text}"
                    fact_key = fact.lower()

                    if fact_key not in seen_facts:
                        facts.append(fact)
                        seen_facts.add(fact_key)

        # Limit context size
        facts = facts[:20]

        if facts:
            return "Knowledge Graph Facts:\n" + "\n".join(f"- {f}" for f in facts)

        return ""

    def _deduplicate_results(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Remove duplicate results, keeping highest scored."""
        seen = {}

        for result in results:
            # Use content hash as key
            key = hash(result.content[:200])

            if key not in seen or result.score > seen[key].score:
                seen[key] = result

        return list(seen.values())

    async def get_entity_explanation(
        self,
        entity_text: str,
        max_facts: int = 10,
    ) -> str:
        """
        Get a natural language explanation of an entity from the graph.

        Useful for providing additional context about entities
        mentioned in user queries.

        Args:
            entity_text: Entity to explain
            max_facts: Maximum facts to include

        Returns:
            Natural language explanation string
        """
        try:
            subgraph = await self.graph_store.get_entity_context(
                entity_text=entity_text,
                max_hops=2,
                max_nodes=30,
            )

            if not subgraph.nodes:
                return f"No information found about '{entity_text}' in the knowledge graph."

            # Find the center entity
            center = None
            for node in subgraph.nodes:
                if entity_text.lower() in node.properties.get("text", "").lower():
                    center = node
                    break

            if not center:
                center = subgraph.nodes[0]

            center_text = center.properties.get("text", entity_text)
            center_type = center.properties.get("type", "entity")

            lines = [f"{center_text} is a {center_type.lower()}."]

            # Add relationship facts
            node_map = {n.id: n for n in subgraph.nodes}
            fact_count = 0

            for edge in subgraph.edges:
                if fact_count >= max_facts:
                    break

                source = node_map.get(edge.source_id)
                target = node_map.get(edge.target_id)

                if source and target:
                    source_text = source.properties.get("text", "")
                    target_text = target.properties.get("text", "")
                    relation = edge.relation_type.lower().replace("_", " ")

                    if center_text.lower() in source_text.lower():
                        lines.append(f"It {relation} {target_text}.")
                    elif center_text.lower() in target_text.lower():
                        lines.append(f"{source_text} {relation} it.")

                    fact_count += 1

            return " ".join(lines)

        except Exception as e:
            logger.error(f"Failed to get entity explanation: {e}")
            return f"Could not retrieve information about '{entity_text}'."
