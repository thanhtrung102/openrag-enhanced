"""
Graph Store for Graph RAG

Neo4j-based storage for knowledge graphs built from
document entities and relationships.

Supports:
- Entity and relationship storage
- Graph traversal queries
- Similarity-based entity matching
- Subgraph extraction for RAG context
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "properties": self.properties,
        }


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
        }


@dataclass
class SubGraph:
    """Extracted subgraph for RAG context."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    center_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "center_node_id": self.center_node_id,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    def to_context_string(self) -> str:
        """Convert subgraph to natural language context."""
        lines = []

        # Add entity descriptions
        for node in self.nodes:
            props = node.properties
            text = props.get("text", node.id)
            entity_type = props.get("type", node.label)
            lines.append(f"- {text} ({entity_type})")

        # Add relationship descriptions
        node_map = {n.id: n for n in self.nodes}
        for edge in self.edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            if source and target:
                source_text = source.properties.get("text", edge.source_id)
                target_text = target.properties.get("text", edge.target_id)
                lines.append(f"- {source_text} {edge.relation_type} {target_text}")

        return "\n".join(lines)


class GraphStore:
    """
    Neo4j-based graph store for knowledge graphs.

    Stores entities and relationships extracted from documents
    and provides graph traversal for enhanced RAG retrieval.

    Example:
        store = GraphStore(uri="bolt://localhost:7687")
        await store.connect()

        # Add entities from extraction
        for entity in extraction.entities:
            await store.add_entity(entity, document_id="doc_123")

        # Add relationships
        for rel in extraction.relationships:
            await store.add_relationship(rel)

        # Query for RAG context
        subgraph = await store.get_entity_context(
            entity_text="Python",
            max_hops=2
        )

    Integration with OpenRAG:
        # In retrieval pipeline
        graph_store = GraphStore(uri=settings.neo4j_uri)

        # Extract entities from query
        query_entities = extractor.extract(query)

        # Get graph context
        contexts = []
        for entity in query_entities.entities:
            subgraph = await graph_store.get_entity_context(entity.text)
            contexts.append(subgraph.to_context_string())

        # Combine with vector retrieval results
        enhanced_context = vector_context + "\\n\\nRelated Knowledge:\\n" + "\\n".join(contexts)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize graph store.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )

            # Verify connection
            async with self._driver.session(database=self.database) as session:
                await session.run("RETURN 1")

            logger.info(f"Connected to Neo4j at {self.uri}")

            # Create indexes
            await self._create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self):
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def session(self):
        """Get a database session."""
        if not self._driver:
            await self.connect()

        async with self._driver.session(database=self.database) as session:
            yield session

    async def _create_indexes(self):
        """Create indexes for efficient queries."""
        indexes = [
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_doc IF NOT EXISTS FOR (e:Entity) ON (e.document_id)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
        ]

        async with self.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                except Exception as e:
                    logger.debug(f"Index creation note: {e}")

    async def add_entity(
        self,
        entity,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Add an entity to the graph.

        Args:
            entity: Entity object from extraction
            document_id: Source document identifier

        Returns:
            Node ID of the created entity
        """
        from .entity_extractor import Entity

        query = """
        MERGE (e:Entity {text: $text, type: $type})
        ON CREATE SET
            e.id = randomUUID(),
            e.confidence = $confidence,
            e.created_at = datetime(),
            e.document_ids = [$document_id]
        ON MATCH SET
            e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END,
            e.document_ids = CASE
                WHEN $document_id IS NOT NULL AND NOT $document_id IN e.document_ids
                THEN e.document_ids + $document_id
                ELSE e.document_ids
            END
        RETURN e.id as id
        """

        async with self.session() as session:
            result = await session.run(
                query,
                text=entity.text,
                type=entity.type.value if hasattr(entity.type, 'value') else str(entity.type),
                confidence=entity.confidence,
                document_id=document_id,
            )
            record = await result.single()
            return record["id"] if record else None

    async def add_relationship(
        self,
        relationship,
        document_id: Optional[str] = None,
    ) -> bool:
        """
        Add a relationship to the graph.

        Args:
            relationship: Relationship object from extraction
            document_id: Source document identifier

        Returns:
            True if relationship was created
        """
        # Sanitize relation type for Neo4j (no spaces, uppercase)
        relation_type = relationship.relation_type.upper().replace(" ", "_")

        query = f"""
        MATCH (source:Entity {{text: $source_text}})
        MATCH (target:Entity {{text: $target_text}})
        MERGE (source)-[r:{relation_type}]->(target)
        ON CREATE SET
            r.confidence = $confidence,
            r.context = $context,
            r.document_id = $document_id,
            r.created_at = datetime()
        RETURN r
        """

        async with self.session() as session:
            result = await session.run(
                query,
                source_text=relationship.source.text,
                target_text=relationship.target.text,
                confidence=relationship.confidence,
                context=relationship.context,
                document_id=document_id,
            )
            record = await result.single()
            return record is not None

    async def add_extraction_result(
        self,
        extraction_result,
    ):
        """
        Add all entities and relationships from an extraction result.

        Args:
            extraction_result: ExtractionResult from entity extractor
        """
        document_id = extraction_result.document_id

        # Add entities first
        for entity in extraction_result.entities:
            await self.add_entity(entity, document_id)

        # Then add relationships
        for relationship in extraction_result.relationships:
            await self.add_relationship(relationship, document_id)

        logger.info(
            f"Added {len(extraction_result.entities)} entities and "
            f"{len(extraction_result.relationships)} relationships "
            f"for document {document_id}"
        )

    async def get_entity_context(
        self,
        entity_text: str,
        max_hops: int = 2,
        max_nodes: int = 50,
    ) -> SubGraph:
        """
        Get subgraph context around an entity.

        Args:
            entity_text: Text of the entity to center on
            max_hops: Maximum relationship hops to traverse
            max_nodes: Maximum nodes to return

        Returns:
            SubGraph containing related entities and relationships
        """
        query = """
        MATCH (center:Entity)
        WHERE toLower(center.text) CONTAINS toLower($text)
        WITH center
        LIMIT 1

        CALL {
            WITH center
            MATCH path = (center)-[*1..$max_hops]-(related:Entity)
            RETURN related, relationships(path) as rels
            LIMIT $max_nodes
        }

        WITH center, collect(DISTINCT related) as related_nodes,
             collect(rels) as all_rels

        RETURN center, related_nodes, all_rels
        """

        async with self.session() as session:
            result = await session.run(
                query,
                text=entity_text,
                max_hops=max_hops,
                max_nodes=max_nodes,
            )
            record = await result.single()

            if not record:
                return SubGraph(nodes=[], edges=[])

            nodes = []
            edges = []
            seen_nodes: Set[str] = set()
            seen_edges: Set[str] = set()

            # Add center node
            center = record["center"]
            if center:
                center_id = center.get("id", center.get("text"))
                nodes.append(GraphNode(
                    id=center_id,
                    label="Entity",
                    properties=dict(center),
                ))
                seen_nodes.add(center_id)

            # Add related nodes
            for node in record.get("related_nodes", []):
                if node:
                    node_id = node.get("id", node.get("text"))
                    if node_id not in seen_nodes:
                        nodes.append(GraphNode(
                            id=node_id,
                            label="Entity",
                            properties=dict(node),
                        ))
                        seen_nodes.add(node_id)

            # Add edges from paths
            for rels in record.get("all_rels", []):
                if rels:
                    for rel in rels:
                        edge_key = f"{rel.start_node.get('text')}-{rel.type}-{rel.end_node.get('text')}"
                        if edge_key not in seen_edges:
                            edges.append(GraphEdge(
                                source_id=rel.start_node.get("id", rel.start_node.get("text")),
                                target_id=rel.end_node.get("id", rel.end_node.get("text")),
                                relation_type=rel.type,
                                properties=dict(rel),
                            ))
                            seen_edges.add(edge_key)

            return SubGraph(
                nodes=nodes,
                edges=edges,
                center_node_id=center_id if center else None,
            )

    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[GraphNode]:
        """
        Search for entities by text.

        Args:
            query_text: Text to search for
            entity_types: Optional filter by entity types
            limit: Maximum results to return

        Returns:
            List of matching entity nodes
        """
        if entity_types:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.text) CONTAINS toLower($text)
            AND e.type IN $types
            RETURN e
            ORDER BY e.confidence DESC
            LIMIT $limit
            """
            params = {"text": query_text, "types": entity_types, "limit": limit}
        else:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.text) CONTAINS toLower($text)
            RETURN e
            ORDER BY e.confidence DESC
            LIMIT $limit
            """
            params = {"text": query_text, "limit": limit}

        async with self.session() as session:
            result = await session.run(query, **params)
            records = await result.data()

            return [
                GraphNode(
                    id=r["e"].get("id", r["e"].get("text")),
                    label="Entity",
                    properties=dict(r["e"]),
                )
                for r in records
            ]

    async def get_document_subgraph(
        self,
        document_id: str,
    ) -> SubGraph:
        """
        Get all entities and relationships for a document.

        Args:
            document_id: Document identifier

        Returns:
            SubGraph containing all document entities
        """
        query = """
        MATCH (e:Entity)
        WHERE $document_id IN e.document_ids
        OPTIONAL MATCH (e)-[r]-(related:Entity)
        WHERE $document_id IN related.document_ids
        RETURN collect(DISTINCT e) as entities, collect(DISTINCT r) as relationships
        """

        async with self.session() as session:
            result = await session.run(query, document_id=document_id)
            record = await result.single()

            if not record:
                return SubGraph(nodes=[], edges=[])

            nodes = [
                GraphNode(
                    id=e.get("id", e.get("text")),
                    label="Entity",
                    properties=dict(e),
                )
                for e in record.get("entities", []) if e
            ]

            edges = [
                GraphEdge(
                    source_id=r.start_node.get("id", r.start_node.get("text")),
                    target_id=r.end_node.get("id", r.end_node.get("text")),
                    relation_type=r.type,
                    properties=dict(r),
                )
                for r in record.get("relationships", []) if r
            ]

            return SubGraph(nodes=nodes, edges=edges)

    async def get_path_between_entities(
        self,
        source_text: str,
        target_text: str,
        max_hops: int = 4,
    ) -> Optional[SubGraph]:
        """
        Find shortest path between two entities.

        Args:
            source_text: Source entity text
            target_text: Target entity text
            max_hops: Maximum path length

        Returns:
            SubGraph containing the path, or None if no path exists
        """
        query = """
        MATCH (source:Entity), (target:Entity)
        WHERE toLower(source.text) CONTAINS toLower($source_text)
        AND toLower(target.text) CONTAINS toLower($target_text)
        WITH source, target
        LIMIT 1

        MATCH path = shortestPath((source)-[*1..$max_hops]-(target))
        RETURN nodes(path) as path_nodes, relationships(path) as path_rels
        """

        async with self.session() as session:
            result = await session.run(
                query,
                source_text=source_text,
                target_text=target_text,
                max_hops=max_hops,
            )
            record = await result.single()

            if not record:
                return None

            nodes = [
                GraphNode(
                    id=n.get("id", n.get("text")),
                    label="Entity",
                    properties=dict(n),
                )
                for n in record.get("path_nodes", []) if n
            ]

            edges = [
                GraphEdge(
                    source_id=r.start_node.get("id", r.start_node.get("text")),
                    target_id=r.end_node.get("id", r.end_node.get("text")),
                    relation_type=r.type,
                    properties=dict(r),
                )
                for r in record.get("path_rels", []) if r
            ]

            return SubGraph(nodes=nodes, edges=edges)

    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        query = """
        MATCH (e:Entity)
        WITH count(e) as entity_count
        MATCH ()-[r]->()
        WITH entity_count, count(r) as relationship_count
        MATCH (e:Entity)
        WITH entity_count, relationship_count, e.type as type, count(*) as count
        RETURN entity_count, relationship_count, collect({type: type, count: count}) as type_distribution
        """

        async with self.session() as session:
            result = await session.run(query)
            record = await result.single()

            if not record:
                return {"entity_count": 0, "relationship_count": 0, "type_distribution": []}

            return {
                "entity_count": record["entity_count"],
                "relationship_count": record["relationship_count"],
                "type_distribution": record["type_distribution"],
            }

    async def clear_document(self, document_id: str):
        """
        Remove all entities unique to a document.

        Args:
            document_id: Document identifier to clear
        """
        query = """
        MATCH (e:Entity)
        WHERE e.document_ids = [$document_id]
        DETACH DELETE e
        """

        async with self.session() as session:
            await session.run(query, document_id=document_id)

        # Also remove document_id from shared entities
        query = """
        MATCH (e:Entity)
        WHERE $document_id IN e.document_ids
        SET e.document_ids = [x IN e.document_ids WHERE x <> $document_id]
        """

        async with self.session() as session:
            await session.run(query, document_id=document_id)

        logger.info(f"Cleared entities for document {document_id}")
