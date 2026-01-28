"""
Graph RAG Components for OpenRAG

This module provides knowledge graph capabilities for enhanced RAG:
- Entity extraction from documents
- Neo4j-based graph storage
- Graph-enhanced retrieval

PR #4: Graph RAG Integration
"""

from .entity_extractor import (
    Entity,
    EntityType,
    Relationship,
    ExtractionResult,
    EntityExtractor,
)

from .graph_store import (
    GraphNode,
    GraphEdge,
    SubGraph,
    GraphStore,
)

from .graph_retriever import (
    RetrievalMode,
    RetrievalResult,
    GraphRetrievalResult,
    GraphRetriever,
)

__all__ = [
    # Entity extraction
    "Entity",
    "EntityType",
    "Relationship",
    "ExtractionResult",
    "EntityExtractor",
    # Graph storage
    "GraphNode",
    "GraphEdge",
    "SubGraph",
    "GraphStore",
    # Graph retrieval
    "RetrievalMode",
    "RetrievalResult",
    "GraphRetrievalResult",
    "GraphRetriever",
]
