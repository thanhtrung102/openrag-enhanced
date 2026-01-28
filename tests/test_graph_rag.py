"""
Tests for Graph RAG Components (PR #4)

Tests entity extraction, graph storage, and graph-enhanced retrieval.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import components
from openrag.components.graph import (
    Entity,
    EntityType,
    Relationship,
    ExtractionResult,
    EntityExtractor,
    GraphNode,
    GraphEdge,
    SubGraph,
    RetrievalMode,
    RetrievalResult,
    GraphRetrievalResult,
)


class TestEntityExtractor:
    """Tests for EntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with mocked spaCy."""
        with patch('openrag.components.graph.entity_extractor.spacy') as mock_spacy:
            # Mock spaCy model
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            extractor = EntityExtractor(lazy_load=True)
            extractor._nlp = mock_nlp

            return extractor

    def test_extract_technical_terms(self, extractor):
        """Test technical term extraction."""
        text = "Python and FastAPI are great for building REST APIs."

        entities = extractor._extract_technical_terms(text)

        # Should find Python, FastAPI, REST, API
        entity_texts = [e.text for e in entities]
        assert "Python" in entity_texts
        assert "FastAPI" in entity_texts
        assert any("API" in t for t in entity_texts)

    def test_extract_technical_terms_case_insensitive(self, extractor):
        """Test case-insensitive technical term extraction."""
        text = "python and JAVASCRIPT are programming languages."

        entities = extractor._extract_technical_terms(text)

        entity_texts = [e.text.lower() for e in entities]
        assert "python" in entity_texts
        assert "javascript" in entity_texts

    def test_deduplicate_entities(self, extractor):
        """Test entity deduplication."""
        entities = [
            Entity(text="Python", type=EntityType.TECHNOLOGY, start=0, end=6, confidence=0.9),
            Entity(text="python", type=EntityType.TECHNOLOGY, start=10, end=16, confidence=0.95),
            Entity(text="Java", type=EntityType.TECHNOLOGY, start=20, end=24, confidence=0.8),
        ]

        deduplicated = extractor._deduplicate_entities(entities)

        # Should keep higher confidence Python
        assert len(deduplicated) == 2
        python_entities = [e for e in deduplicated if "python" in e.text.lower()]
        assert python_entities[0].confidence == 0.95

    def test_map_spacy_type(self, extractor):
        """Test spaCy label mapping."""
        assert extractor._map_spacy_type("PERSON") == EntityType.PERSON
        assert extractor._map_spacy_type("ORG") == EntityType.ORGANIZATION
        assert extractor._map_spacy_type("GPE") == EntityType.LOCATION
        assert extractor._map_spacy_type("UNKNOWN") == EntityType.OTHER


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            text="Python",
            type=EntityType.TECHNOLOGY,
            start=0,
            end=6,
            confidence=0.95,
            metadata={"source": "test"},
        )

        d = entity.to_dict()

        assert d["text"] == "Python"
        assert d["type"] == "TECH"
        assert d["start"] == 0
        assert d["end"] == 6
        assert d["confidence"] == 0.95
        assert d["metadata"]["source"] == "test"


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        source = Entity(text="Guido", type=EntityType.PERSON, start=0, end=5)
        target = Entity(text="Python", type=EntityType.TECHNOLOGY, start=10, end=16)

        rel = Relationship(
            source=source,
            target=target,
            relation_type="created",
            confidence=0.9,
            context="Guido created Python",
        )

        d = rel.to_dict()

        assert d["source"]["text"] == "Guido"
        assert d["target"]["text"] == "Python"
        assert d["relation_type"] == "created"
        assert d["confidence"] == 0.9


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_extraction_result_to_dict(self):
        """Test extraction result serialization."""
        entity = Entity(text="Python", type=EntityType.TECHNOLOGY, start=0, end=6)

        result = ExtractionResult(
            entities=[entity],
            relationships=[],
            document_id="doc_123",
        )

        d = result.to_dict()

        assert d["document_id"] == "doc_123"
        assert d["entity_count"] == 1
        assert d["relationship_count"] == 0
        assert len(d["entities"]) == 1


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_graph_node_to_dict(self):
        """Test graph node serialization."""
        node = GraphNode(
            id="node_1",
            label="Entity",
            properties={"text": "Python", "type": "TECH"},
        )

        d = node.to_dict()

        assert d["id"] == "node_1"
        assert d["label"] == "Entity"
        assert d["properties"]["text"] == "Python"


class TestSubGraph:
    """Tests for SubGraph dataclass."""

    def test_subgraph_to_dict(self):
        """Test subgraph serialization."""
        nodes = [
            GraphNode(id="1", label="Entity", properties={"text": "Python"}),
            GraphNode(id="2", label="Entity", properties={"text": "Guido"}),
        ]
        edges = [
            GraphEdge(source_id="2", target_id="1", relation_type="CREATED", properties={}),
        ]

        subgraph = SubGraph(nodes=nodes, edges=edges, center_node_id="1")

        d = subgraph.to_dict()

        assert d["node_count"] == 2
        assert d["edge_count"] == 1
        assert d["center_node_id"] == "1"

    def test_subgraph_to_context_string(self):
        """Test subgraph context string generation."""
        nodes = [
            GraphNode(id="1", label="Entity", properties={"text": "Python", "type": "TECH"}),
            GraphNode(id="2", label="Entity", properties={"text": "Guido", "type": "PERSON"}),
        ]
        edges = [
            GraphEdge(source_id="2", target_id="1", relation_type="created", properties={}),
        ]

        subgraph = SubGraph(nodes=nodes, edges=edges)

        context = subgraph.to_context_string()

        assert "Python" in context
        assert "Guido" in context
        assert "created" in context


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_retrieval_result_to_dict(self):
        """Test retrieval result serialization."""
        result = RetrievalResult(
            content="Python is a programming language",
            source="graph",
            score=0.9,
            metadata={"entity": "Python"},
        )

        d = result.to_dict()

        assert d["content"] == "Python is a programming language"
        assert d["source"] == "graph"
        assert d["score"] == 0.9


class TestGraphRetrievalResult:
    """Tests for GraphRetrievalResult."""

    def test_get_combined_context(self):
        """Test combined context generation."""
        results = [
            RetrievalResult(content="Python is versatile", source="vector", score=0.9),
            RetrievalResult(content="Python was created by Guido", source="graph", score=0.8),
        ]

        graph_result = GraphRetrievalResult(
            results=results,
            query_entities=["Python"],
            graph_context="Python created_by Guido",
            total_vector_results=1,
            total_graph_results=1,
        )

        context = graph_result.get_combined_context()

        assert "Related Knowledge" in context
        assert "Python is versatile" in context


class TestRetrievalMode:
    """Tests for RetrievalMode enum."""

    def test_retrieval_modes(self):
        """Test retrieval mode values."""
        assert RetrievalMode.VECTOR_ONLY.value == "vector_only"
        assert RetrievalMode.GRAPH_ONLY.value == "graph_only"
        assert RetrievalMode.HYBRID.value == "hybrid"
        assert RetrievalMode.GRAPH_ENHANCED.value == "graph_enhanced"


# Integration tests (require mocking)

class TestEntityExtractorIntegration:
    """Integration tests for EntityExtractor."""

    @pytest.fixture
    def mock_spacy_doc(self):
        """Create mock spaCy doc."""
        mock_doc = MagicMock()

        # Mock entities
        mock_ent = MagicMock()
        mock_ent.text = "Guido van Rossum"
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 16

        mock_doc.ents = [mock_ent]
        mock_doc.sents = []

        return mock_doc

    def test_extract_with_spacy(self, mock_spacy_doc):
        """Test full extraction with mocked spaCy."""
        with patch('openrag.components.graph.entity_extractor.spacy') as mock_spacy:
            mock_nlp = MagicMock()
            mock_nlp.return_value = mock_spacy_doc
            mock_spacy.load.return_value = mock_nlp

            extractor = EntityExtractor(lazy_load=True)
            extractor._nlp = mock_nlp

            result = extractor.extract(
                text="Guido van Rossum created Python",
                document_id="doc_1",
            )

            assert result.document_id == "doc_1"
            assert len(result.entities) > 0

            # Should have both spaCy and technical entities
            entity_texts = [e.text for e in result.entities]
            assert "Guido van Rossum" in entity_texts
            assert "Python" in entity_texts


# Run with: pytest tests/test_graph_rag.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
