"""
Entity Extractor for Graph RAG

Extracts named entities and relationships from documents
for building knowledge graphs.

Supports:
- Named Entity Recognition (NER)
- Relationship extraction
- Coreference resolution
- Technical concept extraction
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Supported entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECH"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    OTHER = "OTHER"


@dataclass
class Entity:
    """Extracted entity."""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Relationship:
    """Extracted relationship between entities."""
    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
        }


@dataclass
class ExtractionResult:
    """Result from entity extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    document_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
        }


class EntityExtractor:
    """
    Entity extractor for building knowledge graphs.

    Extracts entities and relationships from text using:
    - spaCy NER models
    - Custom patterns for technical terms
    - LLM-based extraction for complex relationships

    Example:
        extractor = EntityExtractor()
        result = extractor.extract(
            text="Python was created by Guido van Rossum at CWI.",
            document_id="doc_123"
        )

        for entity in result.entities:
            print(f"{entity.text} ({entity.type.value})")

    Integration with OpenRAG:
        # During document ingestion
        extractor = EntityExtractor()

        for chunk in document_chunks:
            extraction = extractor.extract(chunk.text, chunk.id)
            await graph_store.add_entities(extraction.entities)
            await graph_store.add_relationships(extraction.relationships)
    """

    # Technical terms patterns
    TECH_PATTERNS = [
        r'\b(Python|JavaScript|TypeScript|Java|C\+\+|Rust|Go|Ruby)\b',
        r'\b(React|Vue|Angular|Django|FastAPI|Flask|Express)\b',
        r'\b(PostgreSQL|MySQL|MongoDB|Redis|Neo4j|Milvus)\b',
        r'\b(Docker|Kubernetes|AWS|GCP|Azure)\b',
        r'\b(LLM|GPT|BERT|Transformer|RAG|NLP|ML|AI)\b',
        r'\b(API|REST|GraphQL|gRPC|WebSocket)\b',
    ]

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_gpu: bool = False,
        extract_technical: bool = True,
        lazy_load: bool = True,
    ):
        """
        Initialize entity extractor.

        Args:
            spacy_model: spaCy model to use for NER
            use_gpu: Whether to use GPU for spaCy
            extract_technical: Whether to extract technical terms
            lazy_load: Whether to lazy load the model
        """
        self.spacy_model_name = spacy_model
        self.use_gpu = use_gpu
        self.extract_technical = extract_technical
        self._nlp = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load spaCy model."""
        try:
            import spacy

            logger.info(f"Loading spaCy model: {self.spacy_model_name}")

            if self.use_gpu:
                spacy.require_gpu()

            self._nlp = spacy.load(self.spacy_model_name)
            logger.info("spaCy model loaded successfully")

        except OSError:
            logger.warning(f"Model {self.spacy_model_name} not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", self.spacy_model_name])
            import spacy
            self._nlp = spacy.load(self.spacy_model_name)

        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._load_model()
        return self._nlp

    def extract(
        self,
        text: str,
        document_id: Optional[str] = None,
        extract_relationships: bool = True,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.

        Args:
            text: Text to extract from
            document_id: Optional document identifier
            extract_relationships: Whether to extract relationships

        Returns:
            ExtractionResult with entities and relationships
        """
        entities = []
        relationships = []

        # Extract with spaCy NER
        doc = self.nlp(text)

        for ent in doc.ents:
            entity_type = self._map_spacy_type(ent.label_)
            entities.append(Entity(
                text=ent.text,
                type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9,  # spaCy doesn't provide confidence
            ))

        # Extract technical terms
        if self.extract_technical:
            tech_entities = self._extract_technical_terms(text)
            entities.extend(tech_entities)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Extract relationships
        if extract_relationships and len(entities) >= 2:
            relationships = self._extract_relationships(doc, entities)

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            document_id=document_id,
        )

    def _map_spacy_type(self, spacy_label: str) -> EntityType:
        """Map spaCy entity label to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
        }
        return mapping.get(spacy_label, EntityType.OTHER)

    def _extract_technical_terms(self, text: str) -> List[Entity]:
        """Extract technical terms using patterns."""
        entities = []

        for pattern in self.TECH_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    type=EntityType.TECHNOLOGY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                ))

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}

        for entity in entities:
            key = (entity.text.lower(), entity.type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def _extract_relationships(
        self,
        doc,
        entities: List[Entity],
    ) -> List[Relationship]:
        """
        Extract relationships between entities using dependency parsing.
        """
        relationships = []
        entity_map = {(e.start, e.end): e for e in entities}

        # Find subject-verb-object patterns
        for sent in doc.sents:
            subjects = []
            objects = []
            verbs = []

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(token)
                elif token.dep_ in ("dobj", "pobj", "attr"):
                    objects.append(token)
                elif token.pos_ == "VERB":
                    verbs.append(token)

            # Create relationships
            for subj in subjects:
                for obj in objects:
                    # Find matching entities
                    subj_entity = self._find_entity_for_token(subj, entities)
                    obj_entity = self._find_entity_for_token(obj, entities)

                    if subj_entity and obj_entity and subj_entity != obj_entity:
                        # Find the verb connecting them
                        verb = verbs[0].lemma_ if verbs else "relates_to"

                        relationships.append(Relationship(
                            source=subj_entity,
                            target=obj_entity,
                            relation_type=verb,
                            confidence=0.7,
                            context=sent.text,
                        ))

        return relationships

    def _find_entity_for_token(
        self,
        token,
        entities: List[Entity],
    ) -> Optional[Entity]:
        """Find entity that contains the token."""
        token_start = token.idx
        token_end = token.idx + len(token.text)

        for entity in entities:
            if entity.start <= token_start and entity.end >= token_end:
                return entity
            # Also check if token text matches entity text
            if token.text.lower() in entity.text.lower():
                return entity

        return None

    async def extract_with_llm(
        self,
        text: str,
        llm_client,
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract entities using LLM for better relationship extraction.

        Args:
            text: Text to extract from
            llm_client: LLM client for extraction
            document_id: Optional document identifier

        Returns:
            ExtractionResult with entities and relationships
        """
        prompt = f"""Extract all entities and relationships from the following text.

Text:
{text}

Return a JSON object with:
{{
    "entities": [
        {{"text": "entity name", "type": "PERSON|ORG|TECH|CONCEPT|OTHER"}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "relation": "relationship type"}}
    ]
}}

Only extract clear, factual entities and relationships."""

        try:
            response = await llm_client.generate(prompt, temperature=0)
            import json

            # Parse response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]

            data = json.loads(response.strip())

            # Convert to our format
            entities = []
            for e in data.get("entities", []):
                entity_type = EntityType[e.get("type", "OTHER")]
                entities.append(Entity(
                    text=e["text"],
                    type=entity_type,
                    start=0,
                    end=len(e["text"]),
                    confidence=0.85,
                ))

            relationships = []
            entity_lookup = {e.text.lower(): e for e in entities}

            for r in data.get("relationships", []):
                source = entity_lookup.get(r["source"].lower())
                target = entity_lookup.get(r["target"].lower())

                if source and target:
                    relationships.append(Relationship(
                        source=source,
                        target=target,
                        relation_type=r["relation"],
                        confidence=0.8,
                    ))

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                document_id=document_id,
            )

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fall back to standard extraction
            return self.extract(text, document_id)
