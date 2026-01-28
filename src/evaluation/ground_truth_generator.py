"""
Ground Truth Dataset Generator for RAG Evaluation

Generates Q&A pairs from documents for retrieval and generation evaluation.
Uses LLMs to create diverse, high-quality question-answer pairs.

Usage:
    generator = GroundTruthGenerator(llm_client=openai_client)
    dataset = await generator.generate_from_documents(
        documents=my_documents,
        n_per_doc=5
    )
"""

import asyncio
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """A single question-answer pair."""
    pair_id: str
    question: str
    answer: str
    document_id: str
    document_title: Optional[str] = None
    supporting_quote: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    question_type: str = "factual"  # factual, reasoning, comparison, multi-hop
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair_id": self.pair_id,
            "question": self.question,
            "answer": self.answer,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "supporting_quote": self.supporting_quote,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "metadata": self.metadata,
        }


@dataclass
class GroundTruthDataset:
    """Collection of Q&A pairs for evaluation."""
    dataset_id: str
    name: str
    description: str
    pairs: List[QAPair]
    version: str = "1.0.0"
    language: str = "en"
    domain: str = "general"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "auto-generator"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "language": self.language,
            "domain": self.domain,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "total_pairs": len(self.pairs),
            "pairs": [p.to_dict() for p in self.pairs],
        }

    def to_retrieval_format(self) -> List[Dict[str, Any]]:
        """Convert to format suitable for retrieval evaluation."""
        return [
            {
                "query_id": p.pair_id,
                "question": p.question,
                "relevant_doc_ids": [p.document_id],
                "answer": p.answer,
            }
            for p in self.pairs
        ]


class GroundTruthGenerator:
    """
    Generates ground truth Q&A datasets from documents using LLMs.

    Features:
    - Multiple question types (factual, reasoning, comparison)
    - Difficulty levels
    - Multi-hop question generation
    - Quality filtering

    Example:
        generator = GroundTruthGenerator()

        dataset = await generator.generate_from_documents(
            documents=documents,
            n_per_doc=5,
            question_types=["factual", "reasoning"],
            difficulties=["easy", "medium", "hard"]
        )
    """

    GENERATION_PROMPT = """You are an expert at creating high-quality question-answer pairs for evaluating RAG systems.

Based on the following document, generate {n} diverse question-answer pairs.

## Document
Title: {title}
Content:
{content}

## Requirements
1. Questions should be answerable ONLY using information from this document
2. Include a mix of question types: {question_types}
3. Vary difficulty levels: {difficulties}
4. Each answer must be directly supported by the document
5. Include the exact quote that supports each answer

## Question Type Guidelines
- **factual**: Direct fact extraction (What, Who, When, Where)
- **reasoning**: Requires inference or analysis (Why, How)
- **comparison**: Compares concepts within the document
- **procedural**: Asks about processes or steps

## Output Format
Return a JSON array with exactly {n} objects:
[
    {{
        "question": "Clear, specific question",
        "answer": "Complete answer based on document",
        "supporting_quote": "Exact text from document that supports the answer",
        "difficulty": "easy|medium|hard",
        "question_type": "factual|reasoning|comparison|procedural"
    }},
    ...
]

Generate diverse, high-quality questions that would effectively test a RAG system's ability to retrieve and use this document."""

    MULTI_HOP_PROMPT = """You are an expert at creating multi-hop reasoning questions.

Given these related documents, create questions that require combining information from multiple documents to answer.

## Document 1
Title: {title1}
Content:
{content1}

## Document 2
Title: {title2}
Content:
{content2}

## Requirements
1. Questions must require information from BOTH documents
2. Single-document answers should be insufficient
3. The reasoning chain should be clear

## Output Format
Return a JSON array:
[
    {{
        "question": "Question requiring both documents",
        "answer": "Complete answer combining information",
        "reasoning_chain": ["Step 1: From doc 1...", "Step 2: From doc 2...", "Conclusion:..."],
        "required_documents": ["doc1_id", "doc2_id"],
        "difficulty": "hard"
    }}
]

Generate {n} multi-hop questions."""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the generator.

        Args:
            llm_model: Model to use for generation
            api_key: API key (uses env var if not provided)
            max_retries: Number of retries on failure
        """
        self.llm_model = llm_model
        self.max_retries = max_retries
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        import os

        if "gpt" in self.llm_model.lower():
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key or os.getenv("OPENAI_API_KEY")
            )
            self._client_type = "openai"
        elif "claude" in self.llm_model.lower():
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=self._api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self._client_type = "anthropic"
        else:
            raise ValueError(f"Unsupported model: {self.llm_model}")

        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        client = self._get_client()

        if self._client_type == "openai":
            response = await client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content

        elif self._client_type == "anthropic":
            response = await client.messages.create(
                model=self.llm_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

    async def generate_from_documents(
        self,
        documents: List[Dict[str, Any]],
        n_per_doc: int = 5,
        question_types: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        dataset_name: str = "Generated Dataset",
        dataset_description: str = "Auto-generated Q&A pairs",
        language: str = "en",
        domain: str = "general",
        parallel: bool = True,
        max_concurrent: int = 5,
    ) -> GroundTruthDataset:
        """
        Generate Q&A pairs from a list of documents.

        Args:
            documents: List of dicts with 'id', 'title', 'content'
            n_per_doc: Number of Q&A pairs per document
            question_types: Types of questions to generate
            difficulties: Difficulty levels to include
            dataset_name: Name for the dataset
            dataset_description: Description
            language: Language code
            domain: Domain/topic area
            parallel: Whether to process documents in parallel
            max_concurrent: Max concurrent LLM calls

        Returns:
            GroundTruthDataset
        """
        question_types = question_types or ["factual", "reasoning"]
        difficulties = difficulties or ["easy", "medium", "hard"]

        all_pairs = []

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_doc(doc):
                async with semaphore:
                    return await self._generate_for_document(
                        doc, n_per_doc, question_types, difficulties
                    )

            results = await asyncio.gather(
                *[process_doc(doc) for doc in documents],
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Document processing failed: {result}")
                else:
                    all_pairs.extend(result)
        else:
            for doc in documents:
                try:
                    pairs = await self._generate_for_document(
                        doc, n_per_doc, question_types, difficulties
                    )
                    all_pairs.extend(pairs)
                except Exception as e:
                    logger.warning(f"Failed to process document {doc.get('id')}: {e}")

        return GroundTruthDataset(
            dataset_id=str(uuid.uuid4()),
            name=dataset_name,
            description=dataset_description,
            pairs=all_pairs,
            language=language,
            domain=domain,
        )

    async def _generate_for_document(
        self,
        document: Dict[str, Any],
        n: int,
        question_types: List[str],
        difficulties: List[str],
    ) -> List[QAPair]:
        """Generate Q&A pairs for a single document."""
        doc_id = document.get("id", str(uuid.uuid4()))
        title = document.get("title", "Untitled")
        content = document.get("content", document.get("text", ""))

        # Truncate content if too long
        max_content_len = 4000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "\n[... truncated ...]"

        prompt = self.GENERATION_PROMPT.format(
            n=n,
            title=title,
            content=content,
            question_types=", ".join(question_types),
            difficulties=", ".join(difficulties),
        )

        for attempt in range(self.max_retries):
            try:
                response = await self._call_llm(prompt)

                # Parse JSON
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                qa_list = json.loads(response.strip())

                pairs = []
                for qa in qa_list:
                    pairs.append(QAPair(
                        pair_id=str(uuid.uuid4()),
                        question=qa["question"],
                        answer=qa["answer"],
                        document_id=doc_id,
                        document_title=title,
                        supporting_quote=qa.get("supporting_quote"),
                        difficulty=qa.get("difficulty", "medium"),
                        question_type=qa.get("question_type", "factual"),
                    ))

                return pairs

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.warning(f"Generation error on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(1)

        logger.error(f"Failed to generate Q&A for document {doc_id}")
        return []

    async def generate_multi_hop(
        self,
        document_pairs: List[tuple],
        n_per_pair: int = 2,
    ) -> List[QAPair]:
        """
        Generate multi-hop questions requiring multiple documents.

        Args:
            document_pairs: List of (doc1, doc2) tuples
            n_per_pair: Number of questions per document pair

        Returns:
            List of QAPair for multi-hop questions
        """
        all_pairs = []

        for doc1, doc2 in document_pairs:
            try:
                prompt = self.MULTI_HOP_PROMPT.format(
                    n=n_per_pair,
                    title1=doc1.get("title", "Document 1"),
                    content1=doc1.get("content", "")[:2000],
                    title2=doc2.get("title", "Document 2"),
                    content2=doc2.get("content", "")[:2000],
                )

                response = await self._call_llm(prompt)

                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]

                qa_list = json.loads(response.strip())

                for qa in qa_list:
                    all_pairs.append(QAPair(
                        pair_id=str(uuid.uuid4()),
                        question=qa["question"],
                        answer=qa["answer"],
                        document_id=f"{doc1.get('id')}+{doc2.get('id')}",
                        difficulty="hard",
                        question_type="multi-hop",
                        metadata={
                            "required_documents": qa.get("required_documents", []),
                            "reasoning_chain": qa.get("reasoning_chain", []),
                        },
                    ))

            except Exception as e:
                logger.warning(f"Multi-hop generation failed: {e}")

        return all_pairs

    def filter_quality(
        self,
        pairs: List[QAPair],
        min_question_len: int = 10,
        min_answer_len: int = 20,
        require_quote: bool = False,
    ) -> List[QAPair]:
        """
        Filter Q&A pairs for quality.

        Args:
            pairs: List of QAPair to filter
            min_question_len: Minimum question length
            min_answer_len: Minimum answer length
            require_quote: Whether to require supporting quote

        Returns:
            Filtered list of QAPair
        """
        filtered = []

        for pair in pairs:
            # Length checks
            if len(pair.question) < min_question_len:
                continue
            if len(pair.answer) < min_answer_len:
                continue

            # Quote requirement
            if require_quote and not pair.supporting_quote:
                continue

            # Question should end with ?
            if not pair.question.strip().endswith("?"):
                pair.question = pair.question.strip() + "?"

            filtered.append(pair)

        logger.info(f"Filtered {len(pairs)} pairs to {len(filtered)} quality pairs")
        return filtered

    def save_dataset(
        self,
        dataset: GroundTruthDataset,
        filepath: str,
        format: str = "json",
    ) -> None:
        """
        Save dataset to file.

        Args:
            dataset: Dataset to save
            filepath: Output file path
            format: Output format ('json' or 'jsonl')
        """
        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)

        elif format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                # Write metadata first
                meta = {
                    "type": "metadata",
                    "dataset_id": dataset.dataset_id,
                    "name": dataset.name,
                    "version": dataset.version,
                    "total_pairs": len(dataset.pairs),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                # Write each pair
                for pair in dataset.pairs:
                    f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved dataset with {len(dataset.pairs)} pairs to {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> GroundTruthDataset:
        """
        Load dataset from file.

        Args:
            filepath: Path to dataset file

        Returns:
            GroundTruthDataset
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = [
            QAPair(
                pair_id=p["pair_id"],
                question=p["question"],
                answer=p["answer"],
                document_id=p["document_id"],
                document_title=p.get("document_title"),
                supporting_quote=p.get("supporting_quote"),
                difficulty=p.get("difficulty", "medium"),
                question_type=p.get("question_type", "factual"),
                metadata=p.get("metadata", {}),
            )
            for p in data.get("pairs", [])
        ]

        return GroundTruthDataset(
            dataset_id=data.get("dataset_id", str(uuid.uuid4())),
            name=data.get("name", "Loaded Dataset"),
            description=data.get("description", ""),
            pairs=pairs,
            version=data.get("version", "1.0.0"),
            language=data.get("language", "en"),
            domain=data.get("domain", "general"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            created_by=data.get("created_by", "unknown"),
        )
