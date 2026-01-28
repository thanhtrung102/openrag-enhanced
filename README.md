# OpenRAG Contributions: Enhanced RAG Capabilities

> **Proposed contributions to [linagora/openrag](https://github.com/linagora/openrag)**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains proposed contributions to Linagora's OpenRAG framework, addressing gaps identified in the current evaluation pipeline:

| Gap in OpenRAG | Our Contribution | PR |
|----------------|------------------|-----|
| No explicit hallucination detection | Multi-method hallucination detector | PR #1 |
| Evaluation results not persisted | PostgreSQL storage | PR #2 |
| No visual metrics dashboard | Streamlit evaluation UI | PR #3 |
| No Graph RAG | Neo4j knowledge graph integration | PR #4 |
| No Agentic RAG | Multi-step workflows with tools | PR #5 |

### What OpenRAG Already Has (Not Duplicated)

- Retrieval metrics (Hit Rate, MRR, NDCG, Recall)
- LLM-as-Judge (Completion & Precision scoring)
- Ground truth Q&A generation
- System monitoring endpoints

---

## Project Structure

```
openrag-enhanced/
├── openrag/                          # Files for linagora/openrag PRs
│   ├── components/
│   │   ├── hallucination/            # PR #1: Hallucination Detection
│   │   │   ├── __init__.py
│   │   │   ├── detector.py
│   │   │   ├── nli_checker.py
│   │   │   └── alignment_checker.py
│   │   ├── graph/                    # PR #4: Graph RAG
│   │   │   ├── __init__.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── graph_store.py
│   │   │   └── graph_retriever.py
│   │   └── agents/                   # PR #5: Agentic RAG
│   │       ├── __init__.py
│   │       ├── orchestrator.py
│   │       ├── planner.py
│   │       └── tools.py
│   ├── storage/                      # PR #2: Metrics Persistence
│   │   └── evaluation_store.py
│   ├── routers/
│   │   └── hallucination.py          # PR #1: API endpoints
│   └── ui/
│       └── evaluation_dashboard/     # PR #3: Dashboard
│           └── app.py
├── automatic-evaluation-pipeline/
│   └── benchmark_enhanced.py         # PR #2: Benchmark integration
├── sql/
│   └── init_evaluation.sql           # PR #2: PostgreSQL schema
├── tests/                            # All unit tests
│   ├── test_hallucination.py         # PR #1 tests
│   ├── test_graph_rag.py             # PR #4 tests
│   └── test_agentic_rag.py           # PR #5 tests
├── requirements.txt
└── requirements-eval.txt
```

---

## PR #1: Hallucination Detection

**Location:** `openrag/components/hallucination/`

```python
from openrag.components.hallucination import HallucinationDetector

detector = HallucinationDetector()
result = await detector.detect(
    response="Python was created by Guido van Rossum in 1991.",
    sources=["Python is a programming language created by Guido van Rossum..."]
)

if result.is_hallucinated:
    print(f"Flagged claims: {result.flagged_claims}")
```

**API Endpoint:**
```bash
POST /api/v1/hallucination/detect
{
    "response": "...",
    "sources": ["..."],
    "methods": ["nli", "alignment"],
    "threshold": 0.5
}
```

---

## PR #2: Evaluation Persistence

**Location:** `openrag/storage/evaluation_store.py`

```python
from openrag.storage import EvaluationStore

store = EvaluationStore(connection_string="postgresql://...")
await store.save_evaluation(
    query="What is Python?",
    response="Python is...",
    sources=sources,
    metrics={"hit_rate": 0.85, "hallucination_score": 0.15}
)
```

---

## PR #3: Evaluation Dashboard

**Location:** `openrag/ui/evaluation_dashboard/`

```bash
streamlit run openrag/ui/evaluation_dashboard/app.py
```

Features:
- Real-time hallucination rate monitoring
- Model comparison charts
- Historical trend analysis

---

## PR #4: Graph RAG

**Location:** `openrag/components/graph/`

Knowledge graph-enhanced RAG using Neo4j for entity storage and graph-based retrieval.

```python
from openrag.components.graph import EntityExtractor, GraphStore, GraphRetriever, RetrievalMode

# Extract entities from documents
extractor = EntityExtractor()
result = extractor.extract(
    text="Python was created by Guido van Rossum at CWI.",
    document_id="doc_123"
)

# Store in Neo4j
store = GraphStore(uri="bolt://localhost:7687")
await store.connect()
await store.add_extraction_result(result)

# Graph-enhanced retrieval
retriever = GraphRetriever(
    graph_store=store,
    entity_extractor=extractor,
)
results = await retriever.retrieve(
    query="Who created Python?",
    mode=RetrievalMode.GRAPH_ENHANCED,
)
```

Features:
- Entity extraction with spaCy NER and custom patterns
- Relationship extraction using dependency parsing
- Neo4j-based knowledge graph storage
- Hybrid vector + graph retrieval
- Multi-hop graph traversal for context

---

## PR #5: Agentic RAG

**Location:** `openrag/components/agents/`

Multi-step RAG workflows with planning, tool calling, and iterative refinement.

```python
from openrag.components.agents import (
    AgenticRAGOrchestrator,
    QueryPlanner,
    CalculatorTool,
    WebSearchTool,
)

# Create tools
tools = [CalculatorTool(), WebSearchTool()]

# Create planner
planner = QueryPlanner(llm_client=llm)

# Create orchestrator
orchestrator = AgenticRAGOrchestrator(
    llm_client=llm,
    retriever=retriever,
    planner=planner,
    tools=tools,
)

# Run complex query
result = await orchestrator.run(
    query="Compare Python and Rust performance for web servers"
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Steps taken: {len(result.steps)}")
```

Features:
- Query complexity analysis and planning
- Multi-step execution with intermediate reasoning
- Built-in tools: Calculator, Web Search, Code Executor, DateTime
- Self-reflection and answer refinement
- Execution tracing for debugging

---

## Local Development

```bash
# Clone
git clone https://github.com/thanhtrung102/openrag-enhanced.git
cd openrag-enhanced

# Install
pip install -r requirements.txt
pip install -r requirements-eval.txt
```

---

## Testing

All tests are in the `tests/` directory. Run with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=openrag --cov-report=html
```

### Test by PR

| PR | Test Command |
|----|--------------|
| PR #1: Hallucination Detection | `pytest tests/test_hallucination.py -v` |
| PR #2: Evaluation Persistence | Requires PostgreSQL (see setup below) |
| PR #3: Evaluation Dashboard | `streamlit run openrag/ui/evaluation_dashboard/app.py` |
| PR #4: Graph RAG | `pytest tests/test_graph_rag.py -v` |
| PR #5: Agentic RAG | `pytest tests/test_agentic_rag.py -v` |

### PostgreSQL Setup (PR #2)

```bash
# Start PostgreSQL with Docker
docker run -d --name postgres \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=openrag_eval \
    -p 5432:5432 \
    postgres:15

# Initialize schema
psql -h localhost -U postgres -d openrag_eval -f sql/init_evaluation.sql
```

### Neo4j Setup (PR #4)

```bash
# Start Neo4j with Docker
docker run -d --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

---

## Integration with OpenRAG

```python
# In openrag/api.py
from openrag.routers.hallucination import add_hallucination_routes
add_hallucination_routes(app)

# In RAG pipeline - Hallucination Detection
from openrag.components.hallucination import HallucinationDetector
detector = HallucinationDetector()
result = await detector.detect(response, sources)

# In RAG pipeline - Graph RAG
from openrag.components.graph import GraphRetriever, RetrievalMode
retriever = GraphRetriever(graph_store, entity_extractor, vector_store)
results = await retriever.retrieve(query, mode=RetrievalMode.HYBRID)

# In RAG pipeline - Agentic RAG
from openrag.components.agents import AgenticRAGOrchestrator
orchestrator = AgenticRAGOrchestrator(llm, retriever, planner, tools)
result = await orchestrator.run(complex_query)
```

---

## License

AGPL-3.0 - Aligned with [linagora/openrag](https://github.com/linagora/openrag)

---

**Linagora AI Department Internship Project** | thanhtrung102
