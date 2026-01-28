# OpenRAG Contributions: Hallucination Detection & Evaluation Persistence

> **Proposed contributions to [linagora/openrag](https://github.com/linagora/openrag)**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains proposed contributions to Linagora's OpenRAG framework, addressing gaps identified in the current evaluation pipeline:

| Gap in OpenRAG | Our Contribution |
|----------------|------------------|
| No explicit hallucination detection | **NEW:** Multi-method hallucination detector |
| Evaluation results not persisted | **NEW:** PostgreSQL storage + dashboard |
| No visual metrics dashboard | **NEW:** Streamlit evaluation UI |

### What OpenRAG Already Has (Not Duplicated)

- ✅ Retrieval metrics (Hit Rate, MRR, NDCG, Recall)
- ✅ LLM-as-Judge (Completion & Precision scoring)
- ✅ Ground truth Q&A generation
- ✅ System monitoring endpoints

---

## Proposed Contributions

### 1. Hallucination Detection Module

**Location:** `openrag/components/hallucination/`

Multi-method hallucination detection that OpenRAG currently lacks:

```python
from openrag.components.hallucination import HallucinationDetector

detector = HallucinationDetector()
result = await detector.detect(
    response="Python was created by Guido van Rossum in 1991.",
    sources=["Python is a programming language created by Guido van Rossum..."]
)

if result.is_hallucinated:
    print(f"Warning! Flagged claims: {result.flagged_claims}")
    print(f"Hallucination score: {result.ensemble_score:.2f}")
```

**Methods:**
- **NLI-based:** Uses natural language inference to check if claims are entailed by sources
- **Alignment:** Measures semantic similarity between response and source documents
- **Ensemble:** Combines multiple methods for robust detection

**API Endpoint:**
```bash
POST /api/v1/hallucination/detect
{
    "response": "Python is the fastest language in the world.",
    "sources": ["Python is a high-level programming language..."],
    "methods": ["nli", "alignment"],
    "threshold": 0.5
}
```

### 2. Evaluation Persistence

**Location:** `openrag/storage/evaluation_store.py`

Persistent storage for evaluation metrics (currently OpenRAG only prints results):

```python
from openrag.storage import EvaluationStore

store = EvaluationStore(connection_string="postgresql://...")

# Save evaluation results
await store.save_evaluation(
    query="What is Python?",
    response="Python is a programming language...",
    sources=sources,
    metrics={
        "hit_rate": 0.85,
        "mrr": 0.72,
        "hallucination_score": 0.15,
    }
)

# Query historical results
summary = await store.get_summary(
    start_date="2024-01-01",
    group_by="model_name"
)
```

### 3. Evaluation Dashboard

**Location:** `openrag/ui/evaluation_dashboard/`

Streamlit dashboard for visualizing evaluation metrics:

- Real-time hallucination rate monitoring
- Model comparison charts
- Historical trend analysis
- Query-level drill-down

---

## Project Structure (For PR)

```
openrag/                              # Files to add to linagora/openrag
├── components/
│   └── hallucination/               # PR #1: Hallucination Detection
│       ├── __init__.py
│       ├── detector.py              # Main HallucinationDetector class
│       ├── nli_checker.py           # NLI-based detection
│       └── alignment_checker.py     # Semantic alignment detection
├── storage/
│   └── evaluation_store.py          # PR #2: Metrics Persistence
├── routers/
│   └── hallucination.py             # PR #1: API endpoints
└── ui/
    └── evaluation_dashboard/        # PR #3: Streamlit Dashboard
        └── app.py

sql/
└── evaluation_schema.sql            # PR #2: PostgreSQL schema

# Supporting files (standalone testing)
src/                                 # Original standalone implementation
docker-compose.yml                   # Full stack for testing
```

---

## Pull Request Plan

### PR #1: Hallucination Detection
```
Title: feat(evaluation): Add hallucination detection module

Files:
- openrag/components/hallucination/__init__.py
- openrag/components/hallucination/detector.py
- openrag/components/hallucination/nli_checker.py
- openrag/components/hallucination/alignment_checker.py
- openrag/routers/hallucination.py
- tests/test_hallucination.py
```

### PR #2: Evaluation Persistence
```
Title: feat(evaluation): Add metrics persistence with PostgreSQL

Files:
- openrag/storage/evaluation_store.py
- sql/evaluation_schema.sql
- automatic-evaluation-pipeline/benchmark.py (modify)
```

### PR #3: Evaluation Dashboard
```
Title: feat(ui): Add Streamlit evaluation dashboard

Files:
- openrag/ui/evaluation_dashboard/app.py
- openrag/ui/evaluation_dashboard/requirements.txt
```

---

## Local Development & Testing

```bash
# Clone this repository
git clone https://github.com/thanhtrung102/openrag-enhanced.git
cd openrag-enhanced

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-eval.txt

# Run tests
pytest tests/ -v

# Start full stack (for integration testing)
docker-compose up -d

# Access services
# - API: http://localhost:8000/docs
# - Dashboard: http://localhost:8501
```

---

## Integration with OpenRAG

To add hallucination detection to existing OpenRAG installation:

```python
# In openrag/api.py
from openrag.routers.hallucination import add_hallucination_routes

app = FastAPI(...)

# Add hallucination detection routes
add_hallucination_routes(app)
```

To integrate with RAG pipeline:

```python
# In openrag/components/pipeline.py
from openrag.components.hallucination import HallucinationDetector

detector = HallucinationDetector()

async def generate_with_detection(query: str, context: List[str]):
    response = await llm.generate(query, context)

    # Check for hallucinations
    detection = await detector.detect(response, context)

    return {
        "response": response,
        "hallucination_score": detection.ensemble_score,
        "is_hallucinated": detection.is_hallucinated,
        "flagged_claims": detection.flagged_claims,
    }
```

---

## Alignment with OpenRAG Roadmap

From OpenRAG's README, upcoming features include:
- **Tool Calling** → Future contribution
- **Agentic RAG** → Future contribution
- **MCP Integration** → Future contribution

Our hallucination detection directly supports quality assurance for these advanced features.

---

## License

AGPL-3.0 - Aligned with [linagora/openrag](https://github.com/linagora/openrag)

## Acknowledgments

- [Linagora OpenRAG](https://github.com/linagora/openrag) - Base RAG framework
- [Linagora AI Department](https://linagora.com) - Internship opportunity
- [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) - Project format

---

**Linagora AI Department Internship Project** | thanhtrung102
