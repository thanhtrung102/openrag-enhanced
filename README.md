# OpenRAG Contributions: Hallucination Detection & Evaluation

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
| No Graph RAG | Neo4j integration | PR #4 (planned) |
| No Agentic RAG | Multi-step workflows | PR #5 (planned) |

### What OpenRAG Already Has (Not Duplicated)

- ✅ Retrieval metrics (Hit Rate, MRR, NDCG, Recall)
- ✅ LLM-as-Judge (Completion & Precision scoring)
- ✅ Ground truth Q&A generation
- ✅ System monitoring endpoints

---

## Project Structure

```
openrag-enhanced/
├── openrag/                          # Files for linagora/openrag PRs
│   ├── components/
│   │   └── hallucination/            # PR #1: Hallucination Detection
│   │       ├── __init__.py
│   │       ├── detector.py
│   │       ├── nli_checker.py
│   │       └── alignment_checker.py
│   ├── storage/                      # PR #2: Metrics Persistence
│   │   └── evaluation_store.py
│   ├── routers/
│   │   └── hallucination.py          # PR #1: API endpoints
│   └── ui/
│       └── evaluation_dashboard/     # PR #3: Dashboard
│           ├── app.py
│           └── requirements.txt
├── automatic-evaluation-pipeline/
│   └── benchmark_enhanced.py         # PR #2: Benchmark integration
├── sql/
│   └── init_evaluation.sql           # PR #2: PostgreSQL schema
├── tests/
│   └── test_hallucination.py         # PR #1: Unit tests
├── scripts/
│   └── run_tests.py                  # Test runner
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

## Local Development

```bash
# Clone
git clone https://github.com/thanhtrung102/openrag-enhanced.git
cd openrag-enhanced

# Install
pip install -r requirements.txt
pip install -r requirements-eval.txt

# Test
python scripts/run_tests.py --quick

# Dashboard (demo mode)
streamlit run openrag/ui/evaluation_dashboard/app.py
```

---

## Integration with OpenRAG

```python
# In openrag/api.py
from openrag.routers.hallucination import add_hallucination_routes
add_hallucination_routes(app)

# In RAG pipeline
from openrag.components.hallucination import HallucinationDetector
detector = HallucinationDetector()
result = await detector.detect(response, sources)
```

---

## License

AGPL-3.0 - Aligned with [linagora/openrag](https://github.com/linagora/openrag)

---

**Linagora AI Department Internship Project** | thanhtrung102
