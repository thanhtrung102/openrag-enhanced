#!/usr/bin/env python3
"""
Demo script for OpenRAG Enhanced evaluation features.

Run this script to see the evaluation modules in action.
"""

import asyncio
import json
from datetime import datetime


async def demo_hallucination_detection():
    """Demonstrate hallucination detection."""
    print("\n" + "="*60)
    print("HALLUCINATION DETECTION DEMO")
    print("="*60)

    # Note: In production, this would use actual models
    # This demo shows the API structure

    response = "Python was created by Guido van Rossum in 1991. It is the fastest programming language in the world."

    sources = [
        "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
        "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands.",
    ]

    print(f"\nResponse to check:\n{response}")
    print(f"\nSources:\n{json.dumps(sources, indent=2)}")

    # Simulated result (in production, this would call the actual detector)
    result = {
        "ensemble_score": 0.45,
        "is_hallucinated": False,
        "nli_score": 0.3,
        "alignment_score": 0.6,
        "flagged_claims": [
            "It is the fastest programming language in the world."
        ],
        "methods_used": ["nli", "alignment"],
    }

    print(f"\nDetection Result:")
    print(f"  Ensemble Score: {result['ensemble_score']:.2f}")
    print(f"  Is Hallucinated: {result['is_hallucinated']}")
    print(f"  Flagged Claims: {result['flagged_claims']}")


async def demo_llm_judge():
    """Demonstrate LLM-as-judge evaluation."""
    print("\n" + "="*60)
    print("LLM-AS-JUDGE EVALUATION DEMO")
    print("="*60)

    question = "What is Python used for?"
    context = [
        "Python is widely used in web development, data science, artificial intelligence, and automation.",
        "Python's simple syntax makes it popular for beginners and experts alike.",
    ]
    response = "Python is used for web development, data science, AI, and scripting tasks."

    print(f"\nQuestion: {question}")
    print(f"\nContext:\n{json.dumps(context, indent=2)}")
    print(f"\nResponse: {response}")

    # Simulated result
    result = {
        "faithfulness": 4.5,
        "answer_relevance": 4.8,
        "context_relevance": 4.2,
        "completeness": 4.0,
        "coherence": 4.7,
        "overall_score": 4.44,
        "hallucination_detected": False,
        "explanation": "Response accurately summarizes the key uses of Python from the context.",
    }

    print(f"\nEvaluation Result:")
    print(f"  Faithfulness: {result['faithfulness']}/5")
    print(f"  Answer Relevance: {result['answer_relevance']}/5")
    print(f"  Completeness: {result['completeness']}/5")
    print(f"  Coherence: {result['coherence']}/5")
    print(f"  Overall Score: {result['overall_score']:.2f}/5")


async def demo_retrieval_metrics():
    """Demonstrate retrieval evaluation."""
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION DEMO")
    print("="*60)

    # Simulated retrieval results
    results = [
        {"query": "What is Python?", "retrieved": ["doc1", "doc3", "doc5"], "relevant": "doc1"},
        {"query": "How does RAG work?", "retrieved": ["doc2", "doc1", "doc4"], "relevant": "doc2"},
        {"query": "Best LLMs for 2024?", "retrieved": ["doc5", "doc3", "doc2"], "relevant": "doc3"},
    ]

    print(f"\nEvaluating {len(results)} queries...")

    # Simulated metrics
    metrics = {
        "hit_rate": {1: 0.67, 3: 1.0, 5: 1.0},
        "mrr": 0.72,
        "ndcg": {5: 0.85, 10: 0.88},
        "latency_p95_ms": 125,
    }

    print(f"\nRetrieval Metrics:")
    print(f"  Hit Rate @1: {metrics['hit_rate'][1]:.1%}")
    print(f"  Hit Rate @3: {metrics['hit_rate'][3]:.1%}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  NDCG @5: {metrics['ndcg'][5]:.3f}")
    print(f"  Latency P95: {metrics['latency_p95_ms']}ms")


async def demo_ground_truth_generation():
    """Demonstrate ground truth generation."""
    print("\n" + "="*60)
    print("GROUND TRUTH GENERATION DEMO")
    print("="*60)

    document = {
        "id": "doc1",
        "title": "Introduction to Python",
        "content": "Python is a high-level programming language known for its simple syntax and versatility. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    }

    print(f"\nDocument: {document['title']}")
    print(f"Content: {document['content'][:100]}...")

    # Simulated generated Q&A pairs
    qa_pairs = [
        {
            "question": "Who created Python?",
            "answer": "Guido van Rossum",
            "difficulty": "easy",
            "type": "factual",
        },
        {
            "question": "What programming paradigms does Python support?",
            "answer": "Python supports procedural, object-oriented, and functional programming paradigms.",
            "difficulty": "medium",
            "type": "factual",
        },
        {
            "question": "Why is Python considered beginner-friendly?",
            "answer": "Python is considered beginner-friendly because of its simple syntax.",
            "difficulty": "medium",
            "type": "reasoning",
        },
    ]

    print(f"\nGenerated Q&A Pairs:")
    for i, pair in enumerate(qa_pairs, 1):
        print(f"\n  {i}. [{pair['difficulty']}] [{pair['type']}]")
        print(f"     Q: {pair['question']}")
        print(f"     A: {pair['answer']}")


async def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# OpenRAG Enhanced - Evaluation Demo")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    await demo_hallucination_detection()
    await demo_llm_judge()
    await demo_retrieval_metrics()
    await demo_ground_truth_generation()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo run with actual models, start the Docker stack:")
    print("  docker-compose up -d")
    print("\nThen access:")
    print("  API: http://localhost:8000/docs")
    print("  Dashboard: http://localhost:8501")
    print()


if __name__ == "__main__":
    asyncio.run(main())
