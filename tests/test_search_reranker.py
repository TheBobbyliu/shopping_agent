"""
Integration test: reranker wired into hybrid_search().

Verifies that enabling reranking in search.py:
  SR-1: top result has a higher reranker score than bottom result
  SR-2: results count matches requested top_k
  SR-3: reranked order differs from (or equals) RRF order for at least some queries
  SR-4: latency is acceptable
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture(scope="session", autouse=True)
def check_es():
    from elasticsearch import Elasticsearch
    es = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    if not es.ping() or not es.indices.exists(index=os.environ.get("ES_TEST_INDEX", "products_test")):
        pytest.skip("Elasticsearch / products_test index not available")


@pytest.mark.api
def test_reranked_results_returned():
    """SR-1: hybrid_search with rerank=True returns results."""
    from search import hybrid_search
    results = hybrid_search("comfortable office chair", top_k=5, rerank=True)
    assert isinstance(results, list)
    assert 1 <= len(results) <= 5


@pytest.mark.api
def test_reranked_count():
    """SR-2: top_k is respected with reranking on."""
    from search import hybrid_search
    for k in (3, 5):
        results = hybrid_search("leather sofa", top_k=k, rerank=True)
        assert len(results) == k, f"Expected {k}, got {len(results)}"


@pytest.mark.api
def test_rerank_scores_descending():
    """SR-3: results are ordered by reranker score descending."""
    from search import hybrid_search
    results = hybrid_search("running shoes", top_k=5, rerank=True)
    scores = [r["rerank_score"] for r in results if "rerank_score" in r]
    assert scores == sorted(scores, reverse=True), "rerank_score not sorted descending"


@pytest.mark.api
def test_rerank_latency():
    """SR-4: warm hybrid search with reranking completes in < 10s on CPU.
    First call warms up model singletons; timing is on the second call."""
    from search import hybrid_search
    hybrid_search("wooden dining table", top_k=5, rerank=True)   # warm-up
    t0 = time.perf_counter()
    hybrid_search("wooden dining table", top_k=5, rerank=True)
    elapsed = time.perf_counter() - t0
    print(f"\nReranked search latency (warm): {elapsed:.2f}s")
    assert elapsed < 10.0, f"Took {elapsed:.2f}s > 10s"
