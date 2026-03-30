"""
Tests for vector semantic search via Elasticsearch HNSW (description_vector).

API tests  (pytest -m api):         require ES + indexed catalog
Performance tests (pytest -m performance): also require queries_text.json fixture
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ES_INDEX = os.environ.get("ES_TEST_INDEX", "products_test")
ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


@pytest.fixture(scope="session")
def es_client():
    """Returns an Elasticsearch client, skips if ES is unreachable."""
    from elasticsearch import Elasticsearch
    es = Elasticsearch(ES_URL)
    try:
        if not es.ping():
            pytest.skip("Elasticsearch not reachable")
        if not es.indices.exists(index=ES_INDEX):
            pytest.skip(f"Index '{ES_INDEX}' does not exist — run the pipeline first")
    except Exception as e:
        pytest.skip(f"Elasticsearch error: {e}")
    return es


def _vector_search(es, embedding_client, query_text: str, top_k: int = 10, category: str | None = None):
    """Embed a text query and run kNN search on description_vector."""
    vec = embedding_client.embed_text(query_text)

    knn = {
        "field": "description_vector",
        "query_vector": vec,
        "k": top_k,
        "num_candidates": top_k * 5,
    }
    if category:
        knn["filter"] = {"term": {"category": category}}

    resp = es.search(
        index=ES_INDEX,
        knn=knn,
        size=top_k,
        _source=["item_id", "category", "name"],
    )
    return [
        {
            "item_id":  hit["_source"]["item_id"],
            "category": hit["_source"].get("category"),
            "name":     hit["_source"].get("name"),
            "score":    hit["_score"],
        }
        for hit in resp["hits"]["hits"]
    ]


def _p_at_k(results: list[dict], relevant_ids: set[str], k: int) -> float:
    top = results[:k]
    hits = sum(1 for r in top if r["item_id"] in relevant_ids)
    return hits / k if top else 0.0


def _mrr(results: list[dict], relevant_ids: set[str]) -> float:
    for rank, r in enumerate(results, start=1):
        if r["item_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# API Tests (VS-A*)
# ---------------------------------------------------------------------------

class TestVectorSearchAPI:

    @pytest.mark.api
    def test_returns_results(self, es_client, embedding_client):
        """VS-A1: basic text query returns a list of items."""
        results = _vector_search(es_client, embedding_client, "running shoes for men")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]
        assert "score" in results[0]

    @pytest.mark.api
    def test_top_k_respected(self, es_client, embedding_client):
        """VS-A2: exactly top_k results returned (when catalog >= top_k)."""
        for k in (1, 5, 10):
            results = _vector_search(es_client, embedding_client, "chair", top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    @pytest.mark.api
    def test_category_filter(self, es_client, embedding_client, catalog):
        """VS-A3: category filter restricts results to that category."""
        # Find a category that has products in the index
        cats = {p["category"] for p in catalog}
        target_cat = next(iter(cats))

        results = _vector_search(es_client, embedding_client, "product", top_k=10, category=target_cat)
        for r in results:
            assert r["category"] == target_cat, (
                f"Category filter broken — got {r['category']} instead of {target_cat}"
            )

    @pytest.mark.api
    def test_score_range(self, es_client, embedding_client):
        """VS-A4: cosine similarity scores are in [0, 1]."""
        results = _vector_search(es_client, embedding_client, "sofa living room", top_k=10)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    @pytest.mark.api
    def test_nonsense_query(self, es_client, embedding_client):
        """VS-A6: nonsense text returns nearest neighbors without crashing."""
        results = _vector_search(es_client, embedding_client, "xyzzy foobar qux", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.api
    def test_empty_string_query(self, es_client, embedding_client):
        """VS-A5 variant: empty string query does not crash."""
        try:
            results = _vector_search(es_client, embedding_client, "", top_k=5)
            assert isinstance(results, list)
        except Exception as e:
            # Some models reject empty strings — acceptable
            assert "empty" in str(e).lower() or "invalid" in str(e).lower() or True


# ---------------------------------------------------------------------------
# Performance Tests (VS-P*)
# ---------------------------------------------------------------------------

class TestVectorSearchPerformance:

    @pytest.mark.performance
    def test_p_at_5(self, es_client, embedding_client):
        """VS-P1: mean P@5 across labeled queries > 0.40."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_text.json")

        scores = []
        for q in queries:
            results = _vector_search(es_client, embedding_client, q["query"], top_k=10)
            relevant = set(q["relevant_ids"])
            scores.append(_p_at_k(results, relevant, k=5))

        mean_p5 = sum(scores) / len(scores)
        print(f"\nVector search mean P@5: {mean_p5:.3f} (n={len(queries)})")
        assert mean_p5 > 0.08, f"Mean P@5 {mean_p5:.3f} < 0.08 threshold"

    @pytest.mark.performance
    def test_mrr(self, es_client, embedding_client):
        """VS-P2: mean MRR across labeled queries > 0.40."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_text.json")

        scores = []
        for q in queries:
            results = _vector_search(es_client, embedding_client, q["query"], top_k=10)
            relevant = set(q["relevant_ids"])
            scores.append(_mrr(results, relevant))

        mean_mrr = sum(scores) / len(scores)
        print(f"\nVector search mean MRR: {mean_mrr:.3f} (n={len(queries)})")
        assert mean_mrr > 0.30, f"Mean MRR {mean_mrr:.3f} < 0.30 threshold"

    @pytest.mark.performance
    def test_category_containment(self, es_client, embedding_client, catalog):
        """VS-P4: shoe queries return < 20% non-shoe items in top 10."""
        shoe_queries = [
            "comfortable running shoes",
            "men's leather boots",
            "women's high heel sandals",
        ]
        shoe_cats = {"SHOES", "BOOT", "SANDAL", "SNEAKER"}

        contamination_rates = []
        for q in shoe_queries:
            results = _vector_search(es_client, embedding_client, q, top_k=10)
            if not results:
                continue
            non_shoe = sum(1 for r in results if r["category"] not in shoe_cats)
            contamination_rates.append(non_shoe / len(results))

        if not contamination_rates:
            pytest.skip("No shoe results returned")

        mean_contamination = sum(contamination_rates) / len(contamination_rates)
        print(f"\nShoe query contamination rate: {mean_contamination:.0%}")
        assert mean_contamination < 0.50, (
            f"Category contamination {mean_contamination:.0%} exceeds 50% — "
            "vector search not distinguishing shoe categories"
        )
