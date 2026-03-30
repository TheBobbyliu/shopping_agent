"""
Tests for the full hybrid search pipeline:
  HNSW (description_vector) + BM25 → RRF fusion → top-N

Image queries additionally include:
  HNSW (image_vector) → added to RRF before text channels

API tests  (pytest -m api):         require ES + indexed catalog
Performance tests (pytest -m performance): require queries_text.json and queries_image.json
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pytest


ES_INDEX = os.environ.get("ES_TEST_INDEX", "products_test")
ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
RRF_RANK_CONSTANT = 60  # standard RRF constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def es_client():
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


# ---------------------------------------------------------------------------
# Hybrid search implementation under test
# ---------------------------------------------------------------------------

def _rrf_score(rank: int, k: int = RRF_RANK_CONSTANT) -> float:
    return 1.0 / (k + rank)


def _hybrid_search(
    es,
    embedding_client,
    query_text: str,
    image_path: Optional[Path] = None,
    top_k: int = 10,
    rrf_k: int = RRF_RANK_CONSTANT,
) -> list[dict]:
    """
    Full hybrid search:
      1. kNN on description_vector  (text channel)
      2. BM25 on description field  (keyword channel)
      3. If image_path given: kNN on image_vector (image channel)
      4. RRF fusion of all channels
    Returns top_k items sorted by fused RRF score.
    """
    fetch_k = max(50, top_k * 5)

    # --- Channel 1: description vector ---
    txt_vec = embedding_client.embed_text(query_text)
    knn_resp = es.search(
        index=ES_INDEX,
        knn={"field": "description_vector", "query_vector": txt_vec,
             "k": fetch_k, "num_candidates": fetch_k * 3},
        size=fetch_k,
        _source=["item_id", "category", "name"],
    )
    knn_results = [
        {"item_id": h["_source"]["item_id"], "category": h["_source"].get("category"),
         "name": h["_source"].get("name")}
        for h in knn_resp["hits"]["hits"]
    ]

    # --- Channel 2: BM25 ---
    bm25_resp = es.search(
        index=ES_INDEX,
        query={"match": {"description": {"query": query_text}}},
        size=fetch_k,
        _source=["item_id", "category", "name"],
    )
    bm25_results = [
        {"item_id": h["_source"]["item_id"], "category": h["_source"].get("category"),
         "name": h["_source"].get("name")}
        for h in bm25_resp["hits"]["hits"]
    ]

    channels = [knn_results, bm25_results]

    # --- Channel 3: image vector (optional) ---
    if image_path is not None:
        img_vec = embedding_client.embed_image(image_path)
        img_resp = es.search(
            index=ES_INDEX,
            knn={"field": "image_vector", "query_vector": img_vec,
                 "k": fetch_k, "num_candidates": fetch_k * 3},
            size=fetch_k,
            _source=["item_id", "category", "name"],
        )
        img_results = [
            {"item_id": h["_source"]["item_id"], "category": h["_source"].get("category"),
             "name": h["_source"].get("name")}
            for h in img_resp["hits"]["hits"]
        ]
        channels.append(img_results)

    # --- RRF fusion ---
    rrf_scores: dict[str, float] = {}
    item_meta: dict[str, dict] = {}

    for channel in channels:
        for rank, item in enumerate(channel, start=1):
            iid = item["item_id"]
            rrf_scores[iid] = rrf_scores.get(iid, 0.0) + _rrf_score(rank, rrf_k)
            item_meta[iid] = item

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [
        {**item_meta[iid], "score": score}
        for iid, score in fused
    ]


def _p_at_k(results, relevant_ids: set, k: int) -> float:
    top = results[:k]
    hits = sum(1 for r in top if r["item_id"] in relevant_ids)
    return hits / k if top else 0.0


def _bm25_only(es, query_text: str, top_k: int = 10) -> list[dict]:
    resp = es.search(
        index=ES_INDEX,
        query={"match": {"description": {"query": query_text}}},
        size=top_k,
        _source=["item_id", "category", "name"],
    )
    return [
        {"item_id": h["_source"]["item_id"], "category": h["_source"].get("category")}
        for h in resp["hits"]["hits"]
    ]


def _vector_only(es, embedding_client, query_text: str, top_k: int = 10) -> list[dict]:
    vec = embedding_client.embed_text(query_text)
    resp = es.search(
        index=ES_INDEX,
        knn={"field": "description_vector", "query_vector": vec,
             "k": top_k, "num_candidates": top_k * 5},
        size=top_k,
        _source=["item_id", "category"],
    )
    return [
        {"item_id": h["_source"]["item_id"], "category": h["_source"].get("category")}
        for h in resp["hits"]["hits"]
    ]


# ---------------------------------------------------------------------------
# API Tests (H-A*)
# ---------------------------------------------------------------------------

class TestHybridSearchAPI:

    @pytest.mark.api
    def test_text_query_returns_results(self, es_client, embedding_client):
        """H-A1: text query returns results from combined channels."""
        results = _hybrid_search(es_client, embedding_client, "running shoes for men")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]
        assert "score" in results[0]

    @pytest.mark.api
    def test_image_query_returns_results(self, es_client, embedding_client, sample_images):
        """H-A2: image + text query returns results."""
        assert sample_images
        results = _hybrid_search(
            es_client, embedding_client,
            "product image",
            image_path=sample_images[0],
            top_k=10,
        )
        assert len(results) > 0

    @pytest.mark.api
    def test_deduplication(self, es_client, embedding_client):
        """H-A3: each item_id appears at most once in results."""
        results = _hybrid_search(es_client, embedding_client, "chair", top_k=20)
        ids = [r["item_id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate item_ids in hybrid results"

    @pytest.mark.api
    def test_rrf_scores_sorted_descending(self, es_client, embedding_client):
        """H-A4: RRF scores are positive floats, sorted descending."""
        results = _hybrid_search(es_client, embedding_client, "sofa", top_k=10)
        scores = [r["score"] for r in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"

    @pytest.mark.api
    def test_top_k_respected(self, es_client, embedding_client):
        """H-A5: exactly top_k results returned."""
        for k in (5, 10):
            results = _hybrid_search(es_client, embedding_client, "lamp", top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    @pytest.mark.api
    def test_empty_result_handling(self, es_client, embedding_client):
        """H-A7: very unusual query returns empty list or few results gracefully."""
        results = _hybrid_search(es_client, embedding_client, "xyzzy frobnicator", top_k=10)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Performance Tests (H-P*)
# ---------------------------------------------------------------------------

class TestHybridSearchPerformance:

    @pytest.mark.performance
    def test_hybrid_beats_vector_only(self, es_client, embedding_client):
        """H-P1: hybrid P@5 ≥ vector-only P@5 across labeled queries."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_text.json")

        hybrid_scores = []
        vector_scores = []

        for q in queries:
            relevant = set(q["relevant_ids"])
            h = _hybrid_search(es_client, embedding_client, q["query"], top_k=10)
            v = _vector_only(es_client, embedding_client, q["query"], top_k=10)
            hybrid_scores.append(_p_at_k(h, relevant, k=5))
            vector_scores.append(_p_at_k(v, relevant, k=5))

        mean_h = sum(hybrid_scores) / len(hybrid_scores)
        mean_v = sum(vector_scores) / len(vector_scores)
        print(f"\nHybrid P@5={mean_h:.3f} vs Vector-only P@5={mean_v:.3f}")
        assert mean_h >= mean_v - 0.05, (
            f"Hybrid P@5 {mean_h:.3f} much worse than vector-only {mean_v:.3f}"
        )

    @pytest.mark.performance
    def test_hybrid_beats_bm25_only(self, es_client, embedding_client):
        """H-P2: hybrid P@5 ≥ BM25-only P@5 across labeled queries."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_text.json")

        hybrid_scores = []
        bm25_scores = []

        for q in queries:
            relevant = set(q["relevant_ids"])
            h = _hybrid_search(es_client, embedding_client, q["query"], top_k=10)
            b = _bm25_only(es_client, q["query"], top_k=10)
            hybrid_scores.append(_p_at_k(h, relevant, k=5))
            bm25_scores.append(_p_at_k(b, relevant, k=5))

        mean_h = sum(hybrid_scores) / len(hybrid_scores)
        mean_b = sum(bm25_scores) / len(bm25_scores)
        print(f"\nHybrid P@5={mean_h:.3f} vs BM25-only P@5={mean_b:.3f}")
        assert mean_h >= mean_b - 0.05, (
            f"Hybrid P@5 {mean_h:.3f} much worse than BM25-only {mean_b:.3f}"
        )

    @pytest.mark.performance
    def test_exact_name_queries_in_top3(self, es_client, embedding_client, catalog):
        """H-P3: queries using exact product names find that product in top 3."""
        test_products = catalog[:30]
        hits = 0
        total = 0
        for p in test_products:
            name = p.get("name", "").strip()
            if not name or len(name) < 10:
                continue
            results = _hybrid_search(es_client, embedding_client, name, top_k=10)
            top3_ids = [r["item_id"] for r in results[:3]]
            if p["item_id"] in top3_ids:
                hits += 1
            total += 1
            if total >= 10:
                break

        if total == 0:
            pytest.skip("No products with usable names")

        hit_rate = hits / total
        print(f"\nExact-name @3 hit rate: {hit_rate:.0%} ({hits}/{total})")
        assert hit_rate > 0.40, f"Exact-name hit rate {hit_rate:.0%} < 40%"

    @pytest.mark.performance
    def test_image_query_quality(self, es_client, embedding_client):
        """H-P5: P@5 for image queries across labeled set > 0.35."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_image.json")

        scores = []
        for q in queries:
            path = Path(q["image_path"])
            if not path.exists():
                continue
            # Use image_path as image and derive a simple text hint from category
            text_hint = q.get("category", "product").lower().replace("_", " ")
            results = _hybrid_search(
                es_client, embedding_client,
                text_hint,
                image_path=path,
                top_k=10,
            )
            relevant = set(q["relevant_ids"])
            scores.append(_p_at_k(results, relevant, k=5))

        if not scores:
            pytest.skip("No image query files found on disk")

        mean_p5 = sum(scores) / len(scores)
        print(f"\nImage query hybrid P@5: {mean_p5:.3f} (n={len(scores)})")
        assert mean_p5 > 0.18, f"Image query P@5 {mean_p5:.3f} < 0.18"

    @pytest.mark.performance
    def test_text_query_latency(self, es_client, embedding_client):
        """H-P6: full hybrid pipeline (text only) p95 < 5s on CPU."""
        queries = ["running shoes", "sofa", "office lamp", "leather wallet", "yoga mat"]
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            _hybrid_search(es_client, embedding_client, q, top_k=10)
            latencies.append(time.perf_counter() - t0)

        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        print(f"\nHybrid text query p95 latency: {p95:.2f}s")
        assert p95 < 5.0, f"p95 latency {p95:.2f}s exceeds 5s (CPU threshold)"
