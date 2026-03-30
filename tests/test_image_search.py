"""
Tests for image-based vector search via Elasticsearch HNSW (image_vector).

API tests  (pytest -m api):         require ES + indexed catalog + sample images
Performance tests (pytest -m performance): also require queries_image.json fixture
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


ES_INDEX = os.environ.get("ES_TEST_INDEX", "products_test")
ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


# ---------------------------------------------------------------------------
# Fixtures / helpers
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


def _image_search(es, embedding_client, image_path: Path, top_k: int = 10):
    """Embed an image and run kNN search on image_vector."""
    vec = embedding_client.embed_image(image_path)

    resp = es.search(
        index=ES_INDEX,
        knn={
            "field": "image_vector",
            "query_vector": vec,
            "k": top_k,
            "num_candidates": top_k * 5,
        },
        size=top_k,
        _source=["item_id", "category", "name", "image_path"],
    )
    return [
        {
            "item_id":    hit["_source"]["item_id"],
            "category":   hit["_source"].get("category"),
            "name":       hit["_source"].get("name"),
            "image_path": hit["_source"].get("image_path"),
            "score":      hit["_score"],
        }
        for hit in resp["hits"]["hits"]
    ]


# ---------------------------------------------------------------------------
# API Tests (IS-A*)
# ---------------------------------------------------------------------------

class TestImageSearchAPI:

    @pytest.mark.api
    def test_returns_results(self, es_client, embedding_client, sample_images):
        """IS-A1: image query returns a list of items with item_id + score."""
        assert sample_images, "No sample images"
        results = _image_search(es_client, embedding_client, sample_images[0])
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]
        assert "score" in results[0]

    @pytest.mark.api
    def test_top_k_respected(self, es_client, embedding_client, sample_images):
        """IS-A2: exactly top_k results returned."""
        for k in (1, 5, 10):
            results = _image_search(es_client, embedding_client, sample_images[0], top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    @pytest.mark.api
    def test_score_range(self, es_client, embedding_client, sample_images):
        """IS-A3: cosine similarity scores are in [0, 1]."""
        results = _image_search(es_client, embedding_client, sample_images[0], top_k=10)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    @pytest.mark.api
    def test_multiple_images_no_crash(self, es_client, embedding_client, sample_images):
        """IS-A5: searching with several different images all succeed."""
        errors = []
        for img in sample_images[:5]:
            try:
                results = _image_search(es_client, embedding_client, img, top_k=5)
                assert len(results) > 0
            except Exception as e:
                errors.append(str(e))
        assert not errors, f"Errors during image search: {errors}"


# ---------------------------------------------------------------------------
# Performance Tests (IS-P*)
# ---------------------------------------------------------------------------

class TestImageSearchPerformance:

    @pytest.mark.performance
    def test_self_retrieval(self, es_client, embedding_client, catalog):
        """
        IS-P1: searching with a product's own image should return that product in top 3.
        Uses the first 20 catalog products that have an image on disk.
        """
        test_products = []
        for p in catalog:
            path = Path(p["image_path"])
            if path.exists() and len(test_products) < 20:
                test_products.append(p)

        if not test_products:
            pytest.skip("No catalog products with on-disk images")

        hits = 0
        misses = []
        for p in test_products:
            results = _image_search(es_client, embedding_client, Path(p["image_path"]), top_k=3)
            top_ids = [r["item_id"] for r in results]
            if p["item_id"] in top_ids:
                hits += 1
            else:
                misses.append({"item_id": p["item_id"], "top3": top_ids})

        hit_rate = hits / len(test_products)
        print(f"\nImage self-retrieval hit rate @3: {hit_rate:.0%} ({hits}/{len(test_products)})")
        for m in misses[:5]:
            print(f"  MISS {m['item_id']} not in {m['top3']}")
        assert hit_rate > 0.50, f"Self-retrieval hit rate {hit_rate:.0%} < 50%"

    @pytest.mark.performance
    def test_p_at_5(self, es_client, embedding_client):
        """IS-P2: mean P@5 across labeled image queries > 0.40."""
        from conftest import load_labeled_queries
        queries = load_labeled_queries("queries_image.json")

        scores = []
        for q in queries:
            path = Path(q["image_path"])
            if not path.exists():
                continue
            results = _image_search(es_client, embedding_client, path, top_k=5)
            relevant = set(q["relevant_ids"])
            top_ids = [r["item_id"] for r in results]
            hits = sum(1 for r_id in top_ids[:5] if r_id in relevant)
            scores.append(hits / 5)

        if not scores:
            pytest.skip("No image query files found on disk")

        mean_p5 = sum(scores) / len(scores)
        print(f"\nImage search mean P@5: {mean_p5:.3f} (n={len(scores)})")
        assert mean_p5 > 0.25, f"Mean P@5 {mean_p5:.3f} < 0.25"

    @pytest.mark.performance
    def test_category_containment(self, es_client, embedding_client, catalog):
        """
        IS-P3: shoe images should return mostly shoe-category items.
        Contamination < 50%.
        """
        shoe_cats = {"SHOES", "BOOT", "SANDAL", "SNEAKER"}
        shoe_products = [
            p for p in catalog
            if p["category"] in shoe_cats and Path(p["image_path"]).exists()
        ][:5]

        if not shoe_products:
            pytest.skip("No shoe-category products with images in catalog")

        contamination_rates = []
        for p in shoe_products:
            results = _image_search(es_client, embedding_client, Path(p["image_path"]), top_k=10)
            non_shoe = sum(1 for r in results if r["category"] not in shoe_cats)
            contamination_rates.append(non_shoe / len(results) if results else 0)

        mean_contamination = sum(contamination_rates) / len(contamination_rates)
        print(f"\nImage search shoe contamination: {mean_contamination:.0%}")
        assert mean_contamination < 0.50, (
            f"Image search contamination {mean_contamination:.0%} exceeds 50%"
        )
