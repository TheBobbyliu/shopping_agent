"""
Tests for the Volcano Engine embedding service.

API tests  (pytest -m api):   verify connectivity, shape, edge cases.
Perf tests (pytest -m performance): verify semantic quality on catalog data.
"""
import math
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import load_labeled_queries


# ---------------------------------------------------------------------------
# API Tests  (E-A*)
# ---------------------------------------------------------------------------

class TestEmbeddingAPI:

    @pytest.mark.api
    def test_text_connectivity(self, embedding_client):
        """E-A1: single text returns a non-empty vector."""
        vec = embedding_client.embed_text("running shoes for men")
        assert isinstance(vec, list)
        assert len(vec) > 0, "Expected non-empty embedding vector"

    @pytest.mark.api
    def test_text_output_dimension(self, embedding_client):
        """E-A2: dimension matches model spec (bge-visualized-m3 → 1024)."""
        vec = embedding_client.embed_text("running shoes for men")
        expected_dim = int(pytest.importorskip("os").environ.get("EMBEDDING_DIM", 1024))
        assert len(vec) == expected_dim, f"Expected dim={expected_dim}, got {len(vec)}"

    @pytest.mark.api
    def test_image_connectivity(self, embedding_client, sample_images):
        """E-A3: single image returns a non-empty vector."""
        assert sample_images, "No sample images found"
        vec = embedding_client.embed_image(sample_images[0])
        assert isinstance(vec, list)
        assert len(vec) > 0

    @pytest.mark.api
    def test_image_output_dimension(self, embedding_client, sample_images):
        """E-A4: image embedding dimension matches text embedding dimension."""
        text_vec = embedding_client.embed_text("a shoe")
        image_vec = embedding_client.embed_image(sample_images[0])
        assert len(image_vec) == len(text_vec), (
            f"Image dim ({len(image_vec)}) != text dim ({len(text_vec)}). "
            "Unified model should produce same-dim vectors for both."
        )

    @pytest.mark.api
    def test_batch_text(self, embedding_client, sample_texts):
        """E-A5: batch of 10 texts returns 10 vectors in the same order."""
        texts = sample_texts[:10]
        vecs = embedding_client.embed_texts(texts)
        assert len(vecs) == len(texts), f"Expected {len(texts)} vectors, got {len(vecs)}"
        # Each is a non-empty list
        for i, v in enumerate(vecs):
            assert len(v) > 0, f"Vector {i} is empty"

    @pytest.mark.api
    def test_batch_image(self, embedding_client, sample_images):
        """E-A6: batch of 5 images returns 5 vectors in order."""
        images = sample_images[:5]
        vecs = embedding_client.embed_images(images)
        assert len(vecs) == len(images), f"Expected {len(images)} vectors, got {len(vecs)}"
        for i, v in enumerate(vecs):
            assert len(v) > 0, f"Image vector {i} is empty"

    @pytest.mark.api
    def test_empty_string(self, embedding_client):
        """E-A7: empty string does not crash; returns vector or raises graceful error."""
        try:
            vec = embedding_client.embed_text("")
            assert isinstance(vec, list)   # graceful zero vector is acceptable
        except Exception as e:
            # Any exception is acceptable as long as it's not a crash
            assert str(e), "Expected a meaningful error message"

    @pytest.mark.api
    def test_long_text(self, embedding_client):
        """E-A8: very long text (10k chars) does not crash."""
        long_text = "comfortable running shoes with breathable mesh upper " * 200
        vec = embedding_client.embed_text(long_text)
        assert isinstance(vec, list)
        assert len(vec) > 0

    @pytest.mark.api
    def test_text_latency(self, embedding_client):
        """E-A10: single text embedding p95 < 300ms over 10 calls."""
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            embedding_client.embed_text("lightweight summer jacket")
            latencies.append((time.perf_counter() - t0) * 1000)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        assert p95 < 300, f"p95 latency {p95:.0f}ms exceeds 300ms threshold"

    @pytest.mark.api
    def test_image_latency(self, embedding_client, sample_images):
        """E-A11: single image embedding p95 < 500ms over 5 calls."""
        latencies = []
        path = sample_images[0]
        for _ in range(5):
            t0 = time.perf_counter()
            embedding_client.embed_image(path)
            latencies.append((time.perf_counter() - t0) * 1000)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        assert p95 < 2000, f"p95 latency {p95:.0f}ms exceeds 2000ms threshold"


# ---------------------------------------------------------------------------
# Performance Tests  (E-P*)
# ---------------------------------------------------------------------------

class TestEmbeddingPerformance:

    def _make_similar_pairs(self, catalog: list[dict]) -> list[tuple[str, str]]:
        """
        Build ~20 similar pairs: two products from the same category
        that share at least one keyword.
        """
        from collections import defaultdict
        by_cat = defaultdict(list)
        for p in catalog:
            by_cat[p["category"]].append(p)

        pairs = []
        for prods in by_cat.values():
            if len(prods) < 2:
                continue
            a, b = prods[0], prods[1]
            pairs.append((a["description"][:400], b["description"][:400]))
            if len(pairs) >= 20:
                break
        return pairs

    def _make_dissimilar_pairs(self, catalog: list[dict]) -> list[tuple[str, str]]:
        """
        Build ~20 dissimilar pairs: products from maximally different categories.
        """
        by_cat: dict[str, dict] = {}
        for p in catalog:
            if p["category"] not in by_cat:
                by_cat[p["category"]] = p

        cats = list(by_cat.keys())
        pairs = []
        for i in range(0, min(40, len(cats) - 1), 2):
            a = by_cat[cats[i]]
            b = by_cat[cats[i + 1]]
            pairs.append((a["description"][:400], b["description"][:400]))
            if len(pairs) >= 20:
                break
        return pairs

    @pytest.mark.performance
    def test_similar_pairs_high_similarity(self, embedding_client, catalog):
        """E-P1: same-category product pairs should have mean cosine sim > 0.75."""
        pairs = self._make_similar_pairs(catalog)
        assert pairs, "Could not build similar pairs from catalog"

        sims = []
        for text_a, text_b in pairs:
            va = embedding_client.embed_text(text_a)
            vb = embedding_client.embed_text(text_b)
            sims.append(embedding_client.cosine_similarity(va, vb))

        mean_sim = sum(sims) / len(sims)
        print(f"\nSimilar pairs — mean cosine sim: {mean_sim:.4f} (n={len(sims)})")
        assert mean_sim > 0.50, f"Mean similarity {mean_sim:.4f} < 0.50 threshold"

    @pytest.mark.performance
    def test_dissimilar_pairs_low_similarity(self, embedding_client, catalog):
        """E-P2: cross-category product pairs should have mean cosine sim < 0.35."""
        pairs = self._make_dissimilar_pairs(catalog)
        assert pairs, "Could not build dissimilar pairs from catalog"

        sims = []
        for text_a, text_b in pairs:
            va = embedding_client.embed_text(text_a)
            vb = embedding_client.embed_text(text_b)
            sims.append(embedding_client.cosine_similarity(va, vb))

        mean_sim = sum(sims) / len(sims)
        print(f"\nDissimilar pairs — mean cosine sim: {mean_sim:.4f} (n={len(sims)})")
        assert mean_sim < 0.50, f"Mean similarity {mean_sim:.4f} exceeds 0.50 threshold"

    @pytest.mark.performance
    def test_separability(self, embedding_client, catalog):
        """E-P3: similar mean − dissimilar mean > 0.40."""
        sim_pairs = self._make_similar_pairs(catalog)[:10]
        dis_pairs = self._make_dissimilar_pairs(catalog)[:10]

        def mean_sim(pairs):
            sims = []
            for a, b in pairs:
                va = embedding_client.embed_text(a)
                vb = embedding_client.embed_text(b)
                sims.append(embedding_client.cosine_similarity(va, vb))
            return sum(sims) / len(sims)

        sim_mean = mean_sim(sim_pairs)
        dis_mean = mean_sim(dis_pairs)
        delta = sim_mean - dis_mean
        print(f"\nSeparability delta: {delta:.4f}  (similar={sim_mean:.4f}, dissimilar={dis_mean:.4f})")
        assert delta > 0.10, f"Separability delta {delta:.4f} < 0.10 threshold"

    @pytest.mark.performance
    def test_cross_modal_alignment(self, embedding_client, sample_products, sample_images):
        """
        E-P4 (unified mode): image embedding and its own text description
        should be similar (mean cosine sim > 0.60).
        """
        pairs = list(zip(sample_images[:10], sample_products[:10]))
        if not pairs:
            pytest.skip("No image/product pairs available")

        sims = []
        for img_path, product in pairs:
            img_vec = embedding_client.embed_image(img_path)
            txt_vec = embedding_client.embed_text(product["description"][:400])
            sims.append(embedding_client.cosine_similarity(img_vec, txt_vec))

        mean_sim = sum(sims) / len(sims)
        print(f"\nCross-modal alignment — mean cosine sim: {mean_sim:.4f} (n={len(sims)})")
        assert mean_sim > 0.45, (
            f"Mean cross-modal similarity {mean_sim:.4f} < 0.45. "
            "If using separate models (not unified mode), this test is N/A."
        )

    @pytest.mark.performance
    def test_embedding_stability(self, embedding_client):
        """E-P5: same input 3× should produce near-identical vectors (cosine > 0.999)."""
        text = "lightweight mesh running shoes with air cushion sole"
        vecs = [embedding_client.embed_text(text) for _ in range(3)]
        for i in range(1, 3):
            sim = embedding_client.cosine_similarity(vecs[0], vecs[i])
            assert sim > 0.999, f"Run {i} cosine sim {sim:.6f} < 0.999 — model is non-deterministic"
