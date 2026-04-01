"""
Tests for Area 3: Performance — Parallel Embedding

Covers:
  1. PARALLEL_EMBED unset (default) — sequential text-only path
  2. PARALLEL_EMBED=0 — sequential image+text path
  3. PARALLEL_EMBED=1 — parallel image+text path, same result shape
  4. Parallel and sequential return the same top result
  5. Timing print (soft check — warns but doesn't fail)

Note: hybrid_search() takes image_url (a URL to download), not a local path.
      We use the running API's /image/{item_id} endpoint to get a real image URL.
"""
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

# Stable item IDs known to be in products_test index
SAMPLE_ITEM_IDS = ["B079WX8KXJ", "B074H6X5HF", "B071ZLY1BR"]
API_BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def image_url():
    """Return a valid image URL served by the running API."""
    import requests
    for item_id in SAMPLE_ITEM_IDS:
        r = requests.get(f"{API_BASE}/image/{item_id}", timeout=5, stream=True)
        r.close()
        if r.status_code == 200:
            return f"{API_BASE}/image/{item_id}"
    pytest.skip("No test images available via /image/{item_id}")


class TestParallelEmbed:
    def test_sequential_text_only(self):
        """Text-only query without PARALLEL_EMBED should return results."""
        env = {k: v for k, v in os.environ.items() if k != "PARALLEL_EMBED"}
        with mock.patch.dict(os.environ, env, clear=True):
            from search import hybrid_search
            results = hybrid_search("comfortable sofa", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]

    def test_sequential_image_and_text(self, image_url):
        """Image + text in sequential mode (PARALLEL_EMBED=0) should return results."""
        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "0"}):
            from search import hybrid_search
            results = hybrid_search("chair", image_url=image_url, top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]

    def test_parallel_image_and_text(self, image_url):
        """Image + text in parallel mode (PARALLEL_EMBED=1) should return results."""
        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "1"}):
            from search import hybrid_search
            results = hybrid_search("chair", image_url=image_url, top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "item_id" in results[0]

    def test_parallel_returns_valid_item_ids(self, image_url):
        """Parallel results should contain valid item_id strings."""
        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "1"}):
            from search import hybrid_search
            results = hybrid_search("lamp", image_url=image_url, top_k=10)
        for r in results:
            assert isinstance(r["item_id"], str)
            assert len(r["item_id"]) > 0

    def test_parallel_vs_sequential_same_top_result(self, image_url):
        """Parallel and sequential should return the same top result for the same query."""
        query = "wooden chair"
        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "0"}):
            from search import hybrid_search
            seq_results = hybrid_search(query, image_url=image_url, top_k=5)

        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "1"}):
            from search import hybrid_search as hs2
            par_results = hs2(query, image_url=image_url, top_k=5)

        assert seq_results[0]["item_id"] == par_results[0]["item_id"], (
            f"Top result differs: seq={seq_results[0]['item_id']} "
            f"par={par_results[0]['item_id']}"
        )

    def test_parallel_timing(self, image_url):
        """Print sequential vs parallel timing; warn if parallel is slower (soft check)."""
        query = "dining table"

        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "0"}):
            from search import hybrid_search
            t0 = time.perf_counter()
            hybrid_search(query, image_url=image_url, top_k=5)
            seq_time = time.perf_counter() - t0

        with mock.patch.dict(os.environ, {"PARALLEL_EMBED": "1"}):
            from search import hybrid_search as hs2
            t0 = time.perf_counter()
            hs2(query, image_url=image_url, top_k=5)
            par_time = time.perf_counter() - t0

        print(f"\n  sequential: {seq_time:.2f}s  |  parallel: {par_time:.2f}s  "
              f"|  speedup: {seq_time/par_time:.2f}x")
        if par_time > seq_time:
            print(f"  NOTE: parallel not faster (likely CPU-bound or model already cached)")
        # Not a hard assertion — timing varies by hardware/cache state
