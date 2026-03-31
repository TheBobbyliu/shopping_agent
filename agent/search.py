"""
Search client — wraps Elasticsearch hybrid search used by agent tools.
Shared by tools.py and can be tested independently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

# Pipeline monitoring — imported lazily so search.py works without the monitor
def _stage(name: str, input_summary: dict | None = None):
    """Return a pipeline_stage context manager, or a no-op if monitor unavailable."""
    try:
        from pipeline_monitor import pipeline_stage
        return pipeline_stage(name, input_summary)
    except ImportError:
        from contextlib import nullcontext
        return nullcontext({})

RRF_K    = 60
# ES_INDEX: production default is "products". Tests override via conftest.py.
ES_INDEX = os.environ.get("ES_INDEX", os.environ.get("ES_TEST_INDEX", "products_test"))
ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

# Singletons — loaded once on first use
_reranker         = None
_embedding_client = None
_es_client        = None


def _get_es():
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        _es_client = Elasticsearch(ES_URL)
    return _es_client


def _get_embedding_client():
    global _embedding_client
    if _embedding_client is None:
        from embed import EmbeddingClient
        _embedding_client = EmbeddingClient.from_env()
    return _embedding_client


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    return _reranker


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _fetch_image(image_url: str) -> tuple[str, str]:
    """
    Download an image from a URL and return (base64_str, temp_file_path).
    Caller is responsible for deleting the temp file.
    """
    import base64 as _b64
    import tempfile
    import urllib.request

    with urllib.request.urlopen(image_url) as resp:
        raw = resp.read()

    tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_file.write(raw)
    tmp_file.close()

    b64 = _b64.b64encode(raw).decode()
    return b64, tmp_file.name


def hybrid_search(
    query_text: str,
    image_url: Optional[str] = None,
    top_k: int = 8,
    category: Optional[str] = None,
    rerank: bool = True,
    rerank_fetch_multiplier: int = 4,
) -> list[dict]:
    """
    Hybrid search: description HNSW + BM25 (+ image HNSW if image given) → RRF → reranker.

    When image_url is provided the search engine:
      1. Downloads the image and calls GPT-4o to produce a text description.
      2. Uses that description as query_text if none was supplied.
      3. Extracts an image embedding for the image HNSW channel.

    Args:
        query_text:              Natural language query (may be empty when image_url is given).
        image_url:               URL of a product image to include image HNSW channel.
        top_k:                   Final number of results to return.
        category:                Optional Elasticsearch category filter.
        rerank:                  Whether to apply bge-reranker-v2-m3 after RRF fusion.
        rerank_fetch_multiplier: Fetch top_k * this many candidates before reranking.
    """
    es     = _get_es()
    client = _get_embedding_client()

    # How many candidates to retrieve from each channel before reranking
    fetch_k = max(50, top_k * rerank_fetch_multiplier) if rerank else top_k
    knn_filter = [{"term": {"category": category}}] if category else []

    # --- Image understanding + embedding (if image provided) ---
    img_vec = None
    if image_url:
        with _stage("img_fetch", {"url": image_url[:120]}) as out:
            img_b64, tmp_path = _fetch_image(image_url)
            out["size_bytes"] = len(img_b64)

        try:
            with _stage("describe_image", {"img_size_bytes": len(img_b64)}) as out:
                img_description = describe_image(img_b64)
                out["description_preview"] = img_description[:120]
                out["char_count"] = len(img_description)
            if not query_text or not query_text.strip():
                query_text = img_description

            with _stage("embed_image") as out:
                img_vec = client.embed_image(tmp_path)
                out["vector_dim"] = len(img_vec)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # --- Channel 1: description vector (HNSW) ---
    with _stage("embed_text", {"query_preview": query_text[:120]}) as out:
        txt_vec = client.embed_text(query_text)
        out["vector_dim"] = len(txt_vec)

    knn_body: dict = {
        "field": "description_vector",
        "query_vector": txt_vec,
        "k": fetch_k,
        "num_candidates": fetch_k * 3,
    }
    if knn_filter:
        knn_body["filter"] = {"bool": {"must": knn_filter}}

    with _stage("es_knn_description", {"fetch_k": fetch_k, "has_filter": bool(knn_filter)}) as out:
        knn_hits = es.search(index=ES_INDEX, knn=knn_body, size=fetch_k,
                             _source=True)["hits"]["hits"]
        out["hits_count"] = len(knn_hits)
        out["top_score"]  = round(knn_hits[0]["_score"], 6) if knn_hits else None

    # --- Channel 2: BM25 (lexical) ---
    bm25_q: dict = {"match": {"description": {"query": query_text}}}
    if knn_filter:
        bm25_q = {"bool": {"must": [bm25_q, *knn_filter]}}

    with _stage("es_bm25", {"query_preview": query_text[:80], "fetch_k": fetch_k}) as out:
        bm25_hits = es.search(index=ES_INDEX, query=bm25_q, size=fetch_k,
                               _source=True)["hits"]["hits"]
        out["hits_count"] = len(bm25_hits)
        out["top_score"]  = round(bm25_hits[0]["_score"], 6) if bm25_hits else None

    channels = [knn_hits, bm25_hits]

    # --- Channel 3: image vector (optional) ---
    if img_vec is not None:
        img_body: dict = {
            "field": "image_vector",
            "query_vector": img_vec,
            "k": fetch_k,
            "num_candidates": fetch_k * 3,
        }
        if knn_filter:
            img_body["filter"] = {"bool": {"must": knn_filter}}

        with _stage("es_knn_image", {"fetch_k": fetch_k}) as out:
            img_hits = es.search(index=ES_INDEX, knn=img_body, size=fetch_k,
                          _source=True)["hits"]["hits"]
            out["hits_count"] = len(img_hits)
            out["top_score"]  = round(img_hits[0]["_score"], 6) if img_hits else None
        channels.append(img_hits)

    # --- RRF fusion ---
    with _stage("rrf_fusion", {"channel_count": len(channels)}) as out:
        rrf_scores: dict[str, float] = {}
        sources:    dict[str, dict]  = {}
        for channel in channels:
            for rank, hit in enumerate(channel, start=1):
                iid = hit["_source"]["item_id"]
                rrf_scores[iid] = rrf_scores.get(iid, 0.0) + _rrf(rank)
                sources[iid] = hit["_source"]

        # Take the top fetch_k candidates for reranking (or top_k if no reranking)
        candidate_limit = fetch_k if rerank else top_k
        candidates = sorted(rrf_scores.items(), key=lambda x: -x[1])[:candidate_limit]
        out["unique_candidates"] = len(rrf_scores)
        out["after_limit"]       = len(candidates)
        out["top_rrf_score"]     = round(candidates[0][1], 6) if candidates else None

    # --- Reranker ---
    if rerank and candidates:
        cand_ids = [iid for iid, _ in candidates]
        passages = [
            (sources[iid].get("name", "") + " " + sources[iid].get("description", ""))[:512]
            for iid in cand_ids
        ]
        pairs = [[query_text, p] for p in passages]

        with _stage("reranking", {"candidates": len(cand_ids), "query_preview": query_text[:80]}) as out:
            reranker = _get_reranker()
            scores   = reranker.compute_score(pairs)
            if isinstance(scores, float):
                scores = [scores]
            ranked = sorted(zip(cand_ids, scores), key=lambda x: -x[1])[:top_k]
            out["top_k"]          = len(ranked)
            out["top_rerank_score"] = round(float(ranked[0][1]), 6) if ranked else None
            out["score_range"]    = [
                round(float(ranked[-1][1]), 6),
                round(float(ranked[0][1]),  6),
            ] if ranked else []
    else:
        ranked = [(iid, rrf_scores[iid]) for iid, _ in candidates[:top_k]]

    # --- Build result list ---
    # When reranking, `ranked` contains (iid, rerank_score); look up the original RRF score separately.
    results = []
    for iid, score in ranked:
        src = sources[iid]
        entry = {
            "item_id":     iid,
            "name":        src.get("name", ""),
            "category":    src.get("category", ""),
            "description": src.get("description", "")[:300],
            "color":       src.get("color", ""),
            "material":    src.get("material", ""),
            "brand":       src.get("brand", ""),
            "image_path":  src.get("image_path", ""),
            "image_url":   src.get("image_url", ""),
            "web_url":     src.get("web_url", ""),
            "score":       round(rrf_scores[iid], 6),
        }
        if rerank:
            entry["rerank_score"] = round(float(score), 6)
        results.append(entry)
    return results


def get_product(item_id: str) -> Optional[dict]:
    """Fetch a single product by item_id."""
    es = _get_es()
    try:
        resp = es.get(index=ES_INDEX, id=item_id)
        src  = resp["_source"]
        src.pop("description_vector", None)
        src.pop("image_vector", None)
        return src
    except Exception:
        return None


def describe_image(image_b64: str) -> str:
    """Call GPT-4o to produce a shopping-oriented description of a product image."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "Image understanding unavailable (OPENAI_API_KEY not set)."

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text",
                 "text": (
                     "You are a product search assistant. "
                     "Describe this product image concisely for search purposes. "
                     "Include: product type, color, material (if visible), "
                     "style, and any notable features. "
                     "Be specific. Do not include prices or brand guesses."
                 )},
            ],
        }],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()
