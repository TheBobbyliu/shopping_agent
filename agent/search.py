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

RRF_K = 60
ES_INDEX = os.environ.get("ES_TEST_INDEX", "products_test")
ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


def _get_es():
    from elasticsearch import Elasticsearch
    return Elasticsearch(ES_URL)


def _get_embedding_client():
    from embed import EmbeddingClient
    return EmbeddingClient.from_env()


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def hybrid_search(
    query_text: str,
    image_b64: Optional[str] = None,
    top_k: int = 8,
    category: Optional[str] = None,
) -> list[dict]:
    """
    Run hybrid search: description HNSW + BM25 (+ image HNSW if image given).
    Returns top_k product dicts with score.
    """
    es = _get_es()
    client = _get_embedding_client()

    fetch_k = max(50, top_k * 8)
    knn_filter = [{"term": {"category": category}}] if category else []

    # Channel 1: description vector
    txt_vec = client.embed_text(query_text)
    knn_body: dict = {
        "field": "description_vector",
        "query_vector": txt_vec,
        "k": fetch_k,
        "num_candidates": fetch_k * 3,
    }
    if knn_filter:
        knn_body["filter"] = {"bool": {"must": knn_filter}}
    knn_resp = es.search(index=ES_INDEX, knn=knn_body, size=fetch_k,
                         _source=True)
    knn_hits = knn_resp["hits"]["hits"]

    # Channel 2: BM25
    bm25_query: dict = {"match": {"description": {"query": query_text}}}
    if knn_filter:
        bm25_query = {"bool": {"must": [bm25_query, *[{"term": f} for f in knn_filter]]}}
    bm25_resp = es.search(index=ES_INDEX, query=bm25_query, size=fetch_k,
                           _source=True)
    bm25_hits = bm25_resp["hits"]["hits"]

    channels = [knn_hits, bm25_hits]

    # Channel 3: image vector (optional)
    if image_b64:
        import base64, tempfile
        data = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            img_vec = client.embed_image(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)

        img_body: dict = {
            "field": "image_vector",
            "query_vector": img_vec,
            "k": fetch_k,
            "num_candidates": fetch_k * 3,
        }
        if knn_filter:
            img_body["filter"] = {"bool": {"must": knn_filter}}
        img_resp = es.search(index=ES_INDEX, knn=img_body, size=fetch_k,
                             _source=True)
        channels.append(img_resp["hits"]["hits"])

    # RRF fusion
    scores: dict[str, float] = {}
    sources: dict[str, dict] = {}
    for channel in channels:
        for rank, hit in enumerate(channel, start=1):
            iid = hit["_source"]["item_id"]
            scores[iid] = scores.get(iid, 0.0) + _rrf(rank)
            sources[iid] = hit["_source"]

    fused = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    results = []
    for iid, score in fused:
        src = sources[iid]
        results.append({
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
            "score":       round(score, 6),
        })
    return results


def get_product(item_id: str) -> Optional[dict]:
    """Fetch a single product by item_id."""
    es = _get_es()
    try:
        resp = es.get(index=ES_INDEX, id=item_id)
        src = resp["_source"]
        # Remove vector fields from response
        src.pop("description_vector", None)
        src.pop("image_vector", None)
        return src
    except Exception:
        return None


def describe_image(image_b64: str) -> str:
    """
    Call GPT-4o to produce a shopping-oriented description of a product image.
    image_b64: base64-encoded image bytes (JPEG or PNG).
    """
    import os
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
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "You are a product search assistant. "
                        "Describe this product image concisely for search purposes. "
                        "Include: product type, color, material (if visible), "
                        "style, and any notable features. "
                        "Be specific. Do not include prices or brand guesses."
                    ),
                },
            ],
        }],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()
