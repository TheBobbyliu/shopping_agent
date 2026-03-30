"""
Indexer: bulk-upserts EmbeddedProduct records into Elasticsearch.

Index has two HNSW fields:
  - description_vector  (text embeddings, bge-visualized-m3, 1024-dim)
  - image_vector        (image embeddings, bge-visualized-m3, 1024-dim)

Plus BM25 over the 'description' field.

Usage:
    python index.py --input data/embedded_500.jsonl --index products_test --recreate
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class IndexStats:
    indexed: int
    errors: int


def _get_es_client():
    from elasticsearch import Elasticsearch
    url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    return Elasticsearch(url)


def _index_mapping(dim: int) -> dict:
    return {
        "mappings": {
            "properties": {
                "item_id":     {"type": "keyword"},
                "name":        {"type": "text"},
                "description": {"type": "text", "analyzer": "english"},
                "category":    {"type": "keyword"},
                "brand":       {"type": "keyword"},
                "color":       {"type": "keyword"},
                "material":    {"type": "keyword"},
                "image_path":  {"type": "keyword"},
                "image_url":   {"type": "keyword"},
                "web_url":     {"type": "keyword"},
                "keywords":    {"type": "keyword"},
                "bullet_points": {"type": "text"},
                "description_vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "image_vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }


def index_products(
    products: Iterable[dict],
    es_url: str = "http://localhost:9200",
    index_name: str = "products",
    recreate: bool = False,
    dim: int = 1024,
    batch_size: int = 100,
) -> IndexStats:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch(es_url)

    if recreate and es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"[index] Deleted existing index '{index_name}'", file=sys.stderr)

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=_index_mapping(dim))
        print(f"[index] Created index '{index_name}' (dim={dim})", file=sys.stderr)

    def _actions(products):
        for p in products:
            yield {
                "_index": index_name,
                "_id":    p["item_id"],
                "_source": p,
            }

    indexed = 0
    errors = 0
    batch = []

    for p in products:
        batch.append(p)
        if len(batch) >= batch_size:
            ok, errs = bulk(es, _actions(batch), raise_on_error=False)
            indexed += ok
            errors  += len(errs)
            print(f"[index] {indexed} indexed, {errors} errors", file=sys.stderr)
            batch = []

    if batch:
        ok, errs = bulk(es, _actions(batch), raise_on_error=False)
        indexed += ok
        errors  += len(errs)

    es.indices.refresh(index=index_name)
    print(f"[index] Done — {indexed} indexed, {errors} errors", file=sys.stderr)
    return IndexStats(indexed=indexed, errors=errors)


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Index embedded products into Elasticsearch")
    parser.add_argument("--input",     type=Path, required=True, help="JSONL from embed.py")
    parser.add_argument("--index",     default="products",        help="Elasticsearch index name")
    parser.add_argument("--es-url",    default=os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--recreate",  action="store_true",       help="Drop and recreate index")
    parser.add_argument("--dim",       type=int, default=int(os.environ.get("EMBEDDING_DIM", 1024)))
    args = parser.parse_args()

    products = []
    with open(args.input) as f:
        for line in f:
            products.append(json.loads(line))

    stats = index_products(
        products,
        es_url=args.es_url,
        index_name=args.index,
        recreate=args.recreate,
        dim=args.dim,
    )
    print(f"Indexed {stats.indexed} products ({stats.errors} errors)", file=sys.stderr)


if __name__ == "__main__":
    main()
