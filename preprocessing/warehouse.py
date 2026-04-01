#!/usr/bin/env python3
"""
Warehouse CLI — manage products in the Elasticsearch index.

Usage:
    python warehouse.py add items.json
    python warehouse.py delete items.json
    python warehouse.py check --item-id B123456789
    python warehouse.py check --group ELECTRONICS [--count 10]
    python warehouse.py check --count 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Make index.py importable whether run as a script or imported by tests.
sys.path.insert(0, str(Path(__file__).parent))
from index import get_indexed_ids

ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.environ.get("ES_INDEX", "products")
API_URL  = os.environ.get("API_BASE_URL", "http://localhost:8000")

REQUIRED_ADD_FIELDS = {"item_id", "description", "image_path"}
VECTOR_FIELDS = {"description_vector", "image_vector"}

_es_client = None


def _get_es():
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        _es_client = Elasticsearch(ES_URL)
    return _es_client


def _strip_vectors(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in VECTOR_FIELDS}


def _load_json(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[error] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("[error] JSON must be an array of objects", file=sys.stderr)
        sys.exit(1)
    return data


def _call_embed(text: str, image_path: str) -> dict:
    """POST to /embed on the running API server. Returns {description_vector, image_vector}."""
    import requests
    try:
        resp = requests.post(
            f"{API_URL}/embed",
            json={"text": text, "image_path": image_path},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach embedding service at {API_URL} — is the API server running?"
        )
    if resp.status_code == 400:
        raise ValueError(resp.json().get("detail", "Bad request from /embed"))
    if resp.status_code == 503:
        raise RuntimeError("Embedding service not ready — API server is still starting up")
    resp.raise_for_status()
    return resp.json()


def cmd_check(args):
    es = _get_es()

    if args.item_id:
        try:
            resp = es.get(index=ES_INDEX, id=args.item_id)
            doc = _strip_vectors(resp["_source"])
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[error] Item '{args.item_id}' not found: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.group:
        count = args.count or 10
        resp = es.search(
            index=ES_INDEX,
            query={"term": {"category": args.group}},
            size=count,
            _source=True,
        )
        hits = [_strip_vectors(h["_source"]) for h in resp["hits"]["hits"]]
        print(json.dumps(hits, indent=2, ensure_ascii=False))

    else:
        count = args.count or 10
        resp = es.search(
            index=ES_INDEX,
            query={"match_all": {}},
            size=count,
            _source=True,
        )
        hits = [_strip_vectors(h["_source"]) for h in resp["hits"]["hits"]]
        print(json.dumps(hits, indent=2, ensure_ascii=False))


def cmd_add(args):
    pass  # implemented in Task 4


def cmd_delete(args):
    items = _load_json(args.json_file)

    # Validate required field
    errors = []
    for i, item in enumerate(items):
        if "item_id" not in item:
            errors.append(f"  item[{i}]: missing 'item_id'")
    if errors:
        print("[error] Missing required fields:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # Detect duplicates within input
    seen: dict[str, int] = {}
    duplicates = []
    for i, item in enumerate(items):
        iid = item["item_id"]
        if iid in seen:
            duplicates.append(f"  '{iid}' at index {seen[iid]} and {i}")
        else:
            seen[iid] = i
    if duplicates:
        print("[error] Duplicate item_ids in input:", file=sys.stderr)
        for d in duplicates:
            print(d, file=sys.stderr)
        sys.exit(1)

    all_ids = [item["item_id"] for item in items]
    existing = get_indexed_ids(ES_URL, ES_INDEX, all_ids)

    for iid in all_ids:
        if iid not in existing:
            print(f"[warn] {iid} — not found, skipping", file=sys.stderr)

    es = _get_es()
    deleted = 0
    for iid in all_ids:
        if iid in existing:
            es.delete(index=ES_INDEX, id=iid)
            deleted += 1

    es.indices.refresh(index=ES_INDEX)
    not_found = len(all_ids) - deleted
    print(f"\nDeleted: {deleted}, Not found: {not_found}")


def main():
    parser = argparse.ArgumentParser(
        prog="warehouse",
        description="Manage products in the Elasticsearch index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add items from a JSON file")
    p_add.add_argument("json_file", help="Path to JSON file")

    p_del = sub.add_parser("delete", help="Delete items by item_id from a JSON file")
    p_del.add_argument("json_file", help="Path to JSON file")

    p_check = sub.add_parser("check", help="Query items in the index")
    p_check.add_argument("--item-id", dest="item_id", help="Fetch a single item by ID")
    p_check.add_argument("--group", help="Fetch items by category")
    p_check.add_argument(
        "--count", type=int,
        help="Number of results (default 10); use alone to query across all groups",
    )

    args = parser.parse_args()

    if args.command == "check":
        if args.item_id and args.group:
            parser.error("--item-id and --group are mutually exclusive")
        if not args.item_id and not args.group and args.count is None:
            parser.error("specify at least one of --item-id, --group, or --count")
        cmd_check(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "delete":
        cmd_delete(args)


if __name__ == "__main__":
    main()
