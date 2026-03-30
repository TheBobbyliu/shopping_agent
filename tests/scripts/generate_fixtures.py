"""
Generate labeled query fixtures from the test catalog using GPT-4o.

Outputs:
  tests/fixtures/queries_text.json   - 30 text queries with relevant item_ids
  tests/fixtures/queries_image.json  - 20 image queries (product image → relevant ids)

Usage:
    python tests/scripts/generate_fixtures.py \
        --catalog tests/fixtures/catalog_500.jsonl

Requires OPENAI_API_KEY in environment.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "preprocessing"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import argparse
import os


def load_catalog(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_text_queries(catalog: list[dict], client, n: int = 30) -> list[dict]:
    """Use GPT-4o to generate labeled text queries from catalog."""
    # Group by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for p in catalog:
        by_cat[p["category"]].append(p)

    # Pick diverse products
    sampled = []
    cats = list(by_cat.keys())
    random.shuffle(cats)
    for cat in cats:
        products = by_cat[cat]
        sampled.extend(products[:2])
        if len(sampled) >= 60:
            break

    # Build product summaries for GPT
    summaries = []
    for p in sampled[:60]:
        summaries.append({
            "item_id": p["item_id"],
            "category": p["category"],
            "name": p.get("name", "")[:80],
            "description": p.get("description", "")[:200],
            "color": p.get("color", ""),
            "material": p.get("material", ""),
        })

    prompt = f"""You are a shopping search expert. I have a catalog of {len(catalog)} products.
Here are {len(summaries)} sample products:

{json.dumps(summaries, indent=2)}

Generate exactly {n} natural-language shopping search queries that a real user might type.
For each query, list the item_ids of products from the above list that are relevant.
Aim for 1-4 relevant products per query. Include:
- 8 semantic queries (e.g. "something cozy for the living room")
- 8 attribute queries (e.g. "red leather shoes women")
- 7 category queries (e.g. "comfortable office chair")
- 7 cross-attribute queries (e.g. "lightweight chair home office")

Return ONLY a JSON array, no explanation:
[
  {{"query": "...", "relevant_ids": ["item_id1", "item_id2"], "category_hint": "SHOES"}},
  ...
]
"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        # Handle both {"queries": [...]} and [...]
        if isinstance(data, dict):
            queries = data.get("queries", data.get("results", list(data.values())[0]))
        else:
            queries = data
        return queries[:n]
    except Exception as e:
        print(f"Warning: could not parse GPT response: {e}", file=sys.stderr)
        print(content[:500], file=sys.stderr)
        return []


def generate_image_queries(catalog: list[dict], n: int = 20) -> list[dict]:
    """
    Select n products with clean images as image query fixtures.
    The product itself is always relevant; we add same-category items as additional relevant.
    """
    from collections import defaultdict

    by_cat = defaultdict(list)
    for p in catalog:
        path = Path(p["image_path"])
        if path.exists():
            by_cat[p["category"]].append(p)

    queries = []
    seen_cats = set()

    for cat, products in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        if len(queries) >= n:
            break
        if cat in seen_cats:
            continue
        p = products[0]
        # Find 1-2 additional relevant products in same category
        same_cat = [q["item_id"] for q in products[1:3]]
        queries.append({
            "image_path": str(p["image_path"]),
            "item_id": p["item_id"],
            "relevant_ids": [p["item_id"]] + same_cat,
            "category": cat,
        })
        seen_cats.add(cat)

    return queries[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--text-queries", type=int, default=30)
    parser.add_argument("--image-queries", type=int, default=20)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print(f"Loading catalog from {args.catalog}...", file=sys.stderr)
    catalog = load_catalog(args.catalog)
    print(f"Loaded {len(catalog)} products", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Text queries
    print(f"Generating {args.text_queries} text queries via GPT-4o...", file=sys.stderr)
    text_queries = generate_text_queries(catalog, client, n=args.text_queries)
    out_text = args.out_dir / "queries_text.json"
    with open(out_text, "w") as f:
        json.dump(text_queries, f, indent=2)
    print(f"Wrote {len(text_queries)} text queries to {out_text}", file=sys.stderr)

    # Image queries
    print(f"Generating {args.image_queries} image queries...", file=sys.stderr)
    image_queries = generate_image_queries(catalog, n=args.image_queries)
    out_image = args.out_dir / "queries_image.json"
    with open(out_image, "w") as f:
        json.dump(image_queries, f, indent=2)
    print(f"Wrote {len(image_queries)} image queries to {out_image}", file=sys.stderr)


if __name__ == "__main__":
    main()
