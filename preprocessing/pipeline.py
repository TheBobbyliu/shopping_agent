"""
Full preprocessing pipeline: extract → embed → index.

Usage:
    python pipeline.py --limit 500 --stratify --index-name products_test --recreate
    python pipeline.py --index-name products --recreate   # full catalog
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def main():
    import argparse
    load_dotenv()

    DATA_DIR = Path(__file__).parent.parent / "data"

    parser = argparse.ArgumentParser(description="End-to-end preprocessing pipeline")
    parser.add_argument("--listings-dir", type=Path,
                        default=DATA_DIR / "abo-listings/listings/metadata")
    parser.add_argument("--images-csv",   type=Path,
                        default=DATA_DIR / "abo-images-small/images/metadata/images.csv")
    parser.add_argument("--images-dir",   type=Path,
                        default=DATA_DIR / "abo-images-small/images/small")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--stratify",     action="store_true")
    parser.add_argument("--index-name",   default="products")
    parser.add_argument("--es-url",       default=os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--recreate",     action="store_true")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--out",          type=Path, default=None,
                        help="Optional: save embedded JSONL to this path")
    args = parser.parse_args()

    # --- Step 1: Extract ---
    sys.path.insert(0, str(Path(__file__).parent))
    from extract import extract_products
    from embed import EmbeddingClient, _chunks
    from index import index_products
    from catalog import Product, EmbeddedProduct

    print("[pipeline] Step 1: extracting products…", file=sys.stderr)
    products = list(extract_products(
        listings_dir=args.listings_dir,
        images_csv=args.images_csv,
        images_dir=args.images_dir,
        limit=args.limit,
        stratify=args.stratify,
    ))
    print(f"[pipeline] Extracted {len(products)} products", file=sys.stderr)

    # --- Step 2: Embed ---
    print("[pipeline] Step 2: embedding…", file=sys.stderr)
    client = EmbeddingClient.from_env()

    descriptions = [p.description for p in products]
    image_paths  = [p.image_path  for p in products]

    text_vecs: list[list[float]] = []
    for i, batch in enumerate(_chunks(descriptions, args.batch_size)):
        vecs = client.embed_texts(batch)
        text_vecs.extend(vecs)
        done = min((i + 1) * args.batch_size, len(products))
        print(f"[pipeline]   text {done}/{len(products)}", file=sys.stderr)

    image_vecs: list[list[float]] = []
    for i, batch in enumerate(_chunks(image_paths, 15)):
        vecs = client.embed_images(batch)
        image_vecs.extend(vecs)
        done = min((i + 1) * 15, len(products))
        print(f"[pipeline]   image {done}/{len(products)}", file=sys.stderr)

    embedded = []
    for p, img_vec, txt_vec in zip(products, image_vecs, text_vecs):
        ep = EmbeddedProduct(
            **{k: getattr(p, k) for k in p.__dataclass_fields__},
            image_vector=img_vec,
            description_vector=txt_vec,
        )
        embedded.append(ep)

    # --- Optional: save JSONL ---
    save_path = args.out
    if save_path is None and args.limit:
        save_path = Path(__file__).parent.parent / "tests" / "fixtures" / f"embedded_{args.limit}.jsonl"

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for ep in embedded:
                f.write(json.dumps(ep.to_dict()) + "\n")
        print(f"[pipeline] Saved embedded products to {save_path}", file=sys.stderr)

    # --- Step 3: Index ---
    dim = int(os.environ.get("EMBEDDING_DIM", len(text_vecs[0]) if text_vecs else 1024))
    print(f"[pipeline] Step 3: indexing into '{args.index_name}'…", file=sys.stderr)
    stats = index_products(
        [ep.to_dict() for ep in embedded],
        es_url=args.es_url,
        index_name=args.index_name,
        recreate=args.recreate,
        dim=dim,
    )
    print(f"[pipeline] Done — {stats.indexed} indexed, {stats.errors} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
