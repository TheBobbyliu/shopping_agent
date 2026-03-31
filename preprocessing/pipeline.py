"""
Full preprocessing pipeline: extract → embed → index.

Supports resume: products already present in the Elasticsearch index are
skipped automatically. Use --recreate to drop and rebuild from scratch.

Usage:
    # First run (or full re-run):
    python pipeline.py --index-name products --recreate

    # Resume after interruption:
    python pipeline.py --index-name products

    # Test subset:
    python pipeline.py --limit 500 --stratify --index-name products_test --recreate
"""
from __future__ import annotations

import os
import sys
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
    parser.add_argument("--batch-size",   type=int, default=16,
                        help="Products per batch for embedding (text+image together)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from extract import extract_products
    from embed import EmbeddingClient, _chunks
    from index import ensure_index, get_indexed_ids, index_products
    from catalog import EmbeddedProduct

    # --- Step 1: Extract ---
    print("[pipeline] Step 1: extracting products…", file=sys.stderr)
    products = list(extract_products(
        listings_dir=args.listings_dir,
        images_csv=args.images_csv,
        images_dir=args.images_dir,
        limit=args.limit,
        stratify=args.stratify,
    ))
    print(f"[pipeline] Extracted {len(products)} products", file=sys.stderr)

    # --- Step 2: Embed (with resume via ES) ---
    client = EmbeddingClient.from_env()
    dim = int(os.environ.get("EMBEDDING_DIM", len(client.embed_texts(["warmup"])[0])))

    ensure_index(args.es_url, args.index_name, dim, recreate=args.recreate)

    if args.recreate:
        to_embed = products
    else:
        all_ids = [p.item_id for p in products]
        done_ids = get_indexed_ids(args.es_url, args.index_name, all_ids)
        to_embed = [p for p in products if p.item_id not in done_ids]
        if done_ids:
            print(f"[pipeline] Resuming: {len(done_ids)} already indexed, "
                  f"{len(to_embed)} remaining", file=sys.stderr)

    if not to_embed:
        print("[pipeline] All products already indexed — nothing to do", file=sys.stderr)
        return

    print(f"[pipeline] Step 2: embedding and indexing {len(to_embed)} products…",
          file=sys.stderr)

    total_done = 0
    total = len(to_embed)
    total_errors = 0

    for batch in _chunks(to_embed, args.batch_size):
        texts  = [p.description for p in batch]
        images = [str(p.image_path) for p in batch]

        txt_vecs = client.embed_texts(texts)
        img_vecs = client.embed_images(images)

        embedded = [
            EmbeddedProduct(
                **{k: getattr(p, k) for k in p.__dataclass_fields__},
                description_vector=txt_vec,
                image_vector=img_vec,
            ).to_dict()
            for p, txt_vec, img_vec in zip(batch, txt_vecs, img_vecs)
        ]

        stats = index_products(embedded, es_url=args.es_url, index_name=args.index_name,
                               recreate=False, dim=dim)
        total_done += len(batch)
        total_errors += stats.errors
        print(f"[pipeline]   embedded+indexed {total_done}/{total}", file=sys.stderr)

    print(f"[pipeline] Done — {total_done} indexed, {total_errors} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
