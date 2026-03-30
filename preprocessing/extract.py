"""
Extract and normalize product records from the ABO dataset.

Reusable for both test (limit=500) and production (limit=None) pipelines.

Usage:
    # as a module
    products = list(extract_products(listings_dir, images_csv, images_dir, limit=500))

    # as a script
    python preprocessing/extract.py \
        --listings data/abo-listings/listings/metadata \
        --images-csv data/abo-images-small/images/metadata/images.csv \
        --images-dir data/abo-images-small/images/small \
        --limit 500 --stratify --out data/catalog_500.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from catalog import Product


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def _pick_en(field: list[dict], key: str = "value", lang: str = "en_US") -> str:
    """Return the first value matching lang, or first value of any language."""
    for item in field:
        if item.get("language_tag", "") == lang:
            return item.get(key, "").strip()
    # fallback: any language
    for item in field:
        val = item.get(key, "").strip()
        if val:
            return val
    return ""


def _build_description(record: dict, lang: str = "en_US") -> str:
    """Build a searchable description from the richest available text fields."""
    parts: list[str] = []

    # product_description (HTML stripped)
    if "product_description" in record:
        raw = _pick_en(record["product_description"], lang=lang)
        if raw:
            parts.append(raw)

    # bullet_points joined
    if "bullet_point" in record:
        bullets = [_pick_en([bp], lang=lang) for bp in record["bullet_point"]]
        bullets = [b for b in bullets if b]
        if bullets:
            parts.append(" ".join(bullets))

    # keywords joined
    if "item_keywords" in record:
        kws = [_pick_en([kw], lang=lang) for kw in record["item_keywords"]]
        kws = [k for k in kws if k]
        if kws:
            parts.append(" ".join(kws))

    return " ".join(parts).strip()


def _build_keywords(record: dict, lang: str = "en_US") -> list[str]:
    return [
        _pick_en([kw], lang=lang)
        for kw in record.get("item_keywords", [])
        if _pick_en([kw], lang=lang)
    ]


def _build_bullet_points(record: dict, lang: str = "en_US") -> list[str]:
    return [
        _pick_en([bp], lang=lang)
        for bp in record.get("bullet_point", [])
        if _pick_en([bp], lang=lang)
    ]


def _has_english(record: dict, lang: str = "en_US") -> bool:
    for field_name in ("item_name", "bullet_point", "item_keywords"):
        for item in record.get(field_name, []):
            if item.get("language_tag", "") == lang:
                return True
    return False


# ---------------------------------------------------------------------------
# Image index loader
# ---------------------------------------------------------------------------

def _load_image_index(images_csv: Path) -> dict[str, str]:
    """
    Load images.csv (or images.csv.gz) into a dict: image_id -> relative path.
    E.g. {"81iZlv3bjpL": "8c/8ccb5859.jpg", ...}
    """
    opener = gzip.open if str(images_csv).endswith(".gz") else open
    index: dict[str, str] = {}
    with opener(images_csv, "rt") as f:
        header = f.readline().strip().split(",")
        id_col = header.index("image_id")
        path_col = header.index("path")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > max(id_col, path_col):
                index[parts[id_col]] = parts[path_col]
    return index


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _iter_listing_files(listings_dir: Path) -> Iterator[Path]:
    """Yield all .json and .json.gz listing files in the directory."""
    for p in sorted(listings_dir.iterdir()):
        if p.name.endswith(".json") or p.name.endswith(".json.gz"):
            yield p


def _parse_record(
    record: dict,
    image_index: dict[str, str],
    images_dir: Path,
    lang: str = "en_US",
) -> Optional[Product]:
    """
    Convert one raw ABO record to a Product, or return None if it fails filters.
    Filters: must have main_image_id, must have English content, image file must exist.
    """
    item_id = record.get("item_id", "")
    if not item_id:
        return None

    if not _has_english(record, lang):
        return None

    main_image_id = record.get("main_image_id", "")
    if not main_image_id:
        return None

    rel_path = image_index.get(main_image_id)
    if not rel_path:
        return None

    image_path = images_dir / rel_path
    if not image_path.exists():
        return None

    name = _pick_en(record.get("item_name", []), lang=lang)
    if not name:
        return None

    description = _build_description(record, lang=lang)
    if not description:
        # fallback: use name as description
        description = name

    category = (record.get("product_type") or [{}])[0].get("value", "UNKNOWN")

    brand_raw = record.get("brand", [])
    brand = brand_raw[0].get("value", "") if brand_raw else ""

    color_raw = record.get("color", [])
    if color_raw:
        sv = color_raw[0].get("standardized_values", [])
        color = sv[0] if sv else _pick_en([color_raw[0]], lang=lang)
    else:
        color = ""

    material_raw = record.get("material", [])
    material = _pick_en(material_raw[0:1], lang=lang) if material_raw else ""

    domain = record.get("domain_name", "")
    web_url = f"https://{domain}/dp/{item_id}" if domain else ""

    dims = {}
    if "item_dimensions" in record:
        for axis, val in record["item_dimensions"].items():
            nv = val.get("normalized_value", {})
            if nv:
                dims[axis] = f"{nv.get('value', '')} {nv.get('unit', '')}".strip()

    weight = ""
    if "item_weight" in record:
        w = record["item_weight"]
        if w:
            nv = w[0].get("normalized_value", {})
            if nv:
                weight = f"{nv.get('value', '')} {nv.get('unit', '')}".strip()

    return Product(
        item_id=item_id,
        name=name,
        description=description,
        category=category,
        brand=brand,
        color=color,
        material=material,
        image_path=image_path,
        image_url=rel_path,
        web_url=web_url,
        keywords=_build_keywords(record, lang=lang),
        bullet_points=_build_bullet_points(record, lang=lang),
        metadata={
            "domain": domain,
            "dimensions": dims,
            "weight": weight,
            "model_year": (record.get("model_year") or [{}])[0].get("value"),
        },
    )


def extract_products(
    listings_dir: Path,
    images_csv: Path,
    images_dir: Path,
    limit: Optional[int] = None,
    categories: Optional[list[str]] = None,
    lang: str = "en_US",
    stratify: bool = True,
    seed: int = 42,
) -> Iterator[Product]:
    """
    Extract and normalize products from the ABO dataset.

    Args:
        listings_dir:  Path to directory containing listings_*.json[.gz] files.
        images_csv:    Path to images.csv or images.csv.gz.
        images_dir:    Path to directory containing the image files (images/small/).
        limit:         Max products to yield. None = full catalog.
        categories:    If set, only include products in these product_type values.
        lang:          Language tag to prefer for text fields.
        stratify:      When limit is set, sample proportionally across categories
                       so no single category exceeds 20% of the result.
        seed:          Random seed for reproducible sampling.
    """
    print(f"Loading image index from {images_csv}...", file=sys.stderr)
    image_index = _load_image_index(images_csv)
    print(f"  Loaded {len(image_index):,} image records", file=sys.stderr)

    # --- pass 1: collect all valid products grouped by category ---
    by_category: dict[str, list[Product]] = defaultdict(list)
    total_parsed = 0

    for listing_file in _iter_listing_files(listings_dir):
        opener = gzip.open if str(listing_file).endswith(".gz") else open
        with opener(listing_file, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_parsed += 1
                product = _parse_record(record, image_index, images_dir, lang)
                if product is None:
                    continue
                if categories and product.category not in categories:
                    continue
                by_category[product.category].append(product)

    total_valid = sum(len(v) for v in by_category.values())
    print(
        f"  Parsed {total_parsed:,} records → {total_valid:,} valid products "
        f"across {len(by_category)} categories",
        file=sys.stderr,
    )
    for cat, prods in sorted(by_category.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {len(prods):5d}  {cat}", file=sys.stderr)

    # --- pass 2: sample if limit set ---
    if limit is None:
        for products in by_category.values():
            yield from products
        return

    rng = random.Random(seed)
    result: list[Product] = []

    if not stratify:
        all_products = [p for prods in by_category.values() for p in prods]
        rng.shuffle(all_products)
        yield from all_products[:limit]
        return

    # stratified: cap each category at 20% of limit
    max_per_category = max(1, int(limit * 0.20))
    total = sum(len(v) for v in by_category.values())

    for cat, products in by_category.items():
        share = len(products) / total
        n = min(max_per_category, max(1, round(limit * share)))
        sampled = rng.sample(products, min(n, len(products)))
        result.extend(sampled)

    # fill up to limit if we're short (can happen due to rounding)
    if len(result) < limit:
        used_ids = {p.item_id for p in result}
        remaining = [
            p
            for prods in by_category.values()
            for p in prods
            if p.item_id not in used_ids
        ]
        rng.shuffle(remaining)
        result.extend(remaining[: limit - len(result)])

    rng.shuffle(result)
    result = result[:limit]
    print(f"  Sampled {len(result)} products (stratified)", file=sys.stderr)
    yield from result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract products from ABO dataset")
    parser.add_argument("--listings", type=Path, required=True)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.out, "w") as f:
        for product in extract_products(
            listings_dir=args.listings,
            images_csv=args.images_csv,
            images_dir=args.images_dir,
            limit=args.limit,
            categories=args.categories,
            stratify=args.stratify,
            seed=args.seed,
        ):
            f.write(json.dumps(product.to_dict()) + "\n")
            count += 1

    print(f"Wrote {count} products to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
