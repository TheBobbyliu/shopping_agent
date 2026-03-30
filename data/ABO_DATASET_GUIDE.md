# ABO Dataset Structure Guide

The ABO (Amazon Berkeley Objects) dataset contains product listings and associated images. This guide explains how the dataset is organized and how to cross-reference between metadata and image files.

---

## Directory Structure

```
data/
├── README.md
├── abo-listings.tar               (archived source, 87.5 MB)
├── abo-images-small.tar           (archived source, 3.25 GB)
│
├── abo-listings/                  (137 MB total)
│   ├── LICENSE-CC-BY-4.0.txt
│   └── listings/
│       ├── README.md              (full schema reference)
│       └── metadata/
│           ├── listings_0.json    (decompressed copy)
│           ├── listings_0.json.gz
│           ├── listings_1.json.gz
│           ├── ...
│           └── listings_f.json.gz (16 files total, hex-named 0–f)
│
└── abo-images-small/              (3.5 GB total)
    ├── LICENSE-CC-BY-4.0.txt
    ├── README.md
    └── images/
        ├── metadata/
        │   └── images.csv.gz      (index of all images)
        └── small/
            ├── 00/                (256 hex-named subdirectories)
            ├── 01/
            ├── ...
            └── ff/
```

---

## Metadata Files

### Product Listings — `abo-listings/listings/metadata/listings_*.json.gz`

- **Format:** JSONL (one JSON object per line), gzip-compressed
- **Count:** 16 files (`listings_0` through `listings_f`), ~9,200 products each
- **Total products:** ~147,702

Each line is a JSON object representing one product. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Product ID (e.g. `"B06X9STHNG"`) |
| `domain_name` | string | Amazon marketplace domain (e.g. `"amazon.nl"`) |
| `item_name` | multilingual array | Product name |
| `brand` | multilingual array | Brand name |
| `product_type` | string array | Category (e.g. `"SHOES"`) |
| `color` | multilingual array | Color with optional `standardized_values` |
| `color_code` | string array | HTML hex color codes |
| `material` | multilingual array | Material description |
| `item_dimensions` | object | `{height, width, length}` each with `{unit, value, normalized_value}` |
| `item_weight` | array | `{unit, value, normalized_value}` |
| `main_image_id` | string | **Primary image ID — links to `images.csv`** |
| `other_image_id` | string array | **Additional image IDs — each links to `images.csv`** |
| `spin_id` | string | Reference to 360° spin sequence |
| `3dmodel_id` | string | Reference to 3D model asset |
| `marketplace` | string | Retailer name |
| `country` | string | ISO 3166-1 alpha-2 country code |
| `node` | array | Category tree: `{node_id, node_name, path}` |
| `bullet_point` | multilingual array | Feature bullet points |
| `item_keywords` | multilingual array | Search keywords |
| `product_description` | multilingual array | HTML product description |

**Unique product key:** `(item_id, domain_name)` pair.

**Multilingual fields** are arrays of `{language_tag, value}` objects, where `language_tag` follows the `language_COUNTRY` format (e.g. `"en_US"`, `"nl_NL"`):

```json
"item_name": [
  {"language_tag": "en_US", "value": "Women's Leather Sneaker"},
  {"language_tag": "nl_NL", "value": "Dames Leder Sneaker"}
]
```

---

### Image Index — `abo-images-small/images/metadata/images.csv.gz`

- **Format:** CSV, gzip-compressed
- **Total records:** 398,212 images

| Column | Type | Description |
|--------|------|-------------|
| `image_id` | string | Unique image identifier — matches `main_image_id` / `other_image_id` in listings |
| `height` | integer | Original image height in pixels |
| `width` | integer | Original image width in pixels |
| `path` | string | Relative path under `images/small/` (e.g. `"8c/8ccb5859.jpg"`) |

Sample rows:

```csv
image_id,height,width,path
81iZlv3bjpL,2560,1969,8c/8ccb5859.jpg
91mIRxgziUL,1500,1500,3a/3af21b90.jpg
```

---

## Image Files — `abo-images-small/images/small/`

Images are JPEG files (a few PNG), downscaled to a maximum of 256px on the longest axis.

**Directory structure:** files are spread across 256 subdirectories named with two hex characters (`00`–`ff`). The subdirectory is the first two characters of the image filename.

```
images/small/
    8c/
        8ccb5859.jpg    ← filename is 8-character hex
    3a/
        3af21b90.jpg
    ...
```

**Full path to any image:**

```
abo-images-small/images/small/<path>
```

where `<path>` is the value from the `path` column in `images.csv.gz`.

---

## Cross-Referencing: Listings ↔ Images

The `image_id` field is the shared key between the two metadata files.

### From a product listing → image file

1. Read `main_image_id` (or any entry in `other_image_id`) from a listing record.
2. Look up that ID in `images.csv.gz` to get the `path`, `height`, and `width`.
3. Prepend `abo-images-small/images/small/` to `path` to get the full file location.

```
listings_*.json.gz
  └─ main_image_id: "81iZlv3bjpL"
          │
          ▼
images.csv.gz  →  image_id="81iZlv3bjpL", path="8c/8ccb5859.jpg", height=2560, width=1969
          │
          ▼
abo-images-small/images/small/8c/8ccb5859.jpg
```

### From an image file → product listing

1. Derive the `image_id` by looking up the filename's path in `images.csv.gz` (match on `path` column).
2. Search listing files for records where `main_image_id` or `other_image_id` contains that `image_id`.

---

## Summary Statistics

| Item | Count / Size |
|------|-------------|
| Product listings | ~147,702 |
| Catalog images | 398,212 |
| Listing metadata files | 16 (hex 0–f) |
| Image subdirectories | 256 |
| Listings storage | 137 MB |
| Images storage | 3.5 GB |
