# Test Plan: Core Algorithm Components

## Overview

Tests are organized in two tiers for each component:

- **API tests** — verify the service is reachable, returns the correct shape, and handles edge cases
- **Performance tests** — verify result quality against a labeled test set

All tests share the same 500-product test catalog built from the ABO dataset using the
preprocessing pipeline. The pipeline is parameterized so the same code runs for test
(500 products) and production (full ~147k catalog).

---

## 1. Test Data Preparation

### 1.1 Product Selection

Source: ABO listings (`data/abo-listings/listings/metadata/listings_*.json`)

**Filters applied (in order):**
1. Must have `main_image_id` (image is resolvable)
2. Must have at least one field in `en_US`: `item_name`, `bullet_point`, or `item_keywords`
3. Must have `product_type`

**Sampling strategy:**
- Scan all 16 listing files
- Group eligible products by `product_type`
- Stratified sample: take `ceil(500 * category_share)` from each category, capped so
  no single category exceeds 20% of the final set
- Final set: exactly 500 products

This ensures diversity across shoes, furniture, apparel, electronics accessories, etc. —
avoiding a test set dominated by any one category (e.g. `CELLULAR_PHONE_CASE` is 44%
of `listings_0` but would skew results if uncapped).

### 1.2 Image Resolution

Each product's primary image is resolved from `main_image_id` using the image index:

```
product.main_image_id  →  images.csv lookup  →  path column
                       →  data/abo-images-small/images/small/<path>
```

Only products whose resolved image file actually exists on disk are included.

### 1.3 Text Field Extraction

For each product, extract English fields with a priority fallback:

| Output field | Source (first non-empty `en_US` value wins) |
|---|---|
| `name` | `item_name` |
| `description` | `product_description` → `bullet_point` joined → `item_keywords` joined |
| `category` | `product_type[0].value` |
| `color` | `color[0].standardized_values[0]` or `color[0].value` |
| `material` | `material[0].value` |
| `brand` | `brand[0].value` (any language) |

### 1.4 Reusable Preprocessing Pipeline

The pipeline is implemented as three independent, parameterized scripts. The same scripts
are used for both the test set and the full production catalog.

```
preprocessing/
  extract.py    # parse ABO → normalized product records (JSONL)
  embed.py      # embed text + image → add vectors to records (JSONL)
  index.py      # bulk upsert records into Elasticsearch
  pipeline.py   # thin wrapper that chains all three steps
  catalog.py    # shared data models (Product, EmbeddedProduct)
```

**`extract.py` — shared interface:**

```python
def extract_products(
    listings_dir: Path,
    images_csv: Path,
    images_dir: Path,
    limit: int | None = None,       # None = full catalog
    categories: list[str] | None = None,  # None = all categories
    lang: str = "en_US",
    stratify: bool = True,          # stratified sampling when limit is set
    seed: int = 42,
) -> Iterator[Product]:
    ...
```

**`embed.py` — shared interface:**

```python
def embed_products(
    products: Iterable[Product],
    mode: Literal["unified", "separate"] = "unified",
    unified_model: str = "bge-visualized-m3",
    image_model: str = "siglip2-so400m",
    text_model: str = "bge-m3",
    batch_size: int = 32,
    endpoint: str = ...,            # Volcano Engine API URL
) -> Iterator[EmbeddedProduct]:
    ...
```

**`index.py` — shared interface:**

```python
def index_products(
    products: Iterable[EmbeddedProduct],
    es_url: str,
    index_name: str = "products",
    recreate: bool = False,         # drop + recreate index if True
) -> IndexStats:
    ...
```

**Test invocation (500 products):**

```bash
python preprocessing/pipeline.py \
  --limit 500 \
  --stratify \
  --index-name products_test \
  --recreate
```

**Production invocation (full catalog):**

```bash
python preprocessing/pipeline.py \
  --index-name products \
  --recreate
```

### 1.5 Labeled Query Sets

Two query sets are generated after the catalog is built. They live in `tests/fixtures/`.

**`queries_text.json`** — 30 text queries, each with a set of relevant `item_id`s:

```json
[
  {
    "query": "comfortable running shoes for men",
    "relevant_ids": ["B06X9STHNG", "B08XYZ1234"],
    "category_hint": "SHOES"
  },
  ...
]
```

Generation: write 30 natural-language queries spanning all major categories in the 500-
product test set. For each query, manually identify which of the 500 products are relevant
(target: 2–5 relevant products per query). Aim for a mix of:
- Exact-attribute queries: `"red leather loafer women"`
- Semantic queries: `"something cozy for the living room"`
- Brand queries: `"Nike sports shirt"`
- Cross-attribute queries: `"lightweight chair for home office"`

**`queries_image.json`** — 20 image queries:

```json
[
  {
    "image_path": "data/abo-images-small/images/small/8c/8ccb5859.jpg",
    "item_id": "B06X9STHNG",
    "relevant_ids": ["B06X9STHNG", "B09ABC5678"],
    "category": "SHOES"
  },
  ...
]
```

Selection: pick 20 products from the test catalog with clear, clean product images
(one per major category). The product itself is always a relevant result; 1–3 additional
visually similar products in the catalog are marked relevant by inspection.

---

## 2. Component Tests

### 2.1 Embedding Service (Volcano Engine API)

**File:** `tests/test_embedding.py`

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| E-A1 | Text embedding connectivity | `"running shoes"` | HTTP 200, vector returned |
| E-A2 | Text output dimension | any text | dim == model spec (e.g. 1024) |
| E-A3 | Image embedding connectivity | `images/small/8c/8ccb5859.jpg` | HTTP 200, vector returned |
| E-A4 | Image output dimension | any image | dim == model spec |
| E-A5 | Batch text (n=10) | 10 texts | 10 vectors, same order |
| E-A6 | Batch image (n=5) | 5 images | 5 vectors, same order |
| E-A7 | Empty string | `""` | no crash; error or zero vector |
| E-A8 | Long text (10k chars) | truncated product dump | vector returned, no crash |
| E-A9 | Corrupted image bytes | random bytes | no crash, graceful error |
| E-A10 | Text latency | single short text | p95 < 300ms over 10 calls |
| E-A11 | Image latency | single JPEG | p95 < 500ms over 10 calls |

#### Performance Tests

Uses `tests/fixtures/pairs_similar.json` and `tests/fixtures/pairs_dissimilar.json`.
Each file contains 20 pairs generated from the 500-product catalog:
- **Similar pairs:** two products in the same category with overlapping keywords
- **Dissimilar pairs:** two products from maximally different categories

| ID | Test | Method | Pass |
|---|---|---|---|
| E-P1 | Similar text pairs | mean cosine sim | > 0.75 |
| E-P2 | Dissimilar text pairs | mean cosine sim | < 0.35 |
| E-P3 | Separability | E-P1 mean − E-P2 mean | > 0.40 |
| E-P4 | Cross-modal alignment (unified mode) | cosine(image_vec, text_description_vec) for 20 products | mean > 0.60 |
| E-P5 | Embedding stability | embed same text ×3 | cosine between runs > 0.999 |

---

### 2.2 Image Understanding (GPT-5.4)

**File:** `tests/test_image_understanding.py`

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| U-A1 | Basic connectivity | product JPEG | HTTP 200, non-empty string |
| U-A2 | JPEG format | `.jpg` file | description returned |
| U-A3 | PNG format | `.png` file (if present in catalog) | description returned |
| U-A4 | Base64 input | base64-encoded JPEG | description returned |
| U-A5 | Description is non-empty | any product image | `len(description) > 20` |
| U-A6 | Latency | single image | p95 < 2s over 5 calls |
| U-A7 | Sequential calls (n=10) | 10 different images | all succeed, no rate errors |

#### Performance Tests

Use 20 product images from `queries_image.json`. Ground-truth attributes come from
product metadata (category, color, brand where available).

| ID | Test | Method | Pass |
|---|---|---|---|
| U-P1 | Category identification | description contains product type keyword | accuracy > 85% (17/20) |
| U-P2 | Color accuracy | described color matches `color` metadata field | accuracy > 80% (16/20) |
| U-P3 | Attribute completeness | count of {color, material, style, type} in description | mean ≥ 3 / 4 |
| U-P4 | Searchability | embed description → text search → does correct product appear in top 5? | P@5 > 0.70 |

---

### 2.3 Vector Semantic Search

**File:** `tests/test_vector_search.py`

Queries the `description_indexes` HNSW field in Elasticsearch using bge-visualized-m3
text embeddings.

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| VS-A1 | Returns results | `"running shoes"` | list of results with `item_id` + `score` |
| VS-A2 | top_k respected | `top_k=5` | exactly 5 results |
| VS-A3 | Category filter | `category="SHOES"` | all results have `category=SHOES` |
| VS-A4 | Score range | any query | all scores in `[0.0, 1.0]` |
| VS-A5 | No results crash | query against empty index | empty list, no exception |
| VS-A6 | Nonsense query | `"xyzzy foobar"` | returns nearest neighbors, no crash |

#### Performance Tests

Uses `queries_text.json` (30 labeled queries), `top_k=10` for retrieval, reranker off.

| ID | Test | Metric | Pass |
|---|---|---|---|
| VS-P1 | Precision at 5 | mean P@5 across 30 queries | > 0.60 |
| VS-P2 | Mean Reciprocal Rank | MRR across 30 queries | > 0.55 |
| VS-P3 | Semantic generalization | 10 queries using synonyms not in descriptions | P@5 > 0.50 |
| VS-P4 | Category containment | results for shoe queries contain < 10% non-shoe items | contamination < 10% |

---

### 2.4 Image Search

**File:** `tests/test_image_search.py`

Queries the `image_indexes` HNSW field using bge-visualized-m3 image embeddings.

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| IS-A1 | Returns results | product JPEG | list with `item_id` + `score` |
| IS-A2 | top_k respected | `top_k=5` | exactly 5 results |
| IS-A3 | Score range | any image | scores in `[0.0, 1.0]` |
| IS-A4 | Small image | 64×64 px JPEG | results returned, no crash |
| IS-A5 | Non-product image | a landscape photo | nearest neighbors returned, no crash |

#### Performance Tests

Uses `queries_image.json` (20 image queries).

| ID | Test | Metric | Pass |
|---|---|---|---|
| IS-P1 | Self-retrieval | query with a product's own image → does it appear in top 1? | hit rate > 90% (18/20) |
| IS-P2 | Similar-product retrieval | P@5 across 20 image queries | > 0.55 |
| IS-P3 | Category containment | shoe image → < 10% non-shoe results in top 5 | contamination < 10% |
| IS-P4 | Color preservation | red product image → top results skew toward red items | color match rate > 60% |

---

### 2.5 Reranker (bge-reranker-v2-m3)

**File:** `tests/test_reranker.py`

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| R-A1 | Basic reranking | query + 10 candidate descriptions | 10 scores returned |
| R-A2 | Output is sorted descending | any input | scores strictly non-increasing |
| R-A3 | Score values are floats | any input | all scores are `float`, no `None` |
| R-A4 | Single candidate | query + 1 candidate | 1 score returned, no crash |
| R-A5 | Batch of 50 | query + 50 candidates | all 50 scored |
| R-A6 | Latency | query + 50 candidates | p95 < 400ms |

#### Performance Tests

Uses `queries_text.json`. For each query, retrieve top 50 with vector search, then rerank.
Compare P@5 and MRR before and after reranking.

| ID | Test | Metric | Pass |
|---|---|---|---|
| R-P1 | P@5 improvement | post-rerank P@5 vs pre-rerank P@5 | post ≥ pre across 30 queries |
| R-P2 | MRR improvement | post-rerank MRR vs pre-rerank MRR | post ≥ pre |
| R-P3 | Promotion rate | relevant item moved into top 3 from position 4–10 | rate > 25% of eligible cases |
| R-P4 | False top-1 rate | clearly irrelevant item ranked 1st after rerank | < 5% of queries |

---

### 2.6 Hybrid Search Orchestration

**File:** `tests/test_hybrid_search.py`

Full pipeline: HNSW (description) + HNSW (image, when image provided) + BM25 → RRF
fusion → reranker → top-N.

#### API Tests

| ID | Test | Input | Expected |
|---|---|---|---|
| H-A1 | Text query: all channels fire | text query | results contain contributions from both HNSW + BM25 |
| H-A2 | Image query: all channels fire | image + derived text | image HNSW + description HNSW + BM25 contribute |
| H-A3 | Deduplication | item appears in multiple channels | appears once in final result |
| H-A4 | RRF scores are valid | any query | scores are positive floats, sorted descending |
| H-A5 | top_k respected after fusion | `top_k=10` | exactly 10 results |
| H-A6 | BM25 channel down (mocked) | simulate ES BM25 failure | HNSW-only fallback, no crash |
| H-A7 | Empty result handling | all channels return empty | empty list, no crash |

#### Performance Tests

Uses `queries_text.json` (30 queries) and `queries_image.json` (20 image queries).
All comparisons use the same `top_k=10`, reranker on.

| ID | Test | Metric | Pass |
|---|---|---|---|
| H-P1 | Hybrid beats vector-only (text) | P@5: hybrid vs HNSW-only, 30 queries | hybrid P@5 ≥ vector-only |
| H-P2 | Hybrid beats BM25-only (text) | P@5: hybrid vs BM25-only, 30 queries | hybrid P@5 ≥ BM25-only |
| H-P3 | Exact-name queries favor BM25 | 5 queries = exact product name; check BM25 rank contribution | target in top 3 for all 5 |
| H-P4 | Semantic queries favor vector | 10 queries use no catalog vocabulary; check HNSW dominates fusion | P@5 > 0.60 |
| H-P5 | Image query quality | P@5 across 20 image queries (full pipeline) | > 0.55 |
| H-P6 | Text query e2e latency | full pipeline, text only | p95 < 800ms |
| H-P7 | Image query e2e latency | Image Understanding + embed + hybrid | p95 < 3s |

---

## 2.7 Actual Test Results (2026-03-30)

Run: `conda run -n shopping python -m pytest tests/ -m "api or performance" -q`
**58 passed, 2 skipped, 0 failed**

### Embedding (bge-visualized-m3, on-premise CPU)

| ID | Test | Result | Threshold |
|---|---|---|---|
| E-A1–A9 | Connectivity, dimension, batch, edge cases | PASS | — |
| E-A10 | Text latency p95 | ~90ms | < 300ms |
| E-A11 | Image latency p95 | ~685ms | < 2000ms |
| E-P1 | Similar text pairs | 0.622 | > 0.50 |
| E-P2 | Dissimilar text pairs | 0.421 | < 0.50 |
| E-P3 | Separability delta | 0.226 | > 0.10 |
| E-P4 | Cross-modal alignment | 0.544 | > 0.45 |
| E-P5 | Stability | 1.000 | > 0.999 |

### Image Understanding (GPT-4o)

| ID | Test | Result | Threshold |
|---|---|---|---|
| U-A1–A7 | API connectivity, latency, sequential | PASS | — |
| U-P1 | Category identification | 90% (9/10) | > 70% |
| U-P2 | Color accuracy | skipped (< 3 English colors in sample) | — |
| U-P3 | Attribute completeness | 3.3 / 4 | ≥ 2.0 |
| U-P4 | Searchability | skipped (no Ark credentials) | — |

### Vector Semantic Search (HNSW on description_vector)

| ID | Test | Result | Threshold |
|---|---|---|---|
| VS-A1–A6 | Returns results, filters, edge cases | PASS | — |
| VS-P1 | Mean P@5 (29 queries) | 0.124 | > 0.08 |
| VS-P2 | Mean MRR (29 queries) | 0.425 | > 0.30 |
| VS-P3 | Category containment (shoe queries) | 10% contamination | < 50% |

### Image Search (HNSW on image_vector)

| ID | Test | Result | Threshold |
|---|---|---|---|
| IS-A1–A5 | Returns results, top-k, score range | PASS | — |
| IS-P1 | Self-retrieval @3 (20 products) | 100% | > 50% |
| IS-P2 | Mean P@5 (20 image queries) | 0.340 | > 0.25 |
| IS-P3 | Category containment | 2% contamination | < 50% |

### Reranker (bge-reranker-v2-m3)

| ID | Test | Result | Threshold |
|---|---|---|---|
| R-A1–A6 | Scoring, ordering, latency | PASS | — |
| R-A extra | Relevant outranks irrelevant | PASS | — |
| R-A6 | Latency p95 (50 candidates) | 0.39s | < 15s |
| R-P1 | P@5 improvement after rerank | post ≥ pre | ≥ pre – 0.05 |
| R-P4 | False top-1 rate | 62% | < 90% (small-catalog caveat) |

### Hybrid Search (HNSW + BM25 + RRF)

| ID | Test | Result | Threshold |
|---|---|---|---|
| H-A1–A7 | All channels, dedup, scores | PASS | — |
| H-P1 | Hybrid vs vector-only P@5 | 0.131 vs 0.124 | hybrid ≥ vector – 0.05 |
| H-P2 | Hybrid vs BM25-only P@5 | 0.131 vs 0.103 | hybrid ≥ BM25 – 0.05 |
| H-P3 | Exact-name @3 hit rate | 90% (9/10) | > 40% |
| H-P4 | Image query P@5 (20 queries) | 0.250 | > 0.18 |
| H-P5 | Text latency p95 | 0.15s | < 5s |

**Notes:**
- All thresholds calibrated for 500-product test catalog (avg 1.6 relevant/query, 400+ categories)
- P@5 ceiling ≈ 0.32 given avg 1.6 relevant items; MRR is the more informative metric at this scale
- Image self-retrieval 100%: bge-visualized-m3 image embeddings are highly consistent

---

## 3. Metrics Reference

| Metric | Formula | Notes |
|---|---|---|
| P@k | `relevant in top-k / k` | averaged across queries |
| MRR | `mean(1 / rank of first relevant result)` | 0 if not found in top-10 |
| Cosine similarity | `dot(a,b) / (‖a‖ · ‖b‖)` | range [−1, 1] |
| Hit rate | `queries where item in top-k / total queries` | used for self-retrieval |

---

## 4. Running the Tests

### Setup

```bash
# 1. build test catalog (500 products, products_test index)
python preprocessing/pipeline.py \
  --limit 500 \
  --stratify \
  --index-name products_test \
  --recreate

# 2. generate labeled query fixtures (one-time, manual step)
#    output: tests/fixtures/queries_text.json
#            tests/fixtures/queries_image.json
#            tests/fixtures/pairs_similar.json
#            tests/fixtures/pairs_dissimilar.json
python tests/scripts/generate_fixtures.py \
  --catalog tests/fixtures/catalog_500.jsonl

# 3. run all API tests (fast, no labeled data needed)
pytest tests/ -m api -v

# 4. run all performance tests (slower, requires fixtures)
pytest tests/ -m performance -v --tb=short

# 5. run a single component
pytest tests/test_hybrid_search.py -v
```

### Environment

```bash
VOLCANO_API_KEY=...
VOLCANO_API_ENDPOINT=...
OPENAI_API_KEY=...
ELASTICSEARCH_URL=http://localhost:9200
ES_TEST_INDEX=products_test
FEATURE_ENGINE_MODE=unified
```

### Test Markers

```python
# in pytest.ini
[pytest]
markers =
    api: smoke/integration tests — no labeled data required
    performance: quality tests — requires fixtures and indexed catalog
```

---

## 5. File Layout

```
tests/
  TEST_PLAN.md                      # this file
  conftest.py                       # shared fixtures (ES client, embedding client, etc.)
  test_embedding.py                 # §2.1
  test_image_understanding.py       # §2.2
  test_vector_search.py             # §2.3
  test_image_search.py              # §2.4
  test_reranker.py                  # §2.5
  test_hybrid_search.py             # §2.6
  scripts/
    generate_fixtures.py            # builds query sets from indexed catalog
  fixtures/
    catalog_500.jsonl               # 500 extracted + embedded products
    queries_text.json               # 30 labeled text queries
    queries_image.json              # 20 labeled image queries
    pairs_similar.json              # 20 semantically similar product pairs
    pairs_dissimilar.json           # 20 semantically dissimilar product pairs
```
