# Shopping Agent

An AI-powered shopping assistant for e-commerce, inspired by [Amazon Rufus](https://www.aboutamazon.com/news/retail/amazon-rufus). A single conversational agent that handles product recommendations, image-based search, and general Q&A — backed by the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset.

For system architecture, API reference, and design details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Features

- **General conversation** — ask what the agent can do, get contextual help
- **Text-based product recommendation** — describe what you're looking for in natural language
- **Image-based product search** — upload a photo to find visually and semantically similar items
- **Product Q&A** — ask follow-up questions about recommended products

---

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for Elasticsearch + PostgreSQL)
- `OPENAI_API_KEY` (for the agent LLM and image understanding via GPT-4o)
- `ANTHROPIC_API_KEY` (optional, as Claude fallback for the agent)

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd shopping-demo

# Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend-next
npm install
```

### 2. Start infrastructure

```bash
docker-compose up -d   # starts Elasticsearch + PostgreSQL
```

### 3. Configure environment

```bash
cp .env.example .env
# Fill in:
#   OPENAI_API_KEY      (required — agent LLM + image understanding)
#   ANTHROPIC_API_KEY   (optional — Claude fallback for agent)
#   LANGSMITH_API_KEY   (optional, for tracing)
#   ELASTICSEARCH_URL   (default: http://localhost:9200)
#   DATABASE_URL        (default: postgresql://shopping:shopping@localhost:5432/shopping)
#   API_BASE_URL        (default: http://localhost:8000 — used to build upload image URLs)
#   ES_INDEX            (default: products)
```

### 4. Load the product catalog

The pipeline runs three steps in one command: **extract → embed → index**.
Resume is automatic — if the run is interrupted, re-running picks up from the last indexed batch.

#### Full catalog (~26K products)

```bash
python preprocessing/pipeline.py \
  --index-name products \
  --recreate
```

#### Resume after interruption

```bash
# Same command, without --recreate (checks which IDs are already in ES and skips them)
python preprocessing/pipeline.py \
  --index-name products
```

#### Quick test subset (~20 min)

```bash
python preprocessing/pipeline.py \
  --index-name products_test \
  --recreate \
  --limit 500 \
  --stratify
```

`--stratify` samples proportionally across categories so no single category exceeds 20% of the result.

#### All pipeline options

| Flag | Default | Description |
|------|---------|-------------|
| `--index-name` | `products` | Elasticsearch index to create/populate |
| `--recreate` | off | Drop and recreate the index before indexing |
| `--batch-size` | `16` | Products per embedding batch (text + image together) |
| `--limit` | None | Cap number of products (omit for full catalog) |
| `--stratify` | off | When `--limit` set, sample proportionally across categories |
| `--listings-dir` | `data/abo-listings/listings/metadata` | ABO listing files directory |
| `--images-csv` | `data/abo-images-small/images/metadata/images.csv` | Image metadata CSV |
| `--images-dir` | `data/abo-images-small/images/small` | Image files directory |

#### Relevant `.env` variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_INDEX` | `products` | Index the API serves from |
| `ES_TEST_INDEX` | `products_test` | Index used by pytest |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | BGE text model for local backend |
| `EMBEDDING_WEIGHT` | `Visualized_m3.pth` | Visual weight file (downloaded from HF if missing) |
| `EMBEDDING_DIM` | `1024` | Vector dimension (must match model) |

### 5. Run

```bash
# API server (from repo root)
cd api && uvicorn main:app --reload --port 8000

# Frontend
cd frontend-next && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Warehouse Management

The warehouse CLI lets operational staff add, delete, and inspect products in the Elasticsearch index without touching the database directly. The API server must be running (the CLI routes embedding calls through it).

```bash
cd preprocessing
```

### Add products

Prepare a JSON file (array of objects). Required fields: `item_id`, `description`, `image_path`. All other fields are stored as-is.

```json
[
  {
    "item_id": "B123456789",
    "description": "Samsung 65-inch 4K QLED TV with HDR",
    "image_path": "data/images/tv.jpg",
    "name": "Samsung QN65Q80C",
    "brand": "Samsung",
    "color": "Black"
  }
]
```

```bash
python warehouse.py add items.json
```

Output:
```
[1/1] B123456789 — embedding... done

Added: 1, Skipped: 0, Errors: 0
```

Already-indexed items are skipped automatically:
```
[skip] B123456789 — already indexed

Added: 0, Skipped: 1, Errors: 0
```

### Delete products

```json
[{"item_id": "B123456789"}, {"item_id": "B987654321"}]
```

```bash
python warehouse.py delete items.json
```

Output:
```
Deleted: 2, Not found: 0, Errors: 0
```

Missing items produce a warning and are skipped — other deletions in the same file still proceed.

### Check products

```bash
# Single item by ID (vectors stripped from output)
python warehouse.py check --item-id B123456789

# All items in a category (default 10)
python warehouse.py check --group ELECTRONICS

# All items in a category, custom count
python warehouse.py check --group FURNITURE --count 5

# Any N items across all categories
python warehouse.py check --count 20
```

Output is pretty-printed JSON. Vector fields (`description_vector`, `image_vector`) are always stripped.

---

## License

The source code in this repository is licensed under the MIT License. See
[`LICENSE`](LICENSE).

Data and other third-party content under [`data/`](data/) are licensed
separately. In particular, the Amazon Berkeley Objects dataset remains licensed
under CC BY 4.0 as documented in [`data/README.md`](data/README.md).
