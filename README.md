# Shopping Agent

An AI-powered shopping assistant for e-commerce, inspired by [Amazon Rufus](https://www.aboutamazon.com/news/retail/amazon-rufus). A single conversational agent that handles product recommendations, image-based search, and general Q&A — backed by the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset.

---

## Features

- **General conversation** — ask what the agent can do, get contextual help
- **Text-based product recommendation** — describe what you're looking for in natural language
- **Image-based product search** — upload a photo to find visually and semantically similar items
- **Product Q&A** — ask follow-up questions about recommended products

All three are handled by a single agent using intent analysis and tool routing.

---

## Architecture

### Data Preparation (Offline)

```
ABO Dataset
  └─ Data Extraction (images, metadata, descriptions)
       └─ Data Processing
            ├─ data cleaning & normalization
            ├─ image embedding extraction  (bge-visualized-m3)
            └─ text embedding extraction   (bge-visualized-m3)
                  └─→ Elasticsearch  (description_vector + image_vector per product)
```

### Data Pipeline (Online)

```
Web UI (chat + image upload)
  └─→ API  (saves upload → /uploads/{uuid}.jpg)
        └─→ Agent  [LangGraph ReAct]
              └─ message: "user query\n\n[Image URL: http://.../uploads/{uuid}.jpg]"
                    │
                    ├─→ product_search ─────────────────────→ Search Engine
                    │     - query: str                           ├─ 1. download image from URL
                    │     - image_url: str (optional)            ├─ 2. GPT-4o → text description
                    │     - top_k: int                           ├─ hybrid search
                    │     - return: product candidates           │    a. description vector (HNSW)
                    │                                            │    b. image vector (HNSW)
                    ├─→ get_product_info                         │    c. keyword match (BM25)
                    │     - item_id: str                         │    d. hybrid scoring (RRF)
                    │     - return: full product details         └─→ bge-reranker-v2-m3
                    │                                                  └─→ top-N candidates
                    └─→ understand_image
                          - image_url: str
                          - return: text description (via GPT-4o)
                            [used only when user asks "what is this?",
                             not needed for search — product_search handles it internally]
```

**Context** is managed by LangGraph's PostgresSaver checkpointer. If PostgreSQL is unavailable, the agent falls back to stateless mode (client supplies history).

### Search Flow by Query Type

| Query type | Embedding channels | Reranker |
|---|---|---|
| Text recommendation | description HNSW + BM25 | bge-reranker-v2-m3 |
| Image search | description HNSW + image HNSW + BM25 | bge-reranker-v2-m3 |
| Product Q&A | exact lookup by item_id | — |

Hybrid score fusion uses **Reciprocal Rank Fusion (RRF)** across all active channels before reranking.

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Frontend | Next.js | Fast dev, easy deployment, good streaming support |
| Agent API | FastAPI | Async, auto OpenAPI docs, Python ML ecosystem |
| Orchestration | LangGraph (ReAct) | Prebuilt ReAct agent, checkpointing, native streaming |
| Logging / Tracing | LangSmith | Full trace visualization per request, token usage per step |
| LLM | GPT-5.4 (primary) / Claude (fallback) | Strong instruction following, tool use |
| Image Understanding | GPT-4o | Converts product images to search descriptions (inside search engine) |
| Embeddings (image + text) | bge-visualized-m3 | Shared embedding space for text and images — enables cross-modal search |
| Reranker | bge-reranker-v2-m3 | Cross-encoder, same BGE family, no extra infra |
| Search + Vector DB | Elasticsearch | HNSW vector search + BM25 in one system, mature ops, scalable |
| Context DB | PostgreSQL | LangGraph PostgresSaver checkpointer |
| Product Catalog | ABO Dataset | Real-world Amazon product data with images and rich metadata |

### Embedding Model Config

The pipeline uses the on-premise **Visualized_BGE** model (bge-visualized-m3 architecture):
- BGE-M3 text encoder + EVA-CLIP visual encoder in a shared 1024-dim space
- Weights downloaded automatically from HuggingFace on first run
- Controlled by `EMBEDDING_MODEL` (default: `BAAI/bge-m3`) and `EMBEDDING_WEIGHT` (default: `Visualized_m3.pth`)

---

## Project Structure

```
shopping-demo/
├── README.md
├── docker-compose.yml
├── .env.example
│
├── frontend-next/                   # Next.js
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx                 # Entry point
│   ├── components/
│   │   └── ChatWindow.tsx           # Chat UI (messages + image upload)
│   └── lib/
│       └── api.ts                   # API client
│
├── api/
│   └── main.py                      # FastAPI app — all routes
│
├── agent/
│   ├── agent.py                     # create_agent() — LangGraph ReAct agent
│   ├── tools.py                     # product_search, get_product_info, understand_image
│   ├── search.py                    # hybrid_search(), get_product(), describe_image()
│   └── context.py                   # PostgresSaver checkpointer setup
│
├── preprocessing/
│   ├── pipeline.py                  # End-to-end: extract → embed → index
│   ├── extract.py                   # Parse ABO listings + resolve image paths
│   ├── embed.py                     # EmbeddingClient (vikingdb / ark / local backends)
│   ├── index.py                     # Bulk index into Elasticsearch
│   └── catalog.py                   # Product dataclasses
│
├── tests/
│   └── ...
│
└── data/
    ├── abo-listings/                # Product metadata (JSONL.gz)
    └── abo-images-small/            # Product images (JPEG, max 256px)
```

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

### 4. Prepare data

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

#### Re-index only (re-embed from scratch)

```bash
python preprocessing/pipeline.py \
  --index-name products \
  --recreate
```

If you only need to rebuild the Elasticsearch index from an already-embedded JSONL file:

```bash
python preprocessing/index.py \
  --input data/embedded_full.jsonl \
  --index products \
  --recreate
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

#### How embeddings are stored

Each product document in Elasticsearch has two dense vector fields in the same 1024-dim space:

| Field | Source | Used for |
|-------|--------|----------|
| `description_vector` | `embed_text(description)` | Text query HNSW channel |
| `image_vector` | `embed_image(product_image)` | Image query HNSW channel |

Both fields use bge-visualized-m3, a unified multimodal model whose text and image embeddings share the same latent space. This enables cross-modal search: a text query can retrieve products by visual similarity and vice versa.

### 5. Run

```bash
# API server (from repo root)
cd api && uvicorn main:app --reload --port 8000

# Frontend
cd frontend-next && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## API Reference

### `POST /chat`

Send a message (text and/or image) to the agent.

**Request**

```json
{
  "message": "Recommend a running t-shirt under $30",
  "image_b64": "<base64-encoded image, optional>",
  "session_id": "uuid (optional, omit to start new session)",
  "history": []
}
```

- `session_id`: used as LangGraph `thread_id` for persistent context. If omitted, a new UUID is generated.
- `history`: client-supplied message history used only in stateless mode (when PostgreSQL is unavailable).
- `image_b64`: base64-encoded JPEG or PNG. The API saves the image to `uploads/` and passes a URL to the agent. The search engine downloads the image, calls GPT-4o to understand it, extracts an image embedding, and runs full hybrid search — all internally.

**Response**

```json
{
  "reply": "Here are some lightweight running shirts from the catalog:",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "tool_calls": [
    {
      "tool": "product_search",
      "args": { "query": "running t-shirt under $30", "top_k": 5 }
    }
  ]
}
```

- `session_id` is returned on the first turn; pass it back on subsequent turns to maintain context.
- `tool_calls` logs which tools the agent called (image data is stripped from args).

---

### `POST /chat/upload`

Multipart form endpoint — accepts file upload directly (alternative to base64 in `/chat`).

**Form fields**: `message` (str), `session_id` (str, optional), `history` (JSON string, optional), `image` (file, optional).

---

### `GET /products/{item_id}`

Get full details for a single product.

**Response**

```json
{
  "item_id": "B06X9STHNG",
  "name": "Men's Dri-Fit Running Tee",
  "description": "...",
  "bullet_points": ["Moisture-wicking", "Reflective details"],
  "image_path": "data/abo-images-small/images/small/8c/8ccb5859.jpg",
  "image_url": "...",
  "web_url": "...",
  "brand": "Nike",
  "category": "SHIRT",
  "color": "Black",
  "material": "Polyester"
}
```

---

### `GET /image/{item_id}`

Serve the product image file directly.

---

### `GET /health`

Liveness check.

```json
{ "status": "ok" }
```

---

## Data: ABO Dataset

The product catalog is built from the [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset.

| | |
|---|---|
| Products | ~147,700 |
| Images | ~398,000 (max 256px) |
| Metadata | name, brand, category, color, material, dimensions, keywords, bullet points |
| License | CC BY 4.0 |

The preprocessing pipeline selects English-language listings (`en_US`), resolves `main_image_id` to a local file path, and filters records with missing descriptions or images before indexing.

---

## Tracing

With `LANGSMITH_API_KEY` set, every agent run is traced in [LangSmith](https://smith.langchain.com). Each trace shows:
- which tools fired and in what order
- token usage and latency per step
- full input/output at every node
