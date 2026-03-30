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
                  ├─→ Elasticsearch  (image_indexes + description_indexes)
                  └─→ Elasticsearch  (product metadata as documents)
```

### Data Pipeline (Online)

```
Web UI (chat + image upload)
  └─→ Agent  [LangGraph]
        ├─ intent analysis
        ├─ tool calling
        └─ answer questions
              │
              ├─→ Product Search ──────────────────────→ Search Engine
              │     - image: optional                       ├─ hybrid search
              │     - query: str                            │    1. image vector (HNSW)
              │     - filters: **kwargs                     │    2. description vector (HNSW)
              │     - return: product candidates            │    3. keyword match (BM25)
              │                                             │    4. hybrid scoring (RRF)
              ├─→ Get Product Info                          └─→ bge-reranker-v2-m3
              │     - ids: item_id[]                              └─→ top-N candidates
              │     - return: full product info
              │
              │         [image path]
              ├─→ Image Understanding ── GPT-4o ──→ text description
              └─→ Feature Engine
                    ├─ image feature extraction
                    └─ text feature extraction
                          return: embedding vector
```

**Context** is managed by LangGraph's checkpointer, persisted in PostgreSQL.

### Search Flow by Query Type

| Query type | Feature Engine | Search channels | Reranker |
|---|---|---|---|
| Text recommendation | BGE-Vis-M3 on query text | description HNSW + BM25 | bge-reranker-v2-m3 |
| Image search | GPT-4o describes image → BGE-Vis-M3 on description + BGE-Vis-M3 on image | image HNSW + description HNSW + BM25 | bge-reranker-v2-m3 |
| Product Q&A | — | Get Product Info (exact lookup) | — |

Hybrid score fusion uses **Reciprocal Rank Fusion (RRF)** across all active channels before reranking.

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| Frontend | Next.js | Fast dev, easy deployment, good streaming support |
| Agent API | FastAPI | Async, auto OpenAPI docs, Python ML ecosystem |
| Orchestration | LangGraph | State graph, checkpointing, cycles, native streaming |
| Logging / Tracing | LangSmith | Full trace visualization per request, token usage per step |
| LLM + Image Understanding | GPT-5.4 | Multimodal, strong instruction following, tool use |
| Embeddings (image + text) | bge-visualized-m3 | Shared embedding space for text and images — enables cross-modal search |
| Reranker | bge-reranker-v2-m3 | Cross-encoder, same BGE family, no extra infra |
| Search + Vector DB | Elasticsearch | HNSW vector search + BM25 in one system, mature ops, scalable |
| Metadata + Context DB | PostgreSQL | ACID, LangGraph checkpointer, structured product queries |
| Product Catalog | ABO Dataset | Real-world Amazon product data with images and rich metadata |

### Embedding Model Config

The Feature Engine supports two modes, controlled by `FEATURE_ENGINE_MODE` in config:

```
unified   — bge-visualized-m3 for both text and images
            → single shared vector space, cross-modal search works
            → one HNSW index covers both query types

separate  — independent models for each modality
            → image model: configurable (e.g. siglip2-so400m)
            → text model:  configurable (e.g. bge-m3)
            → two separate HNSW indexes, no cross-modal comparison
```

---

## Project Structure

```
shopping-demo/
├── README.md
├── docker-compose.yml
├── .env.example
│
├── frontend/                        # Next.js
│   ├── app/
│   │   └── page.tsx                 # Chat UI entry point
│   ├── components/
│   │   ├── Chat/                    # Message thread, input, image upload
│   │   └── ProductCard/             # Product result card
│   └── lib/
│       └── api.ts                   # API client
│
├── backend/                         # FastAPI + LangGraph
│   ├── main.py                      # FastAPI app entry point
│   ├── config.py                    # Settings (env vars, feature engine mode)
│   │
│   ├── agent/
│   │   ├── graph.py                 # LangGraph state graph definition
│   │   ├── tools.py                 # product_search, get_product_info tools
│   │   └── prompts.py               # System prompt
│   │
│   ├── search/
│   │   ├── engine.py                # Hybrid search (HNSW + BM25 + RRF)
│   │   └── reranker.py              # bge-reranker-v2-m3 wrapper
│   │
│   ├── feature/
│   │   ├── embeddings.py            # bge-visualized-m3, unified/separate modes
│   │   └── image_understanding.py   # GPT-4o image → text description
│   │
│   ├── db/
│   │   ├── elasticsearch.py         # ES client, index management
│   │   └── postgres.py              # LangGraph checkpointer setup
│   │
│   └── api/
│       └── routes/
│           ├── chat.py              # POST /api/chat
│           └── products.py          # GET /api/products, /api/products/{id}
│
├── data/
│   ├── ABO_DATASET_GUIDE.md
│   ├── abo-listings/                # Product metadata (JSONL.gz)
│   └── abo-images-small/            # Product images (JPEG, max 256px)
│
└── preprocessing/
    ├── extract.py                   # Parse ABO listings + resolve image paths
    ├── embed.py                     # Extract embeddings, write to disk
    └── index.py                     # Bulk index into Elasticsearch
```

---

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (for Elasticsearch + PostgreSQL)
- OpenAI API key (GPT-5.4)

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd shopping-demo

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
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
#   OPENAI_API_KEY
#   LANGSMITH_API_KEY  (optional, for tracing)
#   ELASTICSEARCH_URL
#   POSTGRES_URL
#   FEATURE_ENGINE_MODE  (unified | separate)
```

### 4. Prepare data

```bash
cd preprocessing

# Step 1: extract and clean product records from ABO
python extract.py --listings ../data/abo-listings --images ../data/abo-images-small --out catalog.jsonl

# Step 2: compute embeddings
python embed.py --input catalog.jsonl --out embeddings.jsonl --mode unified

# Step 3: index into Elasticsearch
python index.py --input embeddings.jsonl
```

### 5. Run

```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## API Reference

### `POST /api/chat`

Send a message (text and/or image) to the agent.

**Request**

```json
{
  "message": "Recommend a running t-shirt under $30",
  "image": "<base64-encoded image, optional>",
  "conversation_id": "uuid (optional, omit to start new session)"
}
```

**Response**

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "reply": "Here are some lightweight running shirts from the catalog:",
  "products": [
    {
      "item_id": "B06X9STHNG",
      "name": "Men's Dri-Fit Running Tee",
      "description": "Moisture-wicking fabric, reflective details...",
      "image_url": "http://localhost:8000/images/8c/8ccb5859.jpg",
      "web_url": "https://...",
      "metadata": {
        "brand": "Nike",
        "category": "SHIRT",
        "color": "Black",
        "price": 27.99
      },
      "relevance_score": 0.94
    }
  ]
}
```

**Notes**
- `conversation_id` is returned on the first turn; pass it back on subsequent turns to maintain context.
- `products` is an empty array for general conversation replies.
- Image should be base64-encoded; JPEG and PNG are supported.

---

### `GET /api/products`

List products from the catalog.

**Query parameters**

| Param | Type | Description |
|---|---|---|
| `q` | string | Keyword filter |
| `category` | string | Product type (e.g. `SHOES`, `SHIRT`) |
| `limit` | int | Max results (default: 20) |
| `offset` | int | Pagination offset (default: 0) |

**Response**

```json
{
  "total": 1482,
  "products": [ { "item_id": "...", "name": "...", ... } ]
}
```

---

### `GET /api/products/{item_id}`

Get full details for a single product.

**Response**

```json
{
  "item_id": "B06X9STHNG",
  "name": "Men's Dri-Fit Running Tee",
  "description": "...",
  "bullet_points": ["Moisture-wicking", "Reflective details", "..."],
  "image_url": "...",
  "web_url": "...",
  "metadata": {
    "brand": "Nike",
    "category": "SHIRT",
    "color": "Black",
    "material": "Polyester",
    "dimensions": { "height": "28in", "width": "20in" }
  }
}
```

---

### `GET /api/health`

Liveness check.

```json
{ "status": "ok", "elasticsearch": "ok", "postgres": "ok" }
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
