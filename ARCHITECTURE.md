# Architecture

---

## System Architecture

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

### Embedding Model

The pipeline uses the on-premise **Visualized_BGE** model (bge-visualized-m3 architecture):
- BGE-M3 text encoder + EVA-CLIP visual encoder in a shared 1024-dim space
- Weights downloaded automatically from HuggingFace on first run
- Controlled by `EMBEDDING_MODEL` (default: `BAAI/bge-m3`) and `EMBEDDING_WEIGHT` (default: `Visualized_m3.pth`)

Each product document in Elasticsearch has two dense vector fields in the same 1024-dim space:

| Field | Source | Used for |
|-------|--------|----------|
| `description_vector` | `embed_text(description)` | Text query HNSW channel |
| `image_vector` | `embed_image(product_image)` | Image query HNSW channel |

Both fields use bge-visualized-m3, a unified multimodal model whose text and image embeddings share the same latent space. This enables cross-modal search: a text query can retrieve products by visual similarity and vice versa.

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
- `image_b64`: base64-encoded JPEG or PNG. The API saves the image to `uploads/` and passes a URL to the agent.

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

### `POST /embed`

Generate embeddings for a text description and image. Used by the warehouse CLI to reuse the already-warm model without a cold start.

**Request**

```json
{ "text": "Samsung 65-inch 4K TV", "image_path": "/path/to/image.jpg" }
```

**Response**

```json
{ "description_vector": [...], "image_vector": [...] }
```

Returns 400 if `image_path` does not exist. Returns 503 if the embedding model has not finished loading.

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

### `GET /ready`

Returns 200 once the embedding model has finished loading, 503 otherwise. The frontend polls this before rendering the chat UI.

---

## Project Structure

```
shopping-demo/
├── README.md
├── ARCHITECTURE.md
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
│   ├── catalog.py                   # Product dataclasses
│   └── warehouse.py                 # Warehouse CLI (add / delete / check)
│
├── tests/
│   └── ...
│
└── data/
    ├── abo-listings/                # Product metadata (JSONL.gz)
    └── abo-images-small/            # Product images (JPEG, max 256px)
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
