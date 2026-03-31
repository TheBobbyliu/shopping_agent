"""
FastAPI server for the shopping agent.

Endpoints:
  POST /chat             — send a message (text + optional image) to the agent
  POST /chat/upload      — multipart form variant with file upload
  GET  /products/{id}    — get full product details
  GET  /image/{id}       — serve catalog product image file
  GET  /uploads/{file}   — serve temporarily uploaded images
  GET  /health           — liveness check
"""
from __future__ import annotations

import base64
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Directory for temporarily uploaded images (created on import)
_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)

# Base URL used to build image URLs returned to the agent.
# Override with API_BASE_URL env var in production.
_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel


def _save_upload(image_bytes: bytes) -> str:
    """Save image bytes to the uploads directory and return its public URL."""
    filename = f"{uuid.uuid4().hex}.jpg"
    ((_UPLOAD_DIR) / filename).write_bytes(image_bytes)
    return f"{_API_BASE_URL}/uploads/{filename}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load all heavy singletons in a thread (blocking model I/O)
    # so the first request is never the one paying the cold-start cost.
    import asyncio

    def _startup():
        _get_checkpointer()
        _get_agent()
        from search import _get_embedding_client, _get_reranker
        _get_embedding_client()   # downloads + loads Visualized_BGE
        _get_reranker()           # loads bge-reranker-v2-m3

    await asyncio.to_thread(_startup)
    yield
    from context import close_checkpointer
    close_checkpointer()


app = FastAPI(title="Shopping Agent API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons — loaded once at startup
# ---------------------------------------------------------------------------

_agent       = None
_checkpointer = None


def _get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        db_url = os.environ.get("DATABASE_URL",
                                "postgresql://shopping:shopping@localhost:5432/shopping")
        try:
            from context import make_checkpointer
            _checkpointer = make_checkpointer(db_url)
        except Exception as e:
            import sys
            print(f"[api] Context DB unavailable ({e}); running stateless", file=sys.stderr)
            _checkpointer = None   # fallback: no persistence
    return _checkpointer


def _get_agent():
    global _agent
    if _agent is None:
        from agent import create_agent
        _agent = create_agent(checkpointer=_get_checkpointer())
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message:    str
    image_b64:  Optional[str] = None
    session_id: Optional[str] = None   # used as LangGraph thread_id for persistent memory
    history:    list[dict]    = []     # fallback client-side history (stateless mode)


class ChatResponse(BaseModel):
    reply:      str
    session_id: Optional[str]  = None
    tool_calls: list[dict]     = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/uploads/{filename}")
def serve_upload(filename: str):
    """Serve a temporarily uploaded image."""
    path = _UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message to the shopping agent."""
    import uuid
    agent = _get_agent()

    session_id  = req.session_id or str(uuid.uuid4())
    checkpointer = _get_checkpointer()

    # When checkpointer is available, LangGraph manages history via thread_id.
    # When not available, reconstruct history from the client-supplied list.
    if checkpointer:
        messages = []
    else:
        messages = [
            HumanMessage(content=t["content"]) if t["role"] == "user"
            else AIMessage(content=t["content"])
            for t in req.history
        ]

    # Build current human message.
    # For image uploads, save the image and pass a URL so the agent can forward it
    # to product_search — the search engine handles understanding and embedding.
    if req.image_b64:
        img_bytes = base64.b64decode(req.image_b64)
        img_url   = _save_upload(img_bytes)
        user_text = req.message or "Find products similar to this image."
        messages.append(HumanMessage(content=f"{user_text}\n\n[Image URL: {img_url}]"))
    else:
        messages.append(HumanMessage(content=req.message))

    invoke_kwargs: dict = {"messages": messages}
    if checkpointer:
        from context import thread_config
        config = thread_config(session_id)
    else:
        config = {}

    try:
        result = agent.invoke(invoke_kwargs, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract last AI reply and tool call log
    reply = ""
    tool_calls_log = []
    for msg in result["messages"]:
        if msg.__class__.__name__ == "AIMessage":
            if isinstance(msg.content, str) and msg.content:
                reply = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append({
                        "tool": tc.get("name", ""),
                        "args": {k: v for k, v in tc.get("args", {}).items()
                                 if k != "image_b64"},
                    })

    return ChatResponse(reply=reply, session_id=session_id, tool_calls=tool_calls_log)


@app.post("/chat/upload")
async def chat_with_upload(
    message:    str            = Form(""),
    session_id: Optional[str]  = Form(None),
    history:    str            = Form("[]"),
    image:      Optional[UploadFile] = File(None),
):
    """Multipart form endpoint — accepts file upload directly."""
    import asyncio, json
    image_b64 = base64.b64encode(await image.read()).decode() if image else None
    # Pass image_b64 through ChatRequest so the /chat handler saves it and builds the URL.
    req = ChatRequest(
        message=message or "Find products similar to this image.",
        image_b64=image_b64,
        session_id=session_id,
        history=json.loads(history),
    )
    return await asyncio.to_thread(chat, req)


@app.get("/products/{item_id}")
def get_product(item_id: str):
    """Return full product details."""
    from search import get_product as _get
    product = _get(item_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@app.get("/image/{item_id}")
def product_image(item_id: str):
    """Serve the product image file."""
    from search import get_product as _get
    product = _get(item_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    path = Path(product.get("image_path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(path), media_type="image/jpeg")


# The Next.js frontend (frontend-next/) is served separately on port 3000 and
# proxies /api/* to this server via next.config.ts rewrites in dev.
# For production, run `next start` alongside this API server.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
