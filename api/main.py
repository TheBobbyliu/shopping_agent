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
from fastapi.responses import FileResponse, StreamingResponse
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
        global _startup_complete
        _get_checkpointer()
        _get_agent()
        from search import _get_embedding_client, _get_reranker
        _get_embedding_client()   # downloads + loads Visualized_BGE
        _get_reranker()           # loads bge-reranker-v2-m3
        _startup_complete = True

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

_agent          = None
_checkpointer   = None
_startup_complete = False


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


def _extract_reply_and_tool_calls(messages: list) -> tuple[str, list[dict]]:
    """Extract the last AI reply and the tool call log from an agent message list."""
    reply = ""
    tool_calls_log: list[dict] = []
    for msg in messages:
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
    return reply, tool_calls_log


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness probe — returns 200 only after all models have finished loading."""
    if not _startup_complete:
        raise HTTPException(status_code=503, detail="Starting up")
    return {"status": "ready"}


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
    from pipeline_monitor import PipelineMonitor, PipelineCallback, set_monitor

    agent = _get_agent()

    session_id   = req.session_id or str(uuid.uuid4())
    checkpointer = _get_checkpointer()

    # Initialise a per-request pipeline monitor and expose it via thread-local
    # storage so downstream code (search.py) can record sub-stage timings.
    monitor = PipelineMonitor(session_id=session_id, user_message=req.message or "")
    set_monitor(monitor)

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

    # Attach the pipeline callback so LLM turns and tool boundaries are timed
    config["callbacks"] = [PipelineCallback(monitor)]

    reply = ""
    tool_calls_log: list[dict] = []
    try:
        with monitor.stage("agent_invoke", {"session_id": session_id}) as out:
            result = agent.invoke(invoke_kwargs, config=config)
            messages_out = result.get("messages", [])
            reply, tool_calls_log = _extract_reply_and_tool_calls(messages_out)
            out["message_count"] = len(messages_out)
            out["reply_chars"] = len(reply)
            if reply:
                out["reply_preview"] = reply[:400]
            out["tool_call_count"] = len(tool_calls_log)
    except Exception as e:
        monitor.save()
        set_monitor(None)
        raise HTTPException(status_code=500, detail=str(e))

    log_path = monitor.save()
    set_monitor(None)
    print(f"[monitor] {session_id[:8]} → {log_path}", file=sys.stderr)

    return ChatResponse(reply=reply, session_id=session_id, tool_calls=tool_calls_log)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream the agent response as Server-Sent Events (SSE).

    Event types:
      status  — agent status update   data: {"text": "Thinking..."}
      token   — LLM token chunk       data: {"text": "partial reply"}
      done    — final summary         data: {"reply": "...", "session_id": "...", "tool_calls": [...]}
      error   — fatal error           data: {"detail": "..."}
    """
    import asyncio, json as _json
    from pipeline_monitor import PipelineMonitor, PipelineCallback, set_monitor

    agent        = _get_agent()
    session_id   = req.session_id or str(uuid.uuid4())
    checkpointer = _get_checkpointer()

    monitor = PipelineMonitor(session_id=session_id, user_message=req.message or "")
    set_monitor(monitor)

    if checkpointer:
        messages = []
    else:
        messages = [
            HumanMessage(content=t["content"]) if t["role"] == "user"
            else AIMessage(content=t["content"])
            for t in req.history
        ]

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
    config["callbacks"] = [PipelineCallback(monitor)]

    # Status text per tool name
    _TOOL_STATUS = {
        "product_search":  "Searching for products...",
        "get_product_info": "Looking up product details...",
        "understand_image": "Analyzing image...",
    }

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

    async def _generate():
        # agent.stream() is a blocking synchronous iterator. Running it directly
        # inside an async generator would starve the event loop during each
        # LLM/tool call, blocking all other requests (including /health).
        # Solution: run the iterator in a thread and feed results into an asyncio
        # Queue so the event loop stays free to serve other requests.
        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_agent():
            try:
                for chunk, metadata in agent.stream(
                    invoke_kwargs, config=config, stream_mode="messages"
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, (chunk, metadata))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        agent_task = asyncio.ensure_future(asyncio.to_thread(_run_agent))

        reply_parts: list[str] = []
        tool_calls_log: list[dict] = []
        current_node: str = ""
        full_reply = ""

        try:
            with monitor.stage("agent_invoke", {"session_id": session_id}) as out:
                while True:
                    item = await queue.get()
                    if item is None:
                        break  # sentinel — agent finished
                    if isinstance(item, Exception):
                        raise item

                    chunk, metadata = item
                    node = metadata.get("langgraph_node", "")

                    # Emit status when entering a new node
                    if node != current_node:
                        current_node = node
                        if node == "agent":
                            yield _sse("status", {"text": "Thinking..."})
                        elif node == "tools":
                            yield _sse("status", {"text": "Using tools..."})

                    cls = chunk.__class__.__name__

                    if cls == "AIMessageChunk":
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tc in chunk.tool_call_chunks:
                                name = tc.get("name", "")
                                if name:
                                    status_text = _TOOL_STATUS.get(name, f"Calling {name}...")
                                    yield _sse("status", {"text": status_text})
                        elif isinstance(chunk.content, str) and chunk.content:
                            reply_parts.append(chunk.content)
                            yield _sse("token", {"text": chunk.content})

                    elif cls == "AIMessage":
                        if isinstance(chunk.content, str) and chunk.content:
                            if not reply_parts:
                                reply_parts.append(chunk.content)
                                yield _sse("token", {"text": chunk.content})
                        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                            for tc in chunk.tool_calls:
                                tool_calls_log.append({
                                    "tool": tc.get("name", ""),
                                    "args": {k: v for k, v in tc.get("args", {}).items()
                                             if k != "image_b64"},
                                })
                                status_text = _TOOL_STATUS.get(tc.get("name", ""), "Using tool...")
                                yield _sse("status", {"text": status_text})

                full_reply = "".join(reply_parts)
                out["reply_chars"] = len(full_reply)
                if full_reply:
                    out["reply_preview"] = full_reply[:400]
                out["tool_call_count"] = len(tool_calls_log)

        except asyncio.CancelledError:
            agent_task.cancel()
            raise
        except Exception as e:
            monitor.save()
            set_monitor(None)
            yield _sse("error", {"detail": str(e)})
            return

        full_reply = "".join(reply_parts)
        log_path   = monitor.save()
        set_monitor(None)
        print(f"[monitor/stream] {session_id[:8]} → {log_path}", file=sys.stderr)

        yield _sse("done", {
            "reply":      full_reply,
            "session_id": session_id,
            "tool_calls": tool_calls_log,
        })

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


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


_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">'
    '<rect width="200" height="200" fill="#141c17"/>'
    '<text x="100" y="108" font-family="sans-serif" font-size="11" '
    'fill="#4d6655" text-anchor="middle">No image</text>'
    "</svg>"
).encode()


@app.get("/image/{item_id}")
def product_image(item_id: str):
    """Serve the product image file, or a placeholder SVG when not found."""
    from fastapi.responses import Response as FastAPIResponse
    from search import get_product as _get

    product = _get(item_id)
    if product:
        path = Path(product.get("image_path", ""))
        if path.is_file():
            return FileResponse(str(path), media_type="image/jpeg")

    return FastAPIResponse(content=_PLACEHOLDER_SVG, media_type="image/svg+xml")


# The Next.js frontend (frontend-next/) is served separately on port 3000 and
# proxies /api/* to this server via next.config.ts rewrites in dev.
# For production, run `next start` alongside this API server.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
