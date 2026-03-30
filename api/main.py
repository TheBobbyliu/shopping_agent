"""
FastAPI server for the shopping agent.

Endpoints:
  POST /chat          — send a message (text + optional image) to the agent
  GET  /products/{id} — get full product details
  GET  /health        — liveness check
  GET  /              — serve frontend
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

app = FastAPI(title="Shopping Agent API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the agent (loads embedding model on first request)
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from agent import create_agent
        _agent = create_agent()
    return _agent


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    image_b64: Optional[str] = None   # base64-encoded image
    history: list[dict] = []          # [{role: "user"|"assistant", content: "..."}]


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Send a message to the shopping agent."""
    agent = get_agent()

    # Build message history
    messages = []
    for turn in req.history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # Build current human message
    if req.image_b64:
        # Multimodal message — text + image for Claude
        content = [
            {"type": "text", "text": req.message},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": req.image_b64,
                },
            },
        ]
        # Also embed the image_b64 in a way tools can use it
        # We append a hidden instruction so the agent knows to use the image
        content[0]["text"] = (
            req.message
            + f"\n\n[IMAGE_B64:{req.image_b64[:20]}...]"
            + f"\n\nWhen searching, pass this image_b64 to the tools: {req.image_b64}"
        )
        messages.append(HumanMessage(content=req.message + "\n\n" +
                                     f"[User uploaded an image. Use image_b64='{req.image_b64}' "
                                     f"when calling product_search or understand_image tools.]"))
    else:
        messages.append(HumanMessage(content=req.message))

    try:
        result = agent.invoke({"messages": messages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract assistant reply
    reply = ""
    tool_calls_log = []
    for msg in result["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
            if msg.__class__.__name__ == "AIMessage":
                reply = msg.content
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    "tool": tc.get("name", ""),
                    "args": {k: v for k, v in tc.get("args", {}).items()
                             if k != "image_b64"},  # don't log base64 blobs
                })

    return ChatResponse(reply=reply, tool_calls=tool_calls_log)


@app.post("/chat/upload")
async def chat_with_upload(
    message: str = Form(""),
    history: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
):
    """Multipart form endpoint — accepts file upload directly."""
    import json

    image_b64 = None
    if image:
        data = await image.read()
        image_b64 = base64.b64encode(data).decode()

    history_parsed = json.loads(history)
    req = ChatRequest(message=message or "What is this product?",
                      image_b64=image_b64, history=history_parsed)
    return chat(req)


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


# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
