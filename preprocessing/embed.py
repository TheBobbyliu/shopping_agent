"""
Embedding client for the shopping agent pipeline.

Supports two backends:
  - vikingdb: Volcano Engine VikingDB SDK (requires VOLCANO_AK + VOLCANO_SK)
  - ark:      Volcano Engine Ark API, OpenAI-compatible (requires ARK_API_KEY +
              ARK_EMBEDDING_ENDPOINT for text; ARK_VISION_ENDPOINT for image)

Backend is selected by EMBEDDING_BACKEND env var (default: vikingdb).
Model is selected by EMBEDDING_MODEL env var (default: bge-visualized-m3).

Usage:
    client = EmbeddingClient.from_env()

    # single text
    vec = client.embed_text("running shoes for men")

    # single image (path or base64 bytes)
    vec = client.embed_image(Path("images/shoe.jpg"))

    # batch (more efficient)
    vecs = client.embed_texts(["text1", "text2"])
    vecs = client.embed_images([Path("a.jpg"), Path("b.jpg")])
"""
from __future__ import annotations

import base64
import math
import os
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _load_image_b64(path: Path, max_bytes: int = 900_000) -> str:
    """
    Read an image file and return base64-encoded string.
    Resizes if larger than max_bytes using Pillow if available,
    otherwise truncates (VikingDB recommends < 1MB).
    """
    data = path.read_bytes()
    if len(data) > max_bytes:
        try:
            from PIL import Image
            import io
            img = Image.open(path)
            buf = io.BytesIO()
            # scale down until under limit
            scale = math.sqrt(max_bytes / len(data)) * 0.9
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        except ImportError:
            pass  # PIL not available; send as-is and let API decide
    return base64.b64encode(data).decode("utf-8")


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# VikingDB backend
# ---------------------------------------------------------------------------

class _VikingDBBackend:
    """
    Uses the volcengine VikingDB SDK.
    Credentials: VOLCANO_AK, VOLCANO_SK
    Host:   VOLCANO_HOST   (default: api-vikingdb.volces.com)
    Region: VOLCANO_REGION (default: cn-beijing)
    """

    def __init__(self, model: str = "bge-visualized-m3"):
        from volcengine.viking_db import VikingDBService, EmbModel, RawData

        self._EmbModel = EmbModel
        self._RawData = RawData
        self._model = model

        ak = os.environ.get("VOLCANO_AK", "")
        sk = os.environ.get("VOLCANO_SK", "")
        host = os.environ.get("VOLCANO_HOST", "api-vikingdb.volces.com")
        region = os.environ.get("VOLCANO_REGION", "cn-beijing")

        if not ak or not sk:
            raise EnvironmentError(
                "VikingDB backend requires VOLCANO_AK and VOLCANO_SK env vars. "
                "Set EMBEDDING_BACKEND=ark to use the Ark API instead."
            )

        self._svc = VikingDBService(host=host, region=region, ak=ak, sk=sk)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for batch in _chunks(texts, 100):
            raw = [self._RawData("text", text=t) for t in batch]
            res = self._svc.embedding_v2(self._EmbModel(self._model), raw)
            results.extend(res)
        return results

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        results = []
        for batch in _chunks(paths, 15):   # 15 images/sec rate limit
            raw = [self._RawData("image", image=_load_image_b64(p)) for p in batch]
            res = self._svc.embedding_v2(self._EmbModel(self._model), raw)
            results.extend(res)
            if len(paths) > 15:
                time.sleep(1.1)            # respect rate limit
        return results


# ---------------------------------------------------------------------------
# Ark API backend (OpenAI-compatible)
# ---------------------------------------------------------------------------

class _ArkBackend:
    """
    Uses the Ark API (OpenAI-compatible).
    For text:  ARK_API_KEY + ARK_EMBEDDING_ENDPOINT (endpoint ID, e.g. ep-...)
    For image: ARK_API_KEY + ARK_VISION_ENDPOINT    (must support doubao-embedding-vision)

    If ARK_VISION_ENDPOINT is not set, image embedding falls back to describing
    the image with GPT and embedding the description.
    """

    def __init__(self, text_model: str = "bge-m3", image_model: str = "doubao-embedding-vision"):
        from openai import OpenAI

        api_key = os.environ.get("ARK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("Ark backend requires ARK_API_KEY env var.")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self._text_endpoint = os.environ.get("ARK_EMBEDDING_ENDPOINT", text_model)
        self._image_endpoint = os.environ.get("ARK_VISION_ENDPOINT", "")
        self._image_model = image_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for batch in _chunks(texts, 100):
            resp = self._client.embeddings.create(
                model=self._text_endpoint,
                input=batch,
            )
            # sort by index to preserve order
            ordered = sorted(resp.data, key=lambda x: x.index)
            results.extend([item.embedding for item in ordered])
        return results

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        if not self._image_endpoint:
            raise EnvironmentError(
                "ARK_VISION_ENDPOINT is not set. "
                "Set it to a deployed doubao-embedding-vision endpoint, "
                "or use EMBEDDING_BACKEND=vikingdb."
            )
        results = []
        for path in paths:
            b64 = _load_image_b64(path)
            resp = self._client.embeddings.create(
                model=self._image_endpoint,
                input=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}],
            )
            results.append(resp.data[0].embedding)
        return results


# ---------------------------------------------------------------------------
# Local (on-premise) backend — SigLIP2 via transformers
# ---------------------------------------------------------------------------

class _LocalBackend:
    """
    On-premise embedding using BAAI/bge-visualized (Visualized_BGE).
    Combines BGE-M3 text encoder + EVA-CLIP visual encoder into a shared
    1024-dim embedding space — the same architecture as bge-visualized-m3.

    env vars:
        EMBEDDING_MODEL   bge text model id (default: BAAI/bge-m3)
        EMBEDDING_WEIGHT  path or HF filename for visual weights
                          (default: Visualized_m3.pth from BAAI/bge-visualized)
        EMBEDDING_DEVICE  cpu | cuda | mps  (auto-detected if not set)
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        model_weight: str = "Visualized_m3.pth",
    ):
        import torch
        from huggingface_hub import hf_hub_download
        from FlagEmbedding.visual.modeling import Visualized_BGE

        # Resolve weight path — download from HF if not a local file
        weight_path = Path(model_weight)
        if not weight_path.exists():
            print(f"[LocalBackend] Downloading {model_weight}...", file=sys.stderr)
            weight_path = Path(hf_hub_download(
                repo_id="BAAI/bge-visualized",
                filename=model_weight,
            ))

        print(f"[LocalBackend] Loading Visualized_BGE ({model_id})...", file=sys.stderr)
        self._model = Visualized_BGE(
            model_name_bge=model_id,
            model_weight=str(weight_path),
        )
        self._model.eval()
        self._torch = torch
        print("[LocalBackend] Ready.", file=sys.stderr)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for batch in _chunks(texts, 32):
            tokenized = self._model.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._model.device)
            with self._torch.no_grad():
                vecs = self._model.encode_text(tokenized)
            results.extend(vecs.cpu().float().tolist())
        return results

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        from PIL import Image
        results = []
        for path in paths:
            img_tensor = self._model.preprocess_val(
                Image.open(path).convert("RGB")
            ).unsqueeze(0).to(self._model.device)
            with self._torch.no_grad():
                vec = self._model.encode_image(img_tensor)
            results.append(vec.cpu().float()[0].tolist())
        return results


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """
    Unified embedding client. Backend selected by EMBEDDING_BACKEND env var.

    env vars:
        EMBEDDING_BACKEND   vikingdb | ark  (default: vikingdb)
        EMBEDDING_MODEL     model name      (default: bge-visualized-m3)

    VikingDB credentials:
        VOLCANO_AK, VOLCANO_SK, VOLCANO_HOST, VOLCANO_REGION

    Ark credentials:
        ARK_API_KEY, ARK_EMBEDDING_ENDPOINT, ARK_VISION_ENDPOINT
    """

    def __init__(self, backend):
        self._backend = backend

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        backend_name = os.environ.get("EMBEDDING_BACKEND", "vikingdb").lower()
        model = os.environ.get("EMBEDDING_MODEL", "bge-visualized-m3")

        if backend_name == "vikingdb":
            return cls(_VikingDBBackend(model=model))
        elif backend_name == "ark":
            return cls(_ArkBackend())
        elif backend_name == "local":
            return cls(_LocalBackend(model_id=model))
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend_name!r}. Use 'vikingdb', 'ark', or 'local'.")

    # --- single item convenience wrappers ---

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_image(self, path: Union[Path, str]) -> list[float]:
        return self.embed_images([Path(path)])[0]

    # --- batch ---

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_texts(texts)

    def embed_images(self, paths: list[Union[Path, str]]) -> list[list[float]]:
        return self._backend.embed_images([Path(p) for p in paths])

    # --- utility ---

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        return _cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# CLI for embedding a catalog JSONL and writing vectors back
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json
    from catalog import Product, EmbeddedProduct
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Compute embeddings for a product catalog")
    parser.add_argument("--input", type=Path, required=True, help="JSONL from extract.py")
    parser.add_argument("--out",   type=Path, required=True, help="Output JSONL with vectors")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    client = EmbeddingClient.from_env()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    products = []
    with open(args.input) as f:
        for line in f:
            products.append(Product.from_dict(json.loads(line)))

    print(f"Embedding {len(products)} products...", file=sys.stderr)

    # Text embeddings in batches
    descriptions = [p.description for p in products]
    text_vecs: list[list[float]] = []
    for i, batch in enumerate(_chunks(descriptions, args.batch_size)):
        vecs = client.embed_texts(batch)
        text_vecs.extend(vecs)
        print(f"  text {min((i+1)*args.batch_size, len(products))}/{len(products)}", file=sys.stderr)

    # Image embeddings in batches
    image_paths = [p.image_path for p in products]
    image_vecs: list[list[float]] = []
    for i, batch in enumerate(_chunks(image_paths, 15)):
        vecs = client.embed_images(batch)
        image_vecs.extend(vecs)
        print(f"  image {min((i+1)*15, len(products))}/{len(products)}", file=sys.stderr)

    with open(args.out, "w") as f:
        for product, img_vec, txt_vec in zip(products, image_vecs, text_vecs):
            ep = EmbeddedProduct(
                **{k: getattr(product, k) for k in product.__dataclass_fields__},
                image_vector=img_vec,
                description_vector=txt_vec,
            )
            f.write(json.dumps(ep.to_dict()) + "\n")

    print(f"Wrote {len(products)} embedded products to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
