"""
Embedding client for the shopping agent pipeline.

Uses the local on-premise backend (Visualized_BGE / bge-visualized-m3).
Backend is selected by EMBEDDING_BACKEND env var (default: local).
Model is selected by EMBEDDING_MODEL env var (default: BAAI/bge-m3).

Usage:
    client = EmbeddingClient.from_env()

    # single text
    vec = client.embed_text("running shoes for men")

    # single image
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


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Local (on-premise) backend — Visualized_BGE via FlagEmbedding
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
    Embedding client using the local on-premise backend (Visualized_BGE).

    env vars:
        EMBEDDING_MODEL   bge text model id  (default: BAAI/bge-m3)
        EMBEDDING_WEIGHT  visual weight path (default: Visualized_m3.pth)
    """

    def __init__(self, backend):
        self._backend = backend

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        return cls(_LocalBackend(model_id=model))

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
