"""
LangChain tool definitions for the shopping agent.

Three tools:
  product_search     — hybrid text+image search
  get_product_info   — fetch full details for one product by ID
  understand_image   — describe a product image with GPT-4o
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool


@tool
def product_search(query: str, image_b64: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search the product catalog using natural language, an optional product image, or both.

    Args:
        query:     Natural language description of what the user is looking for.
        image_b64: Base64-encoded JPEG/PNG of a product image (optional).
                   When provided, the search uses visual similarity in addition to text.
        top_k:     Maximum number of results to return (1-10, default 5).

    Returns:
        JSON list of matching products, each with item_id, name, category,
        description snippet, color, material, brand, and image_path.
    """
    from search import hybrid_search

    top_k = max(1, min(top_k, 10))
    results = hybrid_search(query_text=query, image_b64=image_b64, top_k=top_k)
    return json.dumps(results, ensure_ascii=False)


@tool
def get_product_info(item_id: str) -> str:
    """
    Retrieve the full details of a single product by its item_id.

    Use this after product_search to get complete information about a product
    the user wants to know more about.

    Args:
        item_id: The product's unique identifier (returned by product_search).

    Returns:
        JSON object with full product details, or an error message if not found.
    """
    from search import get_product

    product = get_product(item_id)
    if product is None:
        return json.dumps({"error": f"Product '{item_id}' not found."})
    return json.dumps(product, ensure_ascii=False, default=str)


@tool
def understand_image(image_b64: str) -> str:
    """
    Analyze a product image and return a text description suitable for search.

    Use this when the user uploads an image and you need to understand
    what the product is before searching for it or similar items.

    Args:
        image_b64: Base64-encoded JPEG/PNG image.

    Returns:
        A concise product description (type, color, material, style, features).
    """
    from search import describe_image

    return describe_image(image_b64)


TOOLS = [product_search, get_product_info, understand_image]
