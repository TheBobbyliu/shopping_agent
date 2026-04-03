"""
LangChain tool definitions for the shopping agent.

Three tools:
  product_search     — hybrid text+image search
  get_product_info   — fetch full details for one product by ID
  understand_image   — describe a product image with GPT-4o
"""
from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
from contextvars import copy_context
from typing import Optional

from langchain_core.tools import tool
from project_logging import record_error

logger = logging.getLogger(__name__)

# Timeout (seconds) before a tool call is considered stuck and retried once.
_TOOL_TIMEOUT_S = 90
_MAX_RETRIES = 1


def _log_tool_error(tool_name: str, error: BaseException | str) -> None:
    """Write a tool error to the project log, pipeline log, and stderr."""
    error_msg = str(error)
    context: dict[str, str] = {}
    try:
        from pipeline_monitor import get_monitor
        monitor = get_monitor()
        if monitor is not None:
            context["session_id"] = monitor.session_id
            context["user_message"] = monitor.user_message[:200]
            with monitor.stage(f"tool_error_{tool_name}", {"error": error_msg}):
                pass
    except Exception:
        pass
    try:
        log_file = record_error(f"tool.{tool_name}", error, context=context)
        logger.error("[tool:%s] %s (see %s)", tool_name, error_msg, log_file)
    except Exception:
        logger.error("[tool:%s] %s", tool_name, error_msg)


def _tool_with_retry(fn):
    """Wrap a tool function with timeout (90 s) and a single retry on timeout."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_err: str = ""
        last_error: BaseException | str = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                ctx = copy_context()
                future = ex.submit(ctx.run, fn, *args, **kwargs)
                try:
                    return future.result(timeout=_TOOL_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    raise
                finally:
                    ex.shutdown(wait=False)  # don't block waiting for timed-out thread
            except concurrent.futures.TimeoutError:
                last_err = f"Tool '{fn.__name__}' timed out after {_TOOL_TIMEOUT_S}s."
                last_error = TimeoutError(last_err)
                if attempt < _MAX_RETRIES:
                    continue  # retry once on timeout
            except Exception as exc:
                last_err = str(exc)
                last_error = exc
                break  # non-timeout errors don't retry
        _log_tool_error(fn.__name__, last_error or last_err)
        return json.dumps({"error": last_err, "tool": fn.__name__})
    return wrapper


def _product_search_impl(query: Optional[str] = None, image_url: Optional[str] = None, top_k: int = 5) -> str:
    from search import hybrid_search
    top_k = max(1, min(top_k, 10))
    results = hybrid_search(query_text=query, image_url=image_url, top_k=top_k)
    return json.dumps(results, ensure_ascii=False)


@tool
@_tool_with_retry
def product_search(query: Optional[str] = None, image_url: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search the product catalog using natural language, an optional product image URL, or both.

    When image_url is provided the search engine automatically:
      - Calls GPT-4o to understand the image and produce a text description.
      - Extracts an image embedding for visual similarity search.
      - Runs all three channels (description HNSW + image HNSW + BM25) and fuses with RRF.

    Args:
        query:     Natural language description of what the user is looking for.
                   Could be empty when image_url is the primary input.
        image_url: URL of a product image (optional). Triggers visual similarity search.
        top_k:     Maximum number of results to return (1-10, default 5).

    Returns:
        JSON list of matching products, each with item_id, name, category,
        description snippet, color, material, brand, and image_path.
    """
    return _product_search_impl(query, image_url, top_k)


@tool
@_tool_with_retry
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
@_tool_with_retry
def understand_image(image_url: str) -> str:
    """
    Analyze a product image and return a text description.

    Use this when the user wants to know what a product is without searching,
    or to confirm what an uploaded image shows before recommending alternatives.
    For image-based product search, prefer product_search(image_url=...) directly.

    Args:
        image_url: URL of the product image.

    Returns:
        A concise product description (type, color, material, style, features).
    """
    from search import _fetch_image, describe_image
    from pathlib import Path
    img_b64, tmp_path = _fetch_image(image_url)
    try:
        return describe_image(img_b64)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


TOOLS = [product_search, get_product_info, understand_image]
