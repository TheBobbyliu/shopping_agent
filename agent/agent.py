"""
Shopping agent — LangGraph ReAct agent powered by Claude.

The agent handles three shopping use cases in a single conversation:
  1. General product Q&A / recommendations (text)
  2. Text-based product search
  3. Image-based product search (upload an image → find similar items)

Usage:
    from agent import create_agent
    agent = create_agent()
    result = agent.invoke({"messages": [HumanMessage(content="Show me running shoes")]})
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

sys.path.insert(0, str(Path(__file__).parent))
from tools import TOOLS

SYSTEM_PROMPT = """You are a helpful AI shopping assistant — think of yourself as a knowledgeable store clerk who knows every product in the catalog.

You help users with three things:
1. **Product recommendations** — understand what they need and suggest relevant items
2. **Text search** — find products matching a description, category, color, material, etc.
3. **Image search** — when a user uploads an image, analyze it and find visually similar products

## How to use your tools

**product_search**: Your primary tool. Use it for any shopping request.
- For text queries: pass the user's request as `query`
- For image searches: pass both `query` (your description of the image) and `image_b64`
- You can call it multiple times to refine results

**understand_image**: Call this first when the user uploads an image to get a clear description, then use that description in product_search with the image.

**get_product_info**: Use when the user asks for more details about a specific product, or when you need complete product information before making a recommendation.

## Response style
- Be conversational and helpful, not robotic
- Present search results as a curated selection, not a raw list
- Highlight 2–4 best matches with name, key features (color, material), and why it matches
- Always mention the item_id so users can click for more details
- Ask clarifying questions if the request is vague
- If no results match well, explain this and suggest alternatives

## Product catalog
The catalog contains ~500 products across categories including shoes, furniture, apparel, jewelry, home decor, kitchen, electronics accessories, and more. All products are real items from Amazon.
"""


def create_agent(model: str | None = None):
    """Create and return the shopping agent.

    Backend is selected automatically:
      - ANTHROPIC_API_KEY → claude-haiku-4-5-20251001 (preferred)
      - OPENAI_API_KEY    → gpt-4o-mini (fallback)
    Override with AGENT_MODEL env var.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    agent_model   = model or os.environ.get("AGENT_MODEL", "")

    if anthropic_key and not agent_model.startswith("gpt"):
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=agent_model or "claude-haiku-4-5-20251001",
            api_key=anthropic_key,
            temperature=0.3,
            max_tokens=1024,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=agent_model or "gpt-4o-mini",
            api_key=openai_key,
            temperature=0.3,
            max_tokens=1024,
        )
    else:
        raise EnvironmentError(
            "No LLM credentials found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return agent
