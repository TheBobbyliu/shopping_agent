"""
Shared pytest fixtures for all algorithm tests.
"""
import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Make preprocessing importable
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
load_dotenv(Path(__file__).parent.parent / ".env")

# Tests always run against the test index, regardless of what .env sets for production.
os.environ["ES_INDEX"] = "products_test"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG_FILE = FIXTURES_DIR / "catalog_500.jsonl"
IMAGES_DIR = Path(__file__).parent.parent / "data/abo-images-small/images/small"


# ---------------------------------------------------------------------------
# Catalog fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def catalog() -> list[dict]:
    """All 500 products from the test catalog."""
    assert CATALOG_FILE.exists(), f"Run extract.py first: {CATALOG_FILE}"
    with open(CATALOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def sample_products(catalog) -> list[dict]:
    """20 products spread across different categories."""
    seen_cats = set()
    result = []
    for p in catalog:
        if p["category"] not in seen_cats:
            seen_cats.add(p["category"])
            result.append(p)
        if len(result) >= 20:
            break
    return result


@pytest.fixture(scope="session")
def sample_texts(sample_products) -> list[str]:
    return [p["description"][:500] for p in sample_products]


@pytest.fixture(scope="session")
def sample_images(sample_products) -> list[Path]:
    paths = []
    for p in sample_products:
        path = Path(p["image_path"])
        if path.exists():
            paths.append(path)
        if len(paths) >= 10:
            break
    return paths


# ---------------------------------------------------------------------------
# Embedding client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_client():
    """
    Returns a live EmbeddingClient.
    Skips the test if credentials are not configured.
    """
    from embed import EmbeddingClient

    backend = os.environ.get("EMBEDDING_BACKEND", "local")

    if backend == "vikingdb":
        if not os.environ.get("VOLCANO_AK") or not os.environ.get("VOLCANO_SK"):
            pytest.skip("VikingDB credentials not set (VOLCANO_AK / VOLCANO_SK).")
    elif backend == "ark":
        if not os.environ.get("ARK_API_KEY"):
            pytest.skip("ARK_API_KEY not set.")
        if not os.environ.get("ARK_EMBEDDING_ENDPOINT"):
            pytest.skip("ARK_EMBEDDING_ENDPOINT not set.")
    # local backend: no credentials needed, always available

    return EmbeddingClient.from_env()


# ---------------------------------------------------------------------------
# OpenAI client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def openai_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Helpers available to all tests
# ---------------------------------------------------------------------------

def load_labeled_queries(name: str) -> list[dict]:
    """Load a labeled query fixture, skip if file does not exist yet."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture not generated yet: {path}. Run tests/scripts/generate_fixtures.py.")
    with open(path) as f:
        return json.load(f)
