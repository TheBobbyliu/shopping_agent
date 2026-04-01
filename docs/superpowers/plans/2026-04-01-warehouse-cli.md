# Warehouse CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `warehouse` CLI tool for operational staff to add, delete, and check products in the Elasticsearch index, backed by the existing embedding model served through the running FastAPI API.

**Architecture:** A new `preprocessing/warehouse.py` CLI (argparse-based, same pattern as `index.py`) provides three commands: `add` (validate → skip existing → embed via `POST /embed` → index into ES), `delete` (validate → warn missing → bulk delete), and `check` (query ES, strip vectors, print JSON). A new `/embed` endpoint in `api/main.py` exposes the already-loaded `_get_embedding_client()` singleton over HTTP so the CLI reuses the warm model without a cold start.

**Tech Stack:** Python, argparse, elasticsearch-py, requests, pytest, unittest.mock

---

## File Map

| File | Change | Responsibility |
|---|---|---|
| `api/main.py` | Modify | Add `POST /embed` endpoint (2 Pydantic models + 1 route) |
| `preprocessing/warehouse.py` | Create | CLI: argparse entry point, `cmd_add`, `cmd_delete`, `cmd_check`, helpers |
| `tests/test_warehouse.py` | Create | Unit tests (mocked ES + embed) and integration tests (`@pytest.mark.api`) |

---

### Task 1: `/embed` endpoint in `api/main.py`

**Files:**
- Modify: `api/main.py`
- Create: `tests/test_warehouse.py`

- [ ] **Step 1: Create test file with failing `/embed` tests**

Create `tests/test_warehouse.py`:

```python
"""Tests for warehouse CLI and the /embed API endpoint."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))


# ---------------------------------------------------------------------------
# /embed endpoint — integration tests (require running API + loaded models)
# ---------------------------------------------------------------------------

@pytest.mark.api
def test_embed_endpoint_returns_vectors(tmp_path):
    """POST /embed returns description_vector and image_vector of length 1024."""
    import requests
    from PIL import Image

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(str(img_path))

    resp = requests.post("http://localhost:8000/embed", json={
        "text": "a small grey square",
        "image_path": str(img_path),
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "description_vector" in data
    assert "image_vector" in data
    assert len(data["description_vector"]) == 1024
    assert len(data["image_vector"]) == 1024


@pytest.mark.api
def test_embed_endpoint_missing_image_returns_400():
    """POST /embed with a nonexistent image_path returns 400."""
    import requests

    resp = requests.post("http://localhost:8000/embed", json={
        "text": "something",
        "image_path": "/nonexistent/path/image.jpg",
    })
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests — confirm FAIL (404, endpoint missing)**

```bash
cd /Users/bob/Workspace/shopping-demo
pytest tests/test_warehouse.py -v -m api
```

Expected: FAIL — `AssertionError: assert 404 == 200`

- [ ] **Step 3: Add `EmbedRequest`, `EmbedResponse`, and `/embed` route to `api/main.py`**

After the `ChatResponse` model (around line 158 in `api/main.py`), insert:

```python
class EmbedRequest(BaseModel):
    text: str
    image_path: str


class EmbedResponse(BaseModel):
    description_vector: list[float]
    image_vector: list[float]
```

After the `/ready` route (around line 192), insert:

```python
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Embed text + image using the already-loaded singleton model.
    Used by the warehouse CLI so it reuses the warm model without cold start."""
    if not _startup_complete:
        raise HTTPException(status_code=503, detail="Embedding service not ready")
    path = Path(req.image_path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"image_path not found: {req.image_path}")
    from search import _get_embedding_client
    client = _get_embedding_client()
    desc_vec = client.embed_text(req.text)
    img_vec = client.embed_image(str(path))
    return EmbedResponse(
        description_vector=list(desc_vec),
        image_vector=list(img_vec),
    )
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_warehouse.py -v -m api
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_warehouse.py
git commit -m "feat: add /embed endpoint for warehouse CLI"
```

---

### Task 2: `warehouse.py` scaffold + `check` command

**Files:**
- Create: `preprocessing/warehouse.py`
- Modify: `tests/test_warehouse.py`

- [ ] **Step 1: Append failing unit tests for `check` to `tests/test_warehouse.py`**

```python
# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------

def _sample_docs():
    return [
        {
            "item_id": "A001", "name": "Red chair", "category": "FURNITURE",
            "description": "A comfortable red chair",
            "description_vector": [0.1] * 1024,
            "image_vector": [0.2] * 1024,
        },
        {
            "item_id": "A002", "name": "Blue lamp", "category": "LIGHTING",
            "description": "A modern blue lamp",
            "description_vector": [0.3] * 1024,
            "image_vector": [0.4] * 1024,
        },
    ]


def _make_es_mock(docs):
    mock_es = MagicMock()

    def fake_get(index, id):
        for d in docs:
            if d["item_id"] == id:
                return {"_source": d}
        raise Exception(f"Not found: {id}")
    mock_es.get.side_effect = fake_get

    def fake_search(**kwargs):
        query = kwargs.get("query", {})
        size = kwargs.get("size", 10)
        if "term" in query:
            cat = list(query["term"].values())[0]
            hits = [{"_source": d} for d in docs if d.get("category") == cat][:size]
        else:
            hits = [{"_source": d} for d in docs][:size]
        return {"hits": {"hits": hits}}
    mock_es.search.side_effect = fake_search

    return mock_es


# ---------------------------------------------------------------------------
# check — unit tests
# ---------------------------------------------------------------------------

def test_check_item_id_strips_vectors(capsys):
    import warehouse
    mock_es = _make_es_mock(_sample_docs())
    args = MagicMock(item_id="A001", group=None, count=None)
    with patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_check(args)
    data = json.loads(capsys.readouterr().out)
    assert data["item_id"] == "A001"
    assert "description_vector" not in data
    assert "image_vector" not in data


def test_check_item_id_not_found_exits(capsys):
    import warehouse
    mock_es = _make_es_mock(_sample_docs())
    args = MagicMock(item_id="ZZZZ", group=None, count=None)
    with patch("warehouse._get_es", return_value=mock_es):
        with pytest.raises(SystemExit):
            warehouse.cmd_check(args)


def test_check_group_filters_by_category(capsys):
    import warehouse
    mock_es = _make_es_mock(_sample_docs())
    args = MagicMock(item_id=None, group="FURNITURE", count=10)
    with patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_check(args)
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
    assert data[0]["item_id"] == "A001"
    assert "description_vector" not in data[0]


def test_check_count_returns_n_items(capsys):
    import warehouse
    mock_es = _make_es_mock(_sample_docs())
    args = MagicMock(item_id=None, group=None, count=1)
    with patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_check(args)
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
```

- [ ] **Step 2: Run tests — confirm FAIL (ModuleNotFoundError)**

```bash
pytest tests/test_warehouse.py -v -k "test_check"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'warehouse'`

- [ ] **Step 3: Create `preprocessing/warehouse.py` with scaffold and `cmd_check`**

```python
#!/usr/bin/env python3
"""
Warehouse CLI — manage products in the Elasticsearch index.

Usage:
    python warehouse.py add items.json
    python warehouse.py delete items.json
    python warehouse.py check --item-id B123456789
    python warehouse.py check --group ELECTRONICS [--count 10]
    python warehouse.py check --count 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Make index.py importable whether run as a script or imported by tests.
sys.path.insert(0, str(Path(__file__).parent))
from index import get_indexed_ids

ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.environ.get("ES_INDEX", "products")
API_URL  = os.environ.get("API_BASE_URL", "http://localhost:8000")

REQUIRED_ADD_FIELDS = {"item_id", "description", "image_path"}
VECTOR_FIELDS = {"description_vector", "image_vector"}

_es_client = None


def _get_es():
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        _es_client = Elasticsearch(ES_URL)
    return _es_client


def _strip_vectors(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in VECTOR_FIELDS}


def _load_json(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[error] Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("[error] JSON must be an array of objects", file=sys.stderr)
        sys.exit(1)
    return data


def _call_embed(text: str, image_path: str) -> dict:
    """POST to /embed on the running API server. Returns {description_vector, image_vector}."""
    import requests
    try:
        resp = requests.post(
            f"{API_URL}/embed",
            json={"text": text, "image_path": image_path},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach embedding service at {API_URL} — is the API server running?"
        )
    if resp.status_code == 400:
        raise ValueError(resp.json().get("detail", "Bad request from /embed"))
    if resp.status_code == 503:
        raise RuntimeError("Embedding service not ready — API server is still starting up")
    resp.raise_for_status()
    return resp.json()


def cmd_check(args):
    es = _get_es()

    if args.item_id:
        try:
            resp = es.get(index=ES_INDEX, id=args.item_id)
            doc = _strip_vectors(resp["_source"])
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[error] Item '{args.item_id}' not found: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.group:
        count = args.count or 10
        resp = es.search(
            index=ES_INDEX,
            query={"term": {"category": args.group}},
            size=count,
            _source=True,
        )
        hits = [_strip_vectors(h["_source"]) for h in resp["hits"]["hits"]]
        print(json.dumps(hits, indent=2, ensure_ascii=False))

    else:
        count = args.count or 10
        resp = es.search(
            index=ES_INDEX,
            query={"match_all": {}},
            size=count,
            _source=True,
        )
        hits = [_strip_vectors(h["_source"]) for h in resp["hits"]["hits"]]
        print(json.dumps(hits, indent=2, ensure_ascii=False))


def cmd_add(args):
    pass  # implemented in Task 4


def cmd_delete(args):
    pass  # implemented in Task 3


def main():
    parser = argparse.ArgumentParser(
        prog="warehouse",
        description="Manage products in the Elasticsearch index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add items from a JSON file")
    p_add.add_argument("json_file", help="Path to JSON file")

    p_del = sub.add_parser("delete", help="Delete items by item_id from a JSON file")
    p_del.add_argument("json_file", help="Path to JSON file")

    p_check = sub.add_parser("check", help="Query items in the index")
    p_check.add_argument("--item-id", dest="item_id", help="Fetch a single item by ID")
    p_check.add_argument("--group", help="Fetch items by category")
    p_check.add_argument(
        "--count", type=int,
        help="Number of results (default 10); use alone to query across all groups",
    )

    args = parser.parse_args()

    if args.command == "check":
        if args.item_id and args.group:
            parser.error("--item-id and --group are mutually exclusive")
        if not args.item_id and not args.group and args.count is None:
            parser.error("specify at least one of --item-id, --group, or --count")
        cmd_check(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "delete":
        cmd_delete(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_warehouse.py -v -k "test_check"
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add preprocessing/warehouse.py tests/test_warehouse.py
git commit -m "feat: warehouse scaffold + check command"
```

---

### Task 3: `delete` command

**Files:**
- Modify: `preprocessing/warehouse.py`
- Modify: `tests/test_warehouse.py`

- [ ] **Step 1: Append failing unit tests for `delete`**

```python
# ---------------------------------------------------------------------------
# delete — unit tests
# ---------------------------------------------------------------------------

def test_delete_removes_existing_item(capsys):
    import warehouse
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=[{"item_id": "A001"}]), \
         patch("warehouse.get_indexed_ids", return_value={"A001"}), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_delete(MagicMock(json_file=None))
    mock_es.delete.assert_called_once_with(index=warehouse.ES_INDEX, id="A001")
    out = capsys.readouterr().out
    assert "Deleted: 1" in out
    assert "Not found: 0" in out


def test_delete_warns_on_missing_item(capsys):
    import warehouse
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=[{"item_id": "ZZZZ"}]), \
         patch("warehouse.get_indexed_ids", return_value=set()), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_delete(MagicMock(json_file=None))
    err = capsys.readouterr().err
    assert "[warn]" in err
    assert "ZZZZ" in err
    mock_es.delete.assert_not_called()


def test_delete_aborts_on_duplicate_input(capsys):
    import warehouse
    with patch("warehouse._load_json", return_value=[
        {"item_id": "A001"},
        {"item_id": "A001"},
    ]):
        with pytest.raises(SystemExit):
            warehouse.cmd_delete(MagicMock(json_file=None))
    assert "Duplicate" in capsys.readouterr().err


def test_delete_aborts_on_missing_item_id_field(capsys):
    import warehouse
    with patch("warehouse._load_json", return_value=[{"name": "no item_id here"}]):
        with pytest.raises(SystemExit):
            warehouse.cmd_delete(MagicMock(json_file=None))
    assert "item_id" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_warehouse.py -v -k "test_delete"
```

Expected: FAIL — `cmd_delete` is a stub (`pass`)

- [ ] **Step 3: Replace `cmd_delete` stub in `preprocessing/warehouse.py`**

```python
def cmd_delete(args):
    items = _load_json(args.json_file)

    # Validate required field
    errors = []
    for i, item in enumerate(items):
        if "item_id" not in item:
            errors.append(f"  item[{i}]: missing 'item_id'")
    if errors:
        print("[error] Missing required fields:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # Detect duplicates within input
    seen: dict[str, int] = {}
    duplicates = []
    for i, item in enumerate(items):
        iid = item["item_id"]
        if iid in seen:
            duplicates.append(f"  '{iid}' at index {seen[iid]} and {i}")
        else:
            seen[iid] = i
    if duplicates:
        print("[error] Duplicate item_ids in input:", file=sys.stderr)
        for d in duplicates:
            print(d, file=sys.stderr)
        sys.exit(1)

    all_ids = [item["item_id"] for item in items]
    existing = get_indexed_ids(ES_URL, ES_INDEX, all_ids)

    for iid in all_ids:
        if iid not in existing:
            print(f"[warn] {iid} — not found, skipping", file=sys.stderr)

    es = _get_es()
    deleted = 0
    for iid in all_ids:
        if iid in existing:
            es.delete(index=ES_INDEX, id=iid)
            deleted += 1

    es.indices.refresh(index=ES_INDEX)
    not_found = len(all_ids) - deleted
    print(f"\nDeleted: {deleted}, Not found: {not_found}")
```

- [ ] **Step 4: Run tests — confirm PASS**

```bash
pytest tests/test_warehouse.py -v -k "test_delete"
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add preprocessing/warehouse.py tests/test_warehouse.py
git commit -m "feat: warehouse delete command"
```

---

### Task 4: `add` command

**Files:**
- Modify: `preprocessing/warehouse.py`
- Modify: `tests/test_warehouse.py`

- [ ] **Step 1: Append failing unit tests for `add`**

```python
# ---------------------------------------------------------------------------
# add — unit tests
# ---------------------------------------------------------------------------

def _embed_mock(desc_vec=None, img_vec=None):
    return MagicMock(return_value={
        "description_vector": desc_vec or [0.1] * 1024,
        "image_vector":       img_vec  or [0.2] * 1024,
    })


def test_add_aborts_on_missing_required_field(capsys):
    import warehouse
    # item_id and description present but image_path missing
    with patch("warehouse._load_json", return_value=[
        {"item_id": "A001", "description": "a desc"},
    ]):
        with pytest.raises(SystemExit):
            warehouse.cmd_add(MagicMock(json_file=None))
    assert "image_path" in capsys.readouterr().err


def test_add_aborts_on_duplicate_item_ids(capsys):
    import warehouse
    with patch("warehouse._load_json", return_value=[
        {"item_id": "A001", "description": "d1", "image_path": "p1.jpg"},
        {"item_id": "A001", "description": "d2", "image_path": "p2.jpg"},
    ]):
        with pytest.raises(SystemExit):
            warehouse.cmd_add(MagicMock(json_file=None))
    assert "Duplicate" in capsys.readouterr().err


def test_add_skips_already_indexed_item(capsys):
    import warehouse
    items = [
        {"item_id": "A001", "description": "d1", "image_path": "p1.jpg"},
        {"item_id": "A002", "description": "d2", "image_path": "p2.jpg"},
    ]
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=items), \
         patch("warehouse.get_indexed_ids", return_value={"A001"}), \
         patch("warehouse._call_embed", _embed_mock()), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_add(MagicMock(json_file=None))
    out = capsys.readouterr().out
    assert "[skip] A001" in out
    assert "Added: 1" in out
    assert "Skipped: 1" in out
    indexed_ids = [c.kwargs["id"] for c in mock_es.index.call_args_list]
    assert "A002" in indexed_ids
    assert "A001" not in indexed_ids


def test_add_indexes_item_with_vectors(capsys):
    import warehouse
    items = [{"item_id": "A003", "description": "A blue cup", "image_path": "cup.jpg"}]
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=items), \
         patch("warehouse.get_indexed_ids", return_value=set()), \
         patch("warehouse._call_embed", _embed_mock(desc_vec=[0.5]*1024, img_vec=[0.6]*1024)), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_add(MagicMock(json_file=None))
    doc = mock_es.index.call_args.kwargs["document"]
    assert doc["item_id"] == "A003"
    assert doc["description_vector"] == [0.5] * 1024
    assert doc["image_vector"] == [0.6] * 1024
    assert "Added: 1" in capsys.readouterr().out


def test_add_counts_embed_errors(capsys):
    import warehouse
    items = [{"item_id": "A004", "description": "desc", "image_path": "img.jpg"}]
    mock_es = MagicMock()
    with patch("warehouse._load_json", return_value=items), \
         patch("warehouse.get_indexed_ids", return_value=set()), \
         patch("warehouse._call_embed", side_effect=RuntimeError("API down")), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_add(MagicMock(json_file=None))
    assert "Errors: 1" in capsys.readouterr().out
    mock_es.index.assert_not_called()
```

- [ ] **Step 2: Run tests — confirm FAIL**

```bash
pytest tests/test_warehouse.py -v -k "test_add"
```

Expected: FAIL — `cmd_add` is a stub (`pass`)

- [ ] **Step 3: Replace `cmd_add` stub in `preprocessing/warehouse.py`**

```python
def cmd_add(args):
    items = _load_json(args.json_file)

    # Validate required fields
    errors = []
    for i, item in enumerate(items):
        missing = REQUIRED_ADD_FIELDS - set(item.keys())
        if missing:
            errors.append(
                f"  item[{i}] ({item.get('item_id', '?')}): missing {sorted(missing)}"
            )
    if errors:
        print("[error] Missing required fields:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    # Detect duplicates within input
    seen: dict[str, int] = {}
    duplicates = []
    for i, item in enumerate(items):
        iid = item["item_id"]
        if iid in seen:
            duplicates.append(f"  '{iid}' at index {seen[iid]} and {i}")
        else:
            seen[iid] = i
    if duplicates:
        print("[error] Duplicate item_ids in input:", file=sys.stderr)
        for d in duplicates:
            print(d, file=sys.stderr)
        sys.exit(1)

    all_ids = [item["item_id"] for item in items]
    already_indexed = get_indexed_ids(ES_URL, ES_INDEX, all_ids)

    for item in items:
        if item["item_id"] in already_indexed:
            print(f"[skip] {item['item_id']} — already indexed")

    to_add = [item for item in items if item["item_id"] not in already_indexed]
    skipped = len(items) - len(to_add)
    total = len(to_add)
    added = 0
    errors_count = 0

    es = _get_es()
    for idx, item in enumerate(to_add, 1):
        iid = item["item_id"]
        print(f"[{idx}/{total}] {iid} — embedding...", end=" ", flush=True)
        try:
            vectors = _call_embed(item["description"], item["image_path"])
        except Exception as e:
            print(f"error (embed: {e})")
            errors_count += 1
            continue
        doc = {**item, **vectors}
        try:
            es.index(index=ES_INDEX, id=iid, document=doc)
            print("done")
            added += 1
        except Exception as e:
            print(f"error (index: {e})")
            errors_count += 1

    es.indices.refresh(index=ES_INDEX)
    print(f"\nAdded: {added}, Skipped: {skipped}, Errors: {errors_count}")
```

- [ ] **Step 4: Run all unit tests — confirm PASS**

```bash
pytest tests/test_warehouse.py -v -k "not api"
```

Expected: 13 PASS (4 check + 4 delete + 5 add)

- [ ] **Step 5: Commit**

```bash
git add preprocessing/warehouse.py tests/test_warehouse.py
git commit -m "feat: warehouse add command"
```

---

### Task 5: Integration tests

**Files:**
- Modify: `tests/test_warehouse.py`

Note: `conftest.py` already sets `os.environ["ES_INDEX"] = "products_test"`, so `warehouse.ES_INDEX` will be `"products_test"` when tests import `warehouse`.

- [ ] **Step 1: Append integration tests**

```python
# ---------------------------------------------------------------------------
# Integration tests — require ES (products_test index) + API server running
# ---------------------------------------------------------------------------

_INTEG_ITEMS = [
    {
        "item_id": "WH_TEST_001",
        "description": "A red leather sofa for living rooms",
        "image_path": str(
            Path(__file__).parent.parent
            / "data/abo-images-small/images/small/00/00000529.jpg"
        ),
        "name": "Red Leather Sofa",
        "brand": "TestBrand",
        "category": "FURNITURE",
    },
    {
        "item_id": "WH_TEST_002",
        "description": "A stainless steel kitchen knife",
        "image_path": str(
            Path(__file__).parent.parent
            / "data/abo-images-small/images/small/00/00003a93.jpg"
        ),
        "name": "Kitchen Knife",
        "brand": "TestBrand",
    },
]


@pytest.fixture()
def cleanup_integ_items():
    """Delete the two WH_TEST_* items before and after each integration test."""
    from elasticsearch import Elasticsearch

    es = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    index = os.environ.get("ES_INDEX", "products_test")

    def _delete_all():
        for item in _INTEG_ITEMS:
            try:
                es.delete(index=index, id=item["item_id"])
            except Exception:
                pass
        try:
            es.indices.refresh(index=index)
        except Exception:
            pass

    _delete_all()
    yield
    _delete_all()


@pytest.mark.api
def test_integ_add_items(cleanup_integ_items, tmp_path, capsys):
    """add two items → verify they appear in ES with 1024-dim vectors."""
    import warehouse
    from elasticsearch import Elasticsearch

    json_file = tmp_path / "items.json"
    json_file.write_text(json.dumps(_INTEG_ITEMS))

    warehouse.cmd_add(MagicMock(json_file=str(json_file)))

    out = capsys.readouterr().out
    assert "Added: 2" in out
    assert "Skipped: 0" in out
    assert "Errors: 0" in out

    es = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    index = os.environ.get("ES_INDEX", "products_test")
    for item in _INTEG_ITEMS:
        doc = es.get(index=index, id=item["item_id"])["_source"]
        assert doc["item_id"] == item["item_id"]
        assert len(doc.get("description_vector", [])) == 1024
        assert len(doc.get("image_vector", [])) == 1024


@pytest.mark.api
def test_integ_add_skips_existing(cleanup_integ_items, tmp_path, capsys):
    """Re-adding an already-indexed item produces a skip line, not an error."""
    import warehouse

    json_file = tmp_path / "items.json"
    json_file.write_text(json.dumps(_INTEG_ITEMS[:1]))

    warehouse.cmd_add(MagicMock(json_file=str(json_file)))
    capsys.readouterr()  # discard first-run output

    warehouse.cmd_add(MagicMock(json_file=str(json_file)))
    out = capsys.readouterr().out
    assert "[skip]" in out
    assert "Skipped: 1" in out
    assert "Added: 0" in out


@pytest.mark.api
def test_integ_delete_items(cleanup_integ_items, tmp_path, capsys):
    """Delete one existing item and one nonexistent → correct counts + warning."""
    import warehouse
    from elasticsearch import Elasticsearch

    add_file = tmp_path / "add.json"
    add_file.write_text(json.dumps(_INTEG_ITEMS[:1]))
    warehouse.cmd_add(MagicMock(json_file=str(add_file)))
    capsys.readouterr()

    del_file = tmp_path / "delete.json"
    del_file.write_text(json.dumps([
        {"item_id": _INTEG_ITEMS[0]["item_id"]},
        {"item_id": "WH_NONEXISTENT_999"},
    ]))
    warehouse.cmd_delete(MagicMock(json_file=str(del_file)))

    out = capsys.readouterr()
    assert "Deleted: 1" in out.out
    assert "Not found: 1" in out.out
    assert "[warn]" in out.err

    es = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    index = os.environ.get("ES_INDEX", "products_test")
    try:
        es.get(index=index, id=_INTEG_ITEMS[0]["item_id"])
        assert False, "Item should have been deleted"
    except Exception:
        pass  # expected — item is gone


@pytest.mark.api
def test_integ_check_item_id(cleanup_integ_items, tmp_path, capsys):
    """check --item-id returns full doc with vectors stripped."""
    import warehouse

    add_file = tmp_path / "add.json"
    add_file.write_text(json.dumps(_INTEG_ITEMS[:1]))
    warehouse.cmd_add(MagicMock(json_file=str(add_file)))
    capsys.readouterr()

    args = MagicMock(item_id=_INTEG_ITEMS[0]["item_id"], group=None, count=None)
    warehouse.cmd_check(args)

    data = json.loads(capsys.readouterr().out)
    assert data["item_id"] == _INTEG_ITEMS[0]["item_id"]
    assert "description_vector" not in data
    assert "image_vector" not in data


@pytest.mark.api
def test_integ_check_group(cleanup_integ_items, tmp_path, capsys):
    """check --group filters by category field."""
    import warehouse

    add_file = tmp_path / "add.json"
    add_file.write_text(json.dumps(_INTEG_ITEMS))
    warehouse.cmd_add(MagicMock(json_file=str(add_file)))
    capsys.readouterr()

    # WH_TEST_001 has category=FURNITURE; WH_TEST_002 has no category
    args = MagicMock(item_id=None, group="FURNITURE", count=10)
    warehouse.cmd_check(args)

    data = json.loads(capsys.readouterr().out)
    item_ids = [d["item_id"] for d in data]
    assert "WH_TEST_001" in item_ids
    assert "WH_TEST_002" not in item_ids


@pytest.mark.api
def test_integ_check_count(cleanup_integ_items, tmp_path, capsys):
    """check --count N returns at most N items."""
    import warehouse

    add_file = tmp_path / "add.json"
    add_file.write_text(json.dumps(_INTEG_ITEMS))
    warehouse.cmd_add(MagicMock(json_file=str(add_file)))
    capsys.readouterr()

    args = MagicMock(item_id=None, group=None, count=1)
    warehouse.cmd_check(args)

    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/test_warehouse.py -v -m api
```

Expected: all PASS (requires `products_test` ES index + API server with models loaded)

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/test_warehouse.py -v
```

Expected: all 13 unit tests PASS; integration tests PASS or skip if services unavailable

- [ ] **Step 4: Commit**

```bash
git add tests/test_warehouse.py
git commit -m "test: warehouse integration tests"
```
