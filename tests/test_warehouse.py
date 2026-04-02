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
    assert "Not found: 0, Errors: 0" in out


def test_delete_warns_on_missing_item(capsys):
    import warehouse
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=[{"item_id": "ZZZZ"}]), \
         patch("warehouse.get_indexed_ids", return_value=set()), \
         patch("warehouse._get_es", return_value=mock_es):
        warehouse.cmd_delete(MagicMock(json_file=None))
    err = capsys.readouterr().err
    assert "[warn] ZZZZ — not found, skipping" in err
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
    captured = capsys.readouterr()
    assert "[skip] A001" in captured.err
    assert "Added: 1" in captured.out
    assert "Skipped: 1" in captured.out
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


def test_add_exits_on_embed_failure(capsys):
    import warehouse
    items = [{"item_id": "A004", "description": "desc", "image_path": "img.jpg"}]
    mock_es = MagicMock()
    mock_es.indices = MagicMock()
    with patch("warehouse._load_json", return_value=items), \
         patch("warehouse.get_indexed_ids", return_value=set()), \
         patch("warehouse._call_embed", side_effect=RuntimeError("Cannot reach embedding service at http://localhost:8000")), \
         patch("warehouse._get_es", return_value=mock_es):
        with pytest.raises(SystemExit):
            warehouse.cmd_add(MagicMock(json_file=None))
    err = capsys.readouterr().err
    assert "embed failed" in err
    assert "A004" in err
    mock_es.index.assert_not_called()
    mock_es.indices.refresh.assert_called_once_with(index=warehouse.ES_INDEX)


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
    """Delete the WH_TEST_* items before and after each integration test."""
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
    out = capsys.readouterr()
    assert "[skip]" in out.err
    assert "Skipped: 1" in out.out
    assert "Added: 0" in out.out


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

    captured = capsys.readouterr()
    assert "Deleted: 1" in captured.out
    assert "Not found: 1" in captured.out
    assert "[warn]" in captured.err

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
