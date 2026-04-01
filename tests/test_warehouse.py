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
