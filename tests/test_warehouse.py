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
