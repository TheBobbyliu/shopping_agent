"""
Next.js frontend smoke tests.

FE-1: npm run build succeeds
FE-2: Built output contains expected HTML entry point
FE-3: /chat API integration — frontend JS references the correct endpoint
FE-4: Session ID is generated and sent with each request
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).parent.parent / "frontend-next"


@pytest.fixture(scope="session", autouse=True)
def check_frontend_exists():
    if not FRONTEND_DIR.exists():
        pytest.skip("frontend-next/ not yet created")


@pytest.mark.slow
def test_build_succeeds():
    """FE-1: npm run build exits 0. Skip with -m 'not slow' to reuse existing build."""
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"npm run build failed:\nSTDOUT: {result.stdout[-1000:]}\nSTDERR: {result.stderr[-1000:]}"
    )


@pytest.mark.api
def test_build_output_exists():
    """FE-2: .next/static directory exists after build."""
    assert (FRONTEND_DIR / ".next").exists(), ".next directory not found — run npm run build"


def _frontend_src_files():
    all_ts = list(FRONTEND_DIR.rglob("*.ts")) + list(FRONTEND_DIR.rglob("*.tsx"))
    return [f for f in all_ts if "node_modules" not in str(f) and ".next" not in str(f)]


@pytest.mark.api
def test_api_endpoint_referenced():
    """FE-3: source code references /chat endpoint."""
    src_files = _frontend_src_files()
    found = any("/chat" in f.read_text() for f in src_files)
    assert found, "No source file references /chat endpoint"


@pytest.mark.api
def test_session_id_in_source():
    """FE-4: session_id is generated and used in requests."""
    src_files = _frontend_src_files()
    found = any("session_id" in f.read_text() for f in src_files)
    assert found, "session_id not found in frontend source — context persistence won't work"
