"""
Browser regression tests for frontend chat history behavior.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).parent.parent
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://127.0.0.1:3000")
PLAYWRIGHT_CMD = ["npx", "--yes", "--package", "@playwright/cli", "playwright-cli"]
SCRIPT = ROOT / "tests" / "scripts" / "frontend_image_history_regression.js"
UPLOAD_FIXTURE = ROOT / "tests" / "fixtures" / "upload-test.svg"
SESSION = "frontend-image-history-regression"


def _run_playwright(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        PLAYWRIGHT_CMD + args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )


@pytest.fixture(scope="module")
def ensure_frontend_server():
    try:
        resp = requests.get(FRONTEND_URL, timeout=5)
    except Exception:
        pytest.skip(f"Frontend server not available at {FRONTEND_URL}")

    if resp.status_code != 200:
        pytest.skip(f"Frontend server returned HTTP {resp.status_code} at {FRONTEND_URL}")


@pytest.fixture(scope="module")
def ensure_playwright_cli():
    if not UPLOAD_FIXTURE.exists():
        pytest.skip(f"Upload fixture missing: {UPLOAD_FIXTURE}")

    version = _run_playwright(["--version"], timeout=60)
    if version.returncode != 0:
        pytest.skip(f"Playwright CLI unavailable: {version.stderr[-500:]}")


@pytest.mark.api
@pytest.mark.slow
def test_uploaded_image_preview_survives_chat_history_restore(
    ensure_frontend_server,
    ensure_playwright_cli,
):
    env = {
        "FRONTEND_URL": FRONTEND_URL,
        "UPLOAD_FIXTURE_PATH": str(UPLOAD_FIXTURE),
    }

    open_result = _run_playwright([f"-s={SESSION}", "open", "about:blank"], env=env, timeout=60)
    if open_result.returncode != 0:
        pytest.skip(f"Could not open Playwright browser: {open_result.stderr[-500:]}")

    try:
        run_result = _run_playwright(
            [f"-s={SESSION}", "run-code", "--filename", str(SCRIPT)],
            env=env,
            timeout=120,
        )
        assert run_result.returncode == 0, (
            "Frontend image-history regression failed:\n"
            f"STDOUT:\n{run_result.stdout[-2000:]}\n"
            f"STDERR:\n{run_result.stderr[-2000:]}"
        )
    finally:
        _run_playwright([f"-s={SESSION}", "close"], env=env, timeout=30)
