from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _REPO_ROOT / "logs" / "errors"
_WRITE_LOCK = threading.Lock()


def current_error_log_file(now: datetime | None = None) -> Path:
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    return _LOG_DIR / f"{current.strftime('%Y-%m-%d')}.log"


def relative_log_file(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def build_error_response_content(
    log_file: str,
    public_message: str = "Internal server error.",
) -> dict[str, str]:
    message = public_message.strip()
    if not message.endswith("."):
        message = f"{message}."
    return {
        "detail": f"{message} See {log_file}.",
        "log_file": log_file,
    }


def record_error(
    event: str,
    error: BaseException | str,
    *,
    context: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> str:
    current = now.astimezone() if now is not None else datetime.now().astimezone()
    log_path = current_error_log_file(current)
    entry: dict[str, Any] = {
        "timestamp": current.isoformat(),
        "event": event,
        "error_type": error.__class__.__name__ if isinstance(error, BaseException) else "Error",
        "error": str(error),
        "context": dict(context or {}),
    }
    if isinstance(error, BaseException):
        entry["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    with _WRITE_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str))
            fh.write("\n")

    return relative_log_file(log_path)
