from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid


FAILURE_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "analytics_failures.jsonl"


def log_analytics_failure(payload: dict[str, Any]) -> dict[str, Any]:
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "entry_id": uuid.uuid4().hex,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **(payload or {}),
    }
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str, ensure_ascii=True) + "\n")
    return {
        "entry_id": entry["entry_id"],
        "path": str(FAILURE_LOG_PATH),
    }