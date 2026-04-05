"""
system_logger.py
Writes structured operational events to logging_system/logs/system_ops.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_LOGS_DIR = Path(__file__).parent / "logs"


def log_event(
    event: str,
    module: str,
    run_id: str = "",
    company: str = "",
    duration_ms: int = 0,
    status: str = "success",
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    """
    Append one JSON entry to system_ops.jsonl.

    Args:
        event:       Event name (e.g. "pdf_ingestion_complete").
        module:      Source module path (e.g. "ingestion/pdf_parser.py").
        run_id:      Run identifier for correlation.
        company:     Company name associated with the event.
        duration_ms: Wall-clock duration of the operation in milliseconds.
        status:      "success" | "error" | "running".
        error:       Error message string, or None.
        extra:       Optional dict of additional fields to merge in.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        "module": module,
        "company": company,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
    }
    if extra:
        entry.update(extra)

    with open(_LOGS_DIR / "system_ops.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
