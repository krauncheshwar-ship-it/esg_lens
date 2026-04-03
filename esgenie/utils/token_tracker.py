"""
token_tracker.py
Logs token usage and estimated cost for every LLM and embedding call.
Stores a running session log in memory and optionally writes to a JSONL file.
"""

import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()

LOG_TOKENS = os.getenv("LOG_TOKENS", "1") == "1"

# ---- Cost table (USD per 1 000 tokens) ----
_COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
}

_LOG_PATH = Path(__file__).parent.parent / "token_log.jsonl"

# In-memory session accumulator
_session: dict = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_embedding_tokens": 0,
    "total_cost_usd": 0.0,
    "calls": [],
}


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    table = _COST_TABLE.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens / 1000 * table["input"]) + (
        completion_tokens / 1000 * table["output"]
    )


def _append_log(entry: dict) -> None:
    """Append a log entry to the JSONL file if LOG_TOKENS env var is set."""
    if os.environ.get("LOG_TOKENS", "0") == "1":
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def track_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    call_type: str = "llm",
) -> dict:
    """
    Record token usage for a chat completion call.

    Args:
        model:             Model identifier (e.g. "gpt-4o-mini").
        prompt_tokens:     Tokens in the prompt.
        completion_tokens: Tokens in the completion.
        call_type:         Label for the log (e.g. "extraction_agent").

    Returns:
        Dict with the recorded entry.
    """
    cost = _cost(model, prompt_tokens, completion_tokens)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": call_type,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 6),
    }

    _session["total_prompt_tokens"] += prompt_tokens
    _session["total_completion_tokens"] += completion_tokens
    _session["total_cost_usd"] += cost
    _session["calls"].append(entry)

    if LOG_TOKENS:
        print(
            f"[TOKEN LOG] {call_type} | model:{model} | "
            f"in:{prompt_tokens} out:{completion_tokens} | ${cost:.5f}"
        )

    _append_log(entry)
    return entry


def track_embedding(
    model: str,
    tokens_used: int,
    num_texts: int = 1,
) -> dict:
    """
    Record token usage for an embedding call.

    Args:
        model:       Embedding model identifier.
        tokens_used: Total tokens consumed.
        num_texts:   Number of texts in the batch (for logging only).

    Returns:
        Dict with the recorded entry.
    """
    cost = _cost(model, tokens_used, 0)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "embedding",
        "model": model,
        "tokens_used": tokens_used,
        "num_texts": num_texts,
        "cost_usd": round(cost, 6),
    }

    _session["total_embedding_tokens"] += tokens_used
    _session["total_cost_usd"] += cost
    _session["calls"].append(entry)

    if LOG_TOKENS:
        print(
            f"[TOKEN LOG] embedding | model:{model} | "
            f"tokens:{tokens_used} texts:{num_texts} | ${cost:.5f}"
        )

    _append_log(entry)
    return entry


def get_session_summary() -> dict:
    """Return the current session's cumulative token and cost summary."""
    return {
        "total_prompt_tokens": _session["total_prompt_tokens"],
        "total_completion_tokens": _session["total_completion_tokens"],
        "total_embedding_tokens": _session["total_embedding_tokens"],
        "total_cost_usd": round(_session["total_cost_usd"], 4),
        "call_count": len(_session["calls"]),
    }


def reset_session() -> None:
    """Reset the in-memory session counters (useful between Streamlit runs)."""
    _session["total_prompt_tokens"] = 0
    _session["total_completion_tokens"] = 0
    _session["total_embedding_tokens"] = 0
    _session["total_cost_usd"] = 0.0
    _session["calls"] = []