"""
query_router.py
Classifies an incoming query into one or more ESG themes using GPT-4o-mini.
"""

import json
import os
from openai import OpenAI
from utils.token_tracker import track_llm_call

_ROUTER_MODEL = "gpt-4o-mini"
_VALID_THEMES = {"environment", "social", "governance", "general"}

_SYSTEM_PROMPT = """You are an ESG (Environmental, Social, Governance) query classifier.

Given a user query, return a JSON object with:
  - "themes": a list of relevant ESG themes from ["environment", "social", "governance"].
              Use "general" if the query spans all themes or is not theme-specific.
  - "rationale": one sentence explaining the classification.

Always return valid JSON. Example:
{"themes": ["environment", "social"], "rationale": "The query mentions both carbon emissions and workforce diversity."}
"""


def route_query(query: str) -> dict:
    """
    Classify a user query into ESG themes.

    Args:
        query: The raw user question.

    Returns:
        Dict with keys:
            "themes":    list[str]  — subset of {"environment","social","governance","general"}
            "rationale": str        — short explanation
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=_ROUTER_MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    track_llm_call(
        model=_ROUTER_MODEL,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        call_type="query_router",
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Sanitize themes to known values
    themes = [t for t in result.get("themes", ["general"]) if t in _VALID_THEMES]
    if not themes:
        themes = ["general"]

    return {
        "themes": themes,
        "rationale": result.get("rationale", ""),
    }
