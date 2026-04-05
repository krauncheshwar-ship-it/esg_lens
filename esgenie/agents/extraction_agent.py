"""
extraction_agent.py
GPT-4o-mini ESG data extractor. Called ONCE per profile generation on
pre-filtered, theme-bucketed context — never uses external knowledge.
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from utils.token_tracker import track_llm_call

load_dotenv()

_EXTRACT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an ESG data extraction specialist working with sustainability reports.

Your task is to extract specific ESG metrics from the provided document chunks.

RULES:
1. Extract data ONLY from the provided context. Never use external knowledge.
2. If the value is not present in the context, return null — do not estimate.
3. Return structured JSON only — no preamble, no explanation outside JSON.
4. Always identify whether you found the value in a structured table or prose.
5. For NUMERIC values: extract exact number and unit as reported.
6. For BOOLEAN values: return "Yes" if credible evidence found, "No" if explicitly
   stated as absent, "Not Disclosed" if topic not mentioned.
7. For SUBJECTIVE values: classify as HIGH, MEDIUM, or LOW with one-line reason.
8. Report the exact page number where you found the value."""


def _build_user_prompt(
    query: str,
    chunks: list[dict],
    company: str,
    year: int,
    value_type: str,
    unit_expected: str,
) -> str:
    """Assemble context block + extraction instruction for the LLM."""
    lines = [f"Context from {company} ESG Report {year}:"]
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number", chunk.get("page_num", "?"))
        lines.append(f"\n[Chunk {i} - Page {page}]:")
        lines.append(chunk["text"])

    lines.append(f"\nQuery: {query}")
    lines.append(f"Extract the {value_type} value. Expected unit: {unit_expected}")
    return "\n".join(lines)


def extract(
    query: str,
    chunks: list[dict],
    kpi_id: str = "",
    value_type: str = "numeric",
    unit_expected: str = "",
    company: str = "",
    year: int = 2024,
) -> dict:
    """
    Extract a single ESG KPI value from retrieved context chunks.

    Args:
        query:         Natural-language extraction question.
        chunks:        Top-k retrieved chunk dicts (must have "text", "page_number").
        kpi_id:        KPI identifier tag (e.g. "C001").
        value_type:    "numeric" | "boolean" | "subjective".
        unit_expected: Expected unit string hint (e.g. "tCO2e").
        company:       Company name for prompt context.
        year:          Reporting year for prompt context.

    Returns:
        Structured extraction dict with keys:
            kpi_id, metric, value, value_numeric, unit, year,
            source_page, source_type, confidence, direct_quote,
            reasoning, hallucination_flag.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_prompt = _build_user_prompt(query, chunks, company, year, value_type, unit_expected)

    response_schema = {
        "kpi_id": kpi_id,
        "metric": query,
        "value": "<extracted value or null>",
        "value_numeric": "<float or null>",
        "unit": "<unit string or null>",
        "year": year,
        "source_page": "<integer page number or null>",
        "source_type": "<'table' or 'prose' or null>",
        "confidence": "<'high' or 'medium' or 'low' or 'not_found'>",
        "direct_quote": "<short verbatim quote or null>",
        "reasoning": "<one sentence>",
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                f"Return ONLY a JSON object matching this schema:\n"
                f"{json.dumps(response_schema, indent=2)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=_EXTRACT_MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    track_llm_call(
        model=_EXTRACT_MODEL,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        call_type="extraction_agent",
    )

    result = json.loads(response.choices[0].message.content)

    # Ensure required keys exist with correct types
    result.setdefault("kpi_id", kpi_id)
    result.setdefault("year", year)
    result.setdefault("hallucination_flag", 0)

    # Coerce value_numeric to float
    try:
        vn = result.get("value_numeric")
        result["value_numeric"] = float(vn) if vn not in (None, "null", "") else None
    except (TypeError, ValueError):
        result["value_numeric"] = None

    # Coerce source_page to int
    try:
        sp = result.get("source_page")
        result["source_page"] = int(sp) if sp not in (None, "null", "") else None
    except (TypeError, ValueError):
        result["source_page"] = None

    # Hallucination flag: reported page != top chunk's page
    if chunks and result.get("source_page") is not None:
        top_page = chunks[0].get("page_number", chunks[0].get("page_num"))
        result["hallucination_flag"] = int(result["source_page"] != top_page)

    return result
