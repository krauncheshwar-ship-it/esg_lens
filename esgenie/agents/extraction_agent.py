"""
extraction_agent.py
Grounded ESG data extractor.  Given retrieved context chunks and a query,
returns a structured JSON response with cited evidence.
"""

import json
import os
from openai import OpenAI
from utils.token_tracker import track_llm_call

_EXTRACT_MODEL = "gpt-4o"

_SYSTEM_PROMPT = """You are a precise ESG analyst. Your job is to answer questions \
about ESG reports by extracting information ONLY from the provided context.

Rules:
1. Base every claim strictly on the provided context. Do NOT hallucinate.
2. For each extracted fact, include a citation with the source file and page number.
3. If the answer cannot be found in the context, state that explicitly.
4. Return a JSON object with the following schema:
   {
     "answer": "<concise answer to the question>",
     "facts": [
       {
         "statement": "<extracted fact>",
         "source": "<filename>",
         "page": <page_number>,
         "quote": "<verbatim excerpt supporting the fact>"
       }
     ],
     "confidence": "high" | "medium" | "low",
     "limitations": "<any caveats or missing information>"
   }
"""


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page_num", "?")
        lines.append(f"[{i}] Source: {source}, Page: {page}")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def extract(query: str, chunks: list[dict], model: str = _EXTRACT_MODEL) -> dict:
    """
    Run the extraction agent on retrieved chunks.

    Args:
        query:  The user's question.
        chunks: Retrieved context chunks (output of HybridRetriever.retrieve).
        model:  OpenAI model to use (default: gpt-4o).

    Returns:
        Parsed JSON dict with keys: answer, facts, confidence, limitations.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context_block = _build_context_block(chunks)
    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )

    track_llm_call(
        model=model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        call_type="extraction_agent",
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)
    return result


def batch_extract(
    queries: list[str],
    retriever,
    top_k: int = 10,
    model: str = _EXTRACT_MODEL,
) -> list[dict]:
    """
    Run extract() for each query in a list (Mode 2 — full ESG sweep).

    Args:
        queries:   List of predefined ESG questions.
        retriever: HybridRetriever instance.
        top_k:     Chunks per query.
        model:     OpenAI model to use.

    Returns:
        List of result dicts, each with an added "query" key.
    """
    results = []
    for query in queries:
        chunks = retriever.retrieve(query, top_k=top_k)
        result = extract(query, chunks, model=model)
        result["query"] = query
        results.append(result)
    return results
