# ESGenie — Project Context for Claude Code
# READ THIS ENTIRE FILE BEFORE WRITING ANY CODE

## What This Is
ESGenie is an AI-powered ESG data intelligence system built for Damco Group
evaluation. Target: ESG analysts at ETF funds/hedge funds extracting ESG data
into Excel models. Commercial vision: compete with Sustainalytics/Trucost.

Demonstrates: hybrid RAG, LangGraph orchestration, token optimization (97%
reduction via BM25 bucketing), structured JSON→SQL→Excel output, eval metrics.

## Architecture — Do Not Deviate

PDF → PyMuPDF page corpus
    → BM25 thematic bucketing (YAML keywords) → top 5 pages per theme
    → Paragraph-aware chunking (bucketed pages only) + metadata tags
    → FAISS (semantic) + BM25 (keyword) dual indexes cached to disk
    → Hybrid retrieval: RRF merge → top-5 chunks per KPI
    → LangGraph orchestration (Mode 2):
        validate → load_template → route_by_theme
        → [retrieve_climate | retrieve_social | retrieve_governance] parallel
        → extract_all (ONE LLM call on pre-filtered context)
        → score_company → write_database → export_excel → log_audit
    → GPT-4o-mini extraction: JSON only, null if not found, never guess
    → Python rules scoring: AAA-CCC, no LLM for scoring
    → SQLite: companies / extractions / extraction_runs
    → openpyxl Excel: 3 tabs Environment/Social/Governance
    → 3 logs: system_ops.jsonl / ai_metrics.jsonl / token_costs.jsonl
    → Streamlit UI: 3 tabs Analyst Workbench / Configuration / Watchtower

## Critical Design Rules

1. LLM used EXACTLY ONCE per profile — BM25 eliminates 97%+ pages first
2. Router is deterministic Python — never LLM (cost/latency, zero benefit)
3. Scoring is pure Python rules — never LLM (auditable for analysts)
4. No LangChain — custom Python + LangGraph only
5. Every LLM call logs tokens + cost via token_tracker.py
6. All LLM responses: structured JSON only, never free text
7. Null = valid output when data not found — never hallucinate
8. Hallucination flag: extraction source_page != retrieval top chunk page
9. source_type field ("table" or "prose") in every extraction JSON
10. Conflict resolution: table > prose, lower page > higher page

## Three Video Lines — Verbatim, Do Not Change

LINE 1: "We flag potential hallucinations when source attribution doesn't
match retrieval. In production we'd add a verification step where a second
LLM call validates the extraction against the original page text."

LINE 2: "The router is rules-based, not LLM-based, because routing by theme
is a deterministic operation — using an LLM here would add cost and latency
for zero benefit."

LINE 3: "Our orchestration follows a classic router to specialist to aggregator
pattern. The router is deterministic for cost efficiency. The specialist nodes
run parallel retrieval per theme — no LLM cost here, just vector search. The
aggregator is the single LLM extraction call, on pre-filtered theme-bucketed
context. LLM used exactly once per profile generation, on the smallest
possible context."

## Tech Stack

Streamlit, PyMuPDF, rank_bm25, faiss-cpu, openai (text-embedding-3-small +
gpt-4o-mini), pyyaml, pandas, numpy, python-dotenv, tiktoken, openpyxl,
langgraph, langchain-core, plotly, sqlite3

## File Structure

esgenie/
├── CLAUDE.md
├── app.py                         Streamlit 3-tab UI
├── requirements.txt
├── .env                           OPENAI_API_KEY=... LOG_TOKENS=1
├── config/
│   ├── esg_themes_v3_FINAL.yaml  Two-tier YAML: 342 bucketing + 610 KPI terms
│   ├── kpi_template_v8_FINAL.csv 88 KPIs across 6 themes
│   ├── sector_materiality.yaml   Sector exposure weights
│   └── ground_truth.csv          58 Nike confirmed values for eval
├── ingestion/
│   └── pdf_parser.py             PyMuPDF → {page_num: text}
├── processing/
│   ├── thematic_bucketer.py      BM25 page scoring → top 5 per theme
│   └── chunker.py                Paragraph chunks + metadata
├── retrieval/
│   ├── embedder.py               OpenAI embed + FAISS
│   ├── bm25_index.py             rank_bm25 index
│   └── hybrid_retriever.py       RRF merge → top-5
├── agents/
│   ├── query_router.py           Deterministic theme classifier
│   ├── extraction_agent.py       GPT-4o-mini, JSON only
│   ├── orchestrator.py           Mode 1 coordinator
│   └── profile_graph.py          LangGraph Mode 2
├── scoring/
│   └── esg_scorer.py             Pure Python, AAA-CCC
├── database/
│   ├── schema.sql
│   ├── db_manager.py
│   └── esgenie.db
├── export/
│   └── excel_exporter.py         Template-driven 3-tab Excel
├── logging_system/
│   ├── system_logger.py
│   ├── ai_metrics_logger.py
│   └── logs/
├── evaluation/
│   └── eval_runner.py
├── utils/
│   └── token_tracker.py
├── sample_reports/
└── faiss_cache/

## SQL Schema

CREATE TABLE companies (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, ticker TEXT,
    sector TEXT DEFAULT 'apparel_consumer', country TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE extractions (
    id INTEGER PRIMARY KEY, company_id INTEGER REFERENCES companies(id),
    kpi_id TEXT NOT NULL, kpi_canonical_name TEXT NOT NULL, kpi_theme TEXT,
    value_raw TEXT, value_numeric REAL, value_boolean INTEGER,
    value_subjective TEXT, unit TEXT, reporting_year INTEGER,
    source_page INTEGER, source_type TEXT, confidence TEXT,
    hallucination_flag INTEGER DEFAULT 0, extraction_run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE extraction_runs (
    run_id TEXT PRIMARY KEY, company_name TEXT, pdf_filename TEXT,
    total_pages INTEGER, pages_after_bucketing INTEGER,
    token_reduction_pct REAL, total_tokens_used INTEGER,
    total_cost_usd REAL, duration_seconds REAL, status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

## Scoring Formula

Base Score = (40% x Exposure_Relevance) + (60% x Management_Effectiveness)
Controversy Adjustment = CONT001_deduction + CONT002_deduction (0 to -20)
Final = max(0, min(100, Base - Controversy))

Overall ESG = sum(Theme_Score x Theme_Weight)
Weights apparel_consumer: climate=0.35, governance=0.30, employee=0.25, water=0.10

Rating: 85-100=AAA, 75-84=AA, 60-74=A, 45-59=BBB, 30-44=BB, 15-29=B, 0-14=CCC

Management_Effectiveness:
  completeness = found_kpis / total_expected_kpis_for_theme
  avg_confidence = mean([high=1.0, medium=0.6, low=0.3, not_found=0.0])
  management = (completeness x 0.5) + (avg_confidence x 0.5)

Controversy deductions: NONE=0, MILD=-3, MODERATE=-8, SEVERE=-15

## sector_materiality.yaml Structure

sectors:
  apparel_consumer:
    climate: 0.35, governance: 0.30, employee: 0.25, water: 0.10
  energy:
    climate: 0.40, governance: 0.25, employee: 0.20, water: 0.15
  financial_services:
    governance: 0.45, climate: 0.25, employee: 0.20, water: 0.10
  utilities:
    climate: 0.35, water: 0.30, governance: 0.20, employee: 0.15
  technology:
    governance: 0.40, employee: 0.30, climate: 0.20, water: 0.10

## Log Formats

system_ops.jsonl entry:
{"run_id":"run_20240404_143022","timestamp":"...","event":"pdf_ingestion_complete",
"module":"ingestion/pdf_parser.py","company":"Nike","total_pages":180,
"duration_ms":340,"status":"success","error":null}

ai_metrics.jsonl entry:
{"run_id":"...","query":"Scope 1 emissions?","theme_classified":"climate",
"chunks_retrieved":5,"top_chunk_page":23,"top_chunk_score_rrf":0.94,
"precision_at_5":0.8,"mrr":0.95,"extraction_confidence":"high",
"value_found":true,"hallucination_flag":false,
"latency_retrieval_ms":120,"latency_extraction_ms":890}

## Streamlit Tabs

TAB 1: Analyst Workbench
  Left: PDF upload, company name, year selector, Mode 1 query box, Mode 2 button
  Right: Overall score gauge + AAA badge, E/S/G sub-scores with rating badges,
         KPI grid (climate/water/governance), token reduction green callout,
         confidence-coded table (green=high, yellow=medium, red=low/not-found)
         Columns: KPI Name | Canonical | Value | Unit | Year | Page | Confidence
         Download Excel button (Mode 2), session cost at bottom

TAB 2: Configuration
  Editable agent prompts, editable YAML, st.data_editor KPI template, Save buttons

TAB 3: Watchtower
  System Health: runs count, success rate, avg duration, last 10 runs table
  AI Performance: Precision@5, MRR, avg confidence, BM25 agreement, token reduction %
  Cost Dashboard: total tokens, total cost, cost per profile, model distribution pie

## LangGraph State

class ESGState(TypedDict):
    company_name: str
    reporting_year: int
    pdf_path: str
    page_corpus: dict
    thematic_page_map: dict
    retrieved_chunks: dict
    extractions: list
    scores: dict
    run_id: str
    total_cost: float
    errors: list

## Scope for Improvement (Mention in Video, Do Not Build)

Silver-to-gold SQL with unit normalization, entity resolution, peer comparison
quartile ranking, value conflict disambiguation agent, supervised H/M/L
classification (LDA/NMF + Q-Learning), reranker model, streaming ingestion,
OCR fallback (Tesseract/Azure Doc Intelligence), news agent for controversy,
knowledge graph for subjective responses, PostgreSQL migration, LangSmith,
Prometheus+Grafana, Excel add-in for analysts.

## Coding Standards

- load_dotenv() at top of every file using API keys
- Every function: docstring with Args and Returns
- Type hints on all signatures
- Print logs for: token reduction %, cost per call, pages found, chunks indexed
- All file paths: pathlib.Path
- Catch OpenAI errors gracefully, return structured error dict
- No global mutable state — pass explicitly
- Test each module immediately before moving to next

## Build Sequence

1  config/sector_materiality.yaml
2  database/schema.sql
3  utils/token_tracker.py
4  ingestion/pdf_parser.py           TEST: extract_pages() returns page dict
5  processing/thematic_bucketer.py   TEST: log shows token reduction %
6  processing/chunker.py             TEST: chunks have correct metadata
7  retrieval/embedder.py             TEST: FAISS index built and cached
8  retrieval/bm25_index.py           TEST: BM25 index built and cached
9  retrieval/hybrid_retriever.py     TEST: RRF returns ranked chunks
10 agents/query_router.py            TEST: classify 5 test queries correctly
11 agents/extraction_agent.py        TEST: extract Scope 1 from Nike PDF
12 agents/orchestrator.py            TEST: end-to-end Mode 1 single query
13 scoring/esg_scorer.py             TEST: score from mock extractions
14 database/db_manager.py            TEST: write + read one extraction
15 export/excel_exporter.py          TEST: generate one Excel file
16 logging_system/system_logger.py   TEST: file written correctly
17 logging_system/ai_metrics_logger.py TEST: metrics captured
18 agents/profile_graph.py           TEST: full LangGraph Mode 2 run
19 evaluation/eval_runner.py         TEST: Precision@5 + MRR computed
20 app.py                            TEST: all three tabs load correctly
