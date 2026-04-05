# ESGenie

An AI-powered ESG data intelligence system that extracts, scores, and exports structured ESG metrics from corporate sustainability PDFs. Built for ESG analysts at ETF funds and hedge funds who need auditable, Excel-ready ESG data.

## Features

- **Mode 1 — Ad-hoc Q&A**: Ask any question about an ESG report and get a grounded, cited answer with source page references.
- **Mode 2 — Full ESG Profile**: Run a full 88-KPI sweep across 6 ESG themes, generate company scores (AAA–CCC), and export to Excel.
- **Hybrid Retrieval**: Combines dense (OpenAI embeddings + FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion (RRF) for high-recall context selection.
- **BM25 Thematic Bucketing**: Two-tier YAML keyword dictionary (342 bucketing + 610 KPI terms) routes pages to the correct ESG theme before retrieval, achieving 97%+ token reduction.
- **LangGraph Orchestration**: Full Mode 2 pipeline orchestrated as a LangGraph state graph with parallel retrieval nodes per theme and SQLite checkpoint auditing.
- **Rules-Based Scoring**: Pure Python scoring engine (AAA–CCC ratings) using sector materiality weights, management effectiveness, and controversy deductions — no LLM involved.
- **Deterministic Router**: Theme classification via keyword matching — no LLM cost or latency for routing.
- **Grounded Extraction**: GPT-4o-mini extracts structured JSON with source citations (page number + source type). Returns null when data is not found — never hallucinates.
- **Hallucination Detection**: Flags extractions where source page attribution doesn't match retrieval top chunk page.
- **SQLite Persistence**: Companies, extractions, and extraction runs stored in SQLite with full run metadata.
- **Excel Export**: Template-driven multi-tab workbook (Environment / Social / Governance) with colour-coded confidence indicators.
- **Token Tracking**: Logs every LLM and embedding call with token counts and estimated USD cost.
- **Heuristic Evaluation**: Ground-truth comparison with Precision@5 and MRR metrics.
- **3-Tab Streamlit UI**: Analyst Workbench, Configuration, and Watchtower (ops + AI metrics + cost dashboard).

## Architecture

```
PDF → PyMuPDF page corpus
    → BM25 thematic bucketing (YAML keywords) → top 5 pages per theme
    → Paragraph-aware chunking (bucketed pages only) + metadata tags
    → FAISS (semantic) + BM25 (keyword) dual indexes cached to disk
    → Hybrid retrieval: RRF merge → top-5 chunks per KPI
    → LangGraph orchestration (Mode 2):
        validate → load_template → run_ingestion → run_bucketing → run_chunking
        → build_indexes → [retrieve_climate | retrieve_social | retrieve_governance
                           | retrieve_water | retrieve_biodiversity] (parallel)
        → extract_all (ONE LLM call on pre-filtered context)
        → score_esg → persist_database → export_results → log_completion
    → GPT-4o-mini extraction: JSON only, null if not found, never guess
    → Python rules scoring: AAA–CCC, no LLM for scoring
    → SQLite: companies / extractions / extraction_runs
    → openpyxl Excel: 3 tabs (Environment / Social / Governance)
    → 2 structured logs: system_ops.jsonl / ai_metrics.jsonl
    → Streamlit UI: 3 tabs (Analyst Workbench / Configuration / Watchtower)
```

## Project Structure

```
esgenie/
├── app.py                          ← Streamlit 3-tab UI
├── requirements.txt
├── config/
│   ├── esg_themes_v3_FINAL.yaml    ← Two-tier YAML: 342 bucketing + 610 KPI terms
│   ├── kpi_template_v8_FINAL.csv   ← 88 KPIs across 6 themes
│   ├── sector_materiality.yaml     ← Sector exposure weights (5 sectors)
│   └── ground_truth.csv            ← Nike confirmed values for evaluation
├── ingestion/
│   └── pdf_parser.py               ← PyMuPDF → {page_num: text}
├── processing/
│   ├── thematic_bucketer.py        ← BM25 page scoring → top 5 per theme
│   └── chunker.py                  ← Paragraph-aware chunking + metadata
├── retrieval/
│   ├── embedder.py                 ← OpenAI embed + FAISS index (cached)
│   ├── bm25_index.py               ← rank_bm25 index (cached)
│   └── hybrid_retriever.py         ← RRF merge → top-5 chunks
├── agents/
│   ├── query_router.py             ← Deterministic theme classifier (no LLM)
│   ├── extraction_agent.py         ← GPT-4o-mini, structured JSON only
│   ├── orchestrator.py             ← Mode 1 single-query coordinator
│   └── profile_graph.py            ← LangGraph Mode 2 state graph
├── scoring/
│   └── esg_scorer.py               ← Pure Python AAA–CCC scoring
├── database/
│   ├── schema.sql                  ← DDL for companies/extractions/runs
│   └── db_manager.py               ← SQLite persistence layer
├── export/
│   └── excel_exporter.py           ← Template-driven 3-tab Excel export
├── logging_system/
│   ├── system_logger.py            ← Structured system event logging
│   ├── ai_metrics_logger.py        ← Retrieval + extraction metrics
│   └── logs/                       ← system_ops.jsonl, ai_metrics.jsonl
├── evaluation/
│   └── eval_runner.py              ← Precision@5, MRR, ground-truth eval
├── utils/
│   └── token_tracker.py            ← Token counts + cost tracking
├── sample_reports/                 ← Drop PDF reports here
└── faiss_cache/                    ← Auto-generated vector + BM25 index cache
```

## Quick Start

### 1. Install dependencies

```bash
cd esgenie
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the `esgenie/` directory:

```
OPENAI_API_KEY=sk-your-key-here
LOG_TOKENS=1
```

### 3. Add ESG PDFs

Drop one or more ESG / sustainability report PDFs into `sample_reports/`.

### 4. Launch the app

```bash
streamlit run app.py
```

## Usage

### Mode 1 — Ad-hoc Q&A

1. Upload a PDF via the sidebar.
2. Company name and reporting year are auto-detected from the first pages.
3. Type your question (e.g. *"What are the company's Scope 1 emissions?"*).
4. View the grounded answer with source page citations.

### Mode 2 — Full ESG Profile

1. Upload a PDF via the sidebar.
2. Click **Run Full ESG Sweep**.
3. View the overall score gauge (AAA–CCC), per-theme sub-scores, and a confidence-coded KPI grid.
4. Download the Excel workbook.

### Streamlit Tabs

| Tab | Purpose |
|---|---|
| **Analyst Workbench** | PDF upload, Mode 1 Q&A, Mode 2 sweep, score gauges, KPI grid, Excel download |
| **Configuration** | Edit agent prompts, YAML themes, KPI template, sector weight sliders |
| **Watchtower** | System health (runs, success rate), AI performance (Precision@5, MRR), cost dashboard |

## Scoring System

```
Base Score = (40% × Exposure_Relevance) + (60% × Management_Effectiveness)
Controversy Adjustment = CONT001 + CONT002 deductions (0 to -20)
Final = max(0, min(100, Base - Controversy))

Overall ESG = Σ(Theme_Score × Theme_Weight)
Rating: 85-100=AAA, 75-84=AA, 60-74=A, 45-59=BBB, 30-44=BB, 15-29=B, 0-14=CCC
```

Sector materiality weights (e.g. apparel_consumer): climate=0.35, governance=0.30, employee=0.25, water=0.10.

## Running the Evaluator

```bash
python evaluation/eval_runner.py sample_reports/your_report.pdf
```

Edit `evaluation/ground_truth.csv` to add known Q&A pairs for your specific report.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `LOG_TOKENS` | `0` | Set to `1` to write token logs to `token_log.jsonl`. |

## Models Used

| Component | Model |
|---|---|
| Theme routing | Deterministic keyword matching (no LLM) |
| Extraction | `gpt-4o-mini` |
| Embeddings | `text-embedding-3-small` |
| Scoring | Pure Python rules (no LLM) |

## Tech Stack

Streamlit, PyMuPDF, rank_bm25, faiss-cpu, OpenAI (text-embedding-3-small + gpt-4o-mini), LangGraph, SQLite, openpyxl, pandas, numpy, pyyaml, python-dotenv

## License

MIT
