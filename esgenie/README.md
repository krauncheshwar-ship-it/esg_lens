# ESGenie

An AI-powered ESG report analyst that extracts, structures, and answers questions from corporate sustainability PDFs.

## Features

- **Mode 1 — Ad-hoc Q&A**: Ask any question about an ESG report and get a grounded, cited answer.
- **Mode 2 — Full ESG Sweep**: Run a predefined set of Environment, Social, and Governance questions against a report and export results to Excel.
- **Hybrid Retrieval**: Combines dense (OpenAI embeddings + FAISS) and sparse (BM25) retrieval via Reciprocal Rank Fusion for high-recall context selection.
- **Thematic Bucketing**: BM25-based page scoring routes content to the correct ESG pillar before retrieval.
- **Grounded Extraction**: GPT-4o extracts facts with source citations (filename + page number) — no hallucinations.
- **Excel Export**: Multi-tab workbook with colour-coded sheets per ESG theme.
- **Token Tracking**: Logs every LLM and embedding call with token counts and estimated USD cost.

## Project Structure

```
esgenie/
├── app.py                    ← Streamlit UI (Mode 1 + Mode 2)
├── config/esg_themes.yaml    ← ESG keyword dictionary
├── ingestion/                ← PyMuPDF PDF parser
├── processing/               ← BM25 thematic bucketer + chunker
├── retrieval/                ← FAISS embedder, BM25 index, RRF merger
├── agents/                   ← Query router, extraction agent, orchestrator
├── export/                   ← openpyxl Excel exporter
├── evaluation/               ← Eval runner + ground truth CSV
├── utils/                    ← Token & cost tracker
├── sample_reports/           ← Drop PDF reports here
└── faiss_cache/              ← Auto-generated vector index cache
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Add ESG PDFs

Drop one or more ESG / sustainability report PDFs into `sample_reports/`.

### 4. Launch the app

```bash
cd esgenie
streamlit run app.py
```

## Usage

### Mode 1 — Ad-hoc Q&A

1. Select a PDF from the sidebar.
2. Type your question (e.g. *"What are the company's Scope 1 emissions?"*).
3. View the grounded answer with source citations.

### Mode 2 — Full ESG Sweep

1. Select a PDF from the sidebar.
2. Click **Run Full ESG Sweep**.
3. Review results in the app or download the Excel workbook.

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
| Query routing | `gpt-4o-mini` |
| Extraction | `gpt-4o` |
| Embeddings | `text-embedding-3-small` |

## License

MIT
