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

CREATE INDEX idx_extractions_company ON extractions(company_id);
CREATE INDEX idx_extractions_kpi ON extractions(kpi_id);
CREATE INDEX idx_extractions_run ON extractions(extraction_run_id);
