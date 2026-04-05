"""
db_manager.py
SQLite persistence layer for ESGenie. Zero LLM calls.
Handles companies, extractions, and extraction_runs tables.
"""

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB = Path(__file__).parent / "esgenie.db"


def init_db(db_path: str | Path = _DEFAULT_DB) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and apply schema.sql.
    Safe to call repeatedly — CREATE TABLE IF NOT EXISTS semantics applied
    by wrapping each statement.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        Open sqlite3.Connection with row_factory set to Row.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")

    # Wrap each CREATE TABLE / CREATE INDEX to be idempotent
    safe_sql = schema_sql.replace(
        "CREATE TABLE ", "CREATE TABLE IF NOT EXISTS "
    ).replace(
        "CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "
    )

    conn.executescript(safe_sql)
    conn.commit()

    print(f"Database initialized at {db_path}")
    return conn


def get_or_create_company(
    conn: sqlite3.Connection,
    name: str,
    ticker: str | None = None,
    sector: str = "apparel_consumer",
    country: str | None = None,
) -> int:
    """
    Return the company_id for `name`, inserting if it doesn't exist.

    Args:
        conn:   Open database connection.
        name:   Company name (unique key).
        ticker: Stock ticker symbol.
        sector: Sector key from sector_materiality.yaml.
        country: Optional country.

    Returns:
        Integer company id.
    """
    row = conn.execute(
        "SELECT id FROM companies WHERE name = ?", (name,)
    ).fetchone()

    if row:
        return row["id"]

    cursor = conn.execute(
        "INSERT INTO companies (name, ticker, sector, country) VALUES (?, ?, ?, ?)",
        (name, ticker, sector, country),
    )
    conn.commit()
    return cursor.lastrowid


def save_extraction_run(
    conn: sqlite3.Connection,
    run_id: str,
    company_name: str,
    pdf_filename: str,
    total_pages: int,
    pages_after_bucketing: int,
    token_reduction_pct: float,
    total_tokens: int,
    total_cost: float,
    duration_seconds: float,
    status: str,
) -> None:
    """
    Upsert a row into extraction_runs.

    Args:
        conn:                  Open database connection.
        run_id:                Unique run identifier.
        company_name:          Company display name.
        pdf_filename:          Source PDF filename.
        total_pages:           Total pages in PDF.
        pages_after_bucketing: Pages sent to retrieval after BM25 bucketing.
        token_reduction_pct:   % token reduction achieved.
        total_tokens:          Total tokens consumed in run.
        total_cost:            Total API cost in USD.
        duration_seconds:      Wall-clock duration.
        status:                "success" | "error" | "running".
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO extraction_runs
            (run_id, company_name, pdf_filename, total_pages,
             pages_after_bucketing, token_reduction_pct, total_tokens_used,
             total_cost_usd, duration_seconds, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, company_name, pdf_filename, total_pages,
            pages_after_bucketing, token_reduction_pct, total_tokens,
            total_cost, duration_seconds, status,
        ),
    )
    conn.commit()


def save_extraction(
    conn: sqlite3.Connection,
    company_id: int,
    extraction_run_id: str,
    extraction: dict,
) -> None:
    """
    Persist one extraction result dict to the extractions table.

    Args:
        conn:               Open database connection.
        company_id:         FK to companies.id.
        extraction_run_id:  FK to extraction_runs.run_id.
        extraction:         Dict from extraction_agent.extract() or score_company.
                            Expected keys: kpi_id, metric/kpi_canonical_name,
                            kpi_theme/theme, value, value_numeric, unit,
                            year/reporting_year, source_page, source_type,
                            confidence, value_subjective, hallucination_flag.
    """
    # Resolve field aliases across different callers
    kpi_id = extraction.get("kpi_id", "")
    canonical = extraction.get("kpi_canonical_name") or extraction.get("metric", "")
    theme = extraction.get("kpi_theme") or extraction.get("theme", "")
    year = extraction.get("reporting_year") or extraction.get("year")

    value_raw = str(extraction.get("value", "") or "")

    # value_numeric: prefer explicit field, else try float parse of value_raw
    value_numeric = extraction.get("value_numeric")
    if value_numeric is None and value_raw:
        try:
            value_numeric = float(value_raw.replace(",", ""))
        except (ValueError, AttributeError):
            value_numeric = None

    # value_boolean: Yes=1, No=0, else None
    raw_val = str(extraction.get("value", "") or "").strip().lower()
    if raw_val == "yes":
        value_boolean = 1
    elif raw_val == "no":
        value_boolean = 0
    else:
        value_boolean = None

    value_subjective = extraction.get("value_subjective")
    unit = extraction.get("unit", "")
    source_page = extraction.get("source_page")
    source_type = extraction.get("source_type")
    confidence = extraction.get("confidence", "not_found")
    hallucination_flag = int(extraction.get("hallucination_flag", 0))

    conn.execute(
        """
        INSERT INTO extractions
            (company_id, kpi_id, kpi_canonical_name, kpi_theme,
             value_raw, value_numeric, value_boolean, value_subjective,
             unit, reporting_year, source_page, source_type,
             confidence, hallucination_flag, extraction_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id, kpi_id, canonical, theme,
            value_raw, value_numeric, value_boolean, value_subjective,
            unit, year, source_page, source_type,
            confidence, hallucination_flag, extraction_run_id,
        ),
    )
    conn.commit()


def get_extractions_for_company(
    conn: sqlite3.Connection,
    company_name: str,
    year: int | None = None,
) -> list[dict]:
    """
    Return all extractions for a company, optionally filtered by reporting year.

    Args:
        conn:         Open database connection.
        company_name: Company name as stored in companies table.
        year:         Optional reporting year filter.

    Returns:
        List of extraction row dicts.
    """
    if year is not None:
        rows = conn.execute(
            """
            SELECT e.* FROM extractions e
            JOIN companies c ON e.company_id = c.id
            WHERE c.name = ? AND e.reporting_year = ?
            ORDER BY e.created_at DESC
            """,
            (company_name, year),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT e.* FROM extractions e
            JOIN companies c ON e.company_id = c.id
            WHERE c.name = ?
            ORDER BY e.created_at DESC
            """,
            (company_name,),
        ).fetchall()

    return [dict(r) for r in rows]


def get_all_runs(conn: sqlite3.Connection) -> list[dict]:
    """
    Return all extraction_runs ordered by created_at descending.

    Args:
        conn: Open database connection.

    Returns:
        List of run row dicts.
    """
    rows = conn.execute(
        "SELECT * FROM extraction_runs ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_cross_company_kpi(
    conn: sqlite3.Connection,
    kpi_canonical_name: str,
) -> list[dict]:
    """
    Return all extractions for a given KPI across all companies and years.
    Enables peer comparison — a key commercial differentiator vs Sustainalytics.

    Args:
        conn:               Open database connection.
        kpi_canonical_name: Canonical KPI name (e.g. "Scope 1 Emissions").

    Returns:
        List of dicts with company_name, value_numeric, unit, reporting_year,
        confidence, source_page.
    """
    rows = conn.execute(
        """
        SELECT c.name AS company_name, e.value_numeric, e.unit,
               e.reporting_year, e.confidence, e.source_page
        FROM extractions e
        JOIN companies c ON e.company_id = c.id
        WHERE e.kpi_canonical_name = ?
        ORDER BY e.reporting_year DESC, c.name
        """,
        (kpi_canonical_name,),
    ).fetchall()
    return [dict(r) for r in rows]
