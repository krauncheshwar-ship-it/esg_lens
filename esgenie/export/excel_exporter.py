"""
excel_exporter.py
Exports ESG sweep results to a multi-tab Excel workbook using openpyxl.
One tab per ESG theme (Environment, Social, Governance) + a Summary tab.
"""

import io
from datetime import datetime

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# Theme -> (tab name, hex fill colour)
_THEME_STYLE = {
    "environment": ("Environment", "C6EFCE"),   # green
    "social":      ("Social",      "DDEBF7"),   # blue
    "governance":  ("Governance",  "FCE4D6"),   # orange
    "general":     ("General",     "F2F2F2"),   # grey
}

_HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
_HEADER_FILL = PatternFill("solid", fgColor="2F5496")
_WRAP = Alignment(wrap_text=True, vertical="top")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _write_header(ws, columns: list[str]) -> None:
    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _WRAP
        cell.border = _BORDER
    ws.row_dimensions[1].height = 20


def _write_row(ws, row_idx: int, values: list, fill_colour: str) -> None:
    fill = PatternFill("solid", fgColor=fill_colour)
    for col_idx, value in enumerate(values, start=1):
        cell = ws.cell(row=row_idx, column=col_idx, value=str(value) if value is not None else "")
        cell.alignment = _WRAP
        cell.fill = fill
        cell.border = _BORDER


def _auto_column_width(ws) -> None:
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 4, 60)


def _classify_result(result: dict) -> str:
    """Best-effort theme classification for a result dict."""
    query = result.get("query", "").lower()
    theme_keywords = {
        "environment": ["carbon", "emission", "energy", "water", "waste", "scope", "climate", "renewable"],
        "social": ["employee", "diversity", "safety", "injury", "training", "human rights", "workforce", "gender"],
        "governance": ["board", "director", "compensation", "audit", "corruption", "whistleblower", "cyber"],
    }
    for theme, keywords in theme_keywords.items():
        if any(kw in query for kw in keywords):
            return theme
    return "general"


def export_to_excel(
    sweep_results: list[dict],
    report_name: str = "ESG Report",
) -> bytes:
    """
    Build an Excel workbook from sweep results and return as bytes.

    Args:
        sweep_results: List of extraction result dicts from orchestrator.sweep().
        report_name:   Name of the source report (used in Summary tab).

    Returns:
        Raw bytes of the .xlsx file (ready for st.download_button).
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    # Group results by theme
    by_theme: dict[str, list[dict]] = {}
    for result in sweep_results:
        theme = _classify_result(result)
        by_theme.setdefault(theme, []).append(result)

    # Theme tabs
    for theme in ["environment", "social", "governance", "general"]:
        results = by_theme.get(theme, [])
        tab_name, fill_colour = _THEME_STYLE[theme]
        ws = wb.create_sheet(title=tab_name)

        columns = ["Question", "Answer", "Confidence", "Limitations", "Sources"]
        _write_header(ws, columns)

        for row_idx, result in enumerate(results, start=2):
            sources = "; ".join(
                f"{f.get('source','?')} p.{f.get('page','?')}"
                for f in result.get("facts", [])
            )
            values = [
                result.get("query", ""),
                result.get("answer", ""),
                result.get("confidence", ""),
                result.get("limitations", ""),
                sources,
            ]
            _write_row(ws, row_idx, values, fill_colour)

        _auto_column_width(ws)
        ws.freeze_panes = "A2"

    # Summary tab
    ws_summary = wb.create_sheet(title="Summary", index=0)
    ws_summary["A1"] = "ESGenie — Extraction Summary"
    ws_summary["A1"].font = Font(bold=True, size=14)
    ws_summary["A3"] = "Report:"
    ws_summary["B3"] = report_name
    ws_summary["A4"] = "Generated:"
    ws_summary["B4"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ws_summary["A6"] = "Theme"
    ws_summary["B6"] = "Questions Answered"
    ws_summary["A6"].font = Font(bold=True)
    ws_summary["B6"].font = Font(bold=True)

    for i, theme in enumerate(["environment", "social", "governance", "general"], start=7):
        tab_name, _ = _THEME_STYLE[theme]
        ws_summary.cell(row=i, column=1, value=tab_name)
        ws_summary.cell(row=i, column=2, value=len(by_theme.get(theme, [])))

    ws_summary.column_dimensions["A"].width = 20
    ws_summary.column_dimensions["B"].width = 25

    # Serialize to bytes
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
