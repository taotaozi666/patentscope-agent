"""Common table backend contract and compatibility projection for current UI."""

from dataclasses import asdict, dataclass, field
from typing import Protocol

import pymupdf

from app.schemas.pdf import PdfElement
from app.schemas.table import TableFragment


TABLE_BACKEND_VERSION = "table-backends-v1"


@dataclass(frozen=True)
class TableOptions:
    backend: str = "native"
    models_directory: str = ""
    mode: str = "accurate"
    fallback: bool = True
    ocr_enabled: bool = True
    ocr_language: str = "chi_sim+eng"

    def __post_init__(self):
        if self.backend not in {"native", "auto", "docling"}:
            raise ValueError("PDF_TABLE_BACKEND 必须为 native、auto 或 docling")
        if self.mode not in {"accurate", "fast"}:
            raise ValueError("PDF_TABLE_MODE 必须为 accurate 或 fast")


@dataclass
class TableBackendResult:
    fragments: list[TableFragment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)


class TableBackend(Protocol):
    def parse_page(self, page: pymupdf.Page, *, ocr: bool) -> TableBackendResult: ...


def to_element(fragment: TableFragment, content_hash: str) -> PdfElement:
    """Only anchor text populates the grid; merged values are never copied."""
    from hashlib import sha256

    rows = [[None] * fragment.column_count for _ in range(fragment.row_count)]
    occupancy = {}
    for cell in fragment.cells:
        if cell.state == "covered":
            continue
        rows[cell.row][cell.column] = cell.raw_text
        for row in range(cell.row, cell.row + (cell.row_span or 1)):
            for column in range(cell.column, cell.column + (cell.col_span or 1)):
                if (row, column) in occupancy:
                    raise ValueError("表格单元格覆盖冲突")
                occupancy[(row, column)] = cell
    projected = []
    for row in range(fragment.row_count):
        for column in range(fragment.column_count):
            anchor = occupancy.get((row, column))
            identifier = f"{fragment.id}_r{row}c{column}"
            is_anchor = anchor is not None and (anchor.row, anchor.column) == (row, column)
            projected.append({
                "id": anchor.id if is_anchor else identifier, "row": row, "column": column,
                "raw_text": rows[row][column], "fragment_id": fragment.id,
                "row_span": anchor.row_span if is_anchor else 1,
                "col_span": anchor.col_span if is_anchor else 1,
                "source_page_no": fragment.span.page_no,
                "span": asdict(anchor.spans[0]) if anchor and anchor.spans else None,
                "state": anchor.state if is_anchor else "covered" if anchor else "unknown",
                "covered_by_cell_id": anchor.id if anchor and not is_anchor else None,
            })
    complex_table = any((cell.row_span or 1) > 1 or (cell.col_span or 1) > 1 for cell in fragment.cells)
    origins = {cell.text_origin for cell in fragment.cells if cell.raw_text}
    origin = next(iter(origins)) if len(origins) == 1 else "mixed" if origins else "unknown"
    text = "\n".join(" | ".join(value or "" for value in row) for row in rows)
    return PdfElement(fragment.id, "table", text, fragment.span, section_path=fragment.section_path,
        payload={"rows": rows, "cells": projected, "row_count": fragment.row_count,
                 "column_count": fragment.column_count, "header_rows": fragment.header_rows,
                 "structure": "structured_grid", "table_backend": fragment.backend,
                 "requires_review": complex_table or fragment.structure_status != "complete"
                                    or fragment.provenance_status != "complete" or fragment.backend == "pymupdf-text",
                 "doc_type_kwd": "table", "layout_type": "table", "text_origin": origin,
                 "image_id": sha256(f"{content_hash}:{fragment.id}".encode()).hexdigest(),
                 "properties": {"row_count": fragment.row_count, "column_count": fragment.column_count,
                                "structure": "structured_grid", "backend": fragment.backend}})
