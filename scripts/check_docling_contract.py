"""Validate conversion against real Docling Core classes, without neural inference."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf
from docling_core.types.doc import BoundingBox, DoclingDocument, ProvenanceItem, Size, TableCell, TableData

from app.pdf.tables.docling import adapt_document
from tests.pdf_regression.evaluation import load_expectations
from tests.pdf_regression.fixtures import make_pdf


def main():
    for expected in load_expectations()["cases"]:
        if expected["id"] not in {"multi_header", "merged_cells"}:
            continue
        raw = expected["fragments"][0]
        def bbox(values):
            return BoundingBox(l=values[0] * 600, t=values[1] * 800,
                               r=values[2] * 600, b=values[3] * 800, coord_origin="TOPLEFT")
        document = DoclingDocument(name=expected["id"])
        document.add_page(page_no=1, size=Size(width=600, height=800))
        data = TableData(num_rows=raw["row_count"], num_cols=raw["column_count"], table_cells=[
            TableCell(text=cell["raw_text"], bbox=bbox(cell["bbox"]), row_span=cell["row_span"], col_span=cell["col_span"],
                start_row_offset_idx=cell["row"], end_row_offset_idx=cell["row"] + cell["row_span"],
                start_col_offset_idx=cell["column"], end_col_offset_idx=cell["column"] + cell["col_span"],
                column_header=cell["row"] in raw["header_rows"]) for cell in raw["cells"]])
        document.add_table(data=data, prov=ProvenanceItem(page_no=1, bbox=bbox(raw["bbox"]), charspan=(0, 0)))
        with pymupdf.open(stream=make_pdf(expected["id"]), filetype="pdf") as pdf:
            fragments = adapt_document(json.loads(document.model_dump_json()), pdf[0],
                                       backend_version="real-core-contract", text_origin="native")
        assert len(fragments[0].cells) == len(raw["cells"])
        for cell, gold in zip(fragments[0].cells, raw["cells"]):
            assert (cell.raw_text, cell.row_span, cell.col_span) == (gold["raw_text"], gold["row_span"], gold["col_span"])
        print(f"PASS: {expected['id']} -> actual DoclingDocument {document.version} -> unified table")
    print("Contract validation only; no model inference or OCR accuracy claim.")


if __name__ == "__main__":
    main()
