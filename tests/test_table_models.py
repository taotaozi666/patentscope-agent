"""统一表格的状态、跨度、来源持久化和历史数据兼容性。"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.pdf.chunking import build_chunks
from app.pdf.parser import PARSER_VERSION, parse_pdf
from app.pdf.serialization import restore_pdf
from app.schemas import LogicalTable, TableCell, TableFragment
from app.schemas.pdf import SourceSpan
from app.storage.pdf_repository import PdfRepository
from tests.pdf_regression.evaluation import load_expectations
from tests.pdf_regression.fixtures import make_pdf


def annotated_tables(parsed, expected):
    """Build target models from manual gold, not a simulated parser response."""
    fragments = []
    for raw in expected["fragments"]:
        fragment_id = raw["id"]
        cells = []
        for value in raw["cells"]:
            cell_id = f"{fragment_id}:{value['id']}"
            cells.append(TableCell(
                id=cell_id, fragment_id=fragment_id,
                row=value["row"], column=value["column"], raw_text=value["raw_text"],
                state="text" if value["raw_text"] else "blank",
                row_span=value["row_span"], col_span=value["col_span"],
                spans=[SourceSpan(raw["page_no"], "", tuple(value["bbox"]), cell_id, "cell")],
                text_origin="native",
            ))
        for fact in expected["numeric_facts"]:
            if fact["fragment_id"] != fragment_id:
                continue
            cell = next(cell for cell in cells if cell.id == f"{fragment_id}:{fact['cell_id']}")
            cell.row_header_cell_ids = [f"{fragment_id}:{value}" for value in fact["row_header_cell_ids"]]
            cell.column_header_cell_ids = [f"{fragment_id}:{value}" for value in fact["column_header_cell_ids"]]
        fragments.append(TableFragment(
            fragment_id, SourceSpan(raw["page_no"], "", tuple(raw["bbox"]), fragment_id),
            raw["row_count"], raw["column_count"], cells, header_rows=raw["header_rows"],
            backend="manual-annotation", structure_status="complete", provenance_status="complete",
        ))
    parsed.table_fragments = fragments
    parsed.logical_tables = [LogicalTable(f"logical-{index}", ids)
                             for index, ids in enumerate(expected["logical_tables"])]


class TableModelTests(unittest.TestCase):
    def assert_json_equivalent(self, actual, expected):
        # 引擎原始 payload 保持不透明；JSON 将其中 tuple 变为 list 是正常行为。
        self.assertEqual(json.loads(json.dumps(asdict(actual), ensure_ascii=False)),
                         json.loads(json.dumps(asdict(expected), ensure_ascii=False)))

    def test_zero_blank_unreadable_covered_and_unknown_remain_distinct(self):
        zero = TableCell("zero", "t1", 0, 0, "0", state="text", row_span=1, col_span=1)
        blank = TableCell("blank", "t1", 0, 1, "", state="blank", row_span=1, col_span=1)
        unreadable = TableCell("unreadable", "t1", 0, 2, None, state="unreadable")
        covered = TableCell("covered", "t1", 0, 3, None, state="covered", covered_by_cell_id="zero")
        unknown = TableCell("unknown", "t1", 0, 4, None)
        records = json.loads(json.dumps([asdict(cell) for cell in (zero, blank, unreadable, covered, unknown)]))
        self.assertEqual([record["state"] for record in records],
                         ["text", "blank", "unreadable", "covered", "unknown"])
        self.assertEqual(records[0]["raw_text"], "0")
        self.assertEqual(records[1]["raw_text"], "")
        self.assertIsNone(records[2]["raw_text"])
        self.assertIsNone(records[4]["row_span"])

    def test_invalid_spans_indexes_and_covered_cells_are_rejected(self):
        cell = TableCell("c1", "t1", 0, 0, "35.2", state="text")
        for changes in ({"row_span": 0}, {"col_span": -1}, {"row": -1}, {"column": True},
                        {"raw_text": 0}, {"state": "blank"}, {"text_origin": "guessed"},
                        {"normalized_value": 35.2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cell, **changes)
        for changes in ({}, {"covered_by_cell_id": "c2"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TableCell("c2", "t1", 0, 1, None, state="covered", **changes)
        region = SourceSpan(1, "", (.1, .2, .9, .4), "t1")
        for changes in ({"cells": [replace(cell, row_span=3)]},
                        {"cells": [cell, cell]}, {"header_rows": [2]},
                        {"cells": [replace(cell, fragment_id="t2")]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TableFragment("t1", region, 2, 2, **changes)
        with self.assertRaises(ValueError):
            LogicalTable("logical", ["t1", "t1"])

    def test_covered_cell_references_are_validated_and_preserved(self):
        anchor = TableCell("anchor", "t1", 0, 0, "Sample", state="text", row_span=2, col_span=1)
        covered = TableCell("covered", "t1", 1, 0, None, state="covered", covered_by_cell_id="anchor")
        region = SourceSpan(1, "", (.1, .2, .4, .3), "t1")
        fragment = TableFragment("t1", region, 2, 1, [anchor, covered])
        self.assertEqual(fragment.cells[1].covered_by_cell_id, fragment.cells[0].id)
        with self.assertRaisesRegex(ValueError, "锚点"):
            TableFragment("t1", region, 2, 1, [covered])
        with self.assertRaisesRegex(ValueError, "锚点"):
            TableFragment("t1", region, 2, 1, [replace(anchor, raw_text=None, state="covered",
                                                     covered_by_cell_id="covered"), covered])

    def test_multilevel_header_and_merged_row_links_survive_json_roundtrip(self):
        gold = {case["id"]: case for case in load_expectations()["cases"]}
        for case_id in ("multi_header", "merged_cells"):
            with self.subTest(case=case_id):
                parsed = parse_pdf(f"{case_id}.pdf", make_pdf(case_id), ocr_images=False)
                annotated_tables(parsed, gold[case_id])
                restored = restore_pdf(json.loads(json.dumps(asdict(parsed), ensure_ascii=False)))
                self.assert_json_equivalent(restored, parsed)
                cells = restored.table_fragments[0].cells
                if case_id == "multi_header":
                    cell = next(cell for cell in cells if cell.raw_text == "120")
                    self.assertEqual(cell.column_header_cell_ids, ["t1:r0c1", "t1:r1c2"])
                    self.assertEqual(next(cell for cell in cells if cell.id == "t1:r0c1").col_span, 2)
                else:
                    cell = next(cell for cell in cells if cell.raw_text == "40.8")
                    self.assertEqual(cell.row_header_cell_ids, ["t1:r1c0"])
                    self.assertEqual(next(cell for cell in cells if cell.id == "t1:r1c0").row_span, 2)
                self.assertIsInstance(cells[0].spans[0], SourceSpan)
                self.assertIsInstance(cells[0].spans[0].bbox, tuple)

    def test_cross_page_fragments_and_logical_table_persist_in_repository(self):
        expected = next(case for case in load_expectations()["cases"] if case["id"] == "continued_table")
        content = make_pdf("continued_table")
        parsed = parse_pdf("continued_table.pdf", content, ocr_images=False)
        annotated_tables(parsed, expected)
        logical = parsed.logical_tables[0]
        logical.header_paths = {"t1p1:r0c1": ["t1p1:r0c1"]}
        logical.data_rows = [["t1p1:r1c0", "t1p1:r1c1"], ["t1p2:r1c0", "t1p2:r1c1"]]
        logical.repeated_header_cell_ids = ["t1p2:r0c0", "t1p2:r0c1", "t1p2:r0c2"]
        logical.unit_text = "MPa"
        logical.unit_spans = [parsed.table_fragments[0].cells[1].spans[0]]
        logical.footnotes = ["Synthetic footnote"]
        logical.footnote_spans = [SourceSpan(2, "", (.1, .4, .8, .45), "note")]
        chunks = build_chunks(parsed)
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            repository = PdfRepository(directory)
            repository.save(parsed, content, chunks)
            saved = repository.get_run(chunks[0].metadata["pdf_parse_id"])
            self.assertIsNotNone(saved)
            assert saved is not None
            restored = restore_pdf(saved)
        self.assert_json_equivalent(restored, parsed)
        self.assertEqual(restored.logical_tables[0].fragment_ids, ["t1p1", "t1p2"])
        self.assertEqual([fragment.span.page_no for fragment in restored.table_fragments], [1, 2])
        self.assertEqual(restored.table_fragments[1].cells[4].spans[0].page_no, 2)

    def test_parser_recovers_geometry_spans_and_inferred_header_relations(self):
        parsed = parse_pdf("merged_cells.pdf", make_pdf("merged_cells"), ocr_images=False)
        self.assertEqual(parsed.table_schema_version, 1)
        self.assertEqual(len(parsed.table_fragments), 1)
        fragment = parsed.table_fragments[0]
        self.assertEqual(fragment.structure_status, "complete")
        self.assertEqual(fragment.header_rows, [0])
        self.assertEqual(len(parsed.logical_tables), 1)
        self.assertTrue(all(cell.row_span is not None and cell.col_span is not None for cell in fragment.cells))
        merged = next(cell for cell in fragment.cells if cell.raw_text == "Ex1")
        self.assertEqual(merged.row_span, 2)
        self.assert_json_equivalent(restore_pdf(json.loads(json.dumps(asdict(parsed)))), parsed)

    def test_legacy_assets_get_compatibility_view_without_mutation(self):
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        data = asdict(parsed)
        for name in ("table_schema_version", "table_fragments", "logical_tables", "patent", "page_inspections"):
            data.pop(name)
        data["parser_version"] = "native-v9:old-asset"
        original = deepcopy(data)
        restored = restore_pdf(data)
        self.assertEqual(data, original)
        self.assertEqual(restored.table_schema_version, 1)
        self.assertEqual(restored.parser_version, "native-v9:old-asset")
        self.assertEqual(restored.elements, parsed.elements)
        self.assertEqual(len(restored.table_fragments), 1)
        zero = next(cell for cell in restored.table_fragments[0].cells if cell.raw_text == "0")
        self.assertEqual(zero.state, "text")
        self.assertFalse(zero.column_header_cell_ids)
        self.assertFalse(restored.logical_tables)
        self.assert_json_equivalent(restore_pdf(json.loads(json.dumps(asdict(restored)))), restored)

    def test_explicit_empty_fragments_are_not_recreated_and_future_schema_is_rejected(self):
        data = asdict(parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False))
        data["table_fragments"] = []
        self.assertFalse(restore_pdf(data).table_fragments)
        for version in (2, "1", True):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "表格结构版本"):
                restore_pdf({**data, "table_schema_version": version})
        self.assertIn("table-schema-v1", PARSER_VERSION)


if __name__ == "__main__":
    unittest.main()
