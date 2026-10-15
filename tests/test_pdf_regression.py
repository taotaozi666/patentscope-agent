"""Baseline checks plus falsification of the table evaluation harness."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import pymupdf
from app.pdf.ocr import OcrResult

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from tests.pdf_regression.evaluation import evaluate, load_expectations
from tests.pdf_regression.fixtures import CASES, make_pdf


class PdfRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = {case["id"]: case for case in load_expectations()["cases"]}

    def test_specimens_have_native_or_raster_content_on_expected_pages(self):
        self.assertEqual(set(CASES), set(self.gold))
        for case_id in CASES:
            with self.subTest(case=case_id), pymupdf.open(stream=make_pdf(case_id), filetype="pdf") as pdf:
                self.assertEqual(pdf.page_count, len(self.gold[case_id]["page_sources"]))
                if case_id == "scanned_table":
                    self.assertFalse(pdf[0].get_text().strip())
                    self.assertTrue(pdf[0].get_image_info())
                elif case_id == "hybrid_page":
                    self.assertIn("Native paragraph", pdf[0].get_text())
                    self.assertNotIn("35.2", pdf[0].get_text())
                    self.assertTrue(pdf[0].get_image_info())
                else:
                    self.assertIn("35.2", pdf[0].get_text())

    def test_native_control_cells_and_numeric_chunk_provenance(self):
        expected = self.gold["native_grid"]
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        report = evaluate(parsed, expected)
        self.assertTrue(report["measured_checks_passed"], report)
        self.assertFalse(report["target_complete"])
        table = next(item for item in parsed.elements if item.kind == "table")
        chunks = build_chunks(parsed, size=180)
        for raw_text in ("35.2", "40.8", "0"):
            cell = next(cell for cell in table.payload["cells"] if cell["raw_text"] == raw_text)
            values = [chunk.page_content[mapping["start"]:mapping["end"]]
                      for chunk in chunks for mapping in json.loads(chunk.metadata["pdf_serialized_map"])
                      if mapping.get("cell_id") == cell["id"]]
            self.assertEqual(values, [raw_text])

    def test_evaluator_rejects_swapped_values_and_invalid_cell_locations(self):
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        table = next(item for item in parsed.elements if item.kind == "table")
        cells = table.payload["cells"]
        first = next(cell for cell in cells if cell["raw_text"] == "35.2")
        second = next(cell for cell in cells if cell["raw_text"] == "40.8")
        first["raw_text"], second["raw_text"] = second["raw_text"], first["raw_text"]
        report = evaluate(parsed, self.gold["native_grid"])
        self.assertEqual(report["metrics"]["cell_text"]["matched"], 7)
        self.assertFalse(report["measured_checks_passed"])
        first["span"]["page_no"] = 2
        report = evaluate(parsed, self.gold["native_grid"])
        self.assertEqual(report["metrics"]["cell_location"]["matched"], 8)

    def test_mocked_hybrid_ocr_preserves_native_paragraph_once(self):
        blocks = [((68, 205, 210, 235), "Ex1"), ((228, 205, 370, 235), "35.2")]
        with patch("app.pdf.parser.ocr_region", return_value=OcrResult(blocks)):
            parsed = parse_pdf("hybrid_page.pdf", make_pdf("hybrid_page"))
        paragraph = self.gold["hybrid_page"]["paragraphs"][0]["text"]
        self.assertEqual(sum(item.text.count(paragraph) for item in parsed.elements), 1)
        self.assertEqual(sum(item.text.count("35.2") for item in parsed.elements), 1)
        report = evaluate(parsed, self.gold["hybrid_page"])
        self.assertFalse(report["measured_checks_passed"])
        self.assertEqual(report["metrics"]["table_regions"]["matched"], 0)

    def test_scan_without_ocr_is_an_explicit_gap(self):
        with self.assertRaisesRegex(ValueError, "OCR"):
            parse_pdf("scanned_table.pdf", make_pdf("scanned_table"), ocr_images=False)
        report = evaluate(None, self.gold["scanned_table"], error="OCR disabled")
        self.assertEqual(report["metrics"]["cell_text"], {"matched": 0, "expected": 9})
        self.assertFalse(report["target_complete"])

    def test_continuation_and_non_continuation_have_distinct_gold_groups(self):
        self.assertEqual(self.gold["continued_table"]["logical_tables"], [["t1p1", "t1p2"]])
        self.assertEqual(self.gold["distinct_tables"]["logical_tables"], [["t1p1"], ["t1p2"]])
        for case_id in ("continued_table", "distinct_tables"):
            parsed = parse_pdf(f"{case_id}.pdf", make_pdf(case_id), ocr_images=False)
            report = evaluate(parsed, self.gold[case_id])
            self.assertTrue(report["measured_checks_passed"], report)
            self.assertEqual(report["metrics"]["logical_table_grouping"]["matched"],
                             len(self.gold[case_id]["logical_tables"]))

    def test_gold_merged_cells_have_full_geometry_and_no_placeholder_cells(self):
        for case_id in ("merged_cells", "multi_header"):
            fragment = self.gold[case_id]["fragments"][0]
            occupancy = set()
            for cell in fragment["cells"]:
                for row in range(cell["row"], cell["row"] + cell["row_span"]):
                    for col in range(cell["column"], cell["column"] + cell["col_span"]):
                        self.assertNotIn((row, col), occupancy)
                        occupancy.add((row, col))
            self.assertEqual(len(occupancy), fragment["row_count"] * fragment["column_count"])

    def test_numeric_gold_links_reach_headers_across_merged_cells(self):
        for expected in self.gold.values():
            fragments = {fragment["id"]: fragment for fragment in expected["fragments"]}
            for fact in expected["numeric_facts"]:
                fragment = fragments[fact["fragment_id"]]
                cells = {cell["id"]: cell for cell in fragment["cells"]}
                value_cell = cells[fact["cell_id"]]
                self.assertEqual(value_cell["raw_text"], fact["raw_value"])
                self.assertTrue(fact["row_header_cell_ids"])
                self.assertTrue(fact["column_header_cell_ids"])
                for identifier in fact["row_header_cell_ids"]:
                    header = cells[identifier]
                    self.assertEqual(header["column"], 0)
                    self.assertLessEqual(header["row"], value_cell["row"])
                    self.assertLess(value_cell["row"], header["row"] + header["row_span"])
                for identifier in fact["column_header_cell_ids"]:
                    header = cells[identifier]
                    self.assertIn(header["row"], fragment["header_rows"])
                    self.assertLessEqual(header["column"], value_cell["column"])
                    self.assertLess(value_cell["column"], header["column"] + header["col_span"])
        merged_fact = next(fact for fact in self.gold["merged_cells"]["numeric_facts"]
                           if fact["raw_value"] == "40.8")
        self.assertEqual(merged_fact["row_header_cell_ids"], ["r1c0"])
        nested_fact = next(fact for fact in self.gold["multi_header"]["numeric_facts"]
                           if fact["raw_value"] == "120")
        self.assertEqual(nested_fact["column_header_cell_ids"], ["r0c1", "r1c2"])

    def test_evaluator_does_not_accept_duplicated_native_paragraph(self):
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        paragraph = next(item for item in parsed.elements if "Native paragraph" in item.text)
        parsed.elements.append(deepcopy(paragraph))
        report = evaluate(parsed, self.gold["native_grid"])
        self.assertEqual(report["metrics"]["paragraph_once"]["matched"], 0)
        self.assertFalse(report["measured_checks_passed"])


if __name__ == "__main__":
    unittest.main()
