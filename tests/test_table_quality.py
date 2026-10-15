"""Adversarial structure, source, continuation and table usage checks."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import sqlite3
from unittest.mock import patch

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.pdf.preview import render_source_page
from app.pdf.tables.review import review_tables, select_table_evidence, structure_rows
from app.quality import check_pdf
from app.quality.service import inspect_saved_pdf
from app.rag.table_evidence import expand_table_hits, answer_reference_text
from app.schemas import Hit
from app.storage.pdf_repository import PdfRepository
from app.storage.quality_repository import QualityRepository
from tests.pdf_regression.fixtures import make_pdf


class TableQualityTests(unittest.TestCase):
    def parse(self, case="multi_header"):
        content = make_pdf(case)
        parsed = parse_pdf(case + ".pdf", content, ocr_images=False)
        return content, parsed, build_chunks(parsed, size=80)

    def assert_issue(self, parsed, chunks, code, status="blocked"):
        report = check_pdf(parsed, chunks, 80)
        issues = [i for i in report.issues if i.code == code]
        self.assertTrue(issues, [(i.code, i.message) for i in report.issues])
        self.assertTrue(any(i.status == status for i in issues))
        self.assertEqual(report.tables[0].usage, "blocked" if status == "blocked" else "reference_only")
        return report, issues

    def test_native_gold_tables_have_no_blocking_structural_findings(self):
        for case in ("native_grid", "multi_header", "merged_cells", "borderless", "continued_table", "distinct_tables"):
            _, parsed, chunks = self.parse(case)
            report = check_pdf(parsed, chunks, 80)
            self.assertNotEqual(report.status, "blocked", (case, [(i.code, i.message) for i in report.issues]))
            self.assertEqual(len(report.tables), len(parsed.logical_tables))
            if case == "native_grid":
                self.assertEqual(report.tables[0].usage, "traceable")

    def test_overlap_gap_unknown_span_and_wrong_covered_anchor(self):
        _, parsed, chunks = self.parse()
        value = next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2")
        value.row_span = 2
        _, issues = self.assert_issue(parsed, chunks, "table_span_overlap")
        self.assertTrue(issues[0].cell_ids)
        self.assertTrue(issues[0].source_spans)
        _, parsed, chunks = self.parse()
        parsed.table_fragments[0].cells.pop()
        self.assert_issue(parsed, chunks, "table_grid_gap")
        _, parsed, chunks = self.parse()
        parsed.table_fragments[0].cells[0].row_span = None
        self.assert_issue(parsed, chunks, "table_span_unknown", "review")
        _, parsed, chunks = self.parse()
        cell = parsed.table_fragments[0].cells[-1]
        cell.state, cell.raw_text, cell.covered_by_cell_id = "covered", "", "missing"
        self.assert_issue(parsed, chunks, "table_covered_anchor")

    def test_source_numeric_and_semantic_alignment_corruption_block(self):
        for corruption in ("source", "value", "header", "range", "owner"):
            _, parsed, chunks = self.parse()
            value = next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2")
            if corruption == "source":
                value.spans[0].bbox = (2, .2, 3, .3)
                code = "table_cell_location"
            elif corruption == "header":
                value.column_header_cell_ids = [next(c.id for c in parsed.table_fragments[0].cells if c.raw_text == "Elongation %")]
                code = "table_reference_alignment"
            else:
                chunk = next(c for c in chunks if "35.2" in c.page_content)
                if corruption == "value":
                    chunk.page_content = chunk.page_content.replace("35.2", "35.3")
                    code = "table_value_mapping"
                elif corruption == "owner":
                    chunk.metadata["table_id"] = "other-table"
                    code = "table_chunk_owner"
                else:
                    mappings = json.loads(chunk.metadata["pdf_serialized_map"])
                    next(m for m in mappings if m["cell_id"] == value.id)["source_char_start"] = "invalid"
                    chunk.metadata["pdf_serialized_map"] = json.dumps(mappings)
                    code = "table_value_mapping"
            self.assert_issue(parsed, chunks, code)

    def test_unindexed_cells_and_ocr_conflicts_remain_reviewable(self):
        _, parsed, chunks = self.parse("native_grid")
        chunks = [c for c in chunks if c.metadata["chunk_type"] != "table"]
        self.assert_issue(parsed, chunks, "table_unindexed_cell", "review")
        _, parsed, chunks = self.parse("native_grid")
        next(e for e in parsed.elements if e.kind == "table").payload["text_conflicts"] = [{"numeric_conflict": True}]
        self.assert_issue(parsed, chunks, "ocr_conflict", "review")

    def test_ocr_and_model_tables_are_reference_only_without_accuracy_validation(self):
        for kind in ("ocr", "model"):
            _, parsed, chunks = self.parse("native_grid")
            if kind == "ocr":
                next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2").text_origin = "ocr"
            else:
                parsed.table_fragments[0].backend = "docling"
            self.assert_issue(parsed, chunks, "table_recognition_review", "review")

    def test_false_continuations_and_missing_repeated_headers_block(self):
        for corruption in ("unit", "label", "page", "header_as_data", "repeat", "reference"):
            _, parsed, chunks = self.parse("continued_table")
            table = parsed.logical_tables[0]
            second = parsed.table_fragments[1]
            if corruption == "unit":
                next(c for c in second.cells if c.raw_text == "Strength MPa").raw_text = "Strength kPa"
                code = "table_continuation_conflict"
            elif corruption == "label":
                next(e for e in parsed.elements if "continued" in e.text).text = "Table 5 continued"
                code = "table_continuation_conflict"
            elif corruption == "page":
                second.span.page_no = 3
                code = "table_continuation_conflict"
            elif corruption == "header_as_data":
                table.data_rows.append([second.cells[0].id])
                code = "table_header_as_data"
            elif corruption == "repeat":
                table.repeated_header_cell_ids = []
                code = "table_repeated_header"
            else:
                table.data_rows.append(["foreign-cell"])
                code = "table_logical_reference"
            report = check_pdf(parsed, chunks, 80)
            self.assertTrue(any(i.code == code and i.status == "blocked" for i in report.issues), corruption)

    def test_grid_preview_and_cell_highlights_keep_original_evidence(self):
        content, parsed, _ = self.parse()
        run = json.loads(json.dumps(asdict(parsed)))
        before = deepcopy(run)
        fragment = run["table_fragments"][0]
        value = next(c for c in fragment["cells"] if c["raw_text"] == "35.2")
        selection = select_table_evidence(run, run["logical_tables"][0]["id"], fragment["id"], value["id"])
        self.assertEqual({s["highlight_role"] for s in selection["spans"]}, {"value", "row_header", "column_header", "unit"})
        self.assertTrue(render_source_page(content, 1, selection["spans"]).startswith(b"\x89PNG"))
        self.assertIn("↳", structure_rows(fragment)[1]["列 1"])
        self.assertEqual(run, before)
        with self.assertRaises(ValueError):
            select_table_evidence(run, run["logical_tables"][0]["id"], "foreign")
        with self.assertRaises(ValueError):
            render_source_page(content, 0, selection["spans"])
        legacy = deepcopy(run)
        legacy.pop("table_fragments")
        legacy.pop("logical_tables")
        self.assertTrue(review_tables(legacy)[0])

    def test_grid_budget_and_invalid_dimensions_terminate_with_explicit_status(self):
        _, parsed, chunks = self.parse()
        parsed.table_fragments[0].row_count = 100001
        self.assert_issue(parsed, chunks, "table_grid_budget", "review")
        parsed.table_fragments[0].row_count = None
        self.assert_issue(parsed, chunks, "table_grid_budget")

    def test_old_reports_restore_without_table_or_cell_fields(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            _, parsed, chunks = self.parse()
            quality = QualityRepository(directory)
            report = check_pdf(parsed, chunks, 80)
            quality.save(report)
            legacy = asdict(report)
            legacy.pop("tables")
            for item in legacy["issues"]:
                for name in ("table_id", "cell_ids", "source_spans"):
                    item.pop(name)
            with sqlite3.connect(quality.database) as connection:
                connection.execute("UPDATE quality_reports SET payload=? WHERE id=?", (json.dumps(legacy), report.id))
            restored = quality.latest(report.run_id)
            self.assertEqual(restored.tables, [])
            self.assertTrue(all(i.cell_ids == [] and i.source_spans == [] for i in restored.issues))
            quality.review(report.id, "page:1", "passed", "旧范围仍可复核")
            with self.assertRaises(ValueError):
                quality.review(report.id, f"table:{report.tables[0].id}", "passed", "未绑定的表格不接受复核")

    def test_table_review_policy_and_report_chunk_version_isolation(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            content, parsed, chunks = self.parse("native_grid")
            repository, quality = PdfRepository(directory), QualityRepository(directory)
            repository.save(parsed, content, chunks)
            report = check_pdf(parsed, chunks, 80)
            quality.save(report)
            table = report.tables[0]
            target = f"cell:{table.id}:{table.cell_ids[0]}"
            quality.review(report.id, target, "problem", "该格需要修复")
            quality.review(report.id, "document", "passed", "整体已看过")
            chunk = next(c for c in chunks if c.metadata["chunk_type"] == "table")
            run_id = chunk.metadata["pdf_parse_id"]
            policy = repository.get_table_quality(run_id, chunk.metadata["pdf_chunk_id"], table.id)
            self.assertEqual(policy["usage"], "reference_only")
            self.assertEqual(quality.latest(run_id).tables[0].status, "passed")
            with self.assertRaises(ValueError):
                quality.review(report.id, f"cell:{table.id}:foreign", "passed", "")
            broken = deepcopy(chunks)
            next(c for c in broken if "35.2" in c.page_content).page_content = chunk.page_content.replace("35.2", "35.3")
            quality.save(check_pdf(parsed, broken, 80))
            hit = Hit(parsed.source, chunk.page_content, .9, chunk.metadata["chunk"], chunk.metadata)
            expanded = expand_table_hits([hit], repository)[0]
            self.assertEqual(expanded.metadata["table_evidence"]["quality"]["usage"], "blocked")
            self.assertNotIn("35.2", answer_reference_text(expanded))
            self.assertEqual(expanded.text, hit.text)
            with patch.object(repository, "get_object", return_value=None):
                unavailable = expand_table_hits([hit], repository)[0]
                self.assertEqual(unavailable.metadata["table_evidence"]["quality"]["usage"], "blocked")
                self.assertNotIn("35.2", answer_reference_text(unavailable))
            old = deepcopy(chunks)
            for c in old:
                c.metadata["pdf_chunk_id"] += "old"
                c.metadata["pdf_chunker_version"] = "old"
            repository.save(parsed, content, old)
            current = inspect_saved_pdf(directory, run_id)
            self.assertEqual(set(current.chunk_ids), {c.metadata["pdf_chunk_id"] for c in old})
            self.assertNotIn(chunk.metadata["pdf_chunk_id"], current.chunk_ids)


if __name__ == "__main__":
    unittest.main()
