"""Logical-table acceptance and adversarial continuation tests."""

from copy import deepcopy
from dataclasses import asdict
import json
import unittest
import pymupdf

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.pdf.serialization import restore_pdf
from app.pdf.tables.assembly import assemble_tables
from app.quality import check_pdf
from app.schemas.pdf import PdfElement, SourceSpan
from tests.pdf_regression.evaluation import evaluate, load_expectations
from tests.pdf_regression.fixtures import make_pdf


class LogicalTableTests(unittest.TestCase):
    def parse(self, case="continued_table"):
        return parse_pdf(case + ".pdf", make_pdf(case), ocr_images=False)

    def test_fixed_gold_header_paths_and_logical_groups(self):
        gold = {c["id"]: c for c in load_expectations()["cases"]}
        for case in ("native_grid", "multi_header", "merged_cells", "borderless", "continued_table", "distinct_tables"):
            with self.subTest(case=case):
                parsed = self.parse(case)
                report = evaluate(parsed, gold[case])
                self.assertTrue(report["measured_checks_passed"], report["mismatches"])
                self.assertEqual(report["metrics"]["header_association"]["matched"], len(gold[case]["numeric_facts"]))

    def test_merged_stub_and_multilevel_units_reach_numeric_chunks(self):
        for case in ("multi_header", "merged_cells"):
            parsed = self.parse(case)
            before = [(c.id, c.raw_text, c.spans) for f in parsed.table_fragments for c in f.cells]
            chunks = build_chunks(parsed, size=80, target=60)
            for fragment in parsed.table_fragments:
                for cell in fragment.cells:
                    if cell.raw_text not in {"35.2", "40.8", "120", "140"}:
                        continue
                    found = [(chunk, m) for chunk in chunks for m in json.loads(chunk.metadata["pdf_serialized_map"])
                             if m.get("cell_id") == cell.id]
                    self.assertEqual(len(found), 1)
                    chunk, mapping = found[0]
                    self.assertEqual(chunk.page_content[mapping["start"]:mapping["end"]], cell.raw_text)
                    links = json.loads(chunk.metadata["table_cell_relations"])[cell.id]
                    self.assertTrue(links["row_header_cell_ids"])
                    self.assertTrue(links["unit_cell_ids"])
                    self.assertIn("Ex1" if case == "merged_cells" or cell.row == 2 else "Ex2", chunk.metadata["context_text"])
                    self.assertTrue(all(s["spans"] for s in links["support"]))
            self.assertEqual(before, [(c.id, c.raw_text, c.spans) for f in parsed.table_fragments for c in f.cells])
            self.assertTrue(all(len(c.page_content) <= 80 for c in chunks))

    def test_continuation_parent_retains_pages_and_repeated_header_evidence(self):
        parsed = self.parse()
        table = parsed.logical_tables[0]
        self.assertEqual(len(table.fragment_ids), 2)
        self.assertEqual(len(table.data_rows), 4)
        self.assertEqual(len(table.repeated_header_cell_ids), 3)
        chunks = build_chunks(parsed, size=100)
        parents = [o for o in parsed.objects if o.kind == "table"]
        self.assertEqual(len(parents), 1)
        self.assertEqual({s.page_no for s in parents[0].spans}, {1, 2})
        self.assertEqual(parents[0].raw_content.count("Strength MPa"), 1)
        by_id = {c.id: c for f in parsed.table_fragments for c in f.cells}
        for chunk in chunks:
            if chunk.metadata["chunk_type"] != "table":
                continue
            for m in json.loads(chunk.metadata["pdf_serialized_map"]):
                cell = by_id.get(m["cell_id"])
                if cell and cell.raw_text:
                    self.assertEqual(m["span"]["page_no"], cell.spans[0].page_no)
        restored = restore_pdf(json.loads(json.dumps(asdict(parsed))))
        self.assertEqual(asdict(restored.logical_tables[0]), asdict(table))

    def test_same_label_and_headers_without_continuation_is_only_candidate(self):
        parsed = self.parse()
        caption = next(e for e in parsed.elements if "Table 4 continued" in e.text)
        caption.text = "Table 4 Results"
        assemble_tables(parsed)
        self.assertEqual(len(parsed.logical_tables), 2)
        self.assertEqual(parsed.logical_tables[0].continuation_candidate_ids, [parsed.logical_tables[1].id])

    def test_explicit_continuation_cannot_override_incompatible_evidence(self):
        original = self.parse()
        for conflict in ("label", "header_unit", "header_unit_case", "section", "geometry", "page_gap"):
            parsed = deepcopy(original)
            second = parsed.table_fragments[1]
            if conflict == "label":
                next(e for e in parsed.elements if "continued" in e.text).text = "Table 5 continued"
            elif conflict == "header_unit":
                next(c for c in second.cells if c.raw_text == "Strength MPa").raw_text = "Strength kPa"
            elif conflict == "header_unit_case":
                next(c for c in second.cells if c.raw_text == "Strength MPa").raw_text = "Strength mPa"
            elif conflict == "section":
                second.section_path = "Other experiment"
            elif conflict == "geometry":
                second.span.bbox = (.2, .2, .8, .35)
            else:
                second.span.page_no = 3
            assemble_tables(parsed)
            self.assertEqual(len(parsed.logical_tables), 2, conflict)
            self.assertFalse(parsed.logical_tables[0].continuation_candidate_ids, conflict)

    def test_nearby_unit_conflicts_block_merge_and_notes_keep_sources(self):
        parsed = self.parse()
        for page, unit in ((1, "MPa"), (2, "kPa")):
            identifier = f"unit{page}"
            parsed.elements.append(PdfElement(identifier, "paragraph", "单位：" + unit,
                SourceSpan(page, "", (.1, .18, .9, .19), identifier)))
        parsed.elements.append(PdfElement("note", "paragraph", "注：仅适用于本批样品",
            SourceSpan(1, "", (.1, .36, .9, .38), "note")))
        assemble_tables(parsed)
        self.assertEqual(len(parsed.logical_tables), 2)
        first = parsed.logical_tables[0]
        self.assertEqual(first.unit_text, "单位：MPa")
        self.assertEqual(first.unit_spans[0].page_no, 1)
        self.assertEqual(first.footnotes, ["注：仅适用于本批样品"])
        self.assertEqual(first.footnote_spans[0].element_id, "note")

    def test_unknown_header_keeps_first_data_row_and_zero(self):
        parsed = self.parse("native_grid")
        fragment = parsed.table_fragments[0]
        fragment.header_rows = None
        for cell in fragment.cells:
            if cell.raw_text:
                cell.raw_text = "0" if cell.raw_text == "0" else str(cell.row * 10 + cell.column + 1)
        assemble_tables(parsed)
        self.assertIsNone(fragment.header_rows)
        self.assertEqual(len(parsed.logical_tables[0].data_rows), 3)
        chunks = build_chunks(parsed, size=50)
        table_chunks = [c for c in chunks if c.metadata["chunk_type"] == "table"]
        mapped = [m["cell_id"] for c in table_chunks for m in json.loads(c.metadata["pdf_serialized_map"])]
        self.assertIn(fragment.cells[0].id, mapped)
        zero = next(c for c in fragment.cells if c.raw_text == "0")
        values = [c.page_content[m["start"]:m["end"]] for c in table_chunks
                  for m in json.loads(c.metadata["pdf_serialized_map"]) if m["cell_id"] == zero.id]
        self.assertEqual(values, ["0"])

    def test_evaluator_rejects_wrong_header_links_and_incorrect_grouping(self):
        gold = next(c for c in load_expectations()["cases"] if c["id"] == "continued_table")
        parsed = self.parse()
        cell = next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2")
        cell.column_header_cell_ids = [cell.row_header_cell_ids[0]]
        self.assertFalse(evaluate(parsed, gold)["measured_checks_passed"])
        parsed.logical_tables = []
        report = evaluate(parsed, gold)
        self.assertEqual(report["metrics"]["logical_table_grouping"]["matched"], 0)

    def test_unlabelled_page_edge_continuation_with_real_geometry(self):
        from tests.pdf_regression.fixtures import _grid, _rows
        with pymupdf.open() as document:
            for top, sample in ((640, "Ex1"), (160, "Ex2")):
                page = document.new_page(width=600, height=800)
                _grid(page, [60, 220, 380, 540], [top, top+40, top+80, top+120], _rows([
                    ["Sample", "Strength MPa", "Count"], [sample, "35.2", "0"], [sample+"b", "40.8", "1"]]))
            parsed = parse_pdf("edges.pdf", document.tobytes(), ocr_images=False)
        self.assertEqual(len(parsed.logical_tables), 1)
        self.assertEqual(parsed.table_fragments[1].continuation_decision, "joined_page_edges")
        self.assertEqual(len(parsed.logical_tables[0].data_rows), 4)

    def test_quality_reports_candidate_and_blocks_broken_semantic_reference(self):
        parsed = self.parse()
        next(e for e in parsed.elements if "continued" in e.text).text = "Table 4 Results"
        assemble_tables(parsed)
        report = check_pdf(parsed, build_chunks(parsed))
        self.assertIn("table_continuation_candidate", {i.code for i in report.issues})
        cell = next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2")
        cell.unit_cell_ids.append("missing-cell")
        # Check the already-produced chunks: corrupted references must be blocked
        # independently of whether object construction can consume them.
        report = check_pdf(parsed, [])
        self.assertTrue(any(i.code == "table_reference" and i.status == "blocked" for i in report.issues))

    def test_unit_identity_survives_an_intermediate_page_without_declaration(self):
        from tests.pdf_regression.fixtures import _grid, _rows, _text
        with pymupdf.open() as document:
            for index, unit in enumerate(("MPa", None, "kPa")):
                page = document.new_page(width=600, height=800)
                _text(page, 60, 125, "Table 4 Results" if index == 0 else "Table 4 continued")
                if unit:
                    _text(page, 60, 145, "单位：" + unit)
                _grid(page, [60, 220, 380, 540], [160, 200, 240, 280], _rows([
                    ["Sample", "Strength", "Count"], [f"Ex{index}", "35.2", "0"], [f"Ex{index}b", "40.8", "1"]]))
            parsed = parse_pdf("three-pages.pdf", document.tobytes(), ocr_images=False)
        self.assertEqual([len(t.fragment_ids) for t in parsed.logical_tables], [2, 1])

    def test_assembly_keeps_backend_raw_evidence_and_can_be_rebuilt(self):
        parsed = self.parse("multi_header")
        original = deepcopy(parsed.table_fragments[0].raw_output)
        before = asdict(parsed.logical_tables[0])
        assemble_tables(parsed)
        self.assertEqual(original, parsed.table_fragments[0].raw_output)
        self.assertEqual(before, asdict(parsed.logical_tables[0]))
        self.assertEqual(parsed.table_fragments[0].header_method, "numeric_body_transition")


if __name__ == "__main__":
    unittest.main()
