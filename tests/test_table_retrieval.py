"""Row/column budgets and saved evidence reaching retrieval, answers and sources."""

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.pdf.tables.assembly import assemble_tables
from app.rag import RAGService, answer
from app.rag.embeddings import DemoEmbedder
from app.rag.retrieval import merge_candidates
from app.rag.table_evidence import expand_table_hits, answer_reference_text
from app.rag.table_evidence import table_evidence_text
from app.schemas import ChunkRecord, Hit, Settings
from app.storage.pdf_repository import PdfRepository
from tests.pdf_regression.fixtures import make_pdf


class TableRetrievalTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=root, ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.repo = PdfRepository(self.temporary.name)

    def save(self, case="multi_header", size=60):
        raw = make_pdf(case)
        parsed = parse_pdf(case + ".pdf", raw, ocr_images=False)
        chunks = build_chunks(parsed, size=size)
        self.repo.save(parsed, raw, chunks)
        return parsed, chunks

    def hit(self, chunk):
        return Hit(chunk.metadata["source"], chunk.page_content, .9, chunk.metadata["chunk"], deepcopy(chunk.metadata))

    def value_chunk(self, chunks, value="35.2", page=None):
        return next(c for c in chunks if c.metadata["chunk_type"] == "table" and any(
            c.page_content[m["start"]:m["end"]] == value and m["role"] == "source"
            and (page is None or m["span"]["page_no"] == page)
            for m in json.loads(c.metadata["pdf_serialized_map"])))

    def test_column_groups_and_rows_preserve_every_data_character_once(self):
        for case in ("multi_header", "merged_cells", "continued_table", "native_grid"):
            for size in (40, 80, 600):
                parsed = parse_pdf(case + ".pdf", make_pdf(case), ocr_images=False)
                chunks = build_chunks(parsed, size=size)
                evidence = defaultdict(str)
                for chunk in chunks:
                    self.assertLessEqual(len(chunk.page_content), size)
                    if chunk.metadata["chunk_type"] != "table":
                        continue
                    for m in json.loads(chunk.metadata["pdf_serialized_map"]):
                        if m["role"] == "source":
                            evidence[m["cell_id"]] += chunk.page_content[m["start"]:m["end"]]
                for fragment in parsed.table_fragments:
                    for cell in fragment.cells:
                        if cell.row not in fragment.header_rows and cell.raw_text:
                            self.assertEqual(evidence[cell.id], cell.raw_text, (case, size, cell.id))
                if case == "multi_header" and size == 40:
                    self.assertTrue(any(c.metadata.get("table_split") == "column_group" for c in chunks))
                if size == 600:
                    self.assertTrue(any(c.metadata.get("table_split") == "row_group" for c in chunks))

    def test_long_value_split_and_header_only_mapping_offsets(self):
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        fragment = parsed.table_fragments[0]
        value = next(c for c in fragment.cells if c.raw_text == "35.2")
        value.raw_text = "A" * 173
        chunks = build_chunks(parsed, size=40)
        parts = []
        for chunk in chunks:
            for m in json.loads(chunk.metadata["pdf_serialized_map"]):
                if m.get("cell_id") == value.id:
                    self.assertEqual(chunk.page_content[m["start"]:m["end"]],
                                     value.raw_text[m["source_char_start"]:m["source_char_end"]])
                    parts.append(chunk.page_content[m["start"]:m["end"]])
        self.assertEqual("".join(parts), value.raw_text)
        fragment.header_rows = list(range(fragment.row_count))
        assemble_tables(parsed)
        chunks = [c for c in build_chunks(parsed, size=40) if c.metadata["chunk_type"] == "table"]
        by_id = {c.id: c for c in fragment.cells}
        for chunk in chunks:
            for m in json.loads(chunk.metadata["pdf_serialized_map"]):
                self.assertEqual(chunk.page_content[m["start"]:m["end"]],
                    by_id[m["cell_id"]].raw_text[m["source_char_start"]:m["source_char_end"]])

    def test_long_header_remains_indexed_when_it_cannot_fit_above_data(self):
        parsed = parse_pdf("native_grid.pdf", make_pdf("native_grid"), ocr_images=False)
        header = parsed.table_fragments[0].cells[1]
        header.raw_text = "Long property " * 15
        chunks = build_chunks(parsed, size=40)
        pieces = [c.page_content[m["start"]:m["end"]] for c in chunks
                  for m in json.loads(c.metadata["pdf_serialized_map"]) if m.get("cell_id") == header.id]
        self.assertEqual("".join(pieces), header.raw_text)
        self.assertTrue(any("35.2" in c.page_content for c in chunks))

    def test_multilevel_support_is_resolved_from_saved_cells_not_generated_context(self):
        parsed, chunks = self.save()
        hit = self.hit(self.value_chunk(chunks))
        hit.metadata["context_text"] = "Fabricated value 999 MPa"
        hit.metadata["table_cell_relations"] = '{"fake":"Do not use"}'
        before = deepcopy(hit)
        expanded = expand_table_hits([hit], self.repo)[0]
        self.assertEqual(hit, before)
        self.assertEqual(expanded.text, hit.text)
        bundle = expanded.metadata["table_evidence"]
        self.assertEqual(bundle["status"], "complete", bundle)
        value = next(c for c in parsed.table_fragments[0].cells if c.raw_text == "35.2")
        links = bundle["associations"][value.id]
        by_id = {c["id"]: c for c in bundle["cells"]}
        self.assertEqual([by_id[i]["text"] for i in links["column_header_cell_ids"]], ["Properties", "Strength MPa"])
        self.assertEqual([by_id[i]["text"] for i in links["row_header_cell_ids"]], ["Ex1"])
        self.assertTrue(links["unit_cell_ids"])
        self.assertTrue(all(c["spans"] for c in bundle["cells"]))
        text = answer_reference_text(expanded)
        self.assertIn("Properties", text)
        self.assertNotIn("999", text)
        self.assertEqual(json.loads(json.dumps(bundle)), bundle)

    def test_continuation_uses_repeated_header_sources_from_the_same_page(self):
        parsed, chunks = self.save("continued_table", size=40)
        expanded = expand_table_hits([self.hit(self.value_chunk(chunks, page=2))], self.repo)[0]
        bundle = expanded.metadata["table_evidence"]
        self.assertEqual(bundle["status"], "complete", bundle)
        root = next(c for c in parsed.table_fragments[1].cells if c.raw_text == "35.2")
        source_element = next(e for e in parsed.elements if e.kind == "table" and e.span.page_no == 2)
        self.assertEqual(expanded.metadata["image_id"], source_element.payload["image_id"])
        by_id = {c["id"]: c for c in bundle["cells"]}
        for identifier in bundle["associations"][root.id]["column_header_cell_ids"]:
            self.assertEqual(by_id[identifier]["spans"][0]["page_no"], 2)
        self.assertIn("Ex3", answer_reference_text(expanded))
        self.assertNotIn("Ex1", answer_reference_text(expanded))

    def test_version_document_parent_and_raw_text_mismatch_fail_closed(self):
        _, chunks = self.save()
        original = self.hit(self.value_chunk(chunks))
        for field in ("pdf_parse_id", "pdf_version_id", "parent_id", "table_id", "source", "text"):
            changed = deepcopy(original)
            if field in ("source", "text"):
                setattr(changed, field, "other")
            else:
                changed.metadata[field] = "other"
            expanded = expand_table_hits([changed], self.repo)[0]
            self.assertEqual(expanded.metadata["table_evidence"]["status"], "unavailable", field)
            self.assertFalse(expanded.metadata["table_evidence"]["cells"])

    def test_missing_or_legacy_assets_and_budget_do_not_invent_support(self):
        _, chunks = self.save()
        hit = self.hit(self.value_chunk(chunks))
        for repo in (None, Mock(get_chunk=Mock(return_value=None))):
            result = expand_table_hits([hit], repo)[0]
            self.assertEqual(result.metadata["table_evidence"]["status"], "unavailable")
        tiny = expand_table_hits([hit], self.repo, max_cells=1, max_characters=10)[0]
        self.assertEqual(tiny.metadata["table_evidence"]["status"], "partial")
        self.assertIn("evidence_budget", tiny.metadata["table_evidence"]["issues"])
        expanded = expand_table_hits([hit], self.repo)[0]
        self.assertIn("展示预算不足", table_evidence_text(expanded, max_characters=50))
        with patch.object(self.repo, "get_object", wraps=self.repo.get_object) as read:
            expand_table_hits([hit, deepcopy(hit)], self.repo)
            self.assertEqual(read.call_count, 1)
        with patch.object(self.repo, "get_object", return_value={"kind": "table", "payload": {"evidence_schema_version": 0}}):
            result = expand_table_hits([hit], self.repo)[0]
            self.assertEqual(result.metadata["table_evidence"]["status"], "unavailable")

    def test_equal_numbers_at_distinct_cells_remain_distinct_hits(self):
        _, chunks = self.save("continued_table", size=40)
        roots = [self.value_chunk(chunks, page=p) for p in (1, 2)]
        records = [ChunkRecord(str(i), "same-document", "test.pdf", "35.2", i, [], datetime.now(),
            {**c.metadata, "image_id": "same-table-image"}) for i, c in enumerate(roots)]
        hits = merge_candidates([(r, .9) for r in records])
        self.assertEqual(len(hits), 2)
        for record in records:
            record.metadata["pdf_chunk_id"] = None
        self.assertEqual(len(merge_candidates([(r, .9) for r in records])), 2)

    def test_search_and_model_prompt_receive_saved_support_without_extra_hits(self):
        _, chunks = self.save()
        chunk = self.value_chunk(chunks)
        record = ChunkRecord("root", "document", chunk.metadata["source"], chunk.page_content,
                             chunk.metadata["chunk"], [], datetime.now(), chunk.metadata)
        store = Mock()
        store.count.return_value = 1
        store.nearest.return_value = [(record, .9)]
        service = RAGService(DemoEmbedder(), store=store, settings=Settings(
            api_key="", chat_model="test", embedding_model="test", pdf_directory=self.temporary.name))
        hits = service.search("Ex1的拉伸强度", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].metadata["table_evidence"]["status"], "complete")
        with patch("app.rag.rag.generate_reply", return_value="35.2 MPa [1]") as generate:
            reply = answer("强度？", hits, [], client=Mock(), model="test")
            context = generate.call_args.args[1]
        self.assertEqual(reply, "35.2 MPa [1]")
        self.assertIn("Strength MPa", context)
        self.assertIn("Properties", context)
        self.assertIn("Ex1", context)
        self.assertIn("第 1 页", context)
        self.assertEqual(context.count("[1] multi_header.pdf"), 1)


if __name__ == "__main__":
    unittest.main()
