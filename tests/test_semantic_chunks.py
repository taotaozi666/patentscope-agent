"""语义父子分块、跨页来源、图表关联及上下文增强的回归验证。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.pdf.chunking import build_chunks
from app.pdf.context import enhance_context
from app.pdf.parser import parse_pdf
from app.pdf.patent import extract_patent
from app.schemas import Settings
from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan
from app.storage import PdfRepository
from tests.test_pdf_pipeline import native_fixture
from tests.test_patent import patent_fixture


class SemanticChunkTests(unittest.TestCase):
    def element(self, identifier, text, page, kind="paragraph", path="说明书 / 实施例一", bbox=(.1,.2,.9,.3)):
        return PdfElement(identifier, kind, text, SourceSpan(page, "", bbox, identifier), section_path=path)

    def test_cross_page_parent_and_reversible_child_mapping(self):
        elements = [self.element("e1", "接收查询请求。" * 12, 1), self.element("e2", "接着返回结果。" * 12, 2),
                    self.element("e3", "另一实施例。", 2, path="说明书 / 实施例二")]
        parsed = ParsedPdf("test.pdf", "a"*64, "测试", 2, "fixture", elements)
        chunks = build_chunks(parsed, size=100, target=75)
        first = parsed.objects[0]
        children = [chunk for chunk in chunks if chunk.metadata["parent_id"] == first.id]
        self.assertEqual("".join(chunk.page_content for chunk in children), first.raw_content)
        self.assertEqual({span.page_no for span in first.spans}, {1,2})
        by_id = {element.id: element for element in elements}
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content),100)
            for mapping in json.loads(chunk.metadata["pdf_serialized_map"]):
                original = by_id[mapping["element_id"]].text[mapping["source_char_start"]:mapping["source_char_end"]]
                self.assertEqual(chunk.page_content[mapping["start"]:mapping["end"]], original)
        self.assertEqual(children[0].metadata["next_chunk_id"], children[1].metadata["pdf_chunk_id"])
        self.assertEqual(children[-1].metadata["next_chunk_id"], "")

    def test_claim_is_single_complete_parent_even_across_pages(self):
        parsed = patent_fixture()
        parsed.patent = extract_patent(parsed)
        chunks = build_chunks(parsed, size=80)
        parent = next(parent for parent in parsed.objects if parent.kind == "claim" and parent.payload["claim_number"] == 1)
        children = [chunk for chunk in chunks if chunk.metadata["parent_id"] == parent.id]
        self.assertEqual({span.page_no for span in parent.spans}, {2,3})
        self.assertEqual("".join(chunk.page_content for chunk in children), parent.raw_content)
        dependent = next(chunk for chunk in chunks if chunk.metadata.get("claim_number") == 3)
        self.assertEqual(json.loads(dependent.metadata["depends_on"]), [1,2])

    def test_figure_caption_and_remote_description_keep_separate_pages(self):
        image = self.element("image", "", 3, kind="image", bbox=(.1,.2,.8,.6))
        caption = self.element("caption", "图1 系统结构图", 3, bbox=(.1,.62,.8,.65))
        description = self.element("description", "图1为系统结构图，包括检索模块。", 1)
        description.payload["patent_section"] = "drawing_description"
        parsed = ParsedPdf("test.pdf", "a"*64, "测试", 3, "fixture", [description,image,caption])
        chunks = build_chunks(parsed)
        parent = next(parent for parent in parsed.objects if parent.kind == "figure")
        self.assertEqual(parent.payload["image_element_ids"], ["image"])
        self.assertEqual(parent.payload["description_ids"], ["description"])
        self.assertEqual(parent.payload["visual_status"], "not_analyzed")
        figure = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "figure")
        self.assertEqual({span["page_no"] for span in json.loads(figure.metadata["pdf_source_spans"])}, {1,3})
        self.assertIn("图形内容尚未理解", figure.metadata["context_text"])

    def test_wide_table_preserves_cells_and_complete_parent(self):
        parsed = parse_pdf("report.pdf", native_fixture(table=True))
        table = next(element for element in parsed.elements if element.kind == "table")
        table.payload["rows"][1][1] = "数值说明。" * 30
        cell = next(cell for cell in table.payload["cells"] if cell["row"] == 1 and cell["column"] == 1)
        cell["raw_text"] = table.payload["rows"][1][1]
        chunks = build_chunks(parsed, size=60)
        cells = [chunk for chunk in chunks if chunk.metadata.get("table_row") == 1 and chunk.metadata.get("table_column") == 1]
        self.assertEqual("".join(chunk.page_content for chunk in cells), cell["raw_text"])
        self.assertTrue(all(chunk.metadata["table_header"] == "2024" for chunk in cells))
        parent = next(parent for parent in parsed.objects if parent.kind == "table")
        self.assertEqual(parent.payload["rows"][1][1], cell["raw_text"])

    def test_table_units_notes_have_source_and_are_not_added_to_raw_cells(self):
        parsed = parse_pdf("report.pdf", native_fixture(table=True))
        table = next(element for element in parsed.elements if element.kind == "table")
        parsed.elements.extend([
            self.element("table_caption", "表1 财务指标", 1, path=table.section_path, bbox=(.1,.17,.8,.18)),
            self.element("units", "单位：万元", 1, path=table.section_path, bbox=(.1,.18,.8,.19)),
            self.element("note", "注：数据未经审计", 1, path=table.section_path, bbox=(.1,.36,.8,.37)),
        ])
        chunks = build_chunks(parsed)
        parent = next(parent for parent in parsed.objects if parent.kind == "table")
        self.assertEqual(parent.title,"表1 财务指标")
        self.assertEqual(len(parent.payload["related_spans"]),3)
        chunk = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table")
        self.assertIn("单位：万元",chunk.metadata["context_text"])
        self.assertNotIn("单位：万元",chunk.page_content)

    def test_model_context_cached_and_original_unmodified(self):
        root = Path(__file__).resolve().parents[1]/".documents/tests"
        root.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=root,ignore_cleanup_errors=True) as directory:
            parsed = parse_pdf("report.pdf", native_fixture())
            settings = Settings(api_key="test",chat_model="fake",embedding_model="fake",pdf_directory=directory,pdf_context_mode="model")
            chunks = build_chunks(parsed)
            raw = [chunk.page_content for chunk in chunks]
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="本段说明年度报告的财务结果。"))])
            with patch("app.pdf.context.create_client",return_value=client):
                enhance_context(parsed,chunks,settings)
                second = build_chunks(parsed)
                enhance_context(parsed,second,settings)
            self.assertEqual(client.chat.completions.create.call_count,len(chunks))
            self.assertEqual([chunk.page_content for chunk in chunks],raw)
            self.assertTrue(all(chunk.metadata["context_method"] == "model" for chunk in chunks))
            repository = PdfRepository(directory)
            repository.save(parsed,native_fixture(),chunks)
            saved = repository.get_chunk(chunks[0].metadata["pdf_chunk_id"])
            if saved is None:
                self.fail("索引资产未保存")
            self.assertEqual(saved["raw_content"],raw[0])
            self.assertIn("本段说明",saved["index_text"])
            self.assertIsNotNone(repository.get_object(chunks[0].metadata["parent_id"]))
            with patch("app.pdf.context.create_client",side_effect=RuntimeError("unavailable")):
                fallback = build_chunks(parsed)
                enhance_context(parsed,fallback,settings)
            self.assertTrue(all(chunk.metadata["context_method"] == "rule" for chunk in fallback))
            self.assertTrue(any("回退" in warning for warning in parsed.warnings))
