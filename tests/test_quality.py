"""质量规则、报告版本、人工复核及导入阻断的回归测试。"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.pdf.service import prepare_pdf
from app.quality import check_pdf
from app.quality.service import inspect_saved_pdf
from app.schemas import Settings
from app.schemas.pdf import PdfElement, SourceSpan
from app.storage import PdfRepository, QualityRepository
from tests.test_pdf_pipeline import native_fixture


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.content = native_fixture(table=True)
        self.parsed = parse_pdf("report.pdf", self.content)
        self.chunks = build_chunks(self.parsed)
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=root, ignore_cleanup_errors=True)
        self.directory = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_document_and_cell_mappings(self):
        report = check_pdf(self.parsed, self.chunks)
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.pages[0].tables, 1)
        self.assertFalse(report.issues)

    def test_generic_title_and_missing_body_are_located(self):
        self.parsed.title = "(12)发明专利申请"
        report = check_pdf(self.parsed, self.chunks[1:])
        codes = {item.code for item in report.issues}
        self.assertIn("title", codes)
        self.assertIn("unindexed_text", codes)
        self.assertTrue(all(item.page_no == 1 for item in report.issues))

    def test_images_are_not_counted_as_understood(self):
        self.parsed.elements.append(PdfElement("p2_image", "image", "", SourceSpan(2, "", (0.1, 0.1, 0.9, 0.9), "p2_image")))
        self.parsed.page_count = 2
        report = check_pdf(self.parsed, self.chunks)
        self.assertEqual(report.pages[1].text_chars, 0)
        self.assertEqual({issue.code for issue in report.issues if issue.page_no == 2}, {"images", "low_text"})

    def test_invalid_page_and_character_mapping_block(self):
        self.chunks[0].metadata["pdf_source_spans"] = json.dumps([{"page_no": 200, "bbox": [0, 0, 1, 1]}])
        self.chunks[0].metadata["pdf_serialized_map"] = json.dumps([{"start": 0, "end": 9999}])
        report = check_pdf(self.parsed, self.chunks)
        self.assertEqual(report.status, "blocked")
        self.assertIn("source_mapping", {issue.code for issue in report.issues})
        self.assertIn("source_span", {issue.code for issue in report.issues})

    def test_reports_and_reviews_persist_without_overriding_rules(self):
        repository = QualityRepository(self.directory)
        self.parsed.title = "(12)发明专利申请"
        report = check_pdf(self.parsed, self.chunks)
        repository.save(report)
        repository.review(report.id, "document", "passed", "已人工对照正文，标题有误")
        repository.review(report.id, "page:1", "problem", "标题未识别")
        reopened = QualityRepository(self.directory)
        latest = reopened.latest(report.run_id)
        if latest is None:
            self.fail("报告未保存")
        self.assertEqual(latest.status, "review")
        self.assertEqual(len(reopened.reviews(report.id)), 2)
        with self.assertRaises(ValueError):
            repository.review(report.id, "page:200", "passed", "非法范围")
        changed = check_pdf(self.parsed, build_chunks(self.parsed, size=180), 180)
        repository.save(changed)
        self.assertNotEqual(changed.id, report.id)
        self.assertEqual(repository.reviews(changed.id), [])

    def test_historical_check_does_not_mix_chunk_budgets(self):
        repository = PdfRepository(self.directory)
        repository.save(self.parsed, self.content, self.chunks)
        smaller = build_chunks(self.parsed, size=180)
        repository.save(self.parsed, self.content, smaller)
        report = inspect_saved_pdf(self.directory, self.chunks[0].metadata["pdf_parse_id"])
        self.assertEqual(set(report.chunk_ids), {doc.metadata["pdf_chunk_id"] for doc in smaller})

    def test_import_creates_report_and_blocked_assets_are_retained(self):
        settings = Settings(api_key="", chat_model="", embedding_model="", pdf_directory=self.directory)
        chunks, _ = prepare_pdf("report.pdf", self.content, settings)
        report = QualityRepository(self.directory).latest(chunks[0].metadata["pdf_parse_id"])
        self.assertIsNotNone(report)
        broken = build_chunks(self.parsed)
        broken[0].metadata["pdf_source_spans"] = "invalid json"
        with patch("app.pdf.service.build_chunks", return_value=broken):
            with self.assertRaisesRegex(ValueError, "来源映射"):
                prepare_pdf("report.pdf", self.content, settings)
        latest = QualityRepository(self.directory).latest(chunks[0].metadata["pdf_parse_id"])
        if latest is None:
            self.fail("阻断报告未保存")
        self.assertEqual(latest.status, "blocked")
        self.assertEqual(PdfRepository(self.directory).read_original(self.parsed.content_hash), self.content)
