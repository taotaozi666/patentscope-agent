"""用内存 PDF 验证真实坐标、表格映射、扫描拒绝及离线入库，无模型调用。"""

import json
import unittest
import time
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.rag import RAGService
from app.rag.embeddings import DemoEmbedder
from app.schemas import Settings
from app.ingestion import IngestionManager
from app.rag.documents import loaders, split_documents

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.storage.pdf_repository import PdfRepository
from app.pdf.preview import render_source_page


def native_fixture(table: bool = False, rotated: bool = False) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page(width=500, height=600)
        page.insert_text((50, 50), "Annual Report 2024", fontsize=20)
        page.insert_text((50, 90), "The report covers financial results for 2024.", fontsize=11)
        if table:
            xs, ys = [50, 250, 350, 450], [120, 150, 180, 210]
            for x in xs:
                page.draw_line((x, ys[0]), (x, ys[-1]))
            for y in ys:
                page.draw_line((xs[0], y), (xs[-1], y))
            for row, values in enumerate([['Metric', '2024', '2023'], ['Net profit', '320,000.00', '280,000.00'], ['Unit', 'CNY 10,000', 'CNY 10,000']]):
                for column, value in enumerate(values):
                    page.insert_text((xs[column] + 5, ys[row] + 20), value, fontsize=10)
        if rotated:
            page.set_rotation(90)
        return document.tobytes()


class PdfTests(unittest.TestCase):
    def test_image_ocr_keeps_all_blocks_and_does_not_duplicate_native_text(self):
        with pymupdf.open(stream=native_fixture(), filetype="pdf") as source:
            image = source[0].get_pixmap().tobytes("png")
        with pymupdf.open() as document:
            page = document.new_page(width=500, height=600)
            page.insert_text((50, 50), "Native page heading")
            page.insert_image((50, 100, 250, 300), stream=image, keep_proportion=False)
            page.insert_image((280, 100, 480, 300), stream=image, keep_proportion=False)
            page.insert_text((300, 130), "Native image text")
            content = document.tobytes()
        blocks = [((60, 120, 200, 140), "First OCR block"),
                  ((60, 160, 200, 180), "Second OCR block"),
                  ((300, 120, 460, 140), "Duplicate OCR text")]
        with patch("app.pdf.parser._ocr_blocks", return_value=blocks) as ocr:
            parsed = parse_pdf("images.pdf", content)
        ocr.assert_called_once()
        images = [element for element in parsed.elements if element.kind == "image"]
        recognized = next(element for element in images if element.payload["text_origin"] == "ocr")
        self.assertEqual(recognized.text, "First OCR block\nSecond OCR block")
        native = next(element for element in images if element.payload["text_origin"] == "native")
        self.assertEqual(native.text, "Native image text")
        self.assertNotIn("Duplicate OCR text", "\n".join(element.text for element in parsed.elements))

    def test_offline_pdf_worker_skips_model_context_even_with_key(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="configured-key", chat_model="chat", embedding_model="embedding",
                pdf_context_mode="model", pdf_image_ocr=False,
                ingestion_directory=str(Path(directory) / "jobs"),
                chroma_directory=str(Path(directory) / "chroma"),
                pdf_directory=str(Path(directory) / "documents"))
            manager = IngestionManager(settings)
            try:
                with patch("app.pdf.context.create_client", side_effect=AssertionError("offline network call")) as client:
                    with patch.object(manager, "start"):
                        job = manager.submit_batch([("report.pdf", native_fixture())], offline=True)[0]
                    manager._process(job.id)
                    self.assertEqual(manager.get(job.id).status, "completed")
                    client.assert_not_called()
                service = RAGService(DemoEmbedder(), settings=settings)
                records = service.store.query(include_vectors=False).items
                self.assertTrue(records)
                self.assertEqual({record.metadata["pdf_context_mode"] for record in records}, {"rule"})
                self.assertEqual(service.settings.api_key, "")
                self.assertEqual(settings.api_key, "configured-key")
            finally:
                manager.stop()

    def test_native_source_and_rotation(self):
        parsed = parse_pdf("report.pdf", native_fixture(rotated=True))
        self.assertEqual(parsed.page_count, 1)
        self.assertTrue(any(item.kind == "heading" for item in parsed.elements))
        for item in parsed.elements:
            self.assertEqual(item.span.page_no, 1)
            self.assertTrue(all(0 <= value <= 1 for value in item.span.bbox))
        chunks = build_chunks(parsed)
        self.assertNotIn("文档：", chunks[0].page_content)
        self.assertIn("文档：", chunks[0].metadata["_pdf_index_text"])

    def test_table_cells_and_character_mapping(self):
        parsed = parse_pdf("report.pdf", native_fixture(table=True))
        table = next(item for item in parsed.elements if item.kind == "table")
        cell = next(cell for cell in table.payload["cells"] if cell["raw_text"] == "320,000.00")
        self.assertEqual(cell["source_page_no"], 1)
        self.assertEqual(cell["span"]["granularity"], "cell")
        chunks = build_chunks(parsed, size=180)
        located = []
        for chunk in chunks:
            for mapping in json.loads(chunk.metadata["pdf_serialized_map"]):
                if mapping.get("cell_id") == cell["id"]:
                    located.append(chunk.page_content[mapping["start"]:mapping["end"]])
        self.assertEqual(located, ["320,000.00"])
        # fixture 中该格的几何真值，独立于索引序列化算法。
        for actual, expected in zip(cell["span"]["bbox"], (0.5, 0.25, 0.7, 0.3)):
            self.assertAlmostEqual(actual, expected, places=2)
        image = render_source_page(native_fixture(table=True), 1, [cell["span"]])
        self.assertTrue(image.startswith(b"\x89PNG"))

    def test_scan_and_encrypted_are_rejected(self):
        with pymupdf.open(stream=native_fixture(), filetype="pdf") as source:
            image = source[0].get_pixmap().tobytes("png")
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_image(page.rect, stream=image)
            with self.assertRaisesRegex(ValueError, "OCR"):
                parse_pdf("scan.pdf", document.tobytes())
            # PyMuPDF 动态导出加密常量，当前 SDK 类型定义没有列出该属性。
            encrypted = document.tobytes(encryption=getattr(pymupdf, "PDF_ENCRYPT_AES_256"), owner_pw="owner", user_pw="user")
            with self.assertRaisesRegex(ValueError, "加密"):
                parse_pdf("encrypted.pdf", encrypted)

    def test_long_paragraph_keeps_source(self):
        parsed = parse_pdf("report.pdf", native_fixture())
        paragraph = next(item for item in parsed.elements if item.kind == "paragraph")
        paragraph.text = "A complete sentence. " * 80
        chunks = build_chunks(parsed, size=120)
        self.assertEqual("".join(chunk.metadata["_pdf_raw_content"] for chunk in chunks), paragraph.text)
        self.assertEqual(" ".join(chunk.page_content for chunk in chunks), paragraph.text.strip())
        for chunk in chunks:
            for mapping in json.loads(chunk.metadata["pdf_serialized_map"]):
                original = paragraph.text[mapping["source_char_start"]:mapping["source_char_end"]]
                self.assertEqual(chunk.page_content[mapping["start"]:mapping["end"]], original.strip())
        self.assertTrue(all(chunk.metadata["page_start"] == 1 for chunk in chunks))

    def test_real_offline_ingestion_and_duplicate(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="",
                chroma_directory=str(Path(directory) / "chroma"),
                pdf_directory=str(Path(directory) / "documents"))
            service = RAGService(DemoEmbedder(), settings=settings)
            content = native_fixture(table=True)
            count = service.add_file("report.pdf", content)
            self.assertGreater(count, 0)
            self.assertEqual(service.add_file("report.pdf", content), 0)
            records = service.store.query(limit=20, include_vectors=False).items
            self.assertEqual(len(records), count)
            repository = PdfRepository(settings.pdf_directory)
            for record in records:
                self.assertNotIn("_pdf_index_text", record.metadata)
                saved = repository.get_chunk(record.metadata["pdf_chunk_id"])
                self.assertIsNotNone(saved)
                if saved is None:
                    self.fail("已入库片段缺少 PDF 来源记录")
                self.assertEqual(saved["text"], record.text)
                self.assertEqual(repository.read_original(record.metadata["pdf_version_id"]), content)

    def test_registered_loader_keeps_prepared_source_mapping(self):
        documents = loaders.load("report.PDF", native_fixture(table=True))
        chunks = split_documents(documents, size=50, overlap=5)
        self.assertEqual([doc.page_content for doc in chunks], [doc.page_content for doc in documents])
        self.assertEqual([doc.metadata["chunk"] for doc in chunks], list(range(1, len(chunks) + 1)))

    def test_background_pdf_task(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="",
                ingestion_directory=str(Path(directory) / "jobs"),
                chroma_directory=str(Path(directory) / "chroma"),
                pdf_directory=str(Path(directory) / "documents"))
            manager = IngestionManager(settings)
            try:
                job = manager.submit_batch([("report.pdf", native_fixture(table=True))], offline=True)[0]
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    job = manager.get(job.id)
                    if job.status not in {"queued", "processing"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(job.status, "completed", job.error_message)
                self.assertGreater(job.chunk_count, 0)
                self.assertFalse(list(manager.uploads.glob("*.upload")))
                self.assertTrue(list((Path(settings.pdf_directory) / "originals").glob("*.pdf")))
            finally:
                manager.stop()


if __name__ == "__main__":
    unittest.main()
