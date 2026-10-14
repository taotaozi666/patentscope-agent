"""同名源文件保存及知识库列表直接下载，不调用外部服务。"""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
import pymupdf
from streamlit.testing.v1 import AppTest

from app.rag import RAGService
from app.rag.embeddings import DemoEmbedder
from app.schemas import Settings
from app.storage import ChromaKnowledgeStore
from app.storage.source_repository import SourceRepository
from tests.test_pdf_pipeline import native_fixture


class SourceFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(api_key="", chat_model="", embedding_model="",
                                 pdf_image_ocr=False, pdf_directory=self.temporary.name)
        self.store = ChromaKnowledgeStore(DemoEmbedder())
        self.rag = RAGService(DemoEmbedder(), store=self.store, settings=self.settings)

    def page(self):
        page = AppTest.from_string(
            "import streamlit as st\n"
            "from frontend.knowledge_page import render_knowledge_page\n"
            "render_knowledge_page(st.session_state.test_store)"
        )
        page.session_state["test_store"] = self.store
        return page

    def test_same_name_text_download_buttons_preserve_original_bytes(self):
        originals = ["\ufeff# 标题\r\n第一份文本", "# 标题\n第二份文本"]
        for text in originals:
            self.rag.add_file("同名.md", text.encode("utf-8"))
        with patch("frontend.source_components.get_settings", return_value=self.settings):
            page = self.page().run()
            self.assertFalse(page.exception)
            self.assertEqual(len(page.get("download_button")), 2)
            self.assertFalse(any(button.label == "查看源文件" for button in page.button))
            self.assertFalse(page.number_input)
            for document in self.store.documents():
                record = self.store.query(document_id=document.id, limit=1).items[0]
                expected = next(text.encode() for text in originals
                                if sha256(text.encode()).hexdigest() == record.metadata["content_hash"])
                with patch("frontend.source_components.st.download_button") as button:
                    from frontend.source_components import render_download_button
                    render_download_button(self.store, document, "test")
                    self.assertEqual(button.call_args.args[1], expected)
                    self.assertEqual(button.call_args.kwargs["file_name"], "同名.md")
                    self.assertEqual(button.call_args.kwargs["on_click"], "ignore")

    def test_pdf_has_direct_download_without_preview(self):
        content = native_fixture()
        self.rag.add_file("report.pdf", content)
        with patch("frontend.source_components.get_settings", return_value=self.settings):
            page = self.page().run()
            self.assertFalse(page.exception)
            self.assertEqual(len(page.get("download_button")), 1)
            self.assertFalse(page.image)
            self.assertFalse(page.number_input)
        repository = SourceRepository(self.settings.pdf_directory)
        self.assertEqual(repository.read(sha256(content).hexdigest(), ".pdf"), content)

    def test_legacy_text_without_original_has_disabled_button(self):
        self.store.add_documents([Document(page_content="旧片段", metadata={"source": "old.txt"})])
        with patch("frontend.source_components.get_settings", return_value=self.settings):
            page = self.page().run()
            self.assertFalse(page.exception)
            self.assertFalse(page.get("download_button"))
            self.assertTrue(any(button.label == "原文件不可用" and button.disabled for button in page.button))
            self.assertTrue(any("重新上传" in item.value for item in page.caption))

    def test_duplicate_reupload_restores_missing_source_without_extra_chunks(self):
        content = b"original source bytes\r\n"
        self.rag.add_file("report.txt", content)
        record = self.store.query().items[0]
        path = Path(self.settings.pdf_directory) / "originals" / f"{record.metadata['content_hash']}.txt"
        path.unlink()
        self.assertEqual(self.rag.add_file("report.txt", content), 0)
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.store.count(), 1)

    def test_same_name_pdf_documents_keep_separate_sources_and_objects(self):
        first = native_fixture()
        with pymupdf.open(stream=first, filetype="pdf") as pdf:
            pdf[0].insert_text((50, 250), "Different content")
            second = pdf.tobytes()
        self.rag.add_file("same.pdf", first)
        self.rag.add_file("same.pdf", second)
        documents = self.store.documents()
        self.assertEqual(len(documents), 2)
        hashes = {self.store.query(document_id=doc.id, limit=1).items[0].metadata["content_hash"]
                  for doc in documents}
        self.assertEqual(hashes, {sha256(first).hexdigest(), sha256(second).hexdigest()})


if __name__ == "__main__":
    unittest.main()
