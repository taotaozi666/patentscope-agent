"""验证 PDF 来源展示组件迁移后的页面交互和信息点预览。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.schemas import Settings
from app.storage import PdfRepository
from tests.test_pdf_pipeline import native_fixture


class PdfComponentsTests(unittest.TestCase):
    def test_source_preview_and_cell_selection(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="", pdf_directory=directory)
            content = native_fixture(table=True)
            parsed = parse_pdf("report.pdf", content)
            chunks = build_chunks(parsed)
            PdfRepository(directory).save(parsed, content, chunks)
            table = next(chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table")
            script = (
                "from frontend.pdf_components import render_pdf_sources\n"
                f"render_pdf_sources({table.metadata!r})\n"
            )
            with patch("frontend.pdf_components.get_settings", return_value=settings):
                page = AppTest.from_string(script).run()
                self.assertEqual(len(page.exception), 0)
                page.checkbox[0].check().run()
                self.assertEqual(len(page.exception), 0)
                choices = page.selectbox[0].options
                index = next(i for i, label in enumerate(choices) if "320,000.00" in label)
                page.selectbox[0].select(index).run()
                self.assertEqual(len(page.exception), 0)
                self.assertIn("320,000.00", page.text[0].value)
                self.assertEqual(len(page.image), 1)
