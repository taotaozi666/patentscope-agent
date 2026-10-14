"""解析检查页面的原文对照、历史报告生成和复核交互。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from streamlit.testing.v1 import AppTest

from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.schemas import Settings
from app.storage import PdfRepository, QualityRepository
from tests.test_pdf_pipeline import native_fixture
from app.quality.service import inspect_saved_pdf


class QualityPageTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/CN112784004A.pdf").exists(), "真实专利样例未下载")
    def test_patent_navigation_and_clear_page_labels(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            content = (Path(__file__).resolve().parents[1]/"data/CN112784004A.pdf").read_bytes()
            parsed = parse_pdf("CN112784004A.pdf", content)
            chunks = build_chunks(parsed)
            PdfRepository(directory).save(parsed, content, chunks)
            report = inspect_saved_pdf(directory, chunks[0].metadata["pdf_parse_id"])
            app = AppTest.from_string(
                "import streamlit as st\nfrom frontend.quality_page import render_quality_page\n"
                "render_quality_page(st.session_state.settings)"
            )
            app.session_state["settings"] = Settings(api_key="", chat_model="", embedding_model="", pdf_directory=directory,
                                                     chroma_directory=str(Path(directory)/"chroma"))
            app.run()
            self.assertFalse(app.exception)
            run_id = chunks[0].metadata["pdf_parse_id"]
            app.selectbox(key=f"quality_patent_{run_id}").select("claim:8").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key=f"quality_page_{report.id}_claim:8").value, 2)
            captions = [image.caption for item in app.image for image in item.proto.imgs]
            self.assertIn("物理第 2 页；红框为所选问题的来源区域。", captions)
            self.assertTrue(any("区域裁图" in caption for caption in captions))
            self.assertFalse(any("页码标签：无" in caption.value for caption in app.caption))
            self.assertTrue(any("版面页码" in caption.value for caption in app.caption))
            self.assertTrue(any("权利要求" in text.value for text in app.text))

    def test_historical_document_and_review(self):
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            content = native_fixture(table=True)
            parsed = parse_pdf("report.pdf", content)
            parsed.title = "(12)发明专利申请"
            chunks = build_chunks(parsed)
            PdfRepository(directory).save(parsed, content, chunks)
            app = AppTest.from_string(
                "import streamlit as st\n"
                "from frontend.quality_page import render_quality_page\n"
                "render_quality_page(st.session_state.settings)"
            )
            app.session_state["settings"] = Settings(api_key="", chat_model="", embedding_model="", pdf_directory=directory,
                                                     chroma_directory=str(Path(directory)/"chroma"))
            app.run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            captions = [image.caption for item in app.image for image in item.proto.imgs]
            self.assertEqual(len(captions), 2)
            self.assertIn("物理第 1 页；红框为所选问题的来源区域。", captions)
            self.assertIn("第 1 页 · table 区域裁图", captions)
            report = QualityRepository(directory).latest(chunks[0].metadata["pdf_parse_id"])
            if report is None:
                self.fail("历史报告未生成")
            app.selectbox(key=f"quality_issue_{report.id}").select(report.issues[0].id).run()
            self.assertFalse(app.exception)
            app.selectbox(key=f"quality_target_{report.id}").select(f"issue:{report.issues[0].id}").run()
            app.radio[0].set_value("有问题")
            app.text_area[0].set_value("已对照原页，标题错误")
            next(button for button in app.button if button.label == "保存复核").click().run()
            self.assertFalse(app.exception)
            reviews = QualityRepository(directory).reviews(report.id)
            self.assertEqual(reviews[0].decision, "problem")
            self.assertEqual(reviews[0].note, "已对照原页，标题错误")
