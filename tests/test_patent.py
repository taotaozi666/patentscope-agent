"""专利字段、章节、跨页权利要求及引用关系的回归验证。"""

import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from app.pdf.patent import extract_patent
from app.pdf.parser import parse_pdf
from app.pdf.chunking import build_chunks
from app.pdf.serialization import restore_pdf
from app.schemas import Settings
from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan
from app.rag import RAGService
from app.rag.embeddings import DemoEmbedder
from app.storage.index_status import pdf_index_status


def patent_fixture() -> ParsedPdf:
    elements = []
    for index, (page, text) in enumerate([
        (1, "CN 123456789 A"), (1, "(12)发明专利申请"),
        (1, "(54)发明名称"), (1, "一种测试\n检索方法\n(57)摘要"), (1, "通过检索返回资料。"),
        (2, "权 利 要 求 书\n1/2 页"), (2, "1.一种检索方法，包括接收问题；"),
        (3, "权 利 要 求 书\n2/2 页"), (3, "再检索资料并返回结果。"),
        (3, "2.根据权利要求1所述的方法，增加排序。\n3.根据权利要求1至2任一项所述的方法，返回引用。"),
        (4, "说 明 书\n1/1 页"), (4, "技术领域"), (4, "本方法属于检索技术。"),
        (4, "附图说明"), (4, "图1为检索流程。"), (4, "具体实施方式"), (4, "执行检索。"),
        (5, "说 明 书 附 图\n1/1 页"), (5, "图1"),
    ]):
        identifier = f"e{index}"
        elements.append(PdfElement(identifier, "paragraph", text, SourceSpan(page, "", (.1, .1, .8, .2), identifier)))
    return ParsedPdf("test.pdf", "a" * 64, "(12)发明专利申请", 5, "fixture", elements)


class PatentTests(unittest.TestCase):
    def test_fields_cross_page_claims_and_sections(self):
        parsed = patent_fixture()
        parsed.patent = extract_patent(parsed)
        patent = parsed.patent
        if patent is None or patent.publication_number is None:
            self.fail("专利未识别")
        self.assertEqual(parsed.title, "一种测试检索方法")
        self.assertEqual(patent.publication_number.value, "CN123456789A")
        self.assertEqual([claim.number for claim in patent.claims], [1, 2, 3])
        self.assertIn("再检索资料", patent.claims[0].text)
        self.assertEqual({span.page_no for span in patent.claims[0].spans}, {2, 3})
        self.assertEqual(patent.claims[1].depends_on, [1])
        self.assertEqual(patent.claims[2].depends_on, [1, 2])
        self.assertEqual({section.kind for section in patent.sections}, {"abstract", "claims", "description", "drawing_description", "drawings"})
        self.assertFalse(patent.warnings)
        self.assertEqual(restore_pdf(asdict(parsed)), parsed)
        chunks = build_chunks(parsed)
        self.assertTrue(all(doc.metadata["publication_number"] == "CN123456789A" for doc in chunks))
        self.assertTrue(any(doc.metadata.get("claim_number") == 1 for doc in chunks))

    def test_invalid_reference_and_missing_fields_are_reported(self):
        parsed = patent_fixture()
        parsed.elements = [element for element in parsed.elements if "54" not in element.text]
        parsed.elements[6].text = "1.根据权利要求9所述的方法。"
        patent = extract_patent(parsed)
        if patent is None:
            self.fail("专利未识别")
        self.assertIsNone(patent.title)
        self.assertTrue(any("名称" in warning for warning in patent.warnings))
        self.assertTrue(any("引用编号异常" in warning for warning in patent.warnings))

    def test_non_patent_and_legacy_records(self):
        parsed = patent_fixture()
        parsed.elements[0].text = "Annual Report 2024"
        self.assertIsNone(extract_patent(parsed))
        data = asdict(parsed)
        data.pop("patent")
        self.assertIsNone(restore_pdf(data).patent)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/CN112784004A.pdf").exists(), "真实专利样例未下载")
    def test_real_patent_and_actual_index_status(self):
        content = (Path(__file__).resolve().parents[1] / "data/CN112784004A.pdf").read_bytes()
        parsed = parse_pdf("CN112784004A.pdf", content)
        patent = parsed.patent
        if patent is None:
            self.fail("真实专利未识别")
        self.assertEqual(parsed.title, "PDF文档的检索方法、系统、电子设备、存储介质")
        self.assertEqual([claim.number for claim in patent.claims], list(range(1, 16)))
        self.assertEqual({span.page_no for span in patent.claims[7].spans}, {2, 3})
        self.assertEqual(patent.claims[13].depends_on, list(range(1, 8)))
        root = Path(__file__).resolve().parents[1] / ".documents" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root, ignore_cleanup_errors=True) as directory:
            settings = Settings(api_key="", chat_model="", embedding_model="",
                                pdf_directory=str(Path(directory)/"documents"), chroma_directory=str(Path(directory)/"chroma"))
            service = RAGService(DemoEmbedder(), settings=settings)
            count = service.add_file("CN112784004A.pdf", content)
            chunks = build_chunks(parsed)
            run_id = chunks[0].metadata["pdf_parse_id"]
            ids = [doc.metadata["pdf_chunk_id"] for doc in chunks]
            self.assertEqual(pdf_index_status(settings, run_id, ids)["离线集合"], f"完整（{count}/{count}）")
            service.store.clear()
            self.assertEqual(pdf_index_status(settings, run_id, ids)["离线集合"], "未入库")
