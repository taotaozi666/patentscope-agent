"""专利编号段落的换行与跨块来源映射。"""

import json
import unittest

from app.pdf.chunking import build_chunks
from app.pdf.normalization import normalize_pdf_text
from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan


class PdfNormalizationTests(unittest.TestCase):
    def test_numbered_paragraphs_use_one_newline(self):
        text = ("[0002]随着信息技术的不断发展，以提高\n对数据进行处理的效率和效果。\n\n"
                "[0003]相关技术中，多结构文本和图像序列的互\n检任务。\n\n"
                "[0004]因此，如何提高推理准确度是重点问题。")
        result, _ = normalize_pdf_text(text)
        self.assertEqual(result, "[0002]随着信息技术的不断发展，以提高对数据进行处理的效率和效果。\n"
                                "[0003]相关技术中，多结构文本和图像序列的互检任务。\n"
                                "[0004]因此，如何提高推理准确度是重点问题。")

    def test_continuation_with_blank_line_and_list_boundaries(self):
        result, _ = normalize_pdf_text("[0002]提高\n\n处理效率。\n1.接收输入\n2.返回结果\n\n[0003]结束。")
        self.assertEqual(result, "[0002]提高处理效率。\n1.接收输入\n2.返回结果\n[0003]结束。")

    def test_ordinary_paragraph_spacing_is_preserved(self):
        result, _ = normalize_pdf_text("普通第一段。\n\n普通第二段。")
        self.assertEqual(result, "普通第一段。\n\n普通第二段。")

    def test_cross_page_numbered_blocks_preserve_source_locations(self):
        values = [("[0002]提高", 1), ("处理效率。", 2), ("[0003]返回结果。", 2)]
        elements = [PdfElement(f"e{i}", "paragraph", text,
                              SourceSpan(page, "", (.1, .2, .9, .3), f"e{i}"),
                              section_path="背景技术") for i, (text, page) in enumerate(values)]
        parsed = ParsedPdf("test.pdf", "a" * 64, "测试", 2, "fixture", elements)
        chunks = build_chunks(parsed)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "[0002]提高处理效率。\n[0003]返回结果。")
        mappings = json.loads(chunks[0].metadata["pdf_serialized_map"])
        self.assertEqual(len(mappings), 3)
        for element, mapping in zip(elements, mappings):
            self.assertEqual(chunks[0].page_content[mapping["start"]:mapping["end"]], element.text)
            self.assertEqual(mapping["span"]["page_no"], element.span.page_no)
            self.assertEqual((mapping["source_char_start"], mapping["source_char_end"]), (0, len(element.text)))

    def test_multiple_numbered_paragraphs_keep_mapped_offsets(self):
        text = "[0002]第一段。\n\n[0003]第二段。"
        start = text.index("[0003]")
        result, mappings = normalize_pdf_text(text, [{"start": start, "end": len(text),
                                                     "source_char_start": start, "source_char_end": len(text)}])
        self.assertEqual(result[mappings[0]["start"]:mappings[0]["end"]], "[0003]第二段。")
        self.assertEqual(mappings[0]["source_char_start"], start)


if __name__ == "__main__":
    unittest.main()
