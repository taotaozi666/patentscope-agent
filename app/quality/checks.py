"""检查可观察的解析异常；检查通过不代表达到人工标注准确率。"""

import json
import math
import re
from datetime import UTC, datetime
from hashlib import sha256

from langchain_core.documents import Document

from app.schemas.pdf import ParsedPdf
from app.schemas.quality import PageQuality, QualityIssue, QualityReport, QualityStatus

RULE_VERSION = "pdf-quality-v3"


def _valid_span(span: object, page_count: int) -> bool:
    if not isinstance(span, dict):
        return False
    page = span.get("page_no")
    bbox = span.get("bbox")
    if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= page_count:
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1 for value in bbox):
        return False
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]


def check_pdf(parsed: ParsedPdf, chunks: list[Document], size: int = 600) -> QualityReport:
    """按本次分块集合生成报告；规则版本和片段集合决定报告身份。"""
    chunk_ids = [str(chunk.metadata["pdf_chunk_id"]) for chunk in chunks]
    run_id = str(chunks[0].metadata["pdf_parse_id"]) if chunks else sha256(
        f"{parsed.content_hash}:{parsed.source}:{parsed.parser_version}".encode()
    ).hexdigest()
    report_id = sha256(f"{RULE_VERSION}:{run_id}:{size}:{','.join(chunk_ids)}".encode()).hexdigest()
    report = QualityReport(report_id, run_id, RULE_VERSION, datetime.now(UTC).isoformat(), "passed", chunk_ids)

    def issue(code: str, message: str, page: int | None = None, element: str | None = None,
              chunk: str | None = None, status: QualityStatus = "review") -> None:
        identifier = sha256(f"{report_id}:{code}:{page}:{element}:{chunk}:{message}".encode()).hexdigest()[:20]
        report.issues.append(QualityIssue(identifier, code, status, message, page, element, chunk))

    title = re.sub(r"[\s()（）\d]", "", parsed.title)
    if title in {"发明专利申请", "发明专利", "实用新型专利", "授权公告", "申请公布"} or parsed.title == parsed.source:
        first_heading = next((element for element in parsed.elements if element.kind == "heading"), None)
        issue("title", "标题疑似为文档类型标签或文件名，请核对真实名称。",
              first_heading.span.page_no if first_heading else 1,
              first_heading.id if first_heading else None)
    if not chunks:
        issue("empty_chunks", "没有生成可检索的正文片段。", status="blocked")

    covered: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.metadata["pdf_chunk_id"])
        page = chunk.metadata.get("page_start")
        page_no = page if isinstance(page, int) else None
        if len(chunk.page_content) > size:
            issue("chunk_budget", "片段超过本次配置的字符预算。", page_no, chunk=chunk_id)
        if not chunk.page_content.strip():
            issue("empty_chunk", "检索片段为空。", page_no, chunk=chunk_id, status="blocked")
        try:
            spans = json.loads(chunk.metadata.get("pdf_source_spans", "[]"))
            mappings = json.loads(chunk.metadata.get("pdf_serialized_map", "[]"))
            if not isinstance(spans, list) or not spans or not all(_valid_span(span, parsed.page_count) for span in spans):
                issue("source_span", "片段页码或来源区域无效。", page_no, chunk=chunk_id, status="blocked")
            if not isinstance(mappings, list) or not mappings:
                raise ValueError("没有映射")
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    raise ValueError("映射格式错误")
                start, end = mapping.get("start"), mapping.get("end")
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(chunk.page_content):
                    raise ValueError("字符区间越界")
                if mapping.get("element_id"):
                    covered.add(str(mapping["element_id"]))
                if mapping.get("span") is None:
                    issue("cell_location", "部分单元格缺少独立位置，请核对合并单元格。", page_no, chunk=chunk_id)
                elif not _valid_span(mapping["span"], parsed.page_count):
                    raise ValueError("映射位置无效")
        except (ValueError, TypeError):
            issue("source_mapping", "字符映射缺失、格式错误或定位无效。", page_no, chunk=chunk_id, status="blocked")

    for element in parsed.elements:
        if element.kind == "paragraph" and element.text.strip() and element.id not in covered:
            issue("unindexed_text", "该正文元素未在片段字符映射中找到，可能存在漏索引。", element.span.page_no, element.id)
        if element.text and element.text.count("\ufffd") / len(element.text) > 0.01:
            issue("garbled_text", "文字包含较多替换字符，请对照原页检查乱码。", element.span.page_no, element.id)
        if element.kind == "table" and element.payload.get("requires_review"):
            issue("complex_table", "表格结构或合并单元格需要人工核对。", element.span.page_no, element.id)

    for parent in parsed.objects:
        if parent.kind == "figure" and parent.payload.get("requires_review"):
            child = next((chunk for chunk in chunks if chunk.metadata.get("parent_id") == parent.id), None)
            issue("figure_binding", "图注与图形区域绑定不确定，候选区域需人工核对；图形内容尚未理解。",
                  parent.spans[0].page_no, chunk=str(child.metadata["pdf_chunk_id"]) if child else None)

    for page_no in range(1, parsed.page_count + 1):
        elements = [element for element in parsed.elements if element.span.page_no == page_no]
        text_chars = sum(len(element.text.strip()) for element in elements if element.kind != "furniture")
        image_count = sum(element.kind == "image" for element in elements)
        chunk_count = sum(chunk.metadata.get("page_start", 0) <= page_no <= chunk.metadata.get("page_end", 0) for chunk in chunks)
        report.pages.append(PageQuality(page_no, text_chars, sum(element.kind == "paragraph" for element in elements),
                                        sum(element.kind == "table" for element in elements), image_count, chunk_count))
        if image_count:
            issue("images", "本页包含未理解的图片区域；附图内容不属于已解析文字。", page_no)
        if text_chars < 50:
            issue("low_text", "本页提取文字不足 50 字符，可能是附图页、空页或漏提取，请对照原页。", page_no)

    for warning in parsed.warnings:
        if "图片" in warning:
            continue  # 图片风险已按页生成，避免重复展示。
        match = re.search(r"第\s*(\d+)\s*页", warning)
        issue("parser_warning", warning, int(match.group(1)) if match else None)
    report.status = "blocked" if any(item.status == "blocked" for item in report.issues) else "review" if report.issues else "passed"
    return report
