"""原生 PDF 解析适配器；扫描页明确拒绝，后续可替换为 Docling/OCR。"""

import re
from hashlib import sha256
from statistics import median
from typing import cast

import pymupdf

from app.schemas.pdf import NativeImageInfo, NativeTextPage, ParsedPdf, PdfElement, SourceSpan
from .patent import PATENT_VERSION, extract_patent

PARSER_VERSION = f"native-v3:{PATENT_VERSION}:pymupdf-{pymupdf.VersionBind}"


def _span(
    page: pymupdf.Page,
    bbox: tuple[float, float, float, float],
    element_id: str,
    granularity: str = "region",
) -> SourceSpan:
    # 原生坐标先应用旋转矩阵，再归一化到阅读器显示页。
    rect = pymupdf.Rect(bbox) * page.rotation_matrix
    width, height = page.rect.width, page.rect.height
    normalized = [
        max(0.0, min(1.0, value))
        for value in (rect.x0 / width, rect.y0 / height, rect.x1 / width, rect.y1 / height)
    ]
    if page.number is None:
        raise ValueError("PDF 页面未关联文档，无法记录来源页码")
    return SourceSpan(page.number + 1, page.get_label() or "",
                      (normalized[0], normalized[1], normalized[2], normalized[3]), element_id, granularity)


def parse_pdf(source: str, content: bytes) -> ParsedPdf:
    """输出文字、普通表格、图片区域和来源；不把图片描述成已识别正文。"""
    if b"%PDF-" not in content[:1024]:
        raise ValueError("文件不是有效的 PDF，请检查文件内容")
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF 损坏或无法打开") from exc
    with document:
        if document.needs_pass:
            raise ValueError("PDF 已加密，请上传解密后的文件")
        if not 0 < document.page_count <= 500:
            raise ValueError("PDF 页数必须在 1 到 500 页之间")
        elements: list[PdfElement] = []
        warnings: list[str] = []
        sizes: list[float] = []
        toc = cast(list[tuple[int, str, int]], document.get_toc())
        bookmarks = {(page_no, title.strip()): level for level, title, page_no in toc}
        scanned: list[int] = []
        for page in document:
            if page.number is None:
                raise ValueError("PDF 页面未关联文档，无法记录来源页码")
            page_no = page.number + 1
            # get_text 的返回类型取决于模式，SDK 的联合类型无法推断固定的 dict 模式。
            text_page = cast(NativeTextPage, page.get_text("dict", sort=True,
                flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES))
            raw_blocks = text_page["blocks"]
            text_blocks = [block for block in raw_blocks if block["type"] == 0]
            images = cast(list[NativeImageInfo], page.get_image_info())
            if not any(line.get("spans") for block in text_blocks for line in block["lines"]):
                if images or len(page.get_drawings()) > 100:
                    scanned.append(page_no)
                continue
            # 保留网格和单元格来源；合并单元格等复杂结构标记待复核。
            tables = []
            try:
                finder = page.find_tables()
                if finder is not None:
                    tables = finder.tables
            except Exception:
                warnings.append(f"第 {page_no} 页表格检测失败，已保留原生文字，表格需复核")
            table_rects = []
            for table_index, table in enumerate(tables):
                rows = table.extract()
                if len(rows) < 2 or table.col_count < 2:
                    continue
                element_id = f"p{page_no}_table{table_index}"
                cells = []
                for row_index, row in enumerate(rows):
                    for column_index, value in enumerate(row):
                        bbox = table.rows[row_index].cells[column_index]
                        cell_id = f"{element_id}_r{row_index}c{column_index}"
                        cells.append({
                            "id": cell_id, "row": row_index, "column": column_index,
                            "raw_text": value, "source_page_no": page_no,
                            "fragment_id": element_id,
                            "span": vars(_span(page, bbox, cell_id, "cell")) if bbox else None,
                        })
                uncertain = any(cell["span"] is None for cell in cells)
                if uncertain:
                    warnings.append(f"第 {page_no} 页有复杂表格单元格，合并关系和位置需复核")
                elements.append(PdfElement(
                    element_id, "table", "\n".join(" | ".join(value or "" for value in row) for row in rows),
                    _span(page, table.bbox, element_id),
                    payload={"rows": rows, "cells": cells, "requires_review": uncertain,
                             "structure": "basic_grid", "row_count": len(rows)},
                ))
                table_rects.append(pymupdf.Rect(table.bbox))
            for block_index, block in enumerate(text_blocks):
                rect = pymupdf.Rect(block["bbox"])
                if any((rect & table_rect).get_area() >= rect.get_area() * 0.8 for table_rect in table_rects):
                    continue  # 表格文字已保存为结构表，避免再次索引相同数字。
                spans = [span for line in block["lines"] for span in line["spans"]]
                text = "\n".join("".join(span["text"] for span in line["spans"]) for line in block["lines"]).strip()
                if not text:
                    continue
                if text.count("\ufffd") / len(text) > 0.02:
                    raise ValueError(f"第 {page_no} 页文字层乱码，需要 OCR 后再导入")
                font_size = max(float(span["size"]) for span in spans)
                sizes.extend(float(span["size"]) for span in spans if span["text"].strip())
                element_id = f"p{page_no}_text{block_index}"
                elements.append(PdfElement(element_id, "paragraph", text, _span(page, block["bbox"], element_id), font_size))
            for image_index, image in enumerate(images):
                element_id = f"p{page_no}_image{image_index}"
                elements.append(PdfElement(element_id, "image", "", _span(page, image["bbox"], element_id)))
            if images:
                warnings.append(f"第 {page_no} 页包含图片，本版只解析原生文字与普通表格，未执行图片 OCR/理解")
            paragraphs = [item for item in elements if item.span.page_no == page_no and item.kind == "paragraph" and len(item.text) > 40]
            if any(left.span.bbox[2] < right.span.bbox[0] and
                   min(left.span.bbox[3], right.span.bbox[3]) > max(left.span.bbox[1], right.span.bbox[1])
                   for left in paragraphs for right in paragraphs if left is not right):
                warnings.append(f"第 {page_no} 页可能为多栏版面，阅读顺序需人工核对")
        if scanned:
            pages = "、".join(map(str, scanned[:20]))
            raise ValueError(f"PDF 第 {pages} 页没有可用文字层，需要 OCR；当前最小版不导入扫描正文页")
        if not any(element.text for element in elements):
            raise ValueError("PDF 没有可导入的原生文字，需要 OCR 或图像理解")

        # 简单原生版面基线：按页内块顺序排列，复杂双栏不宣称已可靠恢复。
        elements.sort(key=lambda item: (item.span.page_no, item.span.bbox[1], item.span.bbox[0]))
        body_size = median(sizes) if sizes else 11
        repeated: dict[str, set[int]] = {}
        for element in elements:
            if element.kind == "paragraph" and (element.span.bbox[3] < 0.08 or element.span.bbox[1] > 0.94):
                normalized = re.sub(r"\d+", "#", element.text)
                repeated.setdefault(normalized, set()).add(element.span.page_no)
        headings: list[tuple[int, str]] = []
        for element in elements:
            if element.kind != "paragraph":
                element.section_path = " / ".join(title for _, title in headings)
                continue
            normalized = re.sub(r"\d+", "#", element.text)
            if len(repeated.get(normalized, set())) >= max(3, document.page_count * 0.6):
                element.kind = "furniture"
                continue
            level = bookmarks.get((element.span.page_no, element.text), 0)
            if not level and len(element.text) <= 100 and element.font_size > body_size * 1.2:
                level = 1 if element.font_size > body_size * 1.5 else 2
            if level:
                element.kind = "heading"
                element.heading_level = level
                headings = [(depth, title) for depth, title in headings if depth < level]
                headings.append((level, element.text))
            element.section_path = " / ".join(title for _, title in headings)
        title = (document.metadata or {}).get("title", "").strip()
        title = title or next((element.text for element in elements if element.kind == "heading"), source)
        parsed = ParsedPdf(source, sha256(content).hexdigest(), title, document.page_count,
                           PARSER_VERSION, elements, list(dict.fromkeys(warnings)))
        parsed.patent = extract_patent(parsed)
        if parsed.patent:
            parsed.warnings.extend(parsed.patent.warnings)
        return parsed
