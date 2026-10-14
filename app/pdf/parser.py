"""原生 PDF 解析适配器；支持来源定位及可选的 Tesseract 图片/扫描页 OCR。"""

import re
from hashlib import sha256
from statistics import median
from typing import cast

import pymupdf

from app.schemas.pdf import NativeImageInfo, NativeTextPage, ParsedPdf, PdfElement, SourceSpan
from .patent import PATENT_VERSION, extract_patent

PARSER_VERSION = f"native-v5:{PATENT_VERSION}:pymupdf-{pymupdf.VersionBind}"


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


def _ocr_blocks(page: pymupdf.Page, language: str, dpi: int, full: bool) -> list[tuple[tuple[float, float, float, float], str]]:
    """Read only Tesseract-created text blocks, keeping their page rectangles."""
    text_page = page.get_textpage_ocr(language=language, dpi=dpi, full=full)
    data = page.get_text("dict", textpage=text_page, sort=True)
    result = []
    for block in data["blocks"]:
        if block.get("type") != 0:
            continue
        text = "\n".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if "glyphlessfont" in span.get("font", "").lower() and span.get("text", "").strip()
        ).strip()
        if text:
            result.append((tuple(block["bbox"]), text))
    return result


def _deduplicate_image_infos(images: list[NativeImageInfo]) -> list[NativeImageInfo]:
    """Merge duplicate/partial raster placements whose smaller box is mostly covered."""
    groups: list[list[NativeImageInfo]] = []

    def bounds(group: list[NativeImageInfo]) -> tuple[float, float, float, float]:
        boxes = [tuple(float(value) for value in item["bbox"]) for item in group]
        return (min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes))

    for image in images:
        box = tuple(float(value) for value in image["bbox"])
        digest = image.get("digest")
        match = None
        for group in groups:
            area = max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])
            for member in group:
                member_box = tuple(float(value) for value in member["bbox"])
                left, top = max(box[0], member_box[0]), max(box[1], member_box[1])
                right, bottom = min(box[2], member_box[2]), min(box[3], member_box[3])
                intersection = max(0.0, right-left) * max(0.0, bottom-top)
                member_area = max(0.0, member_box[2]-member_box[0]) * max(0.0, member_box[3]-member_box[1])
                coverage = intersection / min(area, member_area) if min(area, member_area) else 0.0
                same_image = bool(digest and member.get("digest") == digest)
                if coverage >= (0.5 if same_image else 0.65):
                    match = group
                    break
            if match is not None:
                break
        if match is None:
            groups.append([image])
        else:
            match.append(image)

    merged: list[NativeImageInfo] = []
    for group in groups:
        # Preserve useful image properties from the largest underlying placement,
        # while rendering the union so clipped/masked fragments are not lost.
        representative = max(group, key=lambda item: (item["bbox"][2]-item["bbox"][0]) *
                             (item["bbox"][3]-item["bbox"][1]))
        merged.append({**representative, "bbox": bounds(group)})
    return merged


def _intersection_ratio(inner: pymupdf.Rect, outer: pymupdf.Rect) -> float:
    area = inner.get_area()
    return ((inner & outer).get_area() / area) if area else 0.0


def parse_pdf(source: str, content: bytes, *, ocr_images: bool = True,
              ocr_language: str = "chi_sim+eng", ocr_dpi: int = 200) -> ParsedPdf:
    """输出有类型和来源位置的元素；可选用 Tesseract 识别图片/扫描页文字。"""
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
        content_hash = sha256(content).hexdigest()
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
            images = _deduplicate_image_infos(cast(list[NativeImageInfo], page.get_image_info()))
            if not any(line.get("spans") for block in text_blocks for line in block["lines"]):
                if images or len(page.get_drawings()) > 100:
                    recognized = []
                    if ocr_images:
                        try:
                            recognized = _ocr_blocks(page, ocr_language, ocr_dpi, full=True)
                        except Exception as exc:
                            warnings.append(f"第 {page_no} 页 OCR 未可用：{type(exc).__name__}。请安装 Tesseract 及 {ocr_language} 语言包。")
                    text = "\n".join(value for _, value in recognized).strip()
                    if text:
                        image_element_id = f"p{page_no}_page_image"
                        image_id = sha256(f"{content_hash}:{image_element_id}".encode()).hexdigest()
                        elements.append(PdfElement(
                            image_element_id, "image", text,
                            SourceSpan(page_no, page.get_label() or "", (0.0, 0.0, 1.0, 1.0), image_element_id, "page"),
                            payload={"doc_type_kwd": "image", "layout_type": "page_image", "image_id": image_id,
                                     "ocr_text": text, "ocr_status": "recognized", "text_origin": "ocr",
                                     "properties": {"width": page.rect.width, "height": page.rect.height,
                                                    "ocr_engine": "tesseract", "ocr_language": ocr_language,
                                                    "ocr_dpi": ocr_dpi}},
                        ))
                        continue
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
                table_image_id = sha256(f"{content_hash}:{element_id}".encode()).hexdigest()
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
                             "structure": "basic_grid", "row_count": len(rows), "column_count": table.col_count,
                             "doc_type_kwd": "table", "layout_type": "table", "text_origin": "native",
                             "image_id": table_image_id,
                             "properties": {"row_count": len(rows), "column_count": table.col_count,
                                            "structure": "basic_grid"}},
                ))
                table_rects.append(pymupdf.Rect(table.bbox))
            image_rects = [pymupdf.Rect(image["bbox"]) for image in images]
            image_native_text = [""] * len(images)
            for block_index, block in enumerate(text_blocks):
                rect = pymupdf.Rect(block["bbox"])
                if any((rect & table_rect).get_area() >= rect.get_area() * 0.8 for table_rect in table_rects):
                    continue  # 表格文字已保存为结构表，避免再次索引相同数字。
                spans = [span for line in block["lines"] for span in line["spans"]]
                text = "\n".join("".join(span["text"] for span in line["spans"]) for line in block["lines"]).strip()
                if not text:
                    continue
                image_matches = [(_intersection_ratio(rect, image_rect), index)
                                 for index, image_rect in enumerate(image_rects)]
                image_matches = [match for match in image_matches if match[0] >= 0.65]
                if image_matches:
                    _, image_index = max(image_matches)
                    image_native_text[image_index] = (image_native_text[image_index] + "\n" + text).strip()
                    continue  # 图片中的可选文字归入图片元素，避免再次作为正文段落切片。
                if text.count("\ufffd") / len(text) > 0.02:
                    raise ValueError(f"第 {page_no} 页文字层乱码，需要 OCR 后再导入")
                font_size = max(float(span["size"]) for span in spans)
                sizes.extend(float(span["size"]) for span in spans if span["text"].strip())
                element_id = f"p{page_no}_text{block_index}"
                elements.append(PdfElement(element_id, "paragraph", text, _span(page, block["bbox"], element_id),
                                           font_size, payload={"doc_type_kwd": "text", "layout_type": "paragraph",
                                                              "text_origin": "native", "properties": {"font_size": font_size}}))
            image_texts = list(image_native_text)
            image_origins = ["native" if text else "none" for text in image_texts]
            image_status = ["not_needed" if text else "disabled" if not ocr_images else "no_text"
                            for text in image_texts]
            if images and ocr_images and any(not text for text in image_texts):
                # 资格取决于原生文字；收到第一个 OCR 块后仍需接收该图的后续块。
                ocr_targets = {index for index, text in enumerate(image_native_text) if not text}
                try:
                    recognized = _ocr_blocks(page, ocr_language, ocr_dpi, full=False)
                    for bbox, text in recognized:
                        text_rect = pymupdf.Rect(bbox)
                        matches = []
                        for image_index, image in enumerate(images):
                            overlap = text_rect & pymupdf.Rect(image["bbox"])
                            if not overlap.is_empty:
                                matches.append((overlap.get_area(), image_index))
                        matches = [match for match in matches if match[1] in ocr_targets]
                        if matches:
                            _, image_index = max(matches)
                            image_texts[image_index] = (image_texts[image_index] + "\n" + text).strip()
                            image_status[image_index] = "recognized"
                            image_origins[image_index] = "ocr"
                except Exception as exc:
                    image_status = ["unavailable" if not image_texts[index] else image_status[index]
                                    for index in range(len(images))]
                    warnings.append(f"第 {page_no} 页图片 OCR 未可用：{type(exc).__name__}。请安装 Tesseract 及 {ocr_language} 语言包。")
            for image_index, image in enumerate(images):
                element_id = f"p{page_no}_image{image_index}"
                image_id = sha256(f"{content_hash}:{element_id}".encode()).hexdigest()
                elements.append(PdfElement(
                    element_id, "image", image_texts[image_index], _span(page, image["bbox"], element_id),
                    payload={"doc_type_kwd": "image", "layout_type": "image", "image_id": image_id,
                             "ocr_text": image_texts[image_index] if image_origins[image_index] == "ocr" else "",
                             "ocr_status": image_status[image_index],
                             "text_origin": image_origins[image_index],
                             "properties": {"xref": image.get("xref"), "width": image.get("width"),
                                            "height": image.get("height"), "colorspace": image.get("cs-name")}},
                ))
            if images and not ocr_images:
                warnings.append(f"第 {page_no} 页包含图片，图片 OCR 已按配置关闭。")
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
            element.payload["layout_type"] = element.kind
        title = (document.metadata or {}).get("title", "").strip()
        title = title or next((element.text for element in elements if element.kind == "heading"), source)
        ocr_suffix = f":ocr-{ocr_language}-{ocr_dpi}" if ocr_images else ":ocr-off"
        parsed = ParsedPdf(source, content_hash, title, document.page_count,
                           PARSER_VERSION + ocr_suffix, elements, list(dict.fromkeys(warnings)))
        parsed.patent = extract_patent(parsed)
        if parsed.patent:
            parsed.warnings.extend(parsed.patent.warnings)
        return parsed
