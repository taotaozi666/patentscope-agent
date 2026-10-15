"""原生 PDF 解析适配器；支持来源定位及可选的 Tesseract 图片/扫描页 OCR。"""

import re
from hashlib import sha256
from statistics import median
from typing import cast

import pymupdf

from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan
from .patent import PATENT_VERSION, extract_patent
from .normalization import is_page_number_region
from .tables.base import TABLE_BACKEND_VERSION, TableOptions, to_element
from .tables.service import backend_identity, parse_tables, update_regions
from .tables.assembly import ASSEMBLY_VERSION, assemble_tables
from pathlib import Path
from app.schemas.table import TABLE_SCHEMA_VERSION
from .geometry import source_span as _span
from .inspection import INSPECTION_VERSION, inspect_page
from .inspection import _remove_overlapping_text as _remove_overlapping_text
from .routing import ROUTING_VERSION, route_regions, record_execution
from .ocr import OCR_VERSION, OcrResult, ocr_region
from .text_fusion import FUSION_VERSION, TextEvidence, fuse_text, serialize_fusion
from dataclasses import asdict
from app.schemas.inspection import PageInspection

PARSER_VERSION = (f"native-v15:{ASSEMBLY_VERSION}:{TABLE_BACKEND_VERSION}:table-schema-v{TABLE_SCHEMA_VERSION}:{INSPECTION_VERSION}:"
                  f"{ROUTING_VERSION}:{OCR_VERSION}:{FUSION_VERSION}:{PATENT_VERSION}:pymupdf-{pymupdf.VersionBind}")


class PdfParseError(ValueError):
    """解析拒绝时保留诊断，调用方可记录失败页面的处理依据。"""

    def __init__(self, message: str, page_inspections: list[PageInspection],
                 page_count: int, parser_version: str):
        super().__init__(message)
        self.page_inspections = page_inspections
        self.page_count = page_count
        self.parser_version = parser_version


def _native_evidence(page, block, prefix):
    evidence = []
    for index, line in enumerate(block["lines"]):
        text = "".join(span["text"] for span in line["spans"]).strip()
        if text:
            bbox = tuple(line.get("bbox", block["bbox"]))
            identifier = f"{prefix}_native{index}"
            evidence.append(TextEvidence(identifier, text, bbox,
                            _span(page, bbox, identifier, "line"), "native"))
    return evidence


def _ocr_evidence(page, result: OcrResult, prefix: str, bbox):
    evidence = []
    bounds = pymupdf.Rect(bbox)
    for index, (box, text) in enumerate(result.blocks):
        rect = pymupdf.Rect(box)
        if not text.strip() or not rect.get_area() or (rect & bounds).get_area() < rect.get_area() * .65:
            continue
        identifier = f"{prefix}_ocr{index}"
        evidence.append(TextEvidence(identifier, text.strip(), box,
                        _span(page, box, identifier, "line"), "ocr"))
    return evidence


def _intersection_ratio(inner: pymupdf.Rect, outer: pymupdf.Rect) -> float:
    area = inner.get_area()
    return ((inner & outer).get_area() / area) if area else 0.0


def parse_pdf(source: str, content: bytes, *, ocr_images: bool = True,
              ocr_language: str = "chi_sim+eng", ocr_dpi: int = 200,
              table_backend: str = "native", table_mode: str = "accurate",
              table_models_directory: str = "", table_fallback: bool = True,
              table_work_directory: str | None = None) -> ParsedPdf:
    """输出有类型和来源位置的元素；可选用 Tesseract 识别图片/扫描页文字。"""
    options = TableOptions(table_backend, table_models_directory, table_mode, table_fallback, ocr_images, ocr_language)
    table_identity = backend_identity(options)
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
        ocr_suffix = f":ocr-{ocr_language}-{ocr_dpi}" if ocr_images else ":ocr-off"
        parser_version = PARSER_VERSION + f":tables-{table_identity}" + ocr_suffix
        sizes: list[float] = []
        toc = cast(list[tuple[int, str, int]], document.get_toc())
        bookmarks = {(page_no, title.strip()): level for level, title, page_no in toc}
        scanned: list[int] = []
        diagnoses: list[PageInspection] = []
        table_fragments = []
        for page in document:
            if page.number is None:
                raise ValueError("PDF 页面未关联文档，无法记录来源页码")
            page_no = page.number + 1
            inspected = inspect_page(page)
            diagnosis = inspected.diagnosis
            diagnoses.append(diagnosis)
            diagnosis.routes = route_regions(diagnosis, ocr_enabled=ocr_images)
            try:
                table_result = parse_tables(page, inspected, options,
                                            table_work_directory or str(Path.cwd() / ".documents" / "table-work"))
            except ValueError as exc:
                diagnosis.metrics["table_backends"] = [{"backend": options.backend, "status": "failed", "reason": str(exc)}]
                diagnosis.warnings.append(str(exc))
                raise PdfParseError(str(exc), diagnoses, document.page_count, parser_version) from exc
            table_fragments.extend(table_result.fragments)
            diagnosis.metrics["table_backends"] = table_result.attempts
            warnings.extend(table_result.warnings)
            diagnosis.warnings.extend(table_result.warnings)
            update_regions(page, inspected, table_result.fragments)
            diagnosis.routes = route_regions(diagnosis, ocr_enabled=ocr_images)
            warnings.extend(diagnosis.warnings)
            text_blocks, images = inspected.text_blocks, inspected.images
            table_rects, reliable_table_rects = [], []
            for fragment in table_result.fragments:
                table_element = to_element(fragment, content_hash)
                elements.append(table_element)
                raw = pymupdf.Rect(fragment.span.bbox[0] * page.rect.width,
                                   fragment.span.bbox[1] * page.rect.height,
                                   fragment.span.bbox[2] * page.rect.width,
                                   fragment.span.bbox[3] * page.rect.height) * page.derotation_matrix
                table_rects.append(raw)
                if fragment.structure_status == fragment.provenance_status == "complete":
                    reliable_table_rects.append(raw)
            if not diagnosis.metrics["native_char_count"]:
                if any("page_ocr" in route.planned_operations for route in diagnosis.routes):
                    result = OcrResult()
                    if ocr_images:
                        try:
                            result = ocr_region(page, tuple(page.rect * page.derotation_matrix), ocr_language, ocr_dpi,
                                                exclude_regions=[tuple(rect) for rect in reliable_table_rects])
                        except Exception as exc:
                            for route in diagnosis.routes:
                                if "page_ocr" in route.planned_operations:
                                    route.failed_operations = ["page_ocr"]
                            warnings.append(f"第 {page_no} 页整页 OCR 失败，未能提取扫描文字。{_ocr_failure_reason(exc, ocr_language)}")
                            diagnosis.warnings.append(warnings[-1])
                    evidence = _ocr_evidence(page, result, f"p{page_no}_scan", tuple(page.rect * page.derotation_matrix))
                    evidence = [item for item in evidence if not any(
                        _intersection_ratio(pymupdf.Rect(item.bbox), rect) >= .8 for rect in table_rects)]
                    fusion = fuse_text([], evidence)
                    if fusion.selected or table_result.fragments:
                        image_element_id = f"p{page_no}_page_image"
                        image_id = sha256(f"{content_hash}:{image_element_id}".encode()).hexdigest()
                        # 背景图保留 OCR 原始通道；仅独立段落进入索引，避免重复。
                        elements.append(PdfElement(
                            image_element_id, "image", "",
                            SourceSpan(page_no, page.get_label() or "", (0.0, 0.0, 1.0, 1.0), image_element_id, "page"),
                            payload={"doc_type_kwd": "image", "layout_type": "page_image", "image_id": image_id,
                                     "native_text": "", "ocr_text": "\n".join(text for _, text in result.blocks),
                                     "ocr_status": "recognized" if evidence else "unavailable" if any(
                                         route.failed_operations for route in diagnosis.routes) else "no_text" if ocr_images else "disabled",
                                     "ocr_scope": "page", "text_origin": "ocr",
                                     "ocr_transform": result.transform, "ocr_blocks": [asdict(item) for item in evidence],
                                     "properties": {"width": page.rect.width, "height": page.rect.height,
                                                    "ocr_engine": "tesseract", "ocr_language": ocr_language, "ocr_dpi": ocr_dpi}},
                        ))
                        for index, item in enumerate(fusion.selected):
                            element_id = f"p{page_no}_scan_text{index}"
                            elements.append(PdfElement(element_id, "paragraph", item.text,
                                _span(page, item.bbox, element_id, "line"),
                                payload={"doc_type_kwd": "text", "layout_type": "paragraph",
                                         "text_origin": "ocr", "ocr_status": "recognized", "ocr_scope": "page",
                                         "image_id": image_id, "scan_region_id": image_element_id}))
                        continue
                    scanned.append(page_no)
                continue
            image_rects = [pymupdf.Rect(image["bbox"]) for image in images]
            image_native_text = [""] * len(images)
            image_native_evidence = [[] for _ in images]
            page_indexed_evidence = []
            for block_index, block in enumerate(text_blocks):
                rect = pymupdf.Rect(block["bbox"])
                if any((rect & table_rect).get_area() >= rect.get_area() * diagnosis.policy["block_table_overlap"] for table_rect in table_rects):
                    continue  # 表格文字已保存为结构表，避免再次索引相同数字。
                spans = [span for line in block["lines"] for span in line["spans"]]
                text = "\n".join("".join(span["text"] for span in line["spans"]) for line in block["lines"]).strip()
                if not text:
                    continue
                image_matches = [(_intersection_ratio(rect, image_rect), index)
                                 for index, image_rect in enumerate(image_rects)]
                image_matches = [match for match in image_matches if match[0] >= diagnosis.policy["block_image_overlap"]]
                if image_matches:
                    _, image_index = max(image_matches)
                    image_native_text[image_index] = (image_native_text[image_index] + "\n" + text).strip()
                    image_native_evidence[image_index].extend(_native_evidence(page, block, f"p{page_no}_b{block_index}"))
                    continue  # 图片中的可选文字归入图片元素，避免再次作为正文段落切片。
                if text.count("\ufffd") / len(text) > diagnosis.policy["replacement_ratio"]:
                    if not ocr_images:
                        raise PdfParseError(f"第 {page_no} 页文字层乱码，需要 OCR 后再导入", diagnoses,
                                            document.page_count, parser_version)
                    element_id = f"p{page_no}_text{block_index}"
                    try:
                        x0, y0, x1, y1 = block["bbox"]
                        result = ocr_region(page, (x0, y0, x1, y1), ocr_language, ocr_dpi)
                        fusion = fuse_text(_native_evidence(page, block, element_id),
                                           _ocr_evidence(page, result, element_id, block["bbox"]))
                        repaired, fragments = serialize_fusion(fusion)
                        if not repaired or "\ufffd" in repaired:
                            raise ValueError("OCR 未恢复异常文字")
                    except Exception as exc:
                        for route in diagnosis.routes:
                            if route.region_id == element_id:
                                route.failed_operations = ["region_ocr"]
                        warnings.append(f"第 {page_no} 页异常文字区域 OCR 未完成。{_ocr_failure_reason(exc, ocr_language)}")
                        diagnosis.warnings.append(warnings[-1])
                        raise PdfParseError(f"第 {page_no} 页文字层乱码，区域 OCR 未能恢复，需要复核后再导入",
                                            diagnoses, document.page_count, parser_version) from exc
                    elements.append(PdfElement(element_id, "paragraph", repaired, _span(page, block["bbox"], element_id),
                        payload={"doc_type_kwd": "text", "layout_type": "paragraph", "text_origin": "ocr",
                                 "native_text": text, "ocr_text": "\n".join(value for _, value in result.blocks),
                                 "ocr_status": "recognized", "ocr_scope": "region", "text_fragments": fragments,
                                 "ocr_transform": result.transform, "text_conflicts": fusion.conflicts}))
                    warnings.append(f"第 {page_no} 页异常文字已使用区域 OCR 恢复，原生文字和备选结果已保留，请复核。")
                    page_indexed_evidence.extend(fusion.selected)
                    continue
                font_size = max(float(span["size"]) for span in spans)
                sizes.extend(float(span["size"]) for span in spans if span["text"].strip())
                element_id = f"p{page_no}_text{block_index}"
                elements.append(PdfElement(element_id, "paragraph", text, _span(page, block["bbox"], element_id),
                                           font_size, payload={"doc_type_kwd": "text", "layout_type": "paragraph",
                                                              "text_origin": "native", "properties": {"font_size": font_size}}))
                page_indexed_evidence.extend(_native_evidence(page, block, element_id))
            existing_evidence = list(page_indexed_evidence)
            ocr_region_ids = {route.region_id for route in diagnosis.routes
                              if "region_ocr" in route.planned_operations
                              and "region_ocr" not in route.disabled_operations}
            for image_index, image in enumerate(images):
                element_id = f"p{page_no}_image{image_index}"
                image_id = sha256(f"{content_hash}:{element_id}".encode()).hexdigest()
                result = OcrResult()
                native = image_native_evidence[image_index]
                status = "not_needed" if native else "disabled" if not ocr_images else "no_text"
                if element_id in ocr_region_ids:
                    try:
                        x0, y0, x1, y1 = image["bbox"]
                        result = ocr_region(page, (x0, y0, x1, y1), ocr_language, ocr_dpi,
                                            exclude_regions=[tuple(rect) for rect in reliable_table_rects])
                        status = "recognized" if result.blocks else "no_text"
                    except Exception as exc:
                        status = "unavailable"
                        warnings.append(f"第 {page_no} 页图片区域 {image_index + 1} OCR 失败，已有原生文字仍保留。{_ocr_failure_reason(exc, ocr_language)}")
                crop_evidence = _ocr_evidence(page, result, element_id, image["bbox"])
                if status == "recognized" and not crop_evidence:
                    status = "no_text"
                # 同一页面的可靠原生表格已经有独立索引，不再次作为图片文字入库。
                evidence = [item for item in crop_evidence if not any(
                    _intersection_ratio(pymupdf.Rect(item.bbox), rect) >= .8 for rect in table_rects)]
                fusion = fuse_text(native, evidence, existing=existing_evidence)
                image_text, fragments = serialize_fusion(fusion)
                existing_evidence.extend(fusion.selected)
                if fusion.conflicts:
                    warnings.append(f"第 {page_no} 页图片区域 {image_index + 1} 原生文字与 OCR 存在冲突，已保留备选来源，请复核。")
                ocr_text = "\n".join(item.text for item in crop_evidence)
                origin = "mixed" if native and ocr_text else "native" if native else "ocr" if ocr_text else "none"
                elements.append(PdfElement(
                    element_id, "image", image_text, _span(page, image["bbox"], element_id),
                    payload={"doc_type_kwd": "image", "layout_type": "image", "image_id": image_id,
                             "native_text": image_native_text[image_index], "ocr_text": ocr_text,
                             "ocr_blocks": [asdict(item) for item in evidence], "native_blocks": [asdict(item) for item in native],
                             "text_fragments": fragments, "text_conflicts": fusion.conflicts,
                             "text_duplicates": fusion.duplicates, "ocr_transform": result.transform,
                             "ocr_status": status, "ocr_scope": "region" if element_id in ocr_region_ids else "none",
                             "text_origin": origin,
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
            raise PdfParseError(f"PDF 第 {pages} 页没有可用文字层，需要 OCR；当前最小版不导入扫描正文页", diagnoses,
                                document.page_count, parser_version)
        if not any(element.text for element in elements):
            raise PdfParseError("PDF 没有可导入的原生文字，需要 OCR 或图像理解", diagnoses,
                                document.page_count, parser_version)

        # 简单原生版面基线：按页内块顺序排列，复杂双栏不宣称已可靠恢复。
        elements.sort(key=lambda item: (item.span.page_no, item.span.bbox[1], item.span.bbox[0]))
        body_size = median(sizes) if sizes else 11
        _mark_repeated_furniture(elements, document.page_count)
        headings: list[tuple[int, str]] = []
        for element in elements:
            if element.kind != "paragraph":
                element.section_path = " / ".join(title for _, title in headings)
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
        parsed = ParsedPdf(source, content_hash, title, document.page_count,
                           parser_version, elements, list(dict.fromkeys(warnings)),
                           page_inspections=diagnoses)
        parsed.patent = extract_patent(parsed)
        if parsed.patent:
            parsed.warnings.extend(parsed.patent.warnings)
        for diagnosis in diagnoses:
            record_execution(diagnosis, [element for element in elements if element.span.page_no == diagnosis.page_no])
        by_element = {element.id: element for element in elements}
        for fragment in table_fragments:
            fragment.section_path = by_element[fragment.id].section_path
        parsed.table_fragments = table_fragments
        assemble_tables(parsed)
        return parsed


def _mark_repeated_furniture(elements: list[PdfElement], page_count: int) -> None:
    """只过滤页边缘短区域，不因正文与页眉同文而删除正文或整页扫描图。"""
    page_numbers = [element for element in elements
                    if element.kind in {"paragraph", "image"}
                    and is_page_number_region(element.text, element.span.page_no, element.span.bbox)]
    for element in page_numbers:
        # 至少三个页面在相同位置出现对应页码，避免孤立正文数字被误判。
        matches = {other.span.page_no for other in page_numbers
                   if all(abs(a - b) < .025 for a, b in zip(element.span.bbox, other.span.bbox))}
        if len(matches) >= 3:
            element.payload["original_kind"] = element.kind
            element.payload["layout_type"] = "furniture"
            element.payload["furniture_type"] = "page_number"
            element.kind = "furniture"
    groups: dict[str, list[PdfElement]] = {}
    for element in elements:
        if element.kind not in {"paragraph", "image"}:
            continue
        if not (element.span.bbox[3] < 0.12 or element.span.bbox[1] > 0.92):
            continue
        text = re.sub(r"\s+", "", element.text)
        if not text or len(text) > 150:
            continue
        normalized = re.sub(r"\d+", "#", text)
        groups.setdefault(normalized, []).append(element)
    for members in groups.values():
        pages = {element.span.page_no for element in members}
        if len(pages) < max(3, page_count * 0.6):
            continue
        for element in members:
            element.payload["original_kind"] = element.kind
            element.payload["layout_type"] = "furniture"
            element.kind = "furniture"


def _ocr_failure_reason(exc: Exception, language: str) -> str:
    detail = str(exc).lower()
    if any(token in detail for token in ("tessdata", "tesseract is not installed", "failed loading language", "traineddata")):
        return f"OCR 环境或语言数据不可用，请检查 Tesseract、{language} 语言包及 TESSDATA_PREFIX 配置。"
    return f"OCR 执行失败（{type(exc).__name__}），需检查引擎或图片处理情况，不能据此判断缺少安装。"
