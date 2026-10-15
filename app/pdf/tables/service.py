"""Select table backends and record actual attempts and fallback outcomes."""

from dataclasses import asdict
from hashlib import sha256
import json
import pymupdf

from app.schemas.inspection import RegionInspection
from ..geometry import source_span

from .base import TABLE_BACKEND_VERSION, TableBackendResult, TableOptions, to_element
from .docling import DOCLING_ADAPTER_VERSION, DoclingBackend, models_identity, package_version
from .native import NATIVE_TABLE_VERSION, native_inputs, parse_native_table


def backend_identity(options: TableOptions) -> str:
    values = {**asdict(options), "adapter": TABLE_BACKEND_VERSION, "native": NATIVE_TABLE_VERSION}
    if options.backend != "native":
        values.update(docling_version=package_version(), models_identity=models_identity(options.models_directory),
                      docling_adapter=DOCLING_ADAPTER_VERSION)
    return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def parse_tables(page, inspected, options: TableOptions, work_directory: str) -> TableBackendResult:
    result = TableBackendResult()
    try:
        result.fragments = [parse_native_table(page, table) for table in native_inputs(page, inspected)]
        result.attempts.append({"backend": "native", "status": "completed", "tables": len(result.fragments)})
    except Exception as exc:
        result.attempts.append({"backend": "native", "status": "failed", "reason": str(exc)})
        result.warnings.append(f"原生表格解析失败，原文保留供复核：{exc}")
    complex_native = any(fragment.structure_status != "complete" or any(
        (cell.row_span or 1) > 1 or (cell.col_span or 1) > 1 for cell in fragment.cells) for fragment in result.fragments)
    should_use_docling = options.backend == "docling" or (options.backend == "auto" and (
        complex_native or bool(inspected.images) or inspected.diagnosis.text_source == "unreliable_text_layer"
        or inspected.diagnosis.metrics["table_detection"] == "failed"))
    if not should_use_docling:
        return result
    try:
        external = DoclingBackend(options, work_directory).parse_page(page,
                   ocr=options.ocr_enabled and bool(inspected.images))
        # Validate compatibility projection before replacing any working native tables.
        for fragment in external.fragments:
            to_element(fragment, "validation")
        if not external.fragments and result.fragments:
            raise ValueError("Docling 未返回表格，不能覆盖已有原生表格")
        result.attempts.append({"backend": "docling", "status": "completed", "tables": len(external.fragments)})
        result.fragments = external.fragments
        result.warnings.extend(external.warnings)
    except Exception as exc:
        result.attempts.append({"backend": "docling", "status": "failed", "reason": str(exc)})
        if not options.fallback:
            raise ValueError(f"Docling 表格解析未完成，且已关闭回退：{exc}") from exc
        result.warnings.append(f"Docling 表格后端未完成，已回退原生解析：{exc}")
    return result


def update_regions(page, inspected, fragments):
    """Rebuild dispatch regions around chosen tables, including fallback text."""
    diagnosis = inspected.diagnosis
    regions = [region for region in diagnosis.regions if region.role not in {"text", "table"}]
    boxes = []
    for fragment in fragments:
        bbox = fragment.span.bbox
        displayed = pymupdf.Rect(bbox[0] * page.rect.width, bbox[1] * page.rect.height,
                                 bbox[2] * page.rect.width, bbox[3] * page.rect.height)
        raw = tuple(displayed * page.derotation_matrix)
        boxes.append(pymupdf.Rect(raw))
        origins = {cell.text_origin for cell in fragment.cells if cell.raw_text}
        source = "native" if origins == {"native"} else "scanned" if not diagnosis.metrics["native_char_count"] else "hybrid"
        regions.append(RegionInspection(fragment.id, fragment.span, raw, "table", source, "observed",
                       {"backend": fragment.backend, "row_count": fragment.row_count,
                        "column_count": fragment.column_count}, ["selected_table_backend_output"]))
    for index, block in enumerate(inspected.text_blocks):
        text = "\n".join("".join(span["text"] for span in line["spans"]) for line in block["lines"]).strip()
        rect = pymupdf.Rect(block["bbox"])
        if not text or not rect.get_area():
            continue
        if any((rect & table).get_area() / rect.get_area() >= diagnosis.policy["block_table_overlap"] for table in boxes):
            continue
        if any((rect & pymupdf.Rect(image["bbox"])).get_area() / rect.get_area() >= diagnosis.policy["block_image_overlap"]
               for image in inspected.images):
            continue
        identifier = f"p{diagnosis.page_no}_text{index}"
        bad = text.count("\ufffd") / len(text) > diagnosis.policy["replacement_ratio"]
        regions.append(RegionInspection(identifier, source_span(page, tuple(rect), identifier), tuple(rect),
                       "text", "unreliable_text_layer" if bad else "native", "candidate" if bad else "observed",
                       {"native_char_count": len(text)}, ["replacement_characters"] if bad else ["extractable_native_text"]))
    for region in regions:
        region.overlaps = [other.id for other in regions if other is not region and
                           (pymupdf.Rect(region.pdf_bbox) & pymupdf.Rect(other.pdf_bbox)).get_area() > 0]
    diagnosis.regions = regions
