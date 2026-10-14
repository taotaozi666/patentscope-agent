"""检索候选去噪、同文档内容合并；不修改入库的原始切片。"""
import re
import json
from app.pdf.normalization import is_page_number_region
from app.patent_identifiers import normalize_publication_number
from app.schemas import Hit


def merge_candidates(ranked):
    merged = {}
    for record, score in ranked:
        # 历史索引中的纯公开号不能提供正文证据；公开号仍保留在文档元数据中。
        if normalize_publication_number(record.text) or not record.text.strip() or _page_number_noise(record):
            continue
        text = re.sub(r"\s+", " ", record.text).strip()
        # 图片 OCR 相同并不证明图片相同，保留各图入口；纯公开号已在上方过滤。
        image = record.metadata.get("image_id", "")
        key = (record.document_id, text, image)
        location = {"chunk": record.chunk, "page_start": record.metadata.get("page_start"),
                    "page_end": record.metadata.get("page_end"),
                    "section": record.metadata.get("section_path", "")}
        if key not in merged:
            metadata = dict(record.metadata)
            metadata["duplicate_locations"] = [location]
            merged[key] = Hit(record.source, record.text, score, record.chunk, metadata)
        elif location not in merged[key].metadata["duplicate_locations"]:
            merged[key].metadata["duplicate_locations"].append(location)
    return list(merged.values())


def _page_number_noise(record):
    """旧索引只有同时提供页底坐标和匹配页码时才过滤，缺少定位信息则保留。"""
    if record.metadata.get("doc_type_kwd") == "table":
        return False
    try:
        spans = json.loads(record.metadata.get("pdf_source_spans", "[]"))
        return bool(spans) and all(is_page_number_region(record.text, int(span["page_no"]), span["bbox"])
                                   for span in spans)
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
