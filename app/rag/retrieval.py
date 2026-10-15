"""检索候选去噪、同文档内容合并；不修改入库的原始切片。"""
import re
import json
from dataclasses import replace
from app.pdf.normalization import is_page_number_region
from app.patent_identifiers import normalize_publication_number
from app.schemas import Hit, ChunkRecord


def fuse_rankings(vector: list[tuple[ChunkRecord, float]], keyword: list[tuple[ChunkRecord, float]],
                  *, constant: int = 60, vector_weight: float = 1.0,
                  keyword_weight: float = 1.0) -> list[tuple[ChunkRecord, float]]:
    """RRF 按稳定切片 ID 融合，两路原始分数仅用于诊断。"""
    records: dict[str, ChunkRecord] = {}
    scores: dict[str, float] = {}
    channels: dict[str, dict[str, dict[str, int | float]]] = {}
    for name, ranked, weight in (("vector", vector, vector_weight), ("keyword", keyword, keyword_weight)):
        seen: set[str] = set()
        for rank, (record, score) in enumerate(ranked, 1):
            if record.id in seen:
                continue
            seen.add(record.id)
            records.setdefault(record.id, record)
            scores[record.id] = scores.get(record.id, 0.0) + weight / (constant + rank)
            channels.setdefault(record.id, {})[name] = {"rank": rank, "score": score}
    result = [(replace(record, metadata={**record.metadata, "retrieval_method": "hybrid_rrf",
                                        "retrieval_channels": channels[identifier]}), scores[identifier])
              for identifier, record in records.items()]
    return sorted(result, key=lambda item: (-item[1], item[0].id))


def merge_candidates(ranked):
    merged = {}
    for record, score in ranked:
        # 历史索引中的纯公开号不能提供正文证据；公开号仍保留在文档元数据中。
        if normalize_publication_number(record.text) or not record.text.strip() or _page_number_noise(record):
            continue
        text = re.sub(r"\s+", " ", record.text).strip()
        # 图片 OCR 相同并不证明图片相同，保留各图入口；纯公开号已在上方过滤。
        image = record.metadata.get("image_id", "")
        # Equal table text may identify different experiments or physical cells.
        table_identity = (record.metadata.get("pdf_chunk_id") or record.id) if record.metadata.get("doc_type_kwd") == "table" else ""
        key = (record.document_id, text, image, table_identity)
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
