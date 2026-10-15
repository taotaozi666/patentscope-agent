"""从完整语义对象生成检索子块，保存父子关系及精确字符来源映射。"""

import json
import re
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256

from langchain_core.documents import Document

from app.schemas.pdf import ParsedPdf, SemanticObject
from .normalization import PDF_TEXT_NORMALIZATION_VERSION, normalize_pdf_text
from .objects import build_objects

CHUNKER_VERSION = "semantic-chunks-v13"


def build_chunks(parsed: ParsedPdf, size: int = 600, target: int = 0) -> list[Document]:
    """语义边界优先；size 为字符安全上限，target 为拆分时的目标预算。"""
    if size < 40 or not 0 <= target <= size:
        raise ValueError("PDF 分块上限至少为 40，目标预算必须介于 0 和上限之间")
    target = target or max(40, int(size * .8))
    objects = build_objects(parsed)
    from .tables.evidence import index_cells, supporting_cells
    table_cells = index_cells(asdict(c) for f in parsed.table_fragments for c in f.cells)
    run_id = sha256(f"{parsed.content_hash}:{parsed.source}:{parsed.parser_version}".encode()).hexdigest()
    output: list[Document] = []
    child_counts: dict[str, int] = defaultdict(int)

    def emit(parent: SemanticObject, text: str, mappings: list[dict], extra: dict | None = None) -> None:
        raw_text = text
        if parent.kind != "table":
            text, mappings = normalize_pdf_text(text, mappings)
        index = child_counts[parent.id]
        child_counts[parent.id] += 1
        context = f"文档：{parsed.title[:100]}；语义对象：{parent.title[:160]}。"
        if parsed.patent:
            if parsed.patent.publication_number:
                context += f"专利公开号：{parsed.patent.publication_number.value}。"
            if parsed.patent.title:
                context += f"专利发明名称：{parsed.patent.title.value[:160]}。"
        if parent.kind == "claim" and parent.payload["depends_on"]:
            context += "本项引用权利要求 " + "、".join(map(str, parent.payload["depends_on"])) + "，解释时需要对应基础条件。"
        if parent.kind == "table":
            context += "；".join([*parent.payload.get("units", []), *parent.payload.get("notes", [])])[:300]
            ids = {m.get("cell_id") for m in mappings}
            relations = {i: {**value, "support": supporting_cells(table_cells[i], table_cells)}
                         for i, value in parent.payload.get("cell_relations", {}).items() if i in ids}
            support = {s["cell_id"]: s["text"] for value in relations.values() for s in value["support"]}
            if support:
                context += "表格关联原文：" + "；".join(v[:160] for v in support.values() if v)[:700] + "。"
        if parent.kind == "figure":
            if parent.payload.get("ocr_status") == "recognized":
                context += "图内 OCR 文字已提取；图形关系和视觉语义仍需结合原图核对。"
            else:
                context += "图形内容尚未理解；以下是图号、图注和说明书中的相关原文。"
        spans = list({(item["span"]["page_no"], item["span"]["element_id"]): item["span"] for item in mappings if item.get("span")}.values())
        if parent.kind == "figure":
            spans = list({(span.page_no, span.element_id): asdict(span) for span in parent.spans}.values())
        if not spans:
            spans = [asdict(span) for span in parent.spans]
        identifier = sha256(f"{run_id}:{CHUNKER_VERSION}:{size}:{target}:{parent.id}:{index}:{text}".encode()).hexdigest()
        pages = [span["page_no"] for span in spans]
        section_path = parent.payload.get("section_path", parent.title)
        doc_type = str(parent.payload.get("doc_type_kwd") or ("image" if parent.kind == "figure" else "table" if parent.kind == "table" else "text"))
        layout_type = str(parent.payload.get("layout_type") or doc_type)
        text_origin = "ocr" if any(item.get("role") == "ocr" for item in mappings) else str(parent.payload.get("text_origin", "native"))
        element_attributes = {
            "doc_type_kwd": doc_type, "ck_type": doc_type, "layout_type": layout_type,
            "chunk_type": parent.kind, "section_path": section_path, "text_origin": text_origin,
            "page_start": min(pages), "page_end": max(pages),
            "source_spans": spans,
            **parent.payload.get("properties", {}),
        }
        metadata = {
            "source": parsed.source, "format": ".pdf", "chunk": len(output)+1,
            "document_id": parent.document_id, "document_title": parsed.title,
            "heading_path": section_path, "section_path": section_path,
            "section_title": section_path.rsplit(" / ", 1)[-1], "object_title": parent.title,
            "section_id": sha256(f"{run_id}:{section_path}".encode()).hexdigest(),
            "parent_id": parent.id, "object_id": parent.id, "child_index": index,
            "previous_chunk_id": "", "next_chunk_id": "",
            "page_start": min(pages), "page_end": max(pages),
            "pdf_version_id": parsed.content_hash, "pdf_parse_id": run_id, "pdf_chunk_id": identifier,
            "pdf_prechunked": True, "chunk_type": parent.kind,
            "doc_type_kwd": doc_type, "ck_type": doc_type, "layout_type": layout_type,
            "text_origin": text_origin,
            "attributes_json": json.dumps(element_attributes, ensure_ascii=False),
            "image_id": str(parent.payload.get("image_id") or ""),
            "ocr_status": str(parent.payload.get("ocr_status") or "not_applicable"),
            "pdf_source_spans": json.dumps(spans, ensure_ascii=False),
            "pdf_serialized_map": json.dumps(mappings, ensure_ascii=False),
            "pdf_context": context, "context_text": context, "context_method": "rule",
            "context_version": "context-v1", "pdf_parser": parsed.parser_version,
            "context_is_evidence": False,
            "context_source_spans": json.dumps(parent.payload.get("related_spans", []), ensure_ascii=False),
            "pdf_chunk_size": size, "pdf_target_size": target, "_pdf_index_text": context + "\n" + text,
            "pdf_chunker_version": CHUNKER_VERSION,
            "_pdf_raw_content": raw_text,
            "text_normalization_version": PDF_TEXT_NORMALIZATION_VERSION,
        }
        for name in ("table_id", "claim_number", "patent_section", "figure_number", "visual_status"):
            if name in parent.payload:
                metadata[name] = parent.payload[name]
        if parent.kind == "table" and "cell_relations" in parent.payload:
            metadata["table_cell_relations"] = json.dumps(relations, ensure_ascii=False)
            metadata["table_fragment_ids"] = json.dumps(parent.payload["fragment_ids"])
            metadata["table_header_status"] = parent.payload["header_status"]
            fragment_ids = list(dict.fromkeys(m["fragment_id"] for m in mappings
                if m.get("fragment_id") and m["role"] != "header" and m["end"] > m["start"]))
            regions = [parent.payload["fragment_regions"][i] for i in fragment_ids]
            metadata["table_region_sources"] = json.dumps(regions, ensure_ascii=False)
            if regions:
                metadata["image_id"] = regions[0]["image_id"]
        if parent.kind == "claim":
            metadata["depends_on"] = json.dumps(parent.payload["depends_on"])
            metadata["depends_on_parent_ids"] = json.dumps(parent.payload["dependency_object_ids"])
        if parent.kind == "figure":
            metadata["figure_id"] = parent.id
            metadata["figure_description"] = parent.payload.get("description_text", "")
            # 通道字段属于完整图片对象，正文切片及定位映射仍按原有边界保存。
            for field in ("native_text", "ocr_text", "caption_text", "description_text"):
                metadata["image_" + field] = str(parent.payload.get(field, ""))
        if parsed.patent and parsed.patent.publication_number:
            metadata["publication_number"] = parsed.patent.publication_number.value
        output.append(Document(page_content=text, metadata={**metadata, **(extra or {})}))

    for parent in objects:
        if parent.kind != "table":
            for start, end in _ranges(parent.raw_content, size, target):
                mappings = []
                for original in parent.mappings:
                    left, right = max(start, original["start"]), min(end, original["end"])
                    if right <= left:
                        continue
                    offset = original.get("source_char_start", 0) + left - original["start"]
                    mapping = {**original, "start": left-start, "end": right-start}
                    if "source_char_start" in original:
                        mapping["source_char_start"] = offset
                        mapping["source_char_end"] = offset + right-left
                    mappings.append(mapping)
                emit(parent, parent.raw_content[start:end], mappings)
            continue
        if "fragment_ids" in parent.payload:
            from .tables.chunks import table_chunks
            for text, mappings, extra in table_chunks(parent, size, target, _ranges):
                if text.strip():
                    emit(parent, text, mappings, extra)
            continue
        rows = parent.payload["rows"]
        cells = {(cell["row"], cell["column"]): cell for cell in parent.payload["cells"]}
        configured_header_rows = parent.payload.get("header_rows")
        # Historical objects retain their first-row candidate; new unknown tables
        # retain every row as data instead of silently losing the first record.
        header_rows: list[int] = (list(configured_header_rows) if configured_header_rows is not None
                                  else [0])
        header = [" / ".join(dict.fromkeys(rows[r][col] for r in header_rows if rows[r][col]))
                  for col in range(len(rows[0]))]
        pending: list[int] = []

        def table_text(indices: list[int]) -> tuple[str, list[dict]]:
            text = ""
            mappings: list[dict] = []
            for row_no in indices:
                if text:
                    text += "\n"
                for column, value in enumerate(rows[row_no]):
                    if column:
                        text += " | "
                    start = len(text)
                    text += value or ""
                    cell = cells[(row_no, column)]
                    mappings.append({"start": start, "end": len(text), "cell_id": cell["id"],
                                     "span": cell["span"], "role": "header" if row_no in header_rows else "source"})
            return text, mappings

        def flush() -> None:
            if pending:
                text, mappings = table_text([*header_rows, *pending])
                emit(parent, text, mappings)
                pending.clear()

        for row_no in range(len(rows)):
            if row_no in header_rows:
                continue
            single, _ = table_text([*header_rows, row_no])
            if len(single) > size:
                flush()
                # 超宽行按单元格生成子块；表头、行标识放入有来源关联的上下文。
                for column, value in enumerate(rows[row_no]):
                    if not value:
                        continue
                    cell = cells[(row_no, column)]
                    for start, end in _ranges(value, size, target):
                        emit(parent, value[start:end], [{"start": 0, "end": end-start, "cell_id": cell["id"],
                             "source_char_start": start, "source_char_end": end, "span": cell["span"], "role": "source"}],
                             {"table_row": row_no, "table_column": column, "table_header": header[column] or "",
                              "table_row_label": rows[row_no][0] or "", "table_split": "cell",
                              "context_table_cell_ids": json.dumps([*(cells[(r, column)]["id"] for r in header_rows), cells[(row_no, 0)]["id"]])})
                        chunk = output[-1]
                        addition = f"行标识：{(rows[row_no][0] or '未确定')[:150]}；列标题候选：{(header[column] or '未确定')[:150]}。"
                        chunk.metadata["context_text"] += addition
                        chunk.metadata["pdf_context"] = chunk.metadata["context_text"]
                        chunk.metadata["_pdf_index_text"] = chunk.metadata["context_text"] + "\n" + chunk.page_content
                continue
            combined, _ = table_text([*header_rows, *pending, row_no])
            if pending and len(combined) > target:
                flush()
            pending.append(row_no)
        flush()
        if header_rows and len(header_rows) == len(rows):
            text, mappings = table_text(header_rows)
            for start, end in _ranges(text, size, target):
                selected = [{**m, "start": max(m["start"], start)-start, "end": min(m["end"], end)-start}
                            for m in mappings if min(m["end"], end) > max(m["start"], start)]
                emit(parent, text[start:end], selected)

    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in output:
        grouped[doc.metadata["parent_id"]].append(doc)
    for siblings in grouped.values():
        for index, doc in enumerate(siblings):
            doc.metadata["previous_chunk_id"] = siblings[index-1].metadata["pdf_chunk_id"] if index else ""
            doc.metadata["next_chunk_id"] = siblings[index+1].metadata["pdf_chunk_id"] if index+1 < len(siblings) else ""
    if not output:
        raise ValueError("PDF 没有可导入的正文或表格")
    return output


def _ranges(text: str, maximum: int, target: int):
    """保留短语义单元；超限优先选段落、条款或句末边界，不添加字符重叠。"""
    start = 0
    while start < len(text):
        if len(text)-start <= maximum:
            yield start, len(text)
            break
        end = min(start + maximum, len(text))
        part = text[start:end]
        boundaries = list(re.finditer(r"\n\n|[；;。！？]|[.!?]\s", part))
        candidates = [match.end() for match in boundaries if match.end() >= target//2]
        if candidates:
            end = start + min(candidates, key=lambda value: abs(value-target))
        yield start, end
        start = end
