"""构建章节、权利要求、完整表格和附图的语义父对象。"""

import re
from dataclasses import asdict
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from app.schemas.pdf import ParsedPdf, PdfElement, SemanticObject, SourceSpan

OBJECT_VERSION = "semantic-objects-v2"


def build_objects(parsed: ParsedPdf) -> list[SemanticObject]:
    source_key = parsed.source.replace("\\", "/").casefold()
    document_id = uuid5(NAMESPACE_URL, f"pdf:{source_key}:{parsed.content_hash}").hex
    run = f"{parsed.content_hash}:{parsed.source}:{parsed.parser_version}"
    result: list[SemanticObject] = []
    by_id = {element.id: element for element in parsed.elements}
    consumed: set[str] = set()
    consumed_images: set[str] = set()

    def add(kind: str, title: str, text: str, spans: list[SourceSpan], mappings: list[dict], payload: dict) -> None:
        identifier = sha256(f"{run}:{OBJECT_VERSION}:{kind}:{len(result)}:{text}".encode()).hexdigest()
        result.append(SemanticObject(identifier, document_id, kind, title, text, spans, mappings, payload))

    if parsed.patent:
        for claim in parsed.patent.claims:
            mappings = []
            for fragment in claim.source_fragments:
                element = by_id[fragment["element_id"]]
                mappings.append({**fragment, "span": asdict(element.span), "role": "source"})
                consumed.add(element.id)
            if not mappings:
                # 旧记录没有字符区间时，不猜测逐字关系，仍可保留完整权利要求。
                continue
            add("claim", f"权利要求 / 第 {claim.number} 项", claim.text, claim.spans, mappings,
                {"claim_number": claim.number, "depends_on": claim.depends_on, "patent_section": "claims"})

    # 图注是图对象的原文证据；不将图片内容伪装成已经识别的文字。
    captions = [(element, match[1]) for element in parsed.elements if element.kind == "paragraph"
                and element.payload.get("patent_section") != "drawing_description" and len(element.text) < 150
                and (match := re.match(r"^\s*图\s*(\d+(?:[a-zA-Z])?)(?![0-9a-zA-Z])", element.text))]
    for caption, number in captions:
        candidates = [element for element in parsed.elements if element.kind == "image"
                      and element.id not in consumed_images
                      and element.span.page_no == caption.span.page_no
                      and element.span.bbox[2] - element.span.bbox[0] > .12
                      and element.span.bbox[3] - element.span.bbox[1] > .08
                      and min(element.span.bbox[2], caption.span.bbox[2]) > max(element.span.bbox[0], caption.span.bbox[0])
                      and 0 <= caption.span.bbox[1] - element.span.bbox[3] <= .25]
        candidates.sort(key=lambda element: caption.span.bbox[1] - element.span.bbox[3])
        spans = [caption.span]
        if candidates:
            spans.insert(0, candidates[0].span)
        elif caption.payload.get("patent_section") == "drawings":
            # 矢量附图没有内嵌图片对象，以图注上方区域作为候选，明确标记待复核。
            previous = [item.span.bbox[3] for item, _ in captions if item.span.page_no == caption.span.page_no
                        and item.span.bbox[3] < caption.span.bbox[1]]
            top = max(previous, default=.12)
            if caption.span.bbox[1] > top + .03:
                spans.insert(0, SourceSpan(caption.span.page_no, caption.span.page_label,
                                          (.08, top, .94, caption.span.bbox[1]), f"figure_region_{caption.id}"))
        descriptions = [element for element in parsed.elements if element.payload.get("patent_section") == "drawing_description"
                        and number in re.findall(r"图\s*(\d+(?:[a-zA-Z])?)", element.text)]
        image = candidates[0] if candidates else None
        evidence = ([image] if image and image.text.strip() else []) + [caption, *descriptions]
        text, mappings = _join(evidence)
        spans.extend(element.span for element in descriptions)
        add("figure", f"图{number}", text, spans, mappings,
            {"figure_number": number, "image_element_ids": [element.id for element in candidates[:1]],
             "image_id": image.payload.get("image_id") if image else sha256(f"{parsed.content_hash}:figure:{caption.id}".encode()).hexdigest(),
             "caption_id": caption.id, "description_ids": [element.id for element in descriptions],
             "visual_status": "not_analyzed", "requires_review": len(candidates) != 1,
             "region_method": "image_caption" if candidates else "caption_candidate",
             "section_path": caption.section_path, "doc_type_kwd": "image", "layout_type": "figure",
             "ocr_text": image.payload.get("ocr_text", "") if image else "",
             "ocr_status": image.payload.get("ocr_status", "not_applicable") if image else "not_applicable",
             "text_origin": image.payload.get("text_origin", "native") if image and image.text.strip() else "native",
             "properties": {"figure_number": number, "caption": caption.text,
                            "image_id": image.payload.get("image_id") if image else sha256(f"{parsed.content_hash}:figure:{caption.id}".encode()).hexdigest(),
                            "has_ocr_text": bool(image and image.payload.get("text_origin") == "ocr" and image.text.strip()),
                            "has_embedded_text": bool(image and image.text.strip())}})
        if image:
            consumed_images.add(image.id)
        consumed.add(caption.id)

    # OCR text inside an uncaptioned image remains retrievable as an image-typed object.
    for image in parsed.elements:
        if image.kind != "image" or image.id in consumed_images or not image.text.strip():
            continue
        text, mappings = _join([image])
        add("figure", f"图片区域 / 第 {image.span.page_no} 页", text, [image.span], mappings,
            {"image_element_ids": [image.id], "image_id": image.payload.get("image_id"),
             "section_path": image.section_path, "doc_type_kwd": "image", "layout_type": "image",
             "ocr_text": image.payload.get("ocr_text", ""),
             "ocr_status": image.payload.get("ocr_status", "unknown"),
             "text_origin": image.payload.get("text_origin", "none"),
             "properties": {**image.payload.get("properties", {}),
                            "has_ocr_text": image.payload.get("text_origin") == "ocr",
                            "has_embedded_text": bool(image.text.strip())}})

    pending: list[PdfElement] = []

    def flush() -> None:
        if pending:
            text, mappings = _join(pending)
            add("text", pending[0].section_path or "未确定章节", text, [element.span for element in pending], mappings,
                {"patent_section": pending[0].payload.get("patent_section", ""), "element_ids": [element.id for element in pending],
                 "doc_type_kwd": "text", "layout_type": "text", "properties": {"section_path": pending[0].section_path}})
            pending.clear()

    for index, element in enumerate(parsed.elements):
        if element.id in consumed or element.kind == "image":
            continue
        if element.kind == "furniture":
            continue  # 页眉页脚不打断跨页正文。
        if element.kind == "heading":
            flush()
            continue
        if element.kind == "table":
            flush()
            neighbors = [item for item in parsed.elements[max(0, index-4):index+5]
                         if item.kind == "paragraph" and item.span.page_no == element.span.page_no
                         and item.section_path == element.section_path
                         and min(abs(item.span.bbox[3] - element.span.bbox[1]), abs(item.span.bbox[1] - element.span.bbox[3])) < .08]
            titles = [item for item in neighbors if re.match(r"^\s*(?:表|Table)\s*\d+", item.text, re.I)]
            units = [item for item in neighbors if re.match(r"^\s*单位\s*[:：]", item.text)]
            notes = [item for item in neighbors if re.match(r"^\s*(?:注|备注)\s*[:：]", item.text)]
            raw_table = "\n".join(" | ".join(value or "" for value in row) for row in element.payload["rows"])
            add("table", titles[0].text if titles else element.section_path or "表格", raw_table, [element.span], [],
                {**element.payload, "table_id": element.id, "section_path": element.section_path,
                 "doc_type_kwd": "table", "layout_type": "table",
                 "caption": titles[0].text if titles else "", "units": [item.text for item in units],
                 "notes": [item.text for item in notes], "related_spans": [asdict(item.span) for item in [*titles, *units, *notes]],
                 "header_status": "candidate_first_row"})
            continue
        if pending and pending[-1].section_path != element.section_path:
            flush()
        pending.append(element)
    flush()
    result.sort(key=lambda item: (min(span.page_no for span in item.spans), item.spans[0].bbox[1]))
    claim_ids = {item.payload["claim_number"]: item.id for item in result if item.kind == "claim"}
    for item in result:
        if item.kind == "claim":
            item.payload["dependency_object_ids"] = [claim_ids[number] for number in item.payload["depends_on"] if number in claim_ids]
    parsed.objects = result
    return result


def _join(elements: list[PdfElement]) -> tuple[str, list[dict]]:
    text = ""
    mappings = []
    for element in elements:
        if text:
            text += "\n\n"
        start = len(text)
        text += element.text
        is_ocr = element.payload.get("text_origin") == "ocr"
        mapping = {"start": start, "end": len(text), "element_id": element.id,
                   "span": asdict(element.span), "role": "ocr" if is_ocr else "source"}
        if not is_ocr:
            mapping.update(source_char_start=0, source_char_end=len(element.text))
        mappings.append(mapping)
    return text, mappings
