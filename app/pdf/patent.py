"""中文原生专利的规则抽取；保留区域证据，不宣称适配所有版式。"""

import re
from collections import defaultdict

from app.schemas.patent import PatentClaim, PatentField, PatentSection, PatentStructure
from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan

PATENT_VERSION = "cn-patent-v2"
SECTION_NAMES = {"abstract": "摘要", "claims": "权利要求", "description": "说明书",
                 "drawing_description": "附图说明", "drawings": "说明书附图", "cover": "著录信息"}
SUBHEADINGS = {"技术领域", "背景技术", "发明内容", "实用新型内容", "附图说明", "具体实施方式"}


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _unique_spans(elements: list[PdfElement]) -> list[SourceSpan]:
    return list({element.id: element.span for element in elements}.values())


def _dependencies(text: str) -> list[int]:
    result: set[int] = set()
    for match in re.finditer(r"权利要求\s*([\d\s、,，至到\-—或及和]+)", text):
        value = match.group(1)
        for start, end in re.findall(r"(\d+)\s*[至到\-—]\s*(\d+)", value):
            first, last = int(start), int(end)
            if 0 < first <= last <= 999:
                result.update(range(first, last + 1))
        result.update(int(number) for number in re.findall(r"\d+", value))
    return sorted(result)


def extract_patent(parsed: ParsedPdf) -> PatentStructure | None:
    """有中国公开号才启用专利规则，避免误处理普通报告。"""
    cover = [element for element in parsed.elements if element.span.page_no == 1 and element.text]
    publication = next(((element, match) for element in cover
                        if (match := re.search(r"\bCN\s*\d{6,12}\s*[A-CUSY](?:\d)?\b", element.text, re.I))), None)
    if publication is None:
        return None
    structure = PatentStructure(PATENT_VERSION)
    pub_element, pub_match = publication
    structure.publication_number = PatentField(_compact(pub_match.group()).upper(), [pub_element.span])
    # 名称可能与 (54) 标签分块，也可能与 (57) 摘要标签同块。
    title_parts: list[str] = []
    title_sources: list[PdfElement] = []
    collecting = False
    for element in cover:
        text = element.text
        if re.search(r"[（(]54[）)]", text):
            collecting = True
            text = re.split(r"[（(]54[）)]", text, maxsplit=1)[1]
            text = re.sub(r"^\s*(?:发明名称|实用新型名称|名称)\s*", "", text)
        if collecting:
            part = re.split(r"[（(]\d{2}[）)]", text, maxsplit=1)[0]
            if part.strip():
                title_parts.append(_compact(part))
                title_sources.append(element)
            if re.search(r"[（(]\d{2}[）)]", text):
                break
    if title_parts:
        structure.title = PatentField("".join(title_parts), _unique_spans(title_sources))
        parsed.title = structure.title.value
    else:
        structure.warnings.append("未识别 (54) 专利名称，请人工核对。")

    page_headers: dict[int, tuple[str, str]] = {}
    for element in parsed.elements:
        text = _compact(element.text)
        match = re.fullmatch(r"(权利要求书|说明书附图|说明书)(\d+/\d+页)?", text)
        if match:
            kind = {"权利要求书": "claims", "说明书": "description", "说明书附图": "drawings"}[match[1]]
            page_headers[element.span.page_no] = (kind, match[2] or "")
            element.kind = "furniture"
            if match[2]:
                element.payload["printed_page_label"] = f"{match[1]} {match[2]}"

    grouped: dict[str, list[tuple[str, PdfElement]]] = defaultdict(list)
    claim_parts: dict[int, list[tuple[str, PdfElement]]] = defaultdict(list)
    current_claim: int | None = None
    section, subsection, page = "cover", "", 0
    seen_numbers: list[int] = []
    for element in parsed.elements:
        if element.span.page_no != page:
            page = element.span.page_no
            if page in page_headers:
                next_section = page_headers[page][0]
                if next_section == "description" and section == "drawing_description":
                    next_section = section
                if next_section != section:
                    current_claim, subsection = None, ""
                section = next_section
        compact = _compact(element.text)
        # 专利页底的数字是页脚，不应成为知识正文。
        if element.kind == "paragraph" and element.span.bbox[1] > 0.88 and compact.isdigit():
            element.kind = "furniture"
        if re.search(r"[（(]57[）)]\s*摘\s*要", element.text):
            section = "abstract"
            text = re.split(r"[（(]57[）)]\s*摘\s*要", element.text, maxsplit=1)[1]
            if text.strip():
                grouped[section].append((text.strip(), element))
        elif compact in SUBHEADINGS and section in {"description", "drawing_description"}:
            subsection = compact
            section = "drawing_description" if compact == "附图说明" else "description"
            element.kind = "heading"
            element.heading_level = 2
        elif element.kind not in {"image", "furniture"}:
            grouped[section].append((element.text, element))
        element.payload["patent_section"] = section
        element.section_path = ("说明书 / " + subsection) if subsection else SECTION_NAMES[section]
        if section != "claims" or element.kind in {"image", "furniture"}:
            continue
        matches = list(re.finditer(r"(?m)^\s*(\d{1,3})\s*[.．、]\s*", element.text))
        if matches:
            prefix = element.text[:matches[0].start()].strip()
            if prefix and current_claim is not None:
                claim_parts[current_claim].append((prefix, element))
            for index, match in enumerate(matches):
                current_claim = int(match[1])
                seen_numbers.append(current_claim)
                end = matches[index + 1].start() if index + 1 < len(matches) else len(element.text)
                claim_parts[current_claim].append((element.text[match.start():end].strip(), element))
            # 多条权利要求同一元素时不强行标成最后一条，精确划分见专利结构。
            if len(matches) == 1:
                element.payload["claim_number"] = current_claim
        elif current_claim is not None:
            claim_parts[current_claim].append((element.text, element))
            element.payload["claim_number"] = current_claim
        if element.payload.get("claim_number"):
            element.section_path = f"权利要求 / 第 {current_claim} 项"

    for kind, parts in grouped.items():
        if kind != "cover":
            structure.sections.append(PatentSection(kind, SECTION_NAMES[kind], "\n\n".join(text for text, _ in parts),
                                                    _unique_spans([element for _, element in parts])))
    for number, parts in sorted(claim_parts.items()):
        text = "\n\n".join(text for text, _ in parts)
        fragments = []
        start = 0
        for part, element in parts:
            offset = element.text.find(part)
            fragments.append({"start": start, "end": start + len(part), "element_id": element.id,
                              "source_char_start": max(0, offset), "source_char_end": max(0, offset) + len(part)})
            start += len(part) + 2
        structure.claims.append(PatentClaim(number, text, _dependencies(text), _unique_spans([element for _, element in parts]), fragments))
    numbers = set(claim_parts)
    if not numbers:
        structure.warnings.append("未提取到编号权利要求，请核对版式和文字层。")
    elif numbers != set(range(1, max(numbers) + 1)) or len(seen_numbers) != len(set(seen_numbers)):
        structure.warnings.append("权利要求编号存在缺号或重复，需人工复核。")
    for claim in structure.claims:
        if any(number not in numbers or number >= claim.number for number in claim.depends_on):
            structure.warnings.append(f"权利要求 {claim.number} 的引用编号异常，需人工复核。")
    for kind in ("abstract", "description"):
        if not any(section.kind == kind and section.text.strip() for section in structure.sections):
            structure.warnings.append(f"未识别{SECTION_NAMES[kind]}正文，请核对。")
    return structure
