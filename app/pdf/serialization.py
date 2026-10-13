"""从已保存的解析 JSON 恢复结构对象，兼容没有专利字段的旧记录。"""

from app.schemas.pdf import ParsedPdf, PdfElement, SourceSpan, SemanticObject
from app.schemas.patent import PatentClaim, PatentField, PatentSection, PatentStructure


def restore_pdf(data: dict) -> ParsedPdf:
    def span(value: dict) -> SourceSpan:
        return SourceSpan(**{**value, "bbox": tuple(value["bbox"])})

    patent = data.get("patent")
    structure = None
    if patent:
        structure = PatentStructure(
            extractor_version=patent["extractor_version"],
            sections=[PatentSection(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in patent["sections"]],
            claims=[PatentClaim(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in patent["claims"]],
            warnings=patent["warnings"],
        )
        for name in ("title", "publication_number"):
            item = patent.get(name)
            if item:
                setattr(structure, name, PatentField(item["value"], [span(value) for value in item["spans"]]))
    elements = [PdfElement(**{**item, "span": span(item["span"])}) for item in data["elements"]]
    objects = [SemanticObject(**{**item, "spans": [span(value) for value in item["spans"]]}) for item in data.get("objects", [])]
    return ParsedPdf(**{**data, "elements": elements, "patent": structure, "objects": objects})
