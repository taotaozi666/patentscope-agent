"""把既有 basic_grid 转为统一模型，不推断跨度、表头或跨页关系。"""

from copy import deepcopy
from typing import cast

from app.schemas.pdf import PdfElement, SourceSpan
from app.schemas.table import CellState, TextOrigin, TableCell, TableFragment


def adapt_basic_grid(element: PdfElement, parser_version: str) -> TableFragment:
    payload = element.payload
    rows = payload.get("rows", [])
    cells = []
    origin = cast(TextOrigin, payload.get("text_origin", "unknown"))
    if origin not in {"native", "ocr", "mixed"}:
        origin = "unknown"
    for raw in payload.get("cells", []):
        source = raw.get("span")
        spans = [SourceSpan(**{**source, "bbox": tuple(source["bbox"])})] if source else []
        text = raw.get("raw_text")
        # None 在旧格式中无法区分空白、合并占位与提取失败。
        state: CellState = "text" if text and text.strip() else "blank" if text == "" and spans else "unknown"
        cells.append(TableCell(
            id=raw["id"], fragment_id=element.id, row=raw["row"], column=raw["column"],
            raw_text=text, state=state, spans=spans, text_origin=origin,
        ))
    return TableFragment(
        id=element.id, span=deepcopy(element.span),
        row_count=payload.get("row_count", len(rows)),
        column_count=payload.get("column_count", max((len(row) for row in rows), default=0)),
        cells=cells, section_path=element.section_path,
        backend="legacy-basic-grid", backend_version=parser_version,
        raw_output=deepcopy(payload),
    )
