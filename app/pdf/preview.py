"""只在内存副本画来源框，原始 PDF 字节不变。"""

import pymupdf
import math

HIGHLIGHT_COLORS: dict[str, tuple[float, float, float]] = {
    "value": (1, 0, 0), "row_header": (0, .3, 1),
    "column_header": (0, .65, .1), "unit": (.65, 0, .8),
}


def render_source_page(content: bytes, page_no: int, spans: list[dict]) -> bytes:
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if type(page_no) is not int or not 1 <= page_no <= document.page_count:
            raise ValueError("来源物理页码无效")
        page = document[page_no - 1]
        for span in spans:
            if span["page_no"] != page_no:
                continue
            box = span.get("bbox")
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                continue
            x0, y0, x1, y1 = span["bbox"]
            rect = pymupdf.Rect(x0 * page.rect.width, y0 * page.rect.height,
                                x1 * page.rect.width, y1 * page.rect.height)
            role = span.get("highlight_role")
            color = HIGHLIGHT_COLORS.get(role, HIGHLIGHT_COLORS["value"]) if isinstance(role, str) else HIGHLIGHT_COLORS["value"]
            page.draw_rect(rect * page.derotation_matrix, color=color, width=1.5)
        return page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).tobytes("png")


def render_source_region(content: bytes, span: dict, dpi: int = 180) -> bytes:
    """Render a source bbox as a standalone crop for image/table chunk previews."""
    with pymupdf.open(stream=content, filetype="pdf") as document:
        page = document[int(span["page_no"]) - 1]
        x0, y0, x1, y1 = span["bbox"]
        rect = pymupdf.Rect(x0 * page.rect.width, y0 * page.rect.height,
                            x1 * page.rect.width, y1 * page.rect.height) & page.rect
        if rect.is_empty:
            raise ValueError("来源区域为空，无法生成裁图")
        return page.get_pixmap(dpi=dpi, clip=rect).tobytes("png")
