"""只在内存副本画来源框，原始 PDF 字节不变。"""

import pymupdf


def render_source_page(content: bytes, page_no: int, spans: list[dict]) -> bytes:
    with pymupdf.open(stream=content, filetype="pdf") as document:
        page = document[page_no - 1]
        for span in spans:
            if span["page_no"] != page_no:
                continue
            x0, y0, x1, y1 = span["bbox"]
            rect = pymupdf.Rect(x0 * page.rect.width, y0 * page.rect.height,
                                x1 * page.rect.width, y1 * page.rect.height)
            page.draw_rect(rect * page.derotation_matrix, color=(1, 0, 0), width=1.5)
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
