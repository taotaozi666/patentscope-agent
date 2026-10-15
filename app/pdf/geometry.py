"""PDF 原生坐标到阅读器显示页来源坐标的统一转换。"""

import pymupdf

from app.schemas.pdf import SourceSpan


def source_span(page: pymupdf.Page, bbox: tuple[float, float, float, float],
                element_id: str, granularity: str = "region") -> SourceSpan:
    rect = pymupdf.Rect(bbox) * page.rotation_matrix
    width, height = page.rect.width, page.rect.height
    normalized = [max(0.0, min(1.0, value)) for value in
                  (rect.x0 / width, rect.y0 / height, rect.x1 / width, rect.y1 / height)]
    if page.number is None:
        raise ValueError("PDF 页面未关联文档，无法记录来源页码")
    return SourceSpan(page.number + 1, page.get_label() or "",
                      (normalized[0], normalized[1], normalized[2], normalized[3]),
                      element_id, granularity)
