"""区域处理计划和执行记录；不把计划中的视觉能力标记为已完成。"""

from app.schemas.inspection import PageInspection, RegionRoute
from app.schemas.pdf import PdfElement


ROUTING_VERSION = "region-routing-v3"


def route_regions(inspection: PageInspection, *, ocr_enabled: bool) -> list[RegionRoute]:
    inspection.routing_version = ROUTING_VERSION
    routes = []
    for region in inspection.regions:
        if region.role == "table":
            operations = ["docling_table" if region.metrics.get("backend") == "docling" else "native_table"]
        elif region.role == "text" and region.text_source == "native":
            operations = ["native_text"]
        elif region.role == "text" and region.text_source == "unreliable_text_layer":
            operations = ["preserve_native_text", "region_ocr", "review_text_layer"]
        elif region.id.endswith("_page_image"):
            operations = ["preserve_image", "page_ocr", "layout_analysis"]
        elif region.role == "image":
            operations = ["preserve_image", "region_ocr", "layout_analysis"]
        else:
            operations = ["preserve_region", "review_layout"]
        disabled = [operation for operation in operations
                    if operation in {"page_ocr", "region_ocr"} and not ocr_enabled]
        routes.append(RegionRoute(region.id, operations, list(region.reasons),
                                  disabled_operations=disabled))
    return routes


def record_execution(inspection: PageInspection, elements: list[PdfElement]) -> None:
    """记录当前兼容解析器的执行情况；区域 OCR 和版面模型将在后续接入。"""
    by_id = {element.id: element for element in elements}
    for route in inspection.routes:
        element = by_id.get(route.region_id)
        executed = []
        if element is not None:
            if element.kind == "table":
                executed.append("docling_table" if element.payload.get("table_backend") == "docling" else "native_table")
            elif element.kind in {"paragraph", "heading", "furniture"}:
                if element.payload.get("ocr_scope") == "region":
                    executed.extend(["preserve_native_text", "region_ocr"])
                else:
                    executed.append("native_text")
            elif element.kind == "image":
                executed.append("preserve_image")
                if element.payload.get("ocr_status") == "unavailable" and element.payload.get("ocr_scope") == "region":
                    route.failed_operations = ["region_ocr"]
                if element.payload.get("ocr_status") in {"recognized", "no_text"} and element.payload.get("ocr_scope") in {"page", "region"}:
                    executed.append("page_ocr" if element.payload.get("ocr_scope") == "page"
                                    else "region_ocr" if element.payload.get("ocr_scope") == "region"
                                    else "legacy_page_ocr")
        route.executed_operations = executed
        route.deferred_operations = [operation for operation in route.planned_operations
                                     if operation not in executed and
                                     operation not in route.disabled_operations and
                                     operation not in route.failed_operations]
