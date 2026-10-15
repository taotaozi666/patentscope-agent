"""页面诊断、区域来源和处理计划；启发式分类不表示已完成 OCR。"""

from dataclasses import dataclass, field
from typing import Any, Literal

from .pdf import SourceSpan


PageSource = Literal["native", "scanned", "hybrid", "unreliable_text_layer", "empty", "unknown"]
RegionRole = Literal["text", "table", "image", "unknown"]


@dataclass
class RegionInspection:
    id: str
    span: SourceSpan
    pdf_bbox: tuple[float, float, float, float]
    role: RegionRole
    text_source: PageSource
    classification_status: Literal["observed", "candidate"] = "candidate"
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    overlaps: list[str] = field(default_factory=list)


@dataclass
class RegionRoute:
    region_id: str
    planned_operations: list[str]
    reasons: list[str] = field(default_factory=list)
    executed_operations: list[str] = field(default_factory=list)
    deferred_operations: list[str] = field(default_factory=list)
    disabled_operations: list[str] = field(default_factory=list)
    failed_operations: list[str] = field(default_factory=list)


@dataclass
class PageInspection:
    page_no: int
    page_label: str
    width: float
    height: float
    rotation: int
    text_source: PageSource
    classification_status: Literal["observed", "candidate"]
    inspection_version: str = "page-inspection-v1"
    routing_version: str = "region-routing-v1"
    metrics: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regions: list[RegionInspection] = field(default_factory=list)
    routes: list[RegionRoute] = field(default_factory=list)
