"""PDF 结构模型，独立于解析引擎及 Chroma 内部格式。"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from .patent import PatentStructure
    from .table import LogicalTable, TableFragment
    from .inspection import PageInspection


class NativeTextSpan(TypedDict):
    """PyMuPDF 原生文字接口中本项目使用的字段。"""

    text: str
    size: float


class NativeTextLine(TypedDict):
    spans: list[NativeTextSpan]


class NativeTextBlock(TypedDict):
    type: int
    bbox: tuple[float, float, float, float]
    lines: list[NativeTextLine]


class NativeTextPage(TypedDict):
    blocks: list[NativeTextBlock]


class NativeImageInfo(TypedDict):
    bbox: tuple[float, float, float, float]


@dataclass
class SourceSpan:
    """显示页左上角原点的归一化区域；这是区域定位，不是逐字定位。"""

    page_no: int
    page_label: str
    bbox: tuple[float, float, float, float]
    element_id: str
    granularity: str = "region"


@dataclass
class PdfElement:
    id: str
    kind: str
    text: str
    span: SourceSpan
    font_size: float = 0
    heading_level: int = 0
    section_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticObject:
    """完整语义父对象；索引子块只引用它，不取代完整原文。"""

    id: str
    document_id: str
    kind: str
    title: str
    raw_content: str
    spans: list[SourceSpan]
    mappings: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPdf:
    source: str
    content_hash: str
    title: str
    page_count: int
    parser_version: str
    elements: list[PdfElement]
    warnings: list[str] = field(default_factory=list)
    patent: "PatentStructure | None" = None
    objects: list[SemanticObject] = field(default_factory=list)
    table_schema_version: int = 1
    table_fragments: list["TableFragment"] = field(default_factory=list)
    logical_tables: list["LogicalTable"] = field(default_factory=list)
    page_inspections: list["PageInspection"] = field(default_factory=list)
