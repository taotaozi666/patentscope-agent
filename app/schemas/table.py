"""解析后端共享的表格结构；原始单元格是证据，逻辑表仅引用它们。"""

from dataclasses import dataclass, field
from typing import Any, Literal

from .pdf import SourceSpan


TABLE_SCHEMA_VERSION = 1
TextOrigin = Literal["native", "ocr", "mixed", "unknown"]
CellState = Literal["text", "blank", "unreadable", "covered", "unknown"]
TableStatus = Literal["unverified", "complete", "partial", "failed"]


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须为大于等于 {minimum} 的整数")


@dataclass
class TableCell:
    """页内单元格；未知跨度使用 None，不能由空格猜测合并关系。

    covered 是已确认的合并占位，必须指向锚点；旧数据缺少位置的
    None 格使用 unknown。原始零值保存为字符串 '0'，不转为缺失值。
    """

    id: str
    fragment_id: str
    row: int
    column: int
    raw_text: str | None
    state: CellState = "unknown"
    row_span: int | None = None
    col_span: int | None = None
    spans: list[SourceSpan] = field(default_factory=list)
    text_origin: TextOrigin = "unknown"
    covered_by_cell_id: str | None = None
    row_header_cell_ids: list[str] = field(default_factory=list)
    column_header_cell_ids: list[str] = field(default_factory=list)
    unit_cell_ids: list[str] = field(default_factory=list)
    normalized_text: str | None = None
    normalized_value: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.fragment_id:
            raise ValueError("单元格必须具有 id 和 fragment_id")
        _integer(self.row, "row")
        _integer(self.column, "column")
        for name in ("row_span", "col_span"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, 1)
        if self.raw_text is not None and not isinstance(self.raw_text, str):
            raise ValueError("raw_text 必须是字符串或 None，数值零使用 '0'")
        for name in ("normalized_text", "normalized_value"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} 必须为字符串或 None，规范化数值保留十进制字符串")
        if self.state not in {"text", "blank", "unreadable", "covered", "unknown"}:
            raise ValueError("未知的单元格状态")
        if self.text_origin not in {"native", "ocr", "mixed", "unknown"}:
            raise ValueError("未知的单元格文字来源")
        if self.state == "text" and not (self.raw_text and self.raw_text.strip()):
            raise ValueError("text 单元格必须具有非空文字")
        if self.state in {"blank", "covered", "unreadable"} and self.raw_text not in {None, ""}:
            raise ValueError("空白、合并占位或未识别单元格不能带有已识别文字")
        if self.state == "covered":
            if not self.covered_by_cell_id or self.covered_by_cell_id == self.id:
                raise ValueError("合并占位必须指向其他锚点单元格")
        elif self.covered_by_cell_id is not None:
            raise ValueError("仅合并占位可以设置 covered_by_cell_id")


@dataclass
class TableFragment:
    """单页物理表区域；位置沿用显示页归一化 SourceSpan。"""

    id: str
    span: SourceSpan
    row_count: int
    column_count: int
    cells: list[TableCell] = field(default_factory=list)
    header_rows: list[int] | None = None
    section_path: str = ""
    backend: str = "unknown"
    backend_version: str = ""
    structure_status: TableStatus = "unverified"

    provenance_status: TableStatus = "unverified"
    raw_output: dict[str, Any] = field(default_factory=dict)
    header_method: str = "unknown"
    continuation_decision: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("表格片段必须具有 id")
        _integer(self.row_count, "row_count", 1)
        _integer(self.column_count, "column_count", 1)
        if self.header_rows is not None:
            for row in self.header_rows:
                _integer(row, "header_row")
                if row >= self.row_count:
                    raise ValueError("表头行索引越界")
        if len({cell.id for cell in self.cells}) != len(self.cells):
            raise ValueError("同一表格片段内的单元格 id 不能重复")
        for cell in self.cells:
            if cell.fragment_id != self.id:
                raise ValueError("单元格 fragment_id 与所属片段不一致")
            if (cell.row + (cell.row_span or 1) > self.row_count
                    or cell.column + (cell.col_span or 1) > self.column_count):
                raise ValueError("单元格行列或跨度越界")
        by_id = {cell.id: cell for cell in self.cells}
        for cell in self.cells:
            if cell.state == "covered":
                anchor = by_id.get(cell.covered_by_cell_id or "")
                if anchor is None or anchor.state == "covered":
                    raise ValueError("合并占位必须引用同一片段内的非占位锚点")
        _status(self.structure_status)
        _status(self.provenance_status)


@dataclass
class LogicalTable:
    """跨页逻辑表；数据行和表头路径引用原片段的单元格，不复制值。"""

    id: str
    fragment_ids: list[str]
    title: str = ""
    table_label: str = ""
    section_path: str = ""
    header_paths: dict[str, list[str]] = field(default_factory=dict)
    data_rows: list[list[str]] = field(default_factory=list)
    repeated_header_cell_ids: list[str] = field(default_factory=list)
    unit_text: str | None = None
    unit_spans: list[SourceSpan] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    footnote_spans: list[SourceSpan] = field(default_factory=list)
    continuation_candidate_ids: list[str] = field(default_factory=list)
    structure_status: TableStatus = "unverified"
    title_spans: list[SourceSpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not self.fragment_ids or any(not value for value in self.fragment_ids):
            raise ValueError("逻辑表必须具有 id 和非空片段引用")
        if len(set(self.fragment_ids)) != len(self.fragment_ids):
            raise ValueError("逻辑表的片段引用不能重复")
        _status(self.structure_status)


def _status(value: TableStatus) -> None:
    if value not in {"unverified", "complete", "partial", "failed"}:
        raise ValueError("未知的表格状态")
