"""解析质量报告及人工复核的数据结构。"""

from dataclasses import dataclass, field
from typing import Literal

type QualityStatus = Literal["passed", "review", "blocked"]
type ReviewDecision = Literal["passed", "problem", "pending"]


@dataclass
class QualityIssue:
    id: str
    code: str
    status: QualityStatus
    message: str
    page_no: int | None = None
    element_id: str | None = None
    chunk_id: str | None = None


@dataclass
class PageQuality:
    page_no: int
    text_chars: int
    paragraphs: int
    tables: int
    images: int
    chunks: int


@dataclass
class QualityReport:
    id: str
    run_id: str
    rule_version: str
    checked_at: str
    status: QualityStatus
    chunk_ids: list[str]
    pages: list[PageQuality] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)


@dataclass
class QualityReview:
    report_id: str
    target: str
    decision: ReviewDecision
    note: str
    updated_at: str
