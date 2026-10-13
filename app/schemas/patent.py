"""专利字段、章节和权利要求；每项保留原 PDF 区域来源。"""

from dataclasses import dataclass, field

from .pdf import SourceSpan


@dataclass
class PatentField:
    value: str
    spans: list[SourceSpan]


@dataclass
class PatentSection:
    kind: str
    title: str
    text: str
    spans: list[SourceSpan]


@dataclass
class PatentClaim:
    number: int
    text: str
    depends_on: list[int]
    spans: list[SourceSpan]
    source_fragments: list[dict] = field(default_factory=list)


@dataclass
class PatentStructure:
    extractor_version: str
    title: PatentField | None = None
    publication_number: PatentField | None = None
    sections: list[PatentSection] = field(default_factory=list)
    claims: list[PatentClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
