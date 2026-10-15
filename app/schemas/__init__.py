"""统一导出各模块的数据结构，仅依赖标准库。"""

from .chat import ChatMessage
from .configuration import Settings
from .ingestion import IngestionJob, IngestionStage, IngestionStatus
from .knowledge import ChunkRecord, DocumentRecord, Hit, QueryResult, StoreStats
from .pdf import ParsedPdf, PdfElement, SourceSpan, SemanticObject
from .quality import QualityIssue, PageQuality, QualityReport, QualityReview
from .patent import PatentField, PatentSection, PatentClaim, PatentStructure
from .table import TABLE_SCHEMA_VERSION, TableCell, TableFragment, LogicalTable
from .inspection import PageInspection, RegionInspection, RegionRoute

__all__ = [
    "ChatMessage", "Settings", "IngestionJob", "IngestionStage", "IngestionStatus",
    "ChunkRecord", "DocumentRecord", "Hit", "QueryResult", "StoreStats",
    "ParsedPdf", "PdfElement", "SourceSpan",
    "SemanticObject",
    "QualityIssue", "PageQuality", "QualityReport", "QualityReview",
    "PatentField", "PatentSection", "PatentClaim", "PatentStructure",
    "TABLE_SCHEMA_VERSION", "TableCell", "TableFragment", "LogicalTable",
    "PageInspection", "RegionInspection", "RegionRoute",
]
