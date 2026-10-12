"""统一导出各模块的数据结构，仅依赖标准库。"""

from .chat import ChatMessage
from .configuration import Settings
from .ingestion import IngestionJob, IngestionStage, IngestionStatus
from .knowledge import ChunkRecord, DocumentRecord, Hit, QueryResult, StoreStats

__all__ = [
    "ChatMessage", "Settings", "IngestionJob", "IngestionStage", "IngestionStatus",
    "ChunkRecord", "DocumentRecord", "Hit", "QueryResult", "StoreStats",
]
