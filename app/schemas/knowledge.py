"""知识库与检索数据模型，供业务、存储和页面共享。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    """一个已入库片段及其向量、来源和元数据，不是数据库 ORM 实体。"""

    id: str
    document_id: str
    source: str
    text: str
    chunk: int
    vector: list[float]
    created_at: datetime
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StoreStats:
    """知识库的文档数量与片段数量。"""

    documents: int
    chunks: int


@dataclass(frozen=True)
class QueryResult:
    """查询结果的一页记录及全部匹配数量。"""

    items: list[ChunkRecord]
    total: int


@dataclass
class Hit:
    """检索结果，包含来源、原文、相似度和片段编号。"""

    source: str
    text: str
    score: float
    chunk: int
