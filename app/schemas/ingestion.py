"""后台文档导入任务状态及进度。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


type IngestionStatus = Literal[
    "queued", "processing", "completed", "skipped", "failed", "cancelled"
]
type IngestionStage = Literal[
    "queued", "loading", "splitting", "embedding", "indexing",
    "completed", "duplicate", "failed", "cancelled"
]


@dataclass(frozen=True)
class IngestionJob:
    """一个文件对应一个可独立重试的任务。"""

    id: str
    batch_id: str
    filename: str
    file_size: int
    embedding_mode: Literal["semantic", "offline"]
    status: IngestionStatus
    stage: IngestionStage
    progress: int
    attempt: int
    chunk_count: int
    error_message: str | None
    result_message: str | None
    created_at: datetime
    updated_at: datetime
