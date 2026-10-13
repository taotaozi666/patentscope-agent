from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """模型配置快照，打印对象时隐藏密钥。"""

    api_key: str = field(repr=False)
    chat_model: str
    embedding_model: str
    embedding_concurrency: int = 2
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chroma_directory: str = str(Path(__file__).resolve().parents[2] / ".chroma")
    chroma_collection: str = "knowledge"
    chunk_size: int = 600
    chunk_overlap: int = 100
    ingestion_directory: str = str(Path(__file__).resolve().parents[2] / ".ingestion")
    ingestion_max_files: int = 20
    ingestion_max_file_bytes: int = 52_428_800
    ingestion_max_batch_bytes: int = 209_715_200
    log_level: str = "INFO"
    pdf_directory: str = str(Path(__file__).resolve().parents[2] / ".documents")
    pdf_target_size: int = 0
    pdf_context_mode: str = "rule"
