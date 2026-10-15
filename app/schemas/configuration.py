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
    pdf_image_ocr: bool = True
    pdf_ocr_language: str = "chi_sim+eng"
    pdf_ocr_dpi: int = 200
    pdf_table_backend: str = "native"
    pdf_table_mode: str = "accurate"
    pdf_table_models_directory: str = ""
    pdf_table_fallback: bool = True
    retrieval_mode: str = "hybrid"
    retrieval_candidates: int = 30
    retrieval_rrf_constant: int = 60
    retrieval_vector_weight: float = 1.0
    retrieval_keyword_weight: float = 1.0
