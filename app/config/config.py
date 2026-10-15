"""统一读取项目配置，同名系统环境变量优先于项目根目录的 .env。"""

import os
import math
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

from app.schemas import Settings

# 配置模块位于 app/config，向上两层是项目根目录。
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Config:
    """提供通用字符串配置读取，不修改进程环境变量。"""

    def __init__(
        self, env_file: str | Path = ENV_FILE, environ: Mapping[str, str] | None = None
    ) -> None:
        self._values = {
            **dotenv_values(env_file),
            **(os.environ if environ is None else environ),
        }

    def get(self, name: str, default: str = "") -> str:
        """读取配置并去除首尾空白，缺失或为空时返回默认值。"""
        value = self._values.get(name)
        return value.strip() if value and value.strip() else default

    def first(self, *names: str, default: str = "") -> str:
        """按顺序读取第一个已定义的名称，显式空值不会回退到别名。"""
        for name in names:
            if name in self._values:
                return self.get(name, default)
        return default


def get_settings(
    env_file: str | Path = ENV_FILE, environ: Mapping[str, str] | None = None
) -> Settings:
    """每次调用重新读取配置，让后续页面操作使用最新的 .env。"""
    config = Config(env_file, environ)
    retrieval_mode = config.get("RETRIEVAL_MODE", "hybrid").lower()
    retrieval_candidates = int(config.get("RETRIEVAL_CANDIDATES", "30"))
    retrieval_rrf_constant = int(config.get("RETRIEVAL_RRF_CONSTANT", "60"))
    retrieval_vector_weight = float(config.get("RETRIEVAL_VECTOR_WEIGHT", "1"))
    retrieval_keyword_weight = float(config.get("RETRIEVAL_KEYWORD_WEIGHT", "1"))
    if retrieval_mode not in {"hybrid", "vector"}:
        raise ValueError("RETRIEVAL_MODE 必须为 hybrid 或 vector")
    if not 1 <= retrieval_candidates <= 1000 or not 1 <= retrieval_rrf_constant <= 10000:
        raise ValueError("候选数必须介于 1 和 1000，RRF 常数必须介于 1 和 10000")
    if any(not math.isfinite(weight) or weight <= 0 for weight in (retrieval_vector_weight, retrieval_keyword_weight)):
        raise ValueError("检索权重必须为有限正数")
    log_level = config.get("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("LOG_LEVEL 必须为 DEBUG、INFO、WARNING 或 ERROR")
    chunk_size = int(config.get("CHUNK_SIZE", "600"))
    chunk_overlap = int(config.get("CHUNK_OVERLAP", "100"))
    pdf_target_size = int(config.get("PDF_TARGET_SIZE", "0"))
    pdf_context_mode = config.get("PDF_CONTEXT_MODE", "rule").lower()
    pdf_image_ocr = config.get("PDF_IMAGE_OCR", "true").lower() in {"1", "true", "yes", "on"}
    pdf_ocr_language = config.get("PDF_OCR_LANGUAGE", "chi_sim+eng")
    pdf_ocr_dpi = int(config.get("PDF_OCR_DPI", "200"))
    pdf_table_backend = config.get("PDF_TABLE_BACKEND", "native").lower()
    pdf_table_mode = config.get("PDF_TABLE_MODE", "accurate").lower()
    if pdf_table_backend not in {"native", "auto", "docling"}:
        raise ValueError("PDF_TABLE_BACKEND 必须为 native、auto 或 docling")
    if pdf_table_mode not in {"accurate", "fast"}:
        raise ValueError("PDF_TABLE_MODE 必须为 accurate 或 fast")
    pdf_table_models_directory = config.get("PDF_TABLE_MODELS_DIRECTORY")
    if pdf_table_models_directory:
        models_path = Path(pdf_table_models_directory).expanduser()
        if not models_path.is_absolute():
            models_path = Path(env_file).resolve().parent / models_path
        pdf_table_models_directory = str(models_path.resolve())
    pdf_table_fallback = config.get("PDF_TABLE_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
    if not 0 <= pdf_target_size <= chunk_size:
        raise ValueError("PDF_TARGET_SIZE 必须介于 0 和 CHUNK_SIZE 之间，0 表示自动使用上限的 80%")
    if pdf_context_mode not in {"rule", "model"}:
        raise ValueError("PDF_CONTEXT_MODE 必须为 rule 或 model")
    if not 72 <= pdf_ocr_dpi <= 600:
        raise ValueError("PDF_OCR_DPI 必须介于 72 和 600 之间")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("CHUNK_OVERLAP 必须非负且小于 CHUNK_SIZE")
    max_files = int(config.get("INGESTION_MAX_FILES", "20"))
    max_file_bytes = int(config.get("INGESTION_MAX_FILE_BYTES", "52428800"))
    max_batch_bytes = int(config.get("INGESTION_MAX_BATCH_BYTES", "209715200"))
    embedding_concurrency = int(config.get("EMBEDDING_CONCURRENCY", "2"))
    if min(max_files, max_file_bytes, max_batch_bytes) <= 0:
        raise ValueError("后台上传数量和文件大小限制必须大于零")
    if not 1 <= embedding_concurrency <= 8:
        raise ValueError("EMBEDDING_CONCURRENCY 必须在 1 到 8 之间")
    directory = Path(config.get("CHROMA_DIRECTORY", ".chroma")).expanduser()
    if not directory.is_absolute():
        directory = Path(env_file).resolve().parent / directory
    ingestion_directory = Path(
        config.get("INGESTION_DIRECTORY", str(Path(env_file).resolve().parent / ".ingestion"))
    ).expanduser()
    if not ingestion_directory.is_absolute():
        ingestion_directory = Path(env_file).resolve().parent / ingestion_directory
    pdf_directory = Path(config.get("PDF_DIRECTORY", ".documents")).expanduser()
    if not pdf_directory.is_absolute():
        pdf_directory = Path(env_file).resolve().parent / pdf_directory
    return Settings(
        api_key=config.first("DASHSCOPE_API_KEY", "API_KEY"),
        retrieval_mode=retrieval_mode, retrieval_candidates=retrieval_candidates,
        retrieval_rrf_constant=retrieval_rrf_constant,
        retrieval_vector_weight=retrieval_vector_weight, retrieval_keyword_weight=retrieval_keyword_weight,
        pdf_directory=str(pdf_directory.resolve()),
        pdf_target_size=pdf_target_size,
        pdf_context_mode=pdf_context_mode,
        pdf_image_ocr=pdf_image_ocr,
        pdf_ocr_language=pdf_ocr_language,
        pdf_ocr_dpi=pdf_ocr_dpi,
        pdf_table_backend=pdf_table_backend,
        pdf_table_mode=pdf_table_mode,
        pdf_table_models_directory=pdf_table_models_directory,
        pdf_table_fallback=pdf_table_fallback,
        chat_model=config.first("CHAT_MODEL", "MODEL", default="qwen3.8-max"),
        embedding_model=config.get("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        embedding_concurrency=embedding_concurrency,
        base_url=config.first(
            "BASE_URL",
            "DASHSCOPE_BASE_URL",
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        chroma_directory=str(directory),
        chroma_collection=config.get("CHROMA_COLLECTION", "knowledge"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ingestion_directory=str(ingestion_directory.resolve()),
        ingestion_max_files=max_files,
        ingestion_max_file_bytes=max_file_bytes,
        ingestion_max_batch_bytes=max_batch_bytes,
        log_level=log_level,
    )
