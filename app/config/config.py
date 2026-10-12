"""统一读取项目配置，同名系统环境变量优先于项目根目录的 .env。"""

import os
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
    log_level = config.get("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("LOG_LEVEL 必须为 DEBUG、INFO、WARNING 或 ERROR")
    chunk_size = int(config.get("CHUNK_SIZE", "600"))
    chunk_overlap = int(config.get("CHUNK_OVERLAP", "100"))
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
    return Settings(
        api_key=config.first("DASHSCOPE_API_KEY", "API_KEY"),
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
