"""处理配置的单一入口；密钥不参与资产身份，阶段版本分别保留。"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from app.schemas import Settings
from .tables.base import TableOptions


@dataclass(frozen=True)
class PdfProfile:
    tables: TableOptions
    ocr_dpi: int
    chunk_size: int
    target_size: int
    context_mode: str
    context_model: str
    context_endpoint: str

    @classmethod
    def from_settings(cls, settings: Settings):
        return cls(TableOptions(
            backend=settings.pdf_table_backend, mode=settings.pdf_table_mode,
            models_directory=settings.pdf_table_models_directory,
            fallback=settings.pdf_table_fallback, ocr_enabled=settings.pdf_image_ocr,
            ocr_language=settings.pdf_ocr_language), settings.pdf_ocr_dpi,
            settings.chunk_size, settings.pdf_target_size, settings.pdf_context_mode,
            settings.chat_model if settings.pdf_context_mode == "model" else "",
            settings.base_url if settings.pdf_context_mode == "model" else "")

    def chunk_metadata(self) -> dict:
        return {"pdf_context_mode": self.context_mode, "pdf_context_model": self.context_model,
                "pdf_context_endpoint": self.context_endpoint}

    def bind_chunks(self, chunks) -> None:
        """模型增强失败时也保留配置身份，避免旧配置资产占用同一切片 ID。"""
        metadata = self.chunk_metadata()
        identities = {}
        for chunk in chunks:
            previous = chunk.metadata["pdf_chunk_id"]
            identifier = (sha256((previous + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()
                          if self.context_mode == "model" else previous)
            identities[previous] = identifier
            chunk.metadata.update(metadata, pdf_chunk_id=identifier)
        for chunk in chunks:
            for key in ("previous_chunk_id", "next_chunk_id"):
                previous = chunk.metadata.get(key, "")
                chunk.metadata[key] = identities.get(previous, previous)

    def parser_arguments(self, directory: str) -> dict:
        return dict(ocr_images=self.tables.ocr_enabled, ocr_language=self.tables.ocr_language,
                    ocr_dpi=self.ocr_dpi, table_backend=self.tables.backend,
                    table_mode=self.tables.mode, table_models_directory=self.tables.models_directory,
                    table_fallback=self.tables.fallback,
                    table_work_directory=str(Path(directory) / "table-work"))


def pipeline_fingerprint(settings: Settings, content_hash: str, file_format: str) -> str:
    """所有导入入口使用相同的配置与阶段版本；模型端点改变也使索引失效。"""
    if file_format == ".pdf":
        from .parser import PARSER_VERSION
        from .chunking import CHUNKER_VERSION
        from app.quality.checks import RULE_VERSION
        from app.prompt.prompt import PDF_CONTEXT_VERSION
        from .tables.service import backend_identity
        profile = PdfProfile.from_settings(settings)
        table_identity = backend_identity(profile.tables)
        strategy = (f"{content_hash}:{PARSER_VERSION}:{CHUNKER_VERSION}:{RULE_VERSION}:{table_identity}:general:{settings.chunk_size}:"
                    f"{settings.pdf_target_size}:{settings.pdf_context_mode}:{PDF_CONTEXT_VERSION}:"
                    f"{settings.pdf_image_ocr}:{settings.pdf_ocr_language}:{settings.pdf_ocr_dpi}:"
                    f"{profile.context_model}:{profile.context_endpoint}")
    else:
        strategy = (f"{content_hash}:{file_format}:general:heading-split-v2:"
                    f"{settings.chunk_size}:{settings.chunk_overlap}")
    return sha256(strategy.encode()).hexdigest()


def chunk_profile(metadata: dict) -> tuple:
    """历史元数据与新元数据共用配置分组，缺失字段按历史默认值解释。"""
    return tuple(metadata.get(key, default) for key, default in (
        ("pdf_chunk_size", 600), ("pdf_target_size", 0),
        ("pdf_chunker_version", "legacy"), ("pdf_parser", "legacy"),
        ("pdf_context_mode", "rule"), ("pdf_context_model", ""), ("pdf_context_endpoint", "")))
