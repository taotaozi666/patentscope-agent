"""向既有文件格式注册表提供 PDF 加载能力。"""

from langchain_core.document_loaders import BaseLoader

from app.config import get_settings
from app.schemas import Settings


class NativePdfLoader(BaseLoader):
    """PDF Loader shares the full preparation, quality and persistence pipeline."""

    def __init__(self, source: str, content: bytes, settings: Settings | None = None) -> None:
        self.source, self.content = source, content
        self.settings = settings or get_settings()
        self.import_message: str | None = None

    def lazy_load(self):
        from .service import prepare_pdf

        documents, self.import_message = prepare_pdf(self.source, self.content, self.settings)
        yield from documents
