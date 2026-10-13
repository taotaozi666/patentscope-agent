"""向既有文件格式注册表提供 PDF 加载能力。"""

from langchain_core.document_loaders import BaseLoader


class NativePdfLoader(BaseLoader):
    def __init__(self, source: str, content: bytes) -> None:
        self.source, self.content = source, content

    def lazy_load(self):
        from .chunking import build_chunks
        from .parser import parse_pdf

        yield from build_chunks(parse_pdf(self.source, self.content))
