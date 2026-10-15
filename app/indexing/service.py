"""独立的文档导入与向量写入编排。"""

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.rag.documents import loaders, split_documents
from app.rag.embeddings import DemoEmbedder, SemanticEmbedder, create_embeddings
from app.schemas import Settings
from app.storage import KnowledgeStore, create_store
from app.storage.source_repository import SourceRepository
from app.trace import debug_step, print_step, trace_operation


class DocumentImportService:
    """加载、去重、向量化和索引写入；不执行查询或回答。"""

    def __init__(
        self,
        embedder: Embeddings | None = None,
        store: KnowledgeStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.last_import_message: str | None = None
        self.embedder = embedder or create_embeddings(
            self.settings, offline=not self.settings.api_key
        )
        if isinstance(self.embedder, DemoEmbedder):
            # 离线检索也约束 PDF 处理，配置了密钥时仍不调用模型增强。
            self.settings = replace(self.settings, api_key="", pdf_context_mode="rule")
        self.store = (
            store
            if store is not None
            else create_store(
                self.embedder, self.settings, offline=isinstance(self.embedder, DemoEmbedder)
            )
        )

    def add_file(
        self,
        source: str,
        content: bytes,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> int:
        """加载器统一返回 Document，后续流程不依赖文件格式。"""
        with trace_operation("文档导入", 来源=source):
            self.last_import_message = None
            is_pdf = Path(source).suffix.lower() == ".pdf"
            loader = loaders.create(source, content, self.settings)
            documents = loader.load()
            if is_pdf:
                message = getattr(loader, "import_message", None)
                self.last_import_message = message or None
            content_hash = sha256(content).hexdigest()
            if not is_pdf:
                SourceRepository(self.settings.pdf_directory).save(content, Path(source).suffix.lower())
            pipeline_fingerprint = self._pipeline_fingerprint(
                content_hash, Path(source).suffix.lower()
            )
            for document in documents:
                document.metadata["content_hash"] = content_hash
                document.metadata["pipeline_fingerprint"] = pipeline_fingerprint
            print_step("文档加载完成", 加载器="LoaderRegistry", 文档数量=len(documents))
            count = self.add_documents(documents, on_progress)
            if is_pdf:
                from app.storage.pdf_repository import PdfRepository

                PdfRepository(self.settings.pdf_directory).mark_indexed(str(documents[0].metadata["pdf_parse_id"]))
            return count

    def add_document(self, source: str, text: str) -> int:
        """程序直接传入文本时也转换为统一文档对象。"""
        content = text.encode("utf-8")
        content_hash = sha256(content).hexdigest()
        original_format = ".md" if Path(source).suffix.lower() == ".md" else ".txt"
        SourceRepository(self.settings.pdf_directory).save(content, original_format)
        return self.add_documents(
            [
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "format": Path(source).suffix.lower() or ".txt",
                        "content_hash": content_hash,
                        "original_format": original_format,
                        "pipeline_fingerprint": self._pipeline_fingerprint(
                            content_hash, Path(source).suffix.lower() or ".txt"
                        ),
                    },
                )
            ]
        )

    def add_documents(
        self,
        documents: list[Document],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> int:
        if on_progress:
            on_progress("splitting", 0, 0)
        chunks = split_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        print_step(
            "文档切分完成",
            分割器="General",
            片段数量=len(chunks),
        )
        debug_step("文档切分预览", lambda: {
            "片段": [
                {"片段编号": index, "字符数": len(doc.page_content), "原文预览": doc.page_content[:160]}
                for index, doc in enumerate(chunks[:3], 1)
            ]
        })
        if not chunks:
            raise ValueError("文档没有可导入的正文")
        file_hashes = {doc.metadata.get("content_hash") for doc in chunks}
        fingerprints = {doc.metadata.get("pipeline_fingerprint") for doc in chunks}
        source_names = {str(doc.metadata.get("source", "")) for doc in chunks}
        if len(source_names) == len(file_hashes) == len(fingerprints) == 1:
            content_hash = next(iter(file_hashes))
            fingerprint = next(iter(fingerprints))
            if isinstance(content_hash, str) and isinstance(fingerprint, str):
                existing_source = self.store.find_complete_document(
                    content_hash, fingerprint, len(chunks)
                )
                if existing_source is not None:
                    source = str(chunks[0].metadata.get("source", ""))
                    source_key = source.replace("\\", "/").casefold()
                    existing_source_key = existing_source.replace("\\", "/").casefold()
                    if source_key == existing_source_key:
                        # 处理上次写入完成但来不及清理旧版本或更新任务状态的中断。
                        self.store.remove_stale_versions(
                            source, content_hash, fingerprint
                        )
                    if on_progress:
                        on_progress("duplicate", len(chunks), len(chunks))
                    print_step(
                        "重复文档已跳过",
                        来源=source,
                        已存在来源=existing_source,
                        内容哈希=content_hash,
                        片段数量=len(chunks),
                    )
                    return 0
        if on_progress:
            on_progress("embedding", 0, len(chunks))
        vectors: list[list[float]] = []
        # PDF 原文用于证据展示；确定性上下文仅用于嵌入，不污染保存的原文。
        texts = [str(document.metadata.get("_pdf_index_text", document.page_content)) for document in chunks]
        if isinstance(self.embedder, SemanticEmbedder):
            # 语义向量适配器内部按批次处理，并在整份文档期间复用同一客户端。
            vectors = self.embedder.embed_documents_with_progress(
                texts,
                lambda completed, total: on_progress("embedding", completed, total)
                if on_progress
                else None,
            )
        else:
            # 其他嵌入实现仍按小批次处理，避免一次性传入过多文本。
            batch_size = 10
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                result = self.embedder.embed_documents(batch)
                if len(result) != len(batch):
                    raise ValueError("嵌入模型返回的批次向量数量与片段数量不一致")
                vectors.extend(result)
                if on_progress:
                    on_progress("embedding", len(vectors), len(chunks))
        if len(vectors) != len(chunks):
            raise ValueError("嵌入模型返回的向量数量与片段数量不一致")
        from .vectors import validate_vectors
        validate_vectors(vectors)
        if on_progress:
            on_progress("indexing", len(chunks), len(chunks))
        for document in chunks:
            document.metadata.pop("_pdf_index_text", None)
            document.metadata.pop("_pdf_raw_content", None)
        count = self.store.add_documents(chunks, vectors)
        if on_progress:
            on_progress("completed", count, count)
        print_step(
            "写入知识库完成",
            存储实现=type(self.store).__name__,
            新增片段=count,
            当前片段数=self.store.count(),
        )
        return count

    def _pipeline_fingerprint(self, content_hash: str, file_format: str) -> str:
        from app.pdf.profile import pipeline_fingerprint
        return pipeline_fingerprint(self.settings, content_hash, file_format)

