"""LangChain 文档导入、Chroma 检索与回答生成。"""

from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.config import get_settings
from app.rag.documents import loaders, split_documents
from app.rag.embeddings import DemoEmbedder, SemanticEmbedder, create_embeddings
from app.history import ChatHistory, as_history
from app.models import generate_reply
from app import prompt
from app.schemas import ChatMessage, Hit, Settings
from app.storage import KnowledgeStore, create_store
from app.trace import debug_step, print_step, trace_operation


class RAGService:
    """编排文档加载、分割、Chroma 检索和回答。"""

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
            if is_pdf:
                from app.pdf.service import prepare_pdf

                documents, message = prepare_pdf(source, content, self.settings)
                self.last_import_message = message or None
            else:
                documents = loaders.load(source, content)
            content_hash = sha256(content).hexdigest()
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
        return self.add_documents(
            [
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "format": Path(source).suffix.lower() or ".txt",
                        "content_hash": content_hash,
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
        chunks = split_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        print_step(
            "文档切分完成",
            分割器="PDF 结构分块" if any(doc.metadata.get("pdf_prechunked") for doc in documents) else "RecursiveCharacterTextSplitter",
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
                    source_key = source.replace("\\", "/").rsplit("/", 1)[
                        -1
                    ].casefold()
                    existing_source_key = existing_source.replace("\\", "/").rsplit(
                        "/", 1
                    )[-1].casefold()
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
                vectors.extend(
                    self.embedder.embed_documents(texts[start : start + batch_size])
                )
                if on_progress:
                    on_progress("embedding", len(vectors), len(chunks))
        if len(vectors) != len(chunks):
            raise ValueError("嵌入模型返回的向量数量与片段数量不一致")
        if on_progress:
            on_progress("indexing", len(chunks), len(chunks))
        for document in chunks:
            document.metadata.pop("_pdf_index_text", None)
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
        """把切分策略纳入指纹，配置改变后同一文件会重新建索引。"""
        if file_format == ".pdf":
            from app.pdf.parser import PARSER_VERSION
            from app.pdf.chunking import CHUNKER_VERSION

            from app.prompt.prompt import PDF_CONTEXT_VERSION

            return sha256(f"{content_hash}:{PARSER_VERSION}:{CHUNKER_VERSION}:{self.settings.chunk_size}:"
                          f"{self.settings.pdf_target_size}:{self.settings.pdf_context_mode}:{PDF_CONTEXT_VERSION}:"
                          f"{self.settings.chat_model if self.settings.pdf_context_mode == 'model' else ''}".encode()).hexdigest()
        strategy = (
            f"{content_hash}:{file_format}:recursive-heading-v1:"
            f"{self.settings.chunk_size}:{self.settings.chunk_overlap}"
        )
        return sha256(strategy.encode("utf-8")).hexdigest()

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        """按余弦相似度返回最相关的若干片段。"""
        count = self.store.count()
        if not count or not question.strip() or top_k <= 0:
            print_step("跳过检索", 原因="知识库为空、问题为空或 Top K 无效")
            return []
        print_step(
            "问题进入检索",
            问题字符数=len(question),
            检索模型=type(self.embedder).__name__,
            top_k=top_k,
            知识库片段数=count,
        )
        query = self.embedder.embed_query(question)
        print_step(
            "问题向量生成完成",
            向量长度=len(query),
        )
        debug_step("问题向量预览", lambda: {"向量预览": query[:8]})
        hits = [
            Hit(record.source, record.text, score, record.chunk)
            for record, score in self.store.nearest(query, top_k)
        ]
        print_step(
            "检索结果",
            结果数量=len(hits),
            排序依据="向量相似度，没有独立重排",
        )
        debug_step("检索原文", lambda: {
            "片段": [
                {"引用编号": i, "来源": hit.source, "片段编号": hit.chunk,
                 "相似度": round(hit.score, 6), "原文": hit.text}
                for i, hit in enumerate(hits, 1)
            ]
        })
        return hits


def answer(
    question: str,
    hits: list[Hit],
    history: ChatHistory | Sequence[ChatMessage],
    client: OpenAI | None = None,
    model: str | None = None,
) -> str:
    """根据本轮资料生成完整回答，无客户端时返回检索原文。"""
    if not hits:
        reply = "知识库中没有找到相关内容，请补充文档或换一种问法。"
        print_step("无检索结果", 流转="跳过聊天模型", 回答字符数=len(reply))
        return reply
    context = prompt.build_context([(hit.source, hit.chunk, hit.text) for hit in hits])
    print_step("资料拼接完成", 引用片段数=len(hits), 资料字符数=len(context))
    debug_step("完整资料", lambda: {"资料": context})
    if client is None:
        reply = "演示模式：以下为检索到的原文，尚未调用大模型生成回答。\n\n" + context
        print_step("演示模式流转", 流转="跳过聊天模型", 回答字符数=len(reply))
        return reply
    print_step(
        "流转至聊天模型",
        问题字符数=len(question),
        历史消息数=as_history(history).context_count(),
    )
    return generate_reply(
        question,
        context,
        as_history(history),
        client,
        model,
    )
