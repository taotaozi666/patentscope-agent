"""LangChain 文档导入、Chroma 检索与回答生成。"""

from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.conf.config import get_settings
from app.documents import loaders
from app.embeddings import create_embeddings
from app.history import ChatHistory, as_history
from app.models import generate_reply
from app.prompt import prompt
from app.schemas import ChatMessage, Hit, Settings
from app.splitters import split_documents
from app.storage import KnowledgeStore, create_store
from app.trace import print_step, trace_operation


class RAGService:
    """编排文档加载、分割、Chroma 检索和回答。"""

    def __init__(
        self,
        embedder: Embeddings | None = None,
        store: KnowledgeStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or create_embeddings(
            self.settings, offline=not self.settings.api_key
        )
        self.store = (
            store if store is not None else create_store(self.embedder, self.settings)
        )

    def add_file(self, source: str, content: bytes) -> int:
        """加载器统一返回 Document，后续流程不依赖文件格式。"""
        with trace_operation("文档导入", 来源=source):
            documents = loaders.load(source, content)
            print_step("文档加载完成", 加载器="LoaderRegistry", 文档数量=len(documents))
            return self.add_documents(documents)

    def add_document(self, source: str, text: str) -> int:
        """程序直接传入文本时也转换为统一文档对象。"""
        return self.add_documents(
            [
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "format": Path(source).suffix.lower() or ".txt",
                    },
                )
            ]
        )

    def add_documents(self, documents: list[Document]) -> int:
        chunks = split_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        print_step(
            "文档切分完成",
            分割器="RecursiveCharacterTextSplitter",
            片段数量=len(chunks),
            片段=[{"原文": doc.page_content, "元数据": doc.metadata} for doc in chunks],
        )
        if not chunks:
            raise ValueError("文档没有可导入的正文")
        count = self.store.add_documents(chunks)
        print_step(
            "写入知识库完成",
            存储实现=type(self.store).__name__,
            新增片段=count,
            当前片段数=self.store.stats().chunks,
        )
        return count

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        """按余弦相似度返回最相关的若干片段。"""
        if not self.store.stats().chunks or not question.strip() or top_k <= 0:
            print_step("跳过检索", 原因="知识库为空、问题为空或 Top K 无效")
            return []
        print_step(
            "问题进入检索",
            问题=question,
            检索模型=type(self.embedder).__name__,
            top_k=top_k,
            知识库片段数=self.store.stats().chunks,
        )
        query = self.embedder.embed_query(question)
        print_step(
            "问题向量生成完成",
            向量长度=len(query),
            向量预览=query[:8],
        )
        hits = [
            Hit(record.source, record.text, score, record.chunk)
            for record, score in self.store.nearest(query, top_k)
        ]
        print_step(
            "检索结果",
            结果数量=len(hits),
            排序依据="向量相似度，没有独立重排",
            片段=[
                {
                    "引用编号": i,
                    "来源": hit.source,
                    "片段编号": hit.chunk,
                    "相似度": round(hit.score, 6),
                    "原文": hit.text,
                }
                for i, hit in enumerate(hits, 1)
            ],
        )
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
        print_step("无检索结果", 流转="跳过聊天模型", 最终回答=reply)
        return reply
    context = prompt.build_context([(hit.source, hit.chunk, hit.text) for hit in hits])
    print_step("资料拼接完成", 引用片段数=len(hits), 资料=context)
    if client is None:
        reply = "演示模式：以下为检索到的原文，尚未调用大模型生成回答。\n\n" + context
        print_step("演示模式流转", 流转="跳过聊天模型", 最终回答=reply)
        return reply
    print_step(
        "流转至聊天模型",
        原始问题=question,
        历史消息数=as_history(history).context_count(),
    )
    return generate_reply(
        question,
        context,
        as_history(history),
        client,
        model,
    )
