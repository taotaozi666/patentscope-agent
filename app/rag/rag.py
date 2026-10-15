"""Chroma 检索与回答编排，保留既有文档导入接口。"""

from collections.abc import Callable, Sequence
import re
import sqlite3

from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from app.storage import KnowledgeStore

from app.rag.retrieval import merge_candidates, fuse_rankings
from app.rag.table_evidence import expand_table_hits, answer_reference_text
from app.history import ChatHistory, as_history
from app.models import generate_reply
from app.patent_identifiers import publication_numbers
from app import prompt
from app.schemas import ChatMessage, ChunkRecord, Hit, Settings
from app.trace import debug_step, print_step


_PATENT_OVERVIEW = re.compile(r"介绍|简介|概述|总结|摘要|发明名称|申请公布号|申请人|发明人|技术内容|技术方案")
_PATENT_DETAIL = re.compile(r"权利要求|第\s*\d+\s*[项条]|图\s*\d+|表\s*\d+|实施例|步骤|参数|数值")
_PATENT_OVERVIEW_QUERY = "专利著录信息 发明名称 摘要 技术领域 发明内容 技术方案 申请人 发明人 申请日 申请公布日 国际分类号"


class RAGService:
    """检索与回答服务；导入委托给独立的索引服务。"""

    def __init__(self, embedder: Embeddings | None = None,
                 store: KnowledgeStore | None = None, settings: Settings | None = None) -> None:
        from app.indexing.service import DocumentImportService
        self.importer = DocumentImportService(embedder, store, settings)
        self.settings = self.importer.settings
        self.embedder = self.importer.embedder
        self.store = self.importer.store

    @property
    def last_import_message(self) -> str | None:
        return self.importer.last_import_message

    def add_file(self, source: str, content: bytes,
                 on_progress: Callable[[str, int, int], None] | None = None) -> int:
        return self.importer.add_file(source, content, on_progress)

    def add_document(self, source: str, text: str) -> int:
        return self.importer.add_document(source, text)

    def add_documents(self, documents: list[Document],
                      on_progress: Callable[[str, int, int], None] | None = None) -> int:
        return self.importer.add_documents(documents, on_progress)

    def _pipeline_fingerprint(self, content_hash: str, file_format: str) -> str:
        return self.importer._pipeline_fingerprint(content_hash, file_format)

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        """向量召回；专利介绍类问题额外召回著录信息与摘要候选。"""
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
        numbers = publication_numbers(question)
        overview_query = (bool(_PATENT_OVERVIEW.search(question))
                          and not _PATENT_DETAIL.search(question)
                          and bool(numbers or re.search(r"专利|发明名称|申请公布号|申请人|发明人", question)))
        queries = [question]
        if overview_query:
            queries.append(f"{' '.join(numbers)} {_PATENT_OVERVIEW_QUERY}".strip())
        keyword_enabled = (self.settings.retrieval_mode == "hybrid"
                           and callable(getattr(type(self.store), "keyword_search", None)))
        candidate_limit = min(count, max(top_k * 3, self.settings.retrieval_candidates) if keyword_enabled else top_k * 3)
        query_vectors = [self.embedder.embed_query(text) for text in queries]
        print_step("问题向量生成完成", 查询向量数=len(queries))
        repository = None
        repository_attempted = False
        while True:
            candidates: dict[str, tuple[ChunkRecord, float]] = {}
            exhausted = True
            for query in query_vectors:
                matches = (self.store.nearest(query, candidate_limit, publication_numbers=numbers)
                           if numbers else self.store.nearest(query, candidate_limit))
                if len(matches) == candidate_limit:
                    exhausted = False
                for record, score in matches:
                    previous = candidates.get(record.id)
                    if previous is None or score > previous[1]:
                        candidates[record.id] = (record, score)
            def priority(item):
                record, score = item
                section = str(record.metadata.get("patent_section", "")).lower()
                boost = {"cover": .10, "abstract": .08, "claims": .025}.get(section, 0.0) if overview_query else 0
                return (score + boost, score)
            ranked = sorted(candidates.values(), key=priority, reverse=True)
            if keyword_enabled:
                try:
                    lexical = (self.store.keyword_search(question, candidate_limit, publication_numbers=numbers)
                               if numbers else self.store.keyword_search(question, candidate_limit))
                    if len(lexical) == candidate_limit:
                        exhausted = False
                    ranked = fuse_rankings(ranked, lexical,
                        constant=self.settings.retrieval_rrf_constant,
                        vector_weight=self.settings.retrieval_vector_weight,
                        keyword_weight=self.settings.retrieval_keyword_weight)
                except (OSError, sqlite3.Error, RuntimeError) as exc:
                    print_step("关键词检索暂不可用，使用向量结果", 异常类型=type(exc).__name__)
            unique = merge_candidates(ranked)
            if any(hit.metadata.get("doc_type_kwd") == "table" for hit in unique):
                if not repository_attempted:
                    from app.storage.pdf_repository import PdfRepository
                    try:
                        repository = PdfRepository(self.settings.pdf_directory)
                    except (OSError, sqlite3.Error):
                        repository = None
                    repository_attempted = True
                unique = expand_table_hits(unique, repository)
                unique = [hit for hit in unique
                          if hit.metadata.get("table_evidence", {}).get("quality", {}).get("usage") != "blocked"]
            if len(unique) >= top_k or exhausted or candidate_limit >= count:
                hits = unique[:top_k]
                break
            candidate_limit = min(count, candidate_limit * 2)
        print_step(
            "检索结果",
            结果数量=len(hits),
            排序依据="公开号限定范围；向量和关键词候选按 RRF 融合（关键词不可用时使用向量）；概览语义候选优先著录信息与摘要；合并同文档重复文本",
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
    context = prompt.build_context([(hit.source, hit.chunk, answer_reference_text(hit)) for hit in hits])
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
