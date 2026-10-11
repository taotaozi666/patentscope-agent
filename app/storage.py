"""知识库存储契约与 Chroma 实现，页面不依赖数据库内部结构。"""

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

import chromadb
from chromadb.api.types import Where
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.embeddings import DemoEmbedder, collection_name
from app.schemas import ChunkRecord, QueryResult, Settings, StoreStats
from app.trace import print_step


class KnowledgeStore(Protocol):
    """新增数据库实现时遵循此接口，业务与页面无需改变。"""

    def add_documents(self, documents: list[Document]) -> int: ...
    def stats(self) -> StoreStats: ...
    def sources(self) -> list[str]: ...
    def query(
        self,
        keyword: str = "",
        source: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> QueryResult: ...
    def get(self, record_id: str) -> ChunkRecord | None: ...
    def nearest(
        self, vector: list[float], top_k: int
    ) -> list[tuple[ChunkRecord, float]]: ...
    def clear(self) -> None: ...


class ChromaKnowledgeStore:
    """通过 LangChain Chroma 写入，通过 Chroma 原生接口查询与分页。"""

    def __init__(
        self,
        embeddings: Embeddings,
        directory: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.collection_name = collection or "session_" + uuid4().hex
        options = ChromaSettings(anonymized_telemetry=False)
        self.client = (
            chromadb.PersistentClient(path=directory, settings=options)
            if directory
            else chromadb.EphemeralClient(settings=options)
        )
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.collection = self.client.get_collection(self.collection_name)

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            raise ValueError("文档分块不能为空")
        ids = [uuid4().hex for _ in documents]
        document_ids: dict[str, str] = {}
        now = datetime.now(UTC).isoformat()
        prepared = []
        for index, document in enumerate(documents, 1):
            source = str(document.metadata.get("source", "未知来源"))
            # 一个批次中同来源片段共享导入文档 ID，重复导入另建 ID。
            document_id = document_ids.setdefault(source, uuid4().hex)
            metadata = {
                **document.metadata,
                "source": source,
                "document_id": document_id,
                "created_at": now,
                "chunk": document.metadata.get("chunk", index),
            }
            prepared.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        print_step(
            "流转至向量存储",
            存储实现=type(self).__name__,
            片段数量=len(prepared),
            集合=self.collection_name,
        )
        try:
            # LangChain 调用 Embeddings.embed_documents 后写入 Chroma。
            batch_size = self.client.get_max_batch_size()
            for start in range(0, len(prepared), batch_size):
                self.vectorstore.add_documents(
                    prepared[start : start + batch_size],
                    ids=ids[start : start + batch_size],
                )
        except Exception:
            # 多批写入途中失败时，回滚本次导入已写入的片段。
            self.collection.delete(ids=ids)
            raise
        print_step("文档向量完成", 向量数量=len(ids), 状态="已写入 Chroma")
        return len(ids)

    def stats(self) -> StoreStats:
        result = self.collection.get(include=["metadatas"])
        return StoreStats(
            len(
                {str(metadata["document_id"]) for metadata in result["metadatas"] or []}
            ),
            len(result["ids"]),
        )

    def sources(self) -> list[str]:
        return sorted(
            {
                str(metadata["source"])
                for metadata in self.collection.get(include=["metadatas"])["metadatas"]
                or []
            }
        )

    @staticmethod
    def _records(result: Any) -> list[ChunkRecord]:
        records = []
        embeddings = result.get("embeddings")
        for index, record_id in enumerate(result["ids"]):
            metadata = dict(result["metadatas"][index])
            vector = (
                [float(value) for value in embeddings[index]]
                if embeddings is not None
                else []
            )
            records.append(
                ChunkRecord(
                    record_id,
                    str(metadata["document_id"]),
                    str(metadata["source"]),
                    result["documents"][index],
                    int(metadata["chunk"]),
                    vector,
                    datetime.fromisoformat(str(metadata["created_at"])),
                    metadata,
                )
            )
        return records

    def query(
        self,
        keyword: str = "",
        source: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> QueryResult:
        if offset < 0 or limit < 1:
            raise ValueError("分页偏移必须非负，每页数量必须为正数")
        where: Where | None = {"source": source} if source is not None else None
        # Chroma 不提供原文和来源的联合模糊匹配；仅扫描文本元数据，不全量加载向量。
        candidates = self._records(
            self.collection.get(where=where, include=["documents", "metadatas"])
        )
        keyword = keyword.strip().casefold()
        filtered = [
            record
            for record in candidates
            if not keyword
            or keyword in record.text.casefold()
            or keyword in record.source.casefold()
        ]
        filtered.sort(
            key=lambda record: (
                record.created_at,
                record.document_id,
                record.chunk,
                record.id,
            )
        )
        page = filtered[offset : offset + limit]
        if page:
            records = self._records(
                self.collection.get(
                    ids=[record.id for record in page],
                    include=["documents", "metadatas", "embeddings"],
                )
            )
            by_id = {record.id: record for record in records}
            page = [by_id[record.id] for record in page]
        return QueryResult(page, len(filtered))

    def get(self, record_id: str) -> ChunkRecord | None:
        records = self._records(
            self.collection.get(
                ids=[record_id], include=["documents", "metadatas", "embeddings"]
            )
        )
        return records[0] if records else None

    def nearest(
        self, vector: list[float], top_k: int
    ) -> list[tuple[ChunkRecord, float]]:
        count = self.collection.count()
        if not count or top_k <= 0:
            return []
        print_step(
            "流转至存储检索",
            存储实现=type(self).__name__,
            知识库片段数=count,
            top_k=top_k,
        )
        # cosine 返回距离，转换为 1-distance 后用于页面显示相似度。
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if any(
            result[key] is None
            for key in ("documents", "metadatas", "distances", "embeddings")
        ):
            raise RuntimeError("Chroma 返回的检索数据不完整")
        assert result["documents"] is not None
        assert result["metadatas"] is not None
        assert result["distances"] is not None
        assert result["embeddings"] is not None
        records = self._records(
            {
                "ids": result["ids"][0],
                "documents": result["documents"][0],
                "metadatas": result["metadatas"][0],
                "embeddings": result["embeddings"][0],
            }
        )
        hits = [
            (record, max(-1.0, min(1.0, 1.0 - float(distance))))
            for record, distance in zip(records, result["distances"][0])
            if 1.0 - float(distance) > 1e-7
        ]
        print_step("Chroma 检索完成", 距离类型="cosine", 返回片段数=len(hits))
        return hits

    def clear(self) -> None:
        # 保留集合配置及维度，只删除当前集合中的记录。
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)


def create_store(embeddings: Embeddings, settings: Settings) -> KnowledgeStore:
    return ChromaKnowledgeStore(
        embeddings,
        settings.chroma_directory,
        collection_name(settings, isinstance(embeddings, DemoEmbedder)),
    )
