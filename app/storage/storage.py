"""知识库存储契约与 Chroma 实现，页面不依赖数据库内部结构。"""

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

import chromadb
from chromadb.api.types import Include, Metadata, PyEmbeddings
from chromadb.api.types import Where
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.schemas import ChunkRecord, DocumentRecord, QueryResult, Settings, StoreStats
from app.trace import print_step


class KnowledgeStore(Protocol):
    """新增数据库实现时遵循此接口，业务与页面无需改变。"""

    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]] | None = None
    ) -> int: ...
    def find_complete_document(
        self, content_hash: str, pipeline_fingerprint: str, expected_chunks: int
    ) -> str | None: ...
    def find_content_hash(self, content_hash: str) -> str | None: ...
    def has_source(self, source: str) -> bool: ...
    def remove_stale_versions(
        self, source: str, content_hash: str, pipeline_fingerprint: str
    ) -> None: ...
    def stats(self) -> StoreStats: ...
    def count(self) -> int: ...
    def sources(self) -> list[str]: ...
    def documents(self) -> list[DocumentRecord]: ...
    def query(
        self,
        keyword: str = "",
        source: str | None = None,
        offset: int = 0,
        limit: int = 20,
        document_id: str | None = None,
        include_vectors: bool = True,
    ) -> QueryResult: ...
    def get(self, record_id: str) -> ChunkRecord | None: ...
    def nearest(
        self, vector: list[float], top_k: int
    ) -> list[tuple[ChunkRecord, float]]: ...
    def clear(self) -> None: ...


class ChromaKnowledgeStore:
    """通过 Chroma 原生集合写入预计算向量，并执行查询与分页。"""

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

    def add_documents(
        self, documents: list[Document], embeddings: list[list[float]] | None = None
    ) -> int:
        if not documents:
            raise ValueError("文档分块不能为空")
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError("向量数量与片段数量不一致")
        ids: list[str] = []
        document_ids: dict[str, str] = {}
        now = datetime.now(UTC).isoformat()
        prepared = []
        for index, document in enumerate(documents, 1):
            source = str(document.metadata.get("source", "未知来源"))
            source_key = source.replace("\\", "/").rsplit("/", 1)[-1].casefold()
            content_hash = document.metadata.get("content_hash")
            pipeline_fingerprint = document.metadata.get("pipeline_fingerprint")
            if content_hash and pipeline_fingerprint:
                # 同一路径的同一版本沿用文档和片段 ID，重试时用 upsert 覆盖半成品。
                document_id = document_ids.setdefault(
                    source_key,
                    str(document.metadata.get("document_id") or uuid5(NAMESPACE_URL, f"{self.collection_name}:{source_key}").hex),
                )
                chunk_number = int(document.metadata.get("chunk", index))
                record_id = uuid5(
                    NAMESPACE_URL,
                    f"{document_id}:{pipeline_fingerprint}:{chunk_number}",
                ).hex
            else:
                # 未带文件版本指纹的旧调用保持独立新增语义。
                document_id = document_ids.setdefault(source, uuid4().hex)
                record_id = uuid4().hex
            ids.append(record_id)
            metadata = {
                **document.metadata,
                "source": source,
                "source_key": source_key,
                "document_id": document_id,
                "created_at": now,
                "chunk": document.metadata.get("chunk", index),
                "chunk_total": len(documents),
            }
            if embeddings is not None:
                metadata["vector_dimensions"] = len(embeddings[index - 1])
            prepared.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        print_step(
            "流转至向量存储",
            存储实现=type(self).__name__,
            片段数量=len(prepared),
            集合=self.collection_name,
        )
        stable_flags = [
            bool(
                doc.metadata.get("content_hash")
                and doc.metadata.get("pipeline_fingerprint")
            )
            for doc in prepared
        ]
        if any(stable_flags) and not all(stable_flags):
            raise ValueError("同一批次不能混合有版本指纹和无版本指纹的片段")
        try:
            batch_size = self.client.get_max_batch_size()
            has_stable_ids = all(stable_flags)
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                batch_ids = ids[start : start + batch_size]
                if embeddings is None and not has_stable_ids:
                    self.vectorstore.add_documents(batch, ids=batch_ids)
                    continue

                if embeddings is None:
                    raise RuntimeError("稳定 ID 写入需要预先生成向量")

                # 明确保存本批向量，避免类型检查器将可选参数推断为 None。
                # SDK 的 PyEmbeddings 类型表示 Python 嵌套序列；这里的向量
                # 正是由 Embeddings 接口生成的浮点数序列，只需在 SDK 边界标注。
                batch_embeddings = cast(
                    PyEmbeddings, embeddings[start : start + len(batch)]
                )
                # 当前 LangChain Chroma 没有 add_embeddings；直接使用公开的
                # Chroma Collection.add，复用已生成的向量，避免再次调用嵌入接口。
                if has_stable_ids:
                    self.collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=[doc.page_content for doc in batch],
                        metadatas=[doc.metadata for doc in batch],
                    )
                else:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=[doc.page_content for doc in batch],
                        metadatas=[doc.metadata for doc in batch],
                    )
            if has_stable_ids:
                source = str(prepared[0].metadata["source"])
                content_hash = str(prepared[0].metadata["content_hash"])
                pipeline_fingerprint = str(
                    prepared[0].metadata["pipeline_fingerprint"]
                )
                self.remove_stale_versions(source, content_hash, pipeline_fingerprint)
        except Exception:
            # 多批写入途中失败时，回滚本次导入已写入的片段。
            self.collection.delete(ids=ids)
            raise
        print_step("文档向量完成", 向量数量=len(ids), 状态="已写入 Chroma")
        return len(ids)

    def find_complete_document(
        self, content_hash: str, pipeline_fingerprint: str, expected_chunks: int
    ) -> str | None:
        """只将分块齐全的同版本文档视为重复，避免跳过崩溃留下的半批数据。"""
        result = self.collection.get(
            where={
                "$and": [
                    {"content_hash": {"$eq": content_hash}},
                    {"pipeline_fingerprint": {"$eq": pipeline_fingerprint}},
                ]
            },
            include=["metadatas"],
        )
        grouped: dict[str, list[Metadata]] = {}
        for metadata in result["metadatas"] or []:
            key = str(metadata.get("document_id", ""))
            grouped.setdefault(key, []).append(metadata)
        for records in grouped.values():
            chunks = {
                int(chunk_value)
                for item in records
                if isinstance(
                    (chunk_value := item.get("chunk", 0)), (str, int, float)
                )
                and not isinstance(chunk_value, bool)
            }
            if len(records) == expected_chunks and chunks == set(
                range(1, expected_chunks + 1)
            ):
                return str(records[0].get("source", ""))
        return None

    def find_content_hash(self, content_hash: str) -> str | None:
        """用于上传预览提示；最终是否跳过仍由完整分块校验决定。"""
        result = self.collection.get(
            where={"content_hash": {"$eq": content_hash}},
            limit=1,
            include=["metadatas"],
        )
        metadatas = result["metadatas"] or []
        return str(metadatas[0].get("source", "")) if metadatas else None

    def has_source(self, source: str) -> bool:
        """检查同名逻辑文件是否已存在，用于提示可能的版本更新。"""
        source_key = source.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        filters: tuple[Where, Where] = (
            {"source_key": {"$eq": source_key}},
            {"source": {"$eq": source}},
        )
        for where in filters:
            result = self.collection.get(where=where, limit=1, include=[])
            if result["ids"]:
                return True
        return False

    def remove_stale_versions(
        self, source: str, content_hash: str, pipeline_fingerprint: str
    ) -> None:
        """新版本写全后清除同一逻辑文件的旧版本与历史遗留片段。"""
        source_key = source.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        records_by_id: dict[str, Metadata] = {}
        filters: tuple[Where, Where] = (
            {"source_key": {"$eq": source_key}},
            {"source": {"$eq": source}},
        )
        for where in filters:
            result = self.collection.get(where=where, include=["metadatas"])
            for record_id, metadata in zip(
                result["ids"], result["metadatas"] or []
            ):
                records_by_id[record_id] = metadata
        stale_ids = [
            record_id
            for record_id, metadata in records_by_id.items()
            if metadata.get("content_hash") != content_hash
            or metadata.get("pipeline_fingerprint") != pipeline_fingerprint
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

    def stats(self) -> StoreStats:
        documents = self.documents()
        return StoreStats(len(documents), sum(doc.chunks for doc in documents))

    def count(self) -> int:
        """仅统计片段数量，不读取全文或元数据。"""
        return self.collection.count()

    def sources(self) -> list[str]:
        return sorted(
            {
                str(metadata["source"])
                for metadata in self.collection.get(include=["metadatas"])["metadatas"]
                or []
            }
        )

    def documents(self) -> list[DocumentRecord]:
        """按文档 ID 汇总实际入库片段，同名的独立导入仍分别展示。"""
        grouped: dict[str, list[Metadata]] = {}
        for metadata in self.collection.get(include=["metadatas"])["metadatas"] or []:
            grouped.setdefault(str(metadata["document_id"]), []).append(metadata)
        documents = []
        for document_id, records in grouped.items():
            first = min(records, key=lambda item: int(str(item["chunk"])))
            documents.append(
                DocumentRecord(
                    id=document_id,
                    source=str(first["source"]),
                    title=str(first.get("document_title", "")),
                    format=str(first.get("format", "")),
                    chunks=len(records),
                    created_at=min(
                        datetime.fromisoformat(str(item["created_at"]))
                        for item in records
                    ),
                )
            )
        return sorted(documents, key=lambda doc: (doc.created_at, doc.id), reverse=True)

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
        document_id: str | None = None,
        include_vectors: bool = True,
    ) -> QueryResult:
        if offset < 0 or limit < 1:
            raise ValueError("分页偏移必须非负，每页数量必须为正数")
        conditions: list[Where] = []
        if source is not None:
            conditions.append({"source": {"$eq": source}})
        if document_id is not None:
            conditions.append({"document_id": {"$eq": document_id}})
        where: Where | None = None
        if conditions:
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        keyword = keyword.strip().casefold()
        # 无关键词时只扫描排序所需的元数据，正文仅加载目标页。
        candidates = self.collection.get(
            where=where, include=["metadatas", "documents"] if keyword else ["metadatas"]
        )
        metadatas = candidates["metadatas"] or []
        ids = [
            record_id for index, record_id in enumerate(candidates["ids"])
            if not keyword
            or keyword in str(metadatas[index]["source"]).casefold()
            or keyword in str((candidates["documents"] or [])[index]).casefold()
        ]
        metadata_by_id = dict(zip(candidates["ids"], metadatas))

        def order(record_id: str) -> tuple:
            metadata = metadata_by_id[record_id]
            chunk = int(str(metadata["chunk"]))
            if document_id is not None:
                return chunk, record_id
            return (
                datetime.fromisoformat(str(metadata["created_at"])),
                str(metadata["document_id"]), chunk, record_id,
            )

        ids.sort(key=order)
        page_ids = ids[offset : offset + limit]
        if not page_ids:
            return QueryResult([], len(ids))
        include: Include = ["documents", "metadatas"]
        if include_vectors:
            include.append("embeddings")
        records = self._records(self.collection.get(ids=page_ids, include=include))
        by_id = {record.id: record for record in records}
        # 两次读取之间可能发生更新或清空，不假定记录仍然存在。
        page = [by_id[record_id] for record_id in page_ids if record_id in by_id]
        return QueryResult(page, len(ids))

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
            include=["documents", "metadatas", "distances"],
        )
        if any(
            result[key] is None
            for key in ("documents", "metadatas", "distances")
        ):
            raise RuntimeError("Chroma 返回的检索数据不完整")
        assert result["documents"] is not None
        assert result["metadatas"] is not None
        assert result["distances"] is not None
        records = self._records(
            {
                "ids": result["ids"][0],
                "documents": result["documents"][0],
                "metadatas": result["metadatas"][0],
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


def create_store(
    embeddings: Embeddings, settings: Settings, offline: bool = False
) -> KnowledgeStore:
    return ChromaKnowledgeStore(
        embeddings,
        settings.chroma_directory,
        collection_name(settings, offline),
    )


def collection_name(settings: Settings, offline: bool = False) -> str:
    """按模型和服务地址分集合，密钥更换不会丢失已有知识库。"""
    # 沿用历史身份前缀以保持已有集合可访问；服务地址和模型名由配置决定。
    identity = (
        "demo-hash-v1-2048"
        if offline
        else f"qwen-compatible:{settings.base_url}:{settings.embedding_model}"
    )
    suffix = sha256(identity.encode()).hexdigest()[:16]
    return f"{settings.chroma_collection}_{suffix}"
