"""真实 Chroma 集成测试，不调用云端向量或聊天服务。"""

import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag.embeddings import DemoEmbedder
from app.rag import RAGService
from app.storage import ChromaKnowledgeStore


class StorageTests(unittest.TestCase):
    def test_same_name_different_content_is_preserved_and_reindex_is_scoped(self):
        rag = RAGService(self.embeddings, store=self.store)
        first, second = "苹果是一种水果。" * 10, "香蕉是一种水果。" * 10
        rag.add_file("同名.txt", first.encode())
        rag.add_file("同名.txt", second.encode())
        self.assertEqual(self.store.stats().documents, 2)
        original_ids = {doc.id for doc in self.store.documents()}
        self.assertEqual(rag.add_file("同名.txt", second.encode()), 0)
        rag = RAGService(self.embeddings, store=self.store, settings=replace(rag.settings, chunk_size=50, chunk_overlap=5))
        rag.add_file("同名.txt", first.encode())
        self.assertEqual({doc.id for doc in self.store.documents()}, original_ids)
        records = self.store.query(limit=100).items
        hashes = {record.metadata["content_hash"] for record in records}
        self.assertEqual(len(hashes), 2)
        for content_hash in hashes:
            self.assertEqual(len({record.metadata["pipeline_fingerprint"] for record in records
                                  if record.metadata["content_hash"] == content_hash}), 1)
        self.assertTrue(any(record.text == second for record in records))

    def test_same_basename_in_different_paths_is_not_replaced(self):
        rag = RAGService(self.embeddings, store=self.store)
        rag.add_file("folder-a/report.txt", "苹果资料".encode())
        rag.add_file("folder-b/report.txt", "香蕉资料".encode())
        self.assertEqual(self.store.stats().documents, 2)
        self.assertEqual(set(self.store.sources()), {"folder-a/report.txt", "folder-b/report.txt"})

    def test_same_name_failed_new_import_keeps_existing_content(self):
        rag = RAGService(self.embeddings, store=self.store)
        rag.add_file("report.txt", b"existing content")
        original = self.store.collection.upsert
        def fail_after_write(**kwargs):
            original(**kwargs)
            raise RuntimeError("write failed")
        with patch.object(self.store.collection, "upsert", side_effect=fail_after_write), self.assertRaises(RuntimeError):
            rag.add_file("report.txt", b"different content")
        self.assertEqual(self.store.stats().documents, 1)
        self.assertEqual(self.store.query().items[0].text, "existing content")

    def setUp(self):
        self.embeddings = DemoEmbedder()
        self.store = ChromaKnowledgeStore(self.embeddings)

    def docs(self, source, texts):
        return [
            Document(page_content=text, metadata={"source": source, "chunk": i})
            for i, text in enumerate(texts, 1)
        ]

    def test_query_pagination_and_duplicate_names(self):
        store = self.store
        store.add_documents(self.docs("同名文档", ["第一段苹果", "第二段香蕉"]))
        store.add_documents(self.docs("同名文档", ["第三段苹果"]))
        self.assertEqual(store.stats().documents, 2)
        self.assertEqual(store.stats().chunks, 3)
        self.assertEqual(store.sources(), ["同名文档"])
        result = store.query("苹果", "同名文档", offset=1, limit=1)
        self.assertEqual(result.total, 2)
        matches = store.query("苹果", "同名文档").items
        self.assertEqual({record.text for record in matches}, {"第一段苹果", "第三段苹果"})
        self.assertEqual(result.items, matches[1:2])
        self.assertEqual(store.get(result.items[0].id), result.items[0])
        self.assertEqual(store.query(source="不存在").total, 0)
        self.assertEqual(store.query(offset=10).items, [])
        self.assertEqual(len({r.id for r in store.query().items}), 3)
        self.assertEqual(len({r.document_id for r in store.query().items}), 2)
        self.assertEqual(len(store.query().items[0].vector), self.embeddings.dimensions)

    def test_nearest_clear_and_isolation(self):
        store = self.store
        with self.assertRaises(ValueError):
            store.add_documents([])
        store.add_documents(self.docs("水果", ["苹果"]))
        self.assertEqual(
            store.nearest(self.embeddings.embed_query("苹果"), 1)[0][0].source, "水果"
        )
        self.assertEqual(ChromaKnowledgeStore(self.embeddings).stats().chunks, 0)
        store.clear()
        self.assertEqual(store.stats().documents, 0)
        self.assertEqual(store.query().total, 0)

    def test_document_listing_and_chunk_scope(self):
        self.store.add_documents(self.docs("同名.md", ["第一段", "第二段", "第三段"]))
        self.store.add_documents(self.docs("同名.md", ["另一份文档"]))
        with patch.object(self.store.collection, "get", wraps=self.store.collection.get) as get:
            documents = self.store.documents()
        self.assertEqual(get.call_args.kwargs["include"], ["metadatas"])
        self.assertEqual(len(documents), 2)
        first = next(doc for doc in documents if doc.chunks == 3)
        second = next(doc for doc in documents if doc.chunks == 1)
        self.assertNotEqual(first.id, second.id)
        result = self.store.query(document_id=first.id, offset=1, limit=1)
        self.assertEqual(result.total, 3)
        self.assertEqual(result.items[0].text, "第二段")
        self.assertEqual(result.items[0].document_id, first.id)
        self.assertEqual(self.store.query("另一份", document_id=first.id).total, 0)
        self.assertEqual(self.store.query(source="不存在", document_id=first.id).total, 0)
        self.assertEqual(self.store.query(document_id="不存在").total, 0)

        # 重试或部分更新后，导入时间可能不同，详情仍按片段编号排列。
        records = self.store.query(document_id=first.id).items
        self.store.collection.update(
            ids=[records[0].id], metadatas=[{"created_at": "2099-01-01T00:00:00+00:00"}]
        )
        self.assertEqual(
            [record.chunk for record in self.store.query(document_id=first.id).items],
            [1, 2, 3],
        )
        self.store.clear()
        self.assertEqual(self.store.documents(), [])

    def test_persistence_and_collection_isolation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first = ChromaKnowledgeStore(self.embeddings, directory, "test_collection")
            first.add_documents(self.docs("文档.md", ["持久化保存雨伞大学"]))
            reopened = ChromaKnowledgeStore(
                self.embeddings, directory, "test_collection"
            )
            self.assertEqual(reopened.stats().chunks, 1)
            self.assertEqual(
                reopened.nearest(self.embeddings.embed_query("雨伞大学"), 1)[0][
                    0
                ].source,
                "文档.md",
            )
            other = ChromaKnowledgeStore(self.embeddings, directory, "other_collection")
            self.assertEqual(other.stats().chunks, 0)
            reopened.clear()
            self.assertEqual(first.stats().chunks, 0)

    def test_injected_store_used_by_rag(self):
        rag = RAGService(self.embeddings, store=self.store)
        rag.add_file("测试.md", "# 标题\n\n知识库保存文档。".encode())
        self.assertIs(rag.store, self.store)
        record = self.store.query().items[0]
        self.assertEqual(record.metadata["format"], ".md")
        self.assertEqual(record.metadata["heading_path"], "标题")
        self.assertEqual(record.metadata["document_title"], "标题")
        self.assertEqual(record.text, "知识库保存文档。")
        self.assertEqual(rag.search("知识库")[0].source, "测试.md")

    def test_failed_import_rolls_back_only_new_records(self):
        self.store.add_documents(self.docs("旧文档", ["原有内容"]))
        original = self.store.vectorstore.add_documents

        def fail_after_write(documents, **kwargs):
            original(documents, **kwargs)
            raise RuntimeError("模拟写入后异常")

        with (
            patch.object(
                self.store.vectorstore, "add_documents", side_effect=fail_after_write
            ),
            self.assertRaises(RuntimeError),
        ):
            self.store.add_documents(self.docs("新文档", ["新内容"]))
        self.assertEqual(self.store.stats().chunks, 1)
        self.assertEqual(self.store.sources(), ["旧文档"])

    def test_embedding_receives_sections_without_document_title(self):
        rag = RAGService(self.embeddings, store=self.store)
        with patch.object(
            self.embeddings, "embed_documents", wraps=self.embeddings.embed_documents
        ) as embed:
            rag.add_file(
                "city.md", "# 城市总标题\n\n## 云朵失物招领\n\n章节正文".encode()
            )
        self.assertEqual(embed.call_args.args[0], ["云朵失物招领\n\n章节正文"])
        record = self.store.query().items[0]
        self.assertEqual(record.metadata["document_title"], "城市总标题")
        self.assertEqual(
            record.metadata["heading_path"], "城市总标题\n云朵失物招领"
        )

    def test_browsing_loads_only_target_page_without_vectors(self):
        self.store.add_documents(self.docs("large.txt", [f"text {i}" for i in range(100)]))
        with patch.object(self.store.collection, "get", wraps=self.store.collection.get) as get:
            result = self.store.query(offset=40, limit=10, include_vectors=False)
        self.assertEqual(result.total, 100)
        self.assertEqual(len(result.items), 10)
        self.assertEqual(result.items[0].text, "text 40")
        self.assertEqual(get.call_args_list[0].kwargs["include"], ["metadatas"])
        self.assertEqual(len(get.call_args_list[1].kwargs["ids"]), 10)
        self.assertNotIn("embeddings", get.call_args_list[1].kwargs["include"])
        self.assertTrue(all(not record.vector for record in result.items))

    def test_query_tolerates_concurrent_deletion(self):
        self.store.add_documents(self.docs("test.txt", ["text"]))
        original = self.store.collection.get
        def remove_after_scan(*args, **kwargs):
            result = original(*args, **kwargs)
            if "ids" not in kwargs:
                self.store.collection.delete(ids=result["ids"])
            return result
        with patch.object(self.store.collection, "get", side_effect=remove_after_scan):
            result = self.store.query()
        self.assertEqual(result.items, [])
        self.assertEqual(self.store.count(), 0)

    def test_count_and_retrieval_avoid_unused_data(self):
        self.store.add_documents(self.docs("test.txt", ["apple"]))
        with patch.object(self.store.collection, "get", wraps=self.store.collection.get) as get:
            self.assertEqual(self.store.count(), 1)
            get.assert_not_called()
        with patch.object(self.store.collection, "query", wraps=self.store.collection.query) as query:
            self.store.nearest(self.embeddings.embed_query("apple"), 1)
        self.assertNotIn("embeddings", query.call_args.kwargs["include"])

    def test_query_preserves_case_insensitive_keyword_matching(self):
        self.store.add_documents(self.docs("UPPER.txt", ["APPLE", "banana"]))
        self.assertEqual(self.store.query("apple").total, 1)
        self.assertEqual(self.store.query("upper").total, 2)


if __name__ == "__main__":
    unittest.main()
