"""真实 Chroma 集成测试，不调用云端向量或聊天服务。"""

import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.embeddings import DemoEmbedder
from app.rag import RAGService
from app.storage import ChromaKnowledgeStore


class StorageTests(unittest.TestCase):
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
        self.assertEqual(result.items[0].text, "第三段苹果")
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
        self.assertEqual(record.metadata["heading_path"], "# 标题")
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
        self.assertEqual(embed.call_args.args[0], ["## 云朵失物招领\n\n章节正文"])
        record = self.store.query().items[0]
        self.assertEqual(record.metadata["document_title"], "城市总标题")
        self.assertEqual(
            record.metadata["heading_path"], "# 城市总标题\n## 云朵失物招领"
        )


if __name__ == "__main__":
    unittest.main()
