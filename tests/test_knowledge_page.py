"""验证知识库两级导航、文档隔离、分页和完整原文展示。"""

import unittest
from html import escape
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from app.rag.embeddings import DemoEmbedder
from app.storage import ChromaKnowledgeStore


class KnowledgePageTests(unittest.TestCase):
    def setUp(self):
        self.store = ChromaKnowledgeStore(DemoEmbedder())
        self.text = "开头 <script>测试</script>\n" + "完整原文" * 100 + "末尾必须保留"
        self.store.add_documents([
            Document(page_content=self.text if i == 1 else f"正文 {i}",
                     metadata={"source": "同名.md", "chunk": i})
            for i in range(1, 24)
        ])
        self.store.add_documents([
            Document(page_content="另一份文档内容", metadata={"source": "同名.md", "chunk": 1})
        ])
        self.document = next(doc for doc in self.store.documents() if doc.chunks == 23)
        self.app = AppTest.from_string(
            "import streamlit as st\n"
            "from frontend.knowledge_page import render_knowledge_page\n"
            "render_knowledge_page(st.session_state.test_store)"
        )
        self.app.session_state["test_store"] = self.store

    def test_navigation_full_text_and_pagination(self):
        app = self.app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.button), 2)
        self.assertFalse(app.get("html"))
        app.button(key=f"knowledge_open_{self.document.id}").click().run()
        self.assertFalse(app.exception)
        table = app.get("html")[0].proto.body
        self.assertIn(escape(self.text), table)
        self.assertNotIn("<script>", table)
        self.assertNotIn("另一份文档内容", table)
        self.assertLess(table.index("<th>文本</th>"), table.index("<th>片段编号</th>"))
        app.number_input(key="knowledge_page_number").set_value(2).run()
        self.assertIn("正文 23", app.get("html")[0].proto.body)
        app.text_input(key="knowledge_chunk_keyword").set_value("末尾必须保留").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="knowledge_page_number").value, 1)
        self.assertIn(escape(self.text), app.get("html")[0].proto.body)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.button), 2)
        self.assertFalse(app.get("html"))

    def test_missing_document_and_empty_library(self):
        self.app.session_state["knowledge_document_id"] = "已被删除的文档"
        app = self.app.run()
        self.assertFalse(app.exception)
        self.assertIn("该文档已不存在", app.info[0].value)
        self.assertEqual(len(app.button), 2)
        self.store.clear()
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("知识库暂无数据", app.info[0].value)
        self.assertFalse(app.button)

    def test_page_queries_once_and_loads_vector_on_demand(self):
        app = self.app.run()
        app.button(key=f"knowledge_open_{self.document.id}").click().run()
        with patch.object(self.store, "query", wraps=self.store.query) as query, patch.object(
            self.store, "get", wraps=self.store.get
        ) as get:
            app.number_input(key="knowledge_page_number").set_value(2).run()
            self.assertFalse(app.exception)
            query.assert_called_once()
            self.assertEqual(query.call_args.kwargs["offset"], 20)
            self.assertFalse(query.call_args.kwargs["include_vectors"])
            get.assert_not_called()
            app.checkbox[0].check().run()
            self.assertFalse(app.exception)
            get.assert_called_once()
            self.assertTrue(any("向量长度：2048" in caption.value for caption in app.caption))

    def test_page_clamps_after_document_shrinks(self):
        app = self.app.run()
        app.button(key=f"knowledge_open_{self.document.id}").click().run()
        app.number_input(key="knowledge_page_number").set_value(2).run()
        records = self.store.query(document_id=self.document.id, limit=50).items
        self.store.collection.delete(ids=[record.id for record in records[1:]])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="knowledge_page_number").value, 1)
        self.assertIn(escape(self.text), app.get("html")[0].proto.body)


if __name__ == "__main__":
    unittest.main()
