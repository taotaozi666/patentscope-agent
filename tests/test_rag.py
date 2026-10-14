import unittest

from app.rag.embeddings import DemoEmbedder
from app.rag import RAGService, answer, direct_reply
from app.rag.documents import split_into_chunks
from app.storage import ChromaKnowledgeStore


def make_rag():
    embeddings = DemoEmbedder()
    return RAGService(embeddings, store=ChromaKnowledgeStore(embeddings))


class RAGTests(unittest.TestCase):
    def test_direct_reply_recognizes_complete_greetings_and_introductions(self):
        for question in ("你好，你是?", "您好！", "你是谁？", "你好，你能做什么？", "Hello!", "Who are you?"):
            with self.subTest(question=question):
                self.assertIsNotNone(direct_reply(question))

    def test_document_questions_and_followups_are_not_intercepted(self):
        for question in (
            "你好，请介绍文档中的散热方法", "你是谁？文档里写了什么？",
            "你能做什么样的散热设计？", "你好是什么意思？", "它有什么特点？",
            "请总结刚才的资料", "hello.txt 写了什么？", "我是谁", "大熊在干嘛", " ",
        ):
            with self.subTest(question=question):
                self.assertIsNone(direct_reply(question))

    def test_markdown_heading_attached_to_each_paragraph(self):
        text = "# 城市\n\n## 04．雨伞大学\n\n第一段正文。\n\n第二段正文。\n\n## 石头商店\n\n第三段正文。"
        self.assertEqual(
            split_into_chunks(text),
            [
                "04．雨伞大学\n\n第一段正文。",
                "04．雨伞大学\n\n第二段正文。",
                "石头商店\n\n第三段正文。",
            ],
        )

    def test_long_body_repeats_heading_with_bounded_size(self):
        self.assertEqual(
            split_into_chunks("## T\n\nabcdefghijklmnop", 16, 2),
            [
                "T\n\nabcdefghijklm",
                "T\n\nlmnop",
            ],
        )
        self.assertTrue(
            all(
                len(chunk) <= 16
                for chunk in split_into_chunks("## T\n\n" + "a" * 40, 16, 12)
            )
        )
        with self.assertRaises(ValueError):
            split_into_chunks("## 很长很长的标题\n\n正文", 6, 2)

    def test_heading_without_blank_lines_and_fenced_code(self):
        self.assertEqual(
            split_into_chunks("## 标题\n正文\n### 子标题\n更多正文"),
            [
                "标题\n\n正文",
                "标题\n子标题\n\n更多正文",
            ],
        )
        self.assertEqual(
            split_into_chunks("```python\n# 代码注释\n\nprint(1)\n```"),
            [
                "```python\n# 代码注释\n\nprint(1)\n```",
            ],
        )
        self.assertEqual(split_into_chunks("# 空标题\n\n## 另一个空标题"), [])

    def test_chunk_overlap_and_empty(self):
        self.assertEqual(split_into_chunks("abcdefghij", 6, 2), ["abcdef", "efghij"])
        self.assertEqual(split_into_chunks(" \n\n "), [])
        with self.assertRaises(ValueError):
            split_into_chunks("x", 2, 2)

    def test_retrieval_and_isolation(self):
        rag = make_rag()
        rag.add_document("水果", "苹果是一种水果。")
        rag.add_document("系统", "向量和文档保存在内存，重启后清空。")
        self.assertEqual(rag.search("文档保存在内存")[0].source, "系统")
        self.assertEqual(make_rag().search("内存"), [])
        self.assertEqual(rag.search("xyzzy"), [])
        self.assertEqual(rag.search("内存", 0), [])

    def test_empty_and_demo_answer(self):
        rag = make_rag()
        with self.assertRaises(ValueError):
            rag.add_document("empty", " ")
        rag.add_document("文档", "RAG 是检索增强生成。")
        reply = answer("RAG", rag.search("RAG"), [])
        self.assertIn("[1] 文档", reply)
        self.assertIn("没有找到", answer("q", [], []))


if __name__ == "__main__":
    unittest.main()
