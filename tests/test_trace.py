"""验证链路打印的操作关联、异常记录及最终结果。"""

import unittest
from contextlib import redirect_stdout
from io import StringIO

from app.embeddings import DemoEmbedder
from app.rag import RAGService, answer
from app.storage import ChromaKnowledgeStore
from app.trace import print_step, trace_operation


class TraceTests(unittest.TestCase):
    def test_chat_steps_share_operation_id(self):
        output = StringIO()
        with redirect_stdout(output):
            embeddings = DemoEmbedder()
            rag = RAGService(embeddings, store=ChromaKnowledgeStore(embeddings))
            rag.add_document("文档", "RAG 是检索增强生成。")
            with trace_operation("聊天", 问题="RAG"):
                hits = rag.search("RAG")
                answer("RAG", hits, [])
        lines = output.getvalue().splitlines()
        chat_lines = lines[
            next(i for i, line in enumerate(lines) if "[聊天开始]" in line) :
        ]
        ids = {line.split("][", 2)[1] for line in chat_lines}
        self.assertEqual(len(ids), 1)
        self.assertIn("问题向量生成完成", output.getvalue())
        self.assertIn("检索结果", output.getvalue())
        self.assertIn("最终回答", output.getvalue())
        self.assertIn("聊天结束", output.getvalue())

    def test_error_does_not_print_exception_body(self):
        output = StringIO()
        with redirect_stdout(output):
            with self.assertRaises(RuntimeError), trace_operation("测试"):
                raise RuntimeError("secret-in-error-body")
            print_step("外部步骤")
        self.assertIn("测试失败", output.getvalue())
        self.assertNotIn("secret-in-error-body", output.getvalue())
        self.assertIn("[独立调用][外部步骤]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
