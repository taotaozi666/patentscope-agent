"""验证链路打印的操作关联、异常记录及最终结果。"""

import unittest
import logging
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from app.rag.embeddings import DemoEmbedder
from app.rag import RAGService, answer
from app.storage import ChromaKnowledgeStore
from app.trace import configure_logging, debug_step, print_step, trace_operation


class TraceTests(unittest.TestCase):
    def setUp(self):
        configure_logging("INFO")
        self.addCleanup(configure_logging, "INFO")

    def test_info_does_not_build_debug_payload_or_serialize_disabled_logs(self):
        with patch("app.trace.trace.json.dumps") as serialize:
            with patch("builtins.print") as output:
                print_step("隐藏内容", level=logging.DEBUG, 原文="private text")
                debug_step("详细内容", lambda: self.fail("DEBUG 数据不应在 INFO 下构建"))
                serialize.assert_not_called()
                output.assert_not_called()

    def test_debug_shows_details_and_reconfiguration_does_not_duplicate_logs(self):
        output = StringIO()
        with redirect_stdout(output):
            configure_logging("DEBUG")
            configure_logging("DEBUG")
            debug_step("详细资料", lambda: {"原文": "完整中文正文"})
        self.assertEqual(output.getvalue().count("[详细资料]"), 1)
        self.assertIn("完整中文正文", output.getvalue())
        self.assertIn("[DEBUG]", output.getvalue())

    def test_warning_keeps_failures_but_hides_normal_steps(self):
        output = StringIO()
        with redirect_stdout(output):
            configure_logging("WARNING")
            print_step("普通步骤")
            with self.assertRaises(RuntimeError), trace_operation("失败操作"):
                raise RuntimeError("private error text")
        self.assertNotIn("普通步骤", output.getvalue())
        self.assertIn("失败操作失败", output.getvalue())
        self.assertNotIn("private error text", output.getvalue())

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
        self.assertIn("回答字符数", output.getvalue())
        self.assertNotIn("RAG 是检索增强生成。", output.getvalue())
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
