"""验证历史存取、追问上下文、模型历史上限及会话隔离。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.history import ChatHistory, get_session_history
from app.schemas import Hit, Settings


class HistoryTests(unittest.TestCase):
    def test_ui_missing_knowledge_does_not_open_general_chat(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings("test-key", "chat", "embedding", chroma_directory=directory)
            with patch("app.config.get_settings", return_value=settings):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/ui.py")).run()
                rag = app.session_state["rag"]
                with patch.object(rag, "search", wraps=rag.search) as search, patch(
                    "app.models.create_client"
                ) as factory:
                    app.chat_input[0].set_value("大熊在干嘛").run()
                    self.assertFalse(app.exception)
                    search.assert_called_once_with("大熊在干嘛", 3)
                    factory.assert_not_called()
                self.assertIn("没有找到", app.session_state["chat_history"].messages()[-1]["content"])

    def test_ui_document_question_calls_model_once_and_closes_client_on_failure(self):
        for response_content in ("文档中的叙述者[1]", "", RuntimeError("model unavailable")):
            with self.subTest(response_content=response_content), TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                settings = Settings("test-key", "chat", "embedding", chroma_directory=directory)
                with patch("app.config.get_settings", return_value=settings):
                    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/ui.py")).run()
                    rag = app.session_state["rag"]
                    with patch.object(rag, "search", return_value=[Hit("文档", "原文", 1.0, 1)]) as search, patch(
                        "app.models.create_client"
                    ) as factory:
                        client = factory.return_value
                        if isinstance(response_content, Exception):
                            client.chat.completions.create.side_effect = response_content
                        else:
                            client.chat.completions.create.return_value = SimpleNamespace(
                                choices=[SimpleNamespace(message=SimpleNamespace(content=response_content))]
                            )
                        question = "请解释文档中的第一人称我到底是谁"
                        app.chat_input[0].set_value(question).run()
                        self.assertFalse(app.exception)
                        search.assert_called_once_with(question, 3)
                        factory.assert_called_once_with(settings)
                        client.chat.completions.create.assert_called_once()
                        client.close.assert_called_once()
                    if response_content == "文档中的叙述者[1]":
                        self.assertTrue(app.session_state["chat_history"].messages()[-1]["hits"])
                    else:
                        self.assertTrue(app.error)
                        self.assertEqual(len(app.session_state["chat_history"]), 0)

    def test_ui_fixed_greeting_skips_models_and_retrieval_with_empty_library(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings("test-key", "chat", "embedding", chroma_directory=directory)
            with patch("app.config.get_settings", return_value=settings):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/ui.py")).run()
                self.assertFalse(app.exception)
                rag = app.session_state["rag"]
                with patch.object(rag, "search") as search, patch.object(
                    rag.embedder, "embed_query"
                ) as embed, patch("app.models.create_client") as create_client:
                    app.chat_input[0].set_value("你好，你是？").run()
                    self.assertFalse(app.exception)
                    search.assert_not_called()
                    embed.assert_not_called()
                    create_client.assert_not_called()
                messages = app.session_state["chat_history"].messages()
                self.assertIn("知识库问答助手", messages[-1]["content"])
                self.assertEqual(messages[-1]["hits"], [])
                self.assertFalse(messages[-2]["retrieval"])
                self.assertFalse(any(expander.label == "查看检索来源" for expander in app.expander))

    def test_ui_greeting_with_documents_then_mixed_question_uses_rag(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings("", "chat", "embedding", chroma_directory=directory)
            with patch("app.config.get_settings", return_value=settings):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/ui.py")).run()
                rag = app.session_state["rag"]
                rag.add_document("fruit.txt", "苹果是一种水果。")
                with patch.object(rag, "search", wraps=rag.search) as search:
                    app.chat_input[0].set_value("你能做什么？").run()
                    search.assert_not_called()
                    question = "你好，请介绍文档中的苹果"
                    app.chat_input[0].set_value(question).run()
                    self.assertFalse(app.exception)
                    search.assert_called_once_with(question, 3)
                self.assertTrue(app.session_state["chat_history"].messages()[-1]["hits"])

    def test_direct_turns_remain_in_chat_but_are_excluded_from_retrieval_history(self):
        history = ChatHistory()
        history.add_turn("苹果是什么？", "一种水果")
        history.add_turn("你好，你是？", "知识库助手", retrieval=False)
        self.assertEqual(len(history), 4)
        self.assertEqual(history.recent_questions(), ["苹果是什么？"])
        self.assertEqual(history.retrieval_question("它有什么特点？"), "苹果是什么？\n它有什么特点？")
        restored = ChatHistory(history.messages())
        self.assertEqual(restored.recent_questions(), ["苹果是什么？"])
        messages = restored.build_model_messages("系统", "当前问题")
        self.assertTrue(any(message.get("content") == "你好，你是？" for message in messages))
        self.assertTrue(all("retrieval" not in message for message in messages))

    def test_ui_followup_performs_one_search(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            settings = Settings("", "chat", "embedding", chroma_directory=directory)
            with patch("app.config.get_settings", return_value=settings):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/ui.py")).run()
                self.assertFalse(app.exception)
                rag = app.session_state["rag"]
                rag.add_document("fruit.txt", "苹果是一种水果。")
                app.session_state["chat_history"].add_turn("苹果是什么？", "苹果是一种水果。")
                app.run()
                with patch.object(rag, "search", wraps=rag.search) as search:
                    app.chat_input[0].set_value("它有什么特点？").run()
                self.assertFalse(app.exception)
                search.assert_called_once_with("苹果是什么？\n它有什么特点？", 3)

    def test_turn_snapshot_clear_and_isolation(self):
        history = ChatHistory()
        history.add_turn("问题", "回答")
        snapshot = history.messages()
        snapshot[0]["content"] = "外部修改"
        snapshot.clear()
        self.assertEqual(history.recent_questions(), ["问题"])
        self.assertEqual(len(history), 2)
        self.assertEqual(len(ChatHistory()), 0)
        history.clear()
        self.assertEqual(history.messages(), [])

    def test_followup_rules(self):
        history = ChatHistory()
        self.assertEqual(history.retrieval_question("它是什么？"), "它是什么？")
        for i in range(3):
            history.add_turn(f"问题{i}", "回答")
        self.assertEqual(history.recent_questions(), ["问题1", "问题2"])
        self.assertEqual(
            history.retrieval_question("它是什么？"), "问题1\n问题2\n它是什么？"
        )
        question = "哆啦A梦的三个秘密道具分别是什么？"
        self.assertEqual(history.retrieval_question(question), question)
        self.assertEqual(history.recent_questions(0), [])

    def test_model_messages_are_limited_and_have_no_metadata(self):
        history = ChatHistory()
        for i in range(8):
            history.add_turn(f"问题{i}", f"回答{i}")
        messages = history.build_model_messages("系统指令", "当前问题与资料")
        self.assertEqual(len(messages), 14)
        self.assertEqual(messages[1].get("content"), "问题2")
        self.assertEqual(messages[-1].get("content"), "当前问题与资料")
        self.assertTrue(all("hits" not in message for message in messages))
        self.assertEqual(history.context_count(), 12)
        self.assertEqual(len(history.build_model_messages("系统", "问题", limit=0)), 2)

    def test_session_history_reuse_and_migration(self):
        session: dict[str, object] = {
            "messages": [{"role": "user", "content": "旧消息"}]
        }
        history = get_session_history(session)
        self.assertEqual(history.recent_questions(), ["旧消息"])
        self.assertNotIn("messages", session)
        self.assertIs(get_session_history(session), history)
        self.assertIsNot(get_session_history({}), history)


if __name__ == "__main__":
    unittest.main()
