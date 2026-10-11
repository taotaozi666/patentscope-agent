"""验证历史存取、追问上下文、模型历史上限及会话隔离。"""

import unittest

from app.history import ChatHistory, get_session_history


class HistoryTests(unittest.TestCase):
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
