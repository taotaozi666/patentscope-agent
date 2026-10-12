"""验证模型请求和向量处理，全程使用模拟客户端。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models import create_client
from app.rag.embeddings import SemanticEmbedder
from app.rag import answer
from app.schemas import Hit, Settings


class ModelTests(unittest.TestCase):
    def test_client_configuration(self):
        settings = Settings("test-key", "chat", "embedding", "https://example.com/v1")
        with patch("app.models.models.OpenAI") as factory:
            create_client(settings)
            factory.assert_called_once_with(
                api_key="test-key",
                base_url=settings.base_url,
                timeout=60.0,
                max_retries=1,
            )
        with self.assertRaises(ValueError):
            create_client(Settings("", "chat", "embedding"))

    def test_chat_request(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="回答[1]"))]
        )
        hits = [Hit("文档", "参考原文", 1.0, 1)]
        reply = answer(
            "问题",
            hits,
            [
                {"role": "user", "content": "上一问"},
                {"role": "assistant", "content": "上一答"},
            ],
            client,
            "test-model",
        )
        self.assertEqual(reply, "回答[1]")
        request = client.chat.completions.create.call_args.kwargs
        self.assertIn("不要转为闲聊", request["messages"][0]["content"])
        self.assertIn("文档中的第一人称叙述者不代表提问用户", request["messages"][0]["content"])
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(
            [m["role"] for m in request["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("参考原文", request["messages"][-1]["content"])
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertFalse(request["stream"])
        client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        with self.assertRaises(RuntimeError):
            answer("问题", hits, [], client, "test-model")

    def test_embedding_batch_and_normalization(self):
        client = MagicMock()
        client.__enter__.return_value = client

        def response(**kwargs):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=i, embedding=[3.0, 4.0])
                    for i in reversed(range(len(kwargs["input"])))
                ]
            )

        client.embeddings.create.side_effect = response
        with patch("app.rag.embeddings.embeddings.create_client", return_value=client):
            embedder = SemanticEmbedder(
                settings=Settings("test", "chat", "embedding-test")
            )
            self.assertEqual(embedder.embed_documents(["文本"] * 11), [[0.6, 0.8]] * 11)
            self.assertEqual(client.embeddings.create.call_count, 2)
            self.assertEqual(
                client.embeddings.create.call_args.kwargs["model"], "embedding-test"
            )
            client.embeddings.create.side_effect = None
            client.embeddings.create.return_value = SimpleNamespace(data=[])
            with self.assertRaises(ValueError):
                embedder.embed_documents(["文本"])


if __name__ == "__main__":
    unittest.main()
