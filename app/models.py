"""集中封装千问客户端创建、聊天调用和文本向量模型。"""

import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from app.conf.config import get_settings
from app.history import ChatHistory, as_history
from app.prompt import prompt
from app.schemas import ChatMessage, Settings
from app.trace import print_step


def create_client(settings: Settings | None = None) -> OpenAI:
    """通过百炼兼容接口创建千问客户端，密钥和地址均来自配置。"""
    settings = settings or get_settings()
    if not settings.api_key:
        raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")
    print_step(
        "创建模型客户端", 配置来源=".env / 系统环境变量", 服务="兼容接口", 超时秒=60
    )
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=1,
    )


class SemanticEmbedder(Embeddings):
    """LangChain 千问嵌入适配器，保留项目配置的兼容接口地址。"""

    def __init__(
        self, model_name: str | None = None, settings: Settings | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        if not self.settings.api_key:
            raise ValueError("语义检索需要配置 DASHSCOPE_API_KEY")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        if not texts:
            return vectors
        with create_client(self.settings) as client:
            # 每批不超过 10 条，兼容千问文本向量接口的批量限制。
            for start in range(0, len(texts), 10):
                batch = texts[start : start + 10]
                print_step(
                    "向量接口请求",
                    模型=self.model_name,
                    批次=start // 10 + 1,
                    输入文本=batch,
                )
                response = client.embeddings.create(
                    model=self.model_name, input=batch, encoding_format="float"
                )
                items = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise ValueError("向量接口返回的片段索引不完整")
                for item in items:
                    norm = math.sqrt(sum(value * value for value in item.embedding))
                    if not norm or not math.isfinite(norm):
                        raise ValueError("向量接口返回无效向量")
                    vectors.append([value / norm for value in item.embedding])
                print_step(
                    "向量接口返回",
                    批次=start // 10 + 1,
                    向量数量=len(items),
                    向量长度=[len(item.embedding) for item in items],
                    状态="已归一化",
                )
        return vectors


def generate_reply(
    question: str,
    context: str,
    history: ChatHistory | Sequence[ChatMessage],
    client: OpenAI,
    model: str | None = None,
) -> str:
    """发送资料和历史消息，一次性返回完整回答。"""
    messages = as_history(history).build_model_messages(
        system_prompt=prompt.SYSTEM_PROMPT,
        current_content=prompt.build_user_prompt(question, context),
    )
    model_name = model or get_settings().chat_model
    print_step(
        "聊天模型请求", 模型=model_name, 消息=messages, 思考模式=False, 流式=False
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        extra_body={"enable_thinking": False},
    )
    reply = response.choices[0].message.content if response.choices else None
    if not reply or not reply.strip():
        print_step("聊天模型返回空结果")
        raise RuntimeError("模型未返回文本，请重试。")
    print_step("聊天模型返回", 最终回答=reply)
    return reply
