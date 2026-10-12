"""共享的模型客户端创建与聊天接口调用。"""

from collections.abc import Sequence

from openai import OpenAI

from app.config import get_settings
from app.history import ChatHistory, as_history
from app import prompt
from app.schemas import ChatMessage, Settings
from app.trace import debug_step, print_step


def create_client(settings: Settings | None = None) -> OpenAI:
    """通过兼容接口创建模型客户端，密钥和地址均来自配置。"""
    settings = settings or get_settings()
    if not settings.api_key:
        raise ValueError("请在 .env 中配置模型服务的 API_KEY")
    print_step(
        "创建模型客户端", 配置来源=".env / 系统环境变量", 服务="兼容接口", 超时秒=60
    )
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=1,
    )


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
        "聊天模型请求", 模型=model_name, 消息数量=len(messages), 思考模式=False, 流式=False
    )
    debug_step("聊天模型消息", lambda: {"消息": messages})
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
    print_step("聊天模型返回", 回答字符数=len(reply))
    return reply
