"""通用聊天历史：存取、追问上下文和模型消息转换集中在此处理。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai.types.chat import ChatCompletionMessageParam

from app.schemas import ChatMessage, Hit
from app.patent_identifiers import publication_numbers


class ChatHistory:
    """保存完整历史，模型上下文仅取最近的指定数量消息。"""

    MODEL_MESSAGE_LIMIT = 12

    def __init__(self, messages: Sequence[ChatMessage] = ()) -> None:
        self._messages = [self._copy(message) for message in messages]

    @staticmethod
    def _copy(message: ChatMessage) -> ChatMessage:
        copied: ChatMessage = {"role": message["role"], "content": message["content"]}
        if "hits" in message:
            copied["hits"] = list(message["hits"])
        if "retrieval" in message:
            copied["retrieval"] = message["retrieval"]
        return copied

    def __len__(self) -> int:
        return len(self._messages)

    def messages(self) -> list[ChatMessage]:
        """返回消息列表副本，供页面展示，避免外部直接修改内部列表。"""
        return [self._copy(message) for message in self._messages]

    def add_turn(
        self, question: str, reply: str, hits: Sequence[Hit] = (), *, retrieval: bool = True
    ) -> None:
        """成功得到回答后一次性保存一轮对话，失败时调用方不执行此方法。"""
        user: ChatMessage = {"role": "user", "content": question}
        if not retrieval:
            user["retrieval"] = False
        self._messages.extend(
            [
                user,
                {"role": "assistant", "content": reply, "hits": list(hits)},
            ]
        )

    def clear(self) -> None:
        self._messages.clear()

    def recent_questions(self, limit: int = 2) -> list[str]:
        """提取最近参与知识库检索的用户问题，直接交流不进入检索上下文。"""
        if limit <= 0:
            return []
        return [
            message["content"]
            for message in self._messages
            if message["role"] == "user" and message.get("retrieval", True)
        ][-limit:]

    def retrieval_question(self, question: str) -> str:
        """短问题或指代问题追加最近两个用户问题，沿用原有追问规则。"""
        if publication_numbers(question):
            return question  # 新问题明确指定专利时，不带入之前的其他专利号。
        recent = self.recent_questions()
        if recent and (
            len(question) < 12
            or any(word in question for word in ("它", "这个", "刚才"))
        ):
            return "\n".join([*recent, question])
        return question

    def context_count(self, limit: int = MODEL_MESSAGE_LIMIT) -> int:
        return min(len(self), max(limit, 0))

    def build_model_messages(
        self, system_prompt: str, current_content: str, limit: int = MODEL_MESSAGE_LIMIT
    ) -> list[ChatCompletionMessageParam]:
        """组装系统指令、最近历史和当前资料，不向模型发送 hits 元数据。"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        recent = self._messages[-limit:] if limit > 0 else []
        for message in recent:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append({"role": "assistant", "content": message["content"]})
        messages.append({"role": "user", "content": current_content})
        return messages


def as_history(value: ChatHistory | Sequence[ChatMessage]) -> ChatHistory:
    """兼容原有消息列表调用方，统一为历史管理对象。"""
    return value if isinstance(value, ChatHistory) else ChatHistory(value)


class HistorySession(Protocol):
    """历史所需的最小会话接口，兼容普通字典和 Streamlit 会话。"""

    def get(self, key: str, /) -> object: ...
    def pop(self, key: str, default: object, /) -> object: ...
    def __setitem__(self, key: str, value: object, /) -> None: ...


def get_session_history(session: HistorySession) -> ChatHistory:
    """获取会话历史；兼容页面热更新前的 messages 列表。"""
    history = session.get("chat_history")
    if not isinstance(history, ChatHistory):
        legacy = session.pop("messages", [])
        history = ChatHistory(legacy if isinstance(legacy, list) else [])
        session["chat_history"] = history
    return history
