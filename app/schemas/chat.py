from typing import Literal, NotRequired, TypedDict
from .knowledge import Hit


class ChatMessage(TypedDict):
    """一条聊天记录，助手消息可以携带检索来源。"""

    role: Literal["user", "assistant"]
    content: str
    hits: NotRequired[list[Hit]]
    retrieval: NotRequired[bool]

    downloads: NotRequired[list[str]]
