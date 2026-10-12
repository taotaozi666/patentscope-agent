"""本地交互规则：问候和助手介绍固定回答，其余问题默认查询知识库。"""

import re

from app import prompt


_GREETING = re.compile(r"^(你好|您好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好)")
_IDENTITY = {
    "你是", "你是谁", "您是谁", "请问你是谁", "你是什么", "你是什么助手",
    "你叫什么", "你叫什么名字", "介绍一下你自己", "自我介绍一下",
    "whoareyou",
}
_CAPABILITIES = {
    "你能做什么", "你可以做什么", "你能帮我做什么", "你会做什么",
    "你有什么功能", "你有哪些功能", "怎么使用你", "如何使用你",
    "whatcanyoudo",
}


def direct_reply(question: str) -> str | None:
    """只匹配完整问候或助手介绍，不拦截附带文档问题的混合表达。"""
    text = re.sub(r"[\s，,。.!！?？、：:;；]+", "", question).casefold()
    if not text:
        return None
    greeting = _GREETING.match(text)
    if greeting:
        text = text[greeting.end():]
        if text in {"", "呀", "啊"}:
            return prompt.INTRODUCTION_REPLY
    if text in _IDENTITY or text in {"在吗", "你在吗"}:
        return prompt.INTRODUCTION_REPLY
    if text in _CAPABILITIES:
        return prompt.HELP_REPLY
    return None
