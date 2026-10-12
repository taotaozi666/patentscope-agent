"""聊天消息与引用来源展示组件。"""

import streamlit as st

from app.schemas import Hit


def render_message(role: str, content: str, hits: list[Hit] | None = None) -> None:
    """消息容器只放正文，来源放在下方，避免影响头像与正文的对齐。"""
    with st.chat_message(role):
        st.markdown(content)
    if role == "assistant":
        render_sources(hits or [])


def render_sources(hits: list[Hit]) -> None:
    """展示当前回答引用的文档原文。"""
    if not hits:
        return
    with st.expander("查看检索来源"):
        st.caption("以下为检索候选片段；检索到片段不代表资料足以回答。答案中的引用编号表示使用的依据。")
        for i, hit in enumerate(hits, 1):
            st.write(f"[{i}] {hit.source} · 片段 {hit.chunk} · 相似度 {hit.score:.3f}")
            st.text(hit.text)
