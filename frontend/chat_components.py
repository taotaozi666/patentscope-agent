"""聊天消息与引用来源展示组件。"""

import re

import streamlit as st

from app.schemas import Hit


@st.cache_resource
def _pdf_repository(directory: str):
    from app.storage.pdf_repository import PdfRepository

    return PdfRepository(directory)


def _get_region_image(image_id: object) -> bytes | None:
    if not image_id:
        return None
    from app.config import get_settings

    try:
        repository = _pdf_repository(get_settings().pdf_directory)
        return repository.get_region_image(str(image_id))
    except (OSError, ValueError):
        return None


def render_message(role: str, content: str, hits: list[Hit] | None = None) -> None:
    """消息容器只放正文，来源放在下方，避免影响头像与正文的对齐。"""
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            render_cited_images(content, hits or [])
    if role == "assistant":
        render_sources(hits or [])


def render_cited_images(content: str, hits: list[Hit]) -> None:
    """在回答下方展示回答实际引用到的图片或表格区域。"""
    cited_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", content)}
    cited_hits = [
        (number, hit)
        for number, hit in enumerate(hits, 1)
        if number in cited_numbers and hit.metadata.get("image_id")
    ]
    if not cited_hits:
        return

    st.caption("引用图片")
    columns = st.columns(min(3, len(cited_hits)))
    for index, (number, hit) in enumerate(cited_hits):
        image = _get_region_image(hit.metadata.get("image_id"))
        with columns[index % len(columns)]:
            if image:
                st.image(image, caption=f"[{number}] {hit.source} · 片段 {hit.chunk}", width=220)
            else:
                st.caption(f"[{number}] 图片暂不可用 · {hit.source} · 片段 {hit.chunk}")


def render_sources(hits: list[Hit]) -> None:
    """展示当前回答引用的文档原文。"""
    if not hits:
        return
    with st.expander("查看检索来源"):
        st.caption("以下为检索候选片段；检索到片段不代表资料足以回答。答案中的引用编号表示使用的依据。")
        for i, hit in enumerate(hits, 1):
            st.write(f"[{i}] {hit.source} · 片段 {hit.chunk} · 相似度 {hit.score:.3f}")
            image_id = hit.metadata.get("image_id")
            image = _get_region_image(image_id) if image_id else None
            if image_id:
                image_column, text_column = st.columns([1, 2], gap="medium")
                with image_column:
                    if image:
                        st.image(image, caption="PDF 来源区域", width=220)
                    else:
                        st.caption("图片区域暂不可用")
                with text_column:
                    st.text(hit.text or "（该图片切片没有可显示的文字）")
            else:
                st.text(hit.text)
