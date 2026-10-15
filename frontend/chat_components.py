"""聊天消息与引用来源展示组件。"""

import streamlit as st
from frontend.image_components import render_image, render_image_text_sources

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
        render_sources(content, hits or [])


def render_sources(content: str, hits: list[Hit]) -> None:
    """在同一个折叠区展示全部检索候选，编号与回答引用一致。"""
    if not hits:
        return
    with st.expander("查看检索来源", expanded=False):
        st.caption("以下为检索候选片段；检索到片段不代表资料足以回答。答案中的引用编号表示使用的依据。")
        for number, hit in enumerate(hits, 1):
            _render_source(number, hit, show_score=True)


def _render_source(number: int, hit: Hit, *, show_score: bool, show_image: bool = True) -> None:
    label = f"[{number}] {hit.source} · 片段 {hit.chunk}"
    if hit.metadata.get("page_start"):
        start = hit.metadata["page_start"]
        end = hit.metadata.get("page_end", start)
        label += f" · 第 {start} 页" if start == end else f" · 第 {start}–{end} 页"
    if show_score:
        score_label = "融合得分" if hit.metadata.get("retrieval_method") == "hybrid_rrf" else "相似度"
        label += f" · {score_label} {hit.score:.3f}"
    st.write(label)
    locations = hit.metadata.get("duplicate_locations", [])
    if len(locations) > 1:
        references = []
        for location in locations:
            start, end = location.get("page_start"), location.get("page_end")
            page = f"第 {start} 页" if start and (not end or end == start) else f"第 {start}–{end} 页" if start else "页码未知"
            references.append(f"片段 {location['chunk']}（{page}）")
        st.caption("相同文本合并展示，召回位置：" + "、".join(references))
    render_image_text_sources(hit.metadata)
    image_id = hit.metadata.get("image_id") if show_image else None
    image = _get_region_image(image_id) if image_id else None
    if image_id:
        image_column, text_column = st.columns([1, 2], gap="medium")
        with image_column:
            if image:
                render_image(image, caption="PDF 来源区域", width=220)
            else:
                st.caption("图片区域暂不可用")
        with text_column:
            st.text(hit.text or "（该图片切片没有可显示的文字）")
    else:
        st.text(hit.text or "（该片段没有可显示的文字）")
    if hit.metadata.get("table_evidence"):
        from app.rag.table_evidence import table_evidence_text
        st.text(table_evidence_text(hit))
