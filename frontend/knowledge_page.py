"""知识库查询页面，仅依赖存储契约。"""

import math
from zoneinfo import ZoneInfo

import streamlit as st

from app.storage import KnowledgeStore


def render_knowledge_page(store: KnowledgeStore) -> None:
    """查询当前知识库，支持来源过滤、关键词、分页与详情。"""
    st.caption("查看当前模型对应的 Chroma 集合。查询不会调用模型。")
    stats = store.stats()
    left, right = st.columns(2)
    left.metric("已导入文档", stats.documents)
    right.metric("文档片段", stats.chunks)
    if not stats.chunks:
        st.info("知识库暂无数据，请先上传文档或载入示例文档。")
        return

    keyword = st.text_input("关键词", placeholder="搜索文件名或片段原文")
    source = st.selectbox(
        "文档来源",
        [None, *store.sources()],
        format_func=lambda value: "全部来源" if value is None else value,
    )
    page_size = st.selectbox("每页数量", [10, 20, 50], index=1)
    filters = (keyword, source, page_size)
    if st.session_state.get("knowledge_filters") != filters:
        st.session_state.knowledge_page_number = 1
        st.session_state.knowledge_filters = filters
    first_page = store.query(keyword, source, limit=page_size)
    if not first_page.total:
        st.info("没有匹配的数据，请调整关键词或来源。")
        return
    pages = math.ceil(first_page.total / page_size)
    if st.session_state.get("knowledge_page_number", 1) > pages:
        st.session_state.knowledge_page_number = pages
    page = int(
        st.number_input(
            "页码", min_value=1, max_value=pages, step=1, key="knowledge_page_number"
        )
    )
    result = (
        first_page
        if page == 1
        else store.query(keyword, source, (page - 1) * page_size, page_size)
    )
    st.caption(f"共 {result.total} 个片段 · 第 {page} / {pages} 页")
    st.dataframe(
        [
            {
                "片段 ID": r.id,
                "文档 ID": r.document_id,
                "来源": r.source,
                "片段编号": r.chunk,
                "字符数": len(r.text),
                "向量长度": len(r.vector),
                "导入时间": r.created_at.astimezone(ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "原文预览": r.text[:100],
            }
            for r in result.items
        ],
        hide_index=True,
        width="stretch",
    )

    record_id = st.selectbox(
        "查看片段详情",
        [r.id for r in result.items],
        format_func=lambda value: next(
            f"{r.source} · 片段 {r.chunk} · {r.id[:8]}"
            for r in result.items
            if r.id == value
        ),
    )
    record = store.get(record_id)
    if record is None:
        st.info("该片段已不存在，请刷新查询。")
        return
    st.text_area("片段原文", value=record.text, height=220, disabled=True)
    with st.expander("查看文档元数据"):
        st.json(record.metadata, expanded=False)
    with st.expander("查看向量数据"):
        st.caption(
            "向量为浮点数数组；离线演示使用固定维度哈希词频，语义模式使用千问模型。"
        )
        st.json(record.vector, expanded=False)
