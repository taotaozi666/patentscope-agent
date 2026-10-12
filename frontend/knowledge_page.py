"""知识库查询：文档主页与片段详情两级页面，仅依赖存储契约。"""

import math
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import streamlit as st

from app.schemas import ChunkRecord, DocumentRecord
from app.storage import KnowledgeStore


def _local_time(value: datetime) -> str:
    return value.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _open_document(document_id: str | None) -> None:
    st.session_state.knowledge_document_id = document_id


def _render_documents(documents: list[DocumentRecord]) -> None:
    st.subheader("已导入文档")
    keyword = (
        st.text_input(
            "搜索文档", placeholder="输入文件名或文档标题", key="knowledge_document_keyword"
        )
        .strip()
        .casefold()
    )
    matches = [
        doc
        for doc in documents
        if not keyword or keyword in doc.source.casefold() or keyword in doc.title.casefold()
    ]
    st.caption(f"共 {len(documents)} 个文档 · 当前显示 {len(matches)} 个 · 点击文档进入片段详情")
    if not matches:
        st.info("没有匹配的文档，请调整关键词。")
        return
    for doc in matches:
        with st.container(border=True):
            st.button(
                doc.source,
                key=f"knowledge_open_{doc.id}",
                on_click=_open_document,
                args=(doc.id,),
                use_container_width=True,
            )
            if doc.title:
                st.text(doc.title)
            st.caption(
                f"{doc.chunks} 个片段 · 格式：{doc.format or '未记录'} · "
                f"导入时间：{_local_time(doc.created_at)}"
            )
            st.caption(f"文档 ID：{doc.id}")


def _render_chunk_table(records: list[ChunkRecord]) -> None:
    """普通 HTML 表格让每行随原文增高，保留换行且不截断文本。"""
    columns = ["文本", "片段编号", "字符数", "章节路径", "向量长度", "片段 ID", "导入时间"]
    rows = []
    for record in records:
        values = [
            record.text,
            record.chunk,
            len(record.text),
            record.metadata.get("heading_path", ""),
            record.metadata.get("vector_dimensions", "未记录"),
            record.id,
            _local_time(record.created_at),
        ]
        rows.append(
            "<tr>"
            + "".join(f"<td>{escape(str(value))}</td>" for value in values)
            + "</tr>"
        )
    st.html(
        """
        <style>
        .knowledge-chunks { overflow-x: auto; }
        .knowledge-chunks table { width: 100%; border-collapse: collapse; }
        .knowledge-chunks th, .knowledge-chunks td {
            border: 1px solid #8885; padding: 0.75rem; text-align: left;
            vertical-align: top; white-space: pre-wrap; overflow-wrap: anywhere;
            min-width: 5rem;
        }
        .knowledge-chunks th { background: #8881; }
        .knowledge-chunks th:first-child, .knowledge-chunks td:first-child {
            width: 60%; min-width: 22rem;
        }
        </style>
        <div class="knowledge-chunks"><table><thead><tr>
        """
        + "".join(f"<th>{column}</th>" for column in columns)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_document(store: KnowledgeStore, document: DocumentRecord) -> None:
    st.button("← 返回文档主页", on_click=_open_document, args=(None,))
    st.subheader(document.source)
    if document.title:
        st.text(document.title)
    st.caption(
        f"文档 ID：{document.id} · {document.chunks} 个片段 · "
        f"导入时间：{_local_time(document.created_at)}"
    )
    keyword = st.text_input("片段关键词", placeholder="搜索片段原文", key="knowledge_chunk_keyword")
    page_size = st.selectbox("每页数量", [10, 20, 50], index=1, key="knowledge_page_size")
    filters = (document.id, keyword, page_size)
    if st.session_state.get("knowledge_filters") != filters:
        st.session_state.knowledge_page_number = 1
        st.session_state.knowledge_filters = filters
    page = int(st.session_state.get("knowledge_page_number", 1))
    result = store.query(
        keyword, offset=(page - 1) * page_size, limit=page_size,
        document_id=document.id, include_vectors=False,
    )
    if not result.total:
        st.info("没有匹配的片段，请调整关键词。")
        return
    pages = math.ceil(result.total / page_size)
    if page > pages:
        st.session_state.knowledge_page_number = pages
        st.rerun()
    st.number_input(
        "页码", min_value=1, max_value=pages, step=1, key="knowledge_page_number"
    )
    st.caption(f"共 {result.total} 个片段 · 第 {page} / {pages} 页 · 文本按原文完整显示")
    _render_chunk_table(result.items)
    if not result.items:
        st.info("文档内容已更新，请刷新查询。")
        return

    record_id = st.selectbox(
        "查看片段属性",
        [r.id for r in result.items],
        format_func=lambda value: next(
            f"片段 {r.chunk} · {r.id[:8]}" for r in result.items if r.id == value
        ),
        key="knowledge_record_id",
    )
    record = next(r for r in result.items if r.id == record_id)
    with st.expander("查看完整元数据"):
        st.json(record.metadata, expanded=False)
    with st.expander("查看向量数据"):
        if st.checkbox("加载该片段的向量", key=f"knowledge_vector_{record.id}"):
            detail = store.get(record.id)
            if detail is None:
                st.info("该片段已更新或删除，请刷新查询。")
            else:
                st.caption(f"向量长度：{len(detail.vector)}")
                st.json(detail.vector, expanded=False)


def render_knowledge_page(store: KnowledgeStore) -> None:
    """主页展示全部已导入文档，点击后进入该文档的片段详情。"""
    st.caption("查看当前模型对应的知识库。查询不会调用模型。")
    documents = store.documents()
    left, right = st.columns(2)
    left.metric("已导入文档", len(documents))
    right.metric("文档片段", sum(doc.chunks for doc in documents))
    if not documents:
        _open_document(None)
        st.info("知识库暂无数据，请先上传文档或载入示例文档。")
        return
    selected_id = st.session_state.get("knowledge_document_id")
    selected = next((doc for doc in documents if doc.id == selected_id), None)
    if selected is not None:
        _render_document(store, selected)
    else:
        if selected_id:
            _open_document(None)
            st.info("该文档已不存在或已切换知识库，请重新选择文档。")
        _render_documents(documents)
