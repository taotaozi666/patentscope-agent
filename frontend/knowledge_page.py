"""知识库查询：文档主页与片段详情两级页面，仅依赖存储契约。"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from frontend.image_components import render_image, render_image_text_sources

from app.schemas import ChunkRecord, DocumentRecord
from app.storage import KnowledgeStore
from frontend.source_components import render_download_button


def _local_time(value: datetime) -> str:
    return value.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _open_document(document_id: str | None) -> None:
    st.session_state.knowledge_document_id = document_id


def _render_documents(store: KnowledgeStore, documents: list[DocumentRecord]) -> None:
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
            name_column, source_column = st.columns([3, 1])
            with name_column:
                st.button(
                    doc.source,
                    key=f"knowledge_open_{doc.id}",
                    on_click=_open_document,
                    args=(doc.id,),
                    width="stretch",
                )
            with source_column:
                render_download_button(store, doc, f"knowledge_download_{doc.id}")
            if doc.title:
                st.text(doc.title)
            st.caption(
                f"{doc.chunks} 个片段 · 格式：{doc.format or '未记录'} · "
                f"导入时间：{_local_time(doc.created_at)}"
            )
            st.caption(f"文档 ID：{doc.id}")


def _render_chunk_table(records: list[ChunkRecord]) -> None:
    """按切片类型展示正文，并为图像切片内联显示 PDF 来源裁图。"""
    repository = None
    for record in records:
        metadata = record.metadata
        chunk_type = str(metadata.get("doc_type_kwd") or metadata.get("ck_type") or "text").lower()
        type_label = {"image": "图片", "table": "表格", "text": "文本"}.get(chunk_type, chunk_type)
        is_ocr = str(metadata.get("text_origin", "")).lower() == "ocr"

        with st.container(border=True):
            # Keep type and OCR provenance visible before the chunk content.
            st.markdown(f"**{type_label}**" + ("　`OCR`" if is_ocr else ""))
            if chunk_type == "image":
                render_image_text_sources(metadata)
                image_id = metadata.get("image_id")
                image = None
                if image_id:
                    if repository is None:
                        from app.config import get_settings
                        from app.storage.pdf_repository import PdfRepository

                        repository = PdfRepository(get_settings().pdf_directory)
                    try:
                        image = repository.get_region_image(str(image_id))
                    except (OSError, ValueError):
                        image = None
                image_column, text_column = st.columns([1, 2], gap="medium")
                with image_column:
                    if image:
                        render_image(image, caption="PDF 图片区域", width=180)
                    else:
                        st.caption("图片裁图暂不可用")
                with text_column:
                    st.caption("OCR 识别文本" if is_ocr else "切片文本")
                    st.text(record.text or "（无可显示文本）")
            else:
                st.text(record.text)

            page_range = (
                f"第 {metadata['page_start']}–{metadata['page_end']} 页"
                if "page_start" in metadata else ""
            )
            section_path = metadata.get("heading_path") or metadata.get("section_path") or ""
            st.caption(
                f"片段 {record.chunk} · {len(record.text)} 字符 · {page_range} · "
                f"章节：{section_path or '未记录'} · 向量：{metadata.get('vector_dimensions', '未记录')} · "
                f"ID：{record.id} · 导入：{_local_time(record.created_at)}"
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
    if record.metadata.get("pdf_chunk_id"):
        from frontend.pdf_components import render_pdf_sources

        render_pdf_sources(record.metadata)
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
        _render_documents(store, documents)
