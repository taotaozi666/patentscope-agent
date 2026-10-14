"""知识库列表与聊天共用的原文件下载按钮。"""
from pathlib import PureWindowsPath

import streamlit as st

from app.config import get_settings
from app.schemas import DocumentRecord
from app.storage import KnowledgeStore
from app.storage.source_repository import SourceRepository


def render_download_button(store: KnowledgeStore, document: DocumentRecord, key: str) -> None:
    records = store.query(document_id=document.id, limit=1, include_vectors=False).items
    metadata = records[0].metadata if records else {}
    content_hash = str(metadata.get("content_hash") or metadata.get("pdf_version_id") or "")
    file_format = str(metadata.get("original_format") or metadata.get("format") or ".txt").lower()
    try:
        content = SourceRepository(get_settings().pdf_directory).read(content_hash, file_format)
    except (OSError, ValueError):
        st.button("原文件不可用", key=key, disabled=True, width="stretch",
                  help="源文件尚未保存或无法读取，请重新上传原文件。")
        st.caption("请重新上传原文件以启用下载。")
        return
    filename = PureWindowsPath(document.source.replace("/", "\\")).name
    mime = {".pdf": "application/pdf", ".md": "text/markdown", ".txt": "text/plain"}[file_format]
    st.download_button("下载源文件", content, file_name=filename, mime=mime,
                       key=key, on_click="ignore", width="stretch")


def render_downloads(store: KnowledgeStore, document_ids: list[str], key: str) -> None:
    documents = {document.id: document for document in store.documents()}
    for document_id in document_ids:
        document = documents.get(document_id)
        if document is None:
            st.info("该文件已从知识库移除。")
            continue
        st.write(document.title or document.source)
        st.caption(document.source)
        render_download_button(store, document, f"{key}_{document.id}")
