"""PDF 来源展示组件，由知识库页面调用。"""

import json

import streamlit as st

from app.config import get_settings

from app.storage.pdf_repository import PdfRepository
from app.pdf.preview import render_source_page


def render_pdf_sources(metadata: dict) -> None:
    chunk_id = metadata.get("pdf_chunk_id")
    if not chunk_id:
        return
    with st.expander("查看 PDF 原文位置"):
        if not st.checkbox("加载原 PDF 及定位预览", key=f"pdf_load_{chunk_id}"):
            return
        repository = PdfRepository(get_settings().pdf_directory)
        saved = repository.get_chunk(str(chunk_id))
        if not saved:
            st.warning("解析来源记录不存在，请检查文档资产目录或重新导入。")
            return
        data = saved["metadata"]
        st.caption(f"PDF 第 {data['page_start']}–{data['page_end']} 页 · {data['section_path'] or '未确定章节'}")
        run = repository.get_run(data["pdf_parse_id"])
        if run and run["warnings"]:
            st.warning("；".join(run["warnings"]))
        try:
            original = repository.read_original(data["pdf_version_id"])
        except FileNotFoundError:
            st.warning("原 PDF 已丢失，请重新导入。")
            return
        st.download_button("下载原 PDF", original, file_name=data["source"], mime="application/pdf", key=f"pdf_download_{chunk_id}")
        spans = json.loads(data["pdf_source_spans"])
        mappings = json.loads(data["pdf_serialized_map"])
        candidates = [item for item in mappings if item.get("span") and item.get("cell_id") and item["end"] > item["start"]]
        if candidates:
            cell_index = st.selectbox(
                "定位表格信息点", range(len(candidates) + 1),
                format_func=lambda index: "整个片段" if index == 0 else
                    f"{saved['text'][candidates[index - 1]['start']:candidates[index - 1]['end']]} · {candidates[index - 1]['cell_id']}",
                key=f"pdf_cell_{chunk_id}",
            )
            if cell_index:
                spans = [candidates[cell_index - 1]["span"]]
        pages = sorted({span["page_no"] for span in spans})
        selected = st.selectbox("来源物理页码", pages, key=f"pdf_page_{chunk_id}")
        if selected is not None:
            st.image(render_source_page(original, selected, spans), caption="红框标出片段或所选单元格的原始来源区域。")
        else:
            st.warning("该片段没有可用的来源页码。")
        st.text(saved["text"])
        if data.get("parent_id"):
            parent = repository.get_object(data["parent_id"])
            if parent:
                with st.expander("完整语义对象及检索上下文"):
                    st.caption("上下文只帮助检索，不作为原文事实；完整对象保留跨页来源和关系。")
                    st.text(saved.get("context_text", data.get("context_text", "")))
                    st.text(parent["raw_content"])
                    st.json(parent)
        with st.expander("单元格和文字来源映射"):
            st.json(json.loads(data["pdf_serialized_map"]))
        if run:
            with st.expander("完整解析结构（含表格与图片区域）"):
                st.json(run, expanded=False)
