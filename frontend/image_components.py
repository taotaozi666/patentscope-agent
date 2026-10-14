"""保留原始图片像素，避免 Streamlit 按缩略显示宽度重采样。"""
import base64
import streamlit as st


def render_image(content: bytes, **kwargs) -> None:
    # data URL 会直接传给浏览器；width 仅控制布局，放大仍使用原图。
    mime = "image/jpeg" if content.startswith(b"\xff\xd8") else "image/png"
    url = f"data:{mime};base64," + base64.b64encode(content).decode("ascii")
    st.image(url, **kwargs)


def render_image_text_sources(metadata: dict) -> None:
    fields = [("image_native_text", "PDF 原生文字"), ("image_ocr_text", "OCR 识别文字（需核对原图）"),
              ("image_caption_text", "原文图注"), ("image_description_text", "原文相关说明"),
              ("generated_context", "模型生成的检索上下文（非原文证据）")]
    if not any(metadata.get(field) for field, _ in fields):
        return
    with st.expander("图片文字来源（完整图片对象）", expanded=False):
        for field, label in fields:
            if metadata.get(field):
                st.caption(label)
                st.text(metadata[field])
