"""对照布局：由左栏决定行高，右栏只在内部滚动。"""

import streamlit as st


def sync_quality_panel_height() -> None:
    """只约束右栏；PDF 图片的宽度、缩放和显示方式保持原样。"""
    st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"]:has(.st-key-quality_text_scroll) {
        align-items: stretch;
    }
    [data-testid="stColumn"]:has(.st-key-quality_text_scroll) {
        contain: size;
        min-height: 0;
        align-self: stretch;
    }
    [data-testid="stColumn"]:has(.st-key-quality_text_scroll) > [data-testid="stVerticalBlock"] {
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
        height: 100%;
        min-height: 0;
    }
    [data-testid="stLayoutWrapper"]:has(> .st-key-quality_text_scroll) {
        height: 100% !important;
        min-height: 0 !important;
        overflow: hidden;
    }
    .st-key-quality_text_scroll {
        height: 100% !important;
        max-height: 100% !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        box-sizing: border-box;
    }
    </style>
    """, unsafe_allow_html=True)
