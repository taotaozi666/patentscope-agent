"""Chroma 知识库聊天页面，配置由服务端统一读取。"""

import logging
import sys
from pathlib import Path

# Streamlit 直接执行此文件时，确保项目根目录位于模块搜索路径中。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from app.conf.config import get_settings
from app.embeddings import create_embeddings
from app.history import get_session_history
from app.models import create_client
from app.rag import RAGService, answer
from app.trace import print_step, trace_operation
from frontend.chat_components import render_message
from frontend.ingestion_page import render_ingestion_page
from frontend.knowledge_page import render_knowledge_page

logger = logging.getLogger(__name__)
# 每次页面重新执行都会读取最新配置，密钥不进入页面控件。
settings = get_settings()
history = get_session_history(st.session_state)


st.set_page_config(page_title="RAG 知识库聊天", page_icon="📚")
page = st.sidebar.radio(
    "页面", ["聊天", "知识库查询", "导入任务"], key="current_page"
)
st.title({"聊天": "📚 RAG 知识库聊天", "知识库查询": "📋 知识库查询", "导入任务": "📥 文档导入任务"}[page])
st.caption("上传文档后提问。知识库由 Chroma 保存到本地，聊天记录保存在当前会话内存。")

with st.sidebar:
    st.header("设置")
    backend = st.selectbox(
        "检索模式",
        ["离线演示（字符匹配）", "中文语义向量"],
        index=1 if settings.api_key else 0,
    )
    top_k = st.slider("检索片段数", 1, 8, 3)
    st.caption("语义检索和回答生成使用千问服务，相关资料和最近聊天会发送给模型服务。")
    st.caption("同一检索模型的本地知识库由各会话共享；清空只作用于当前模型的集合。")

backend_config = (
    backend,
    settings.embedding_model if backend == "中文语义向量" else None,
    settings.base_url if backend == "中文语义向量" else None,
    settings.api_key if backend == "中文语义向量" else None,
    settings.chroma_directory,
    settings.chroma_collection,
    settings.chunk_size,
    settings.chunk_overlap,
)
# 切换模型时连接对应集合，原有集合的数据继续保留。
if st.session_state.get("backend_config") != backend_config:
    try:
        with st.spinner("初始化检索模型…"):
            embedder = create_embeddings(settings, offline=backend != "中文语义向量")
            service = RAGService(embedder, settings=settings)
        st.session_state.rag = service
        history.clear()
        st.session_state.backend_config = backend_config
        print_step(
            "检索模型初始化完成",
            检索模式=backend,
            存储实现=type(st.session_state.rag.store).__name__,
        )
    except Exception:
        logger.exception("初始化向量模型失败")
        st.error("检索服务暂时不可用，请联系管理员或切换离线演示。")
        st.stop()

rag: RAGService = st.session_state.rag
with st.sidebar:
    st.caption("批量上传及后台向量解析请使用“导入任务”页面。")
    if st.button("载入示例文档"):
        try:
            rag.add_document(
                "示例知识库.txt",
                "RAG 是检索增强生成。先加载文档，再分块并计算向量，存入 Chroma，根据问题检索相关片段，最后交给大模型生成回答。\n\n本项目使用 LangChain Document 和递归分割器。Chroma 本地持久化文档和向量，聊天记录仍保存在当前会话内存中。",
            )
            st.success("示例已载入")
        except Exception as exc:
            logger.exception("导入示例失败")
            st.error(f"示例导入失败：{exc}")
    st.write(f"知识库：{rag.store.stats().chunks} 个片段")
    if st.button("清空聊天"):
        history.clear()
        st.rerun()
    if st.button("清空知识库和聊天"):
        rag.store.clear()
        print_step("清空知识库", 当前片段数=rag.store.stats().chunks)
        history.clear()
        st.rerun()

if page == "知识库查询":
    render_knowledge_page(rag.store)
    st.stop()
if page == "导入任务":
    render_ingestion_page(
        settings, rag.store, backend == "离线演示（字符匹配）"
    )
    st.stop()

if not settings.api_key:
    st.info("当前为演示模式，回答展示检索原文。")
for message in history.messages():
    render_message(message["role"], message["content"], message.get("hits", []))

if question := st.chat_input("输入问题", disabled=not rag.store.stats().chunks):
    with trace_operation("聊天", 问题=question, 检索模式=backend):
        render_message("user", question)
        try:
            with st.spinner("正在检索知识库…"):
                hits = rag.search(question, top_k)
                retrieval_question = history.retrieval_question(question)
                if retrieval_question != question:
                    print_step(
                        "追问流转",
                        最近问题=history.recent_questions(),
                        新检索问题=retrieval_question,
                    )
                    hits = rag.search(retrieval_question, top_k)
            client = None
            if settings.api_key:
                client = create_client(settings)
            try:
                with st.spinner("正在生成回答…"):
                    reply = answer(
                        question,
                        hits,
                        history,
                        client,
                        settings.chat_model,
                    )
                render_message("assistant", reply, hits)
            finally:
                # 请求失败时也关闭客户端，释放网络连接。
                if client:
                    client.close()
                    print_step("聊天客户端已关闭")
            history.add_turn(question, reply, hits)
            print_step(
                "页面接收最终结果",
                回答=reply,
                来源数量=len(hits),
                历史消息数=len(history),
            )
        except Exception:
            print_step("聊天异常", 结果="未保存本轮聊天，请查看服务端异常日志")
            logger.exception("生成知识库回答失败")
            st.error(
                "回答未完成，本轮未保存。请重新发送问题；若持续失败，请联系管理员。"
            )
