"""Chroma 知识库问答页面，配置由服务端统一读取。"""

import logging
import sys
from pathlib import Path

# Streamlit 直接执行此文件时，确保项目根目录位于模块搜索路径中。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from app.config import get_settings
from app.rag.embeddings import create_embeddings
from app.history import get_session_history
from app.models import create_client
from app.rag import RAGService, answer, direct_reply
from app.trace import configure_logging, debug_step, print_step, trace_operation
from app.downloads import is_download_request, resolve_downloads
from frontend.source_components import render_downloads
from frontend.chat_components import render_message
from frontend.ingestion_page import render_ingestion_page
from frontend.knowledge_page import render_knowledge_page
from frontend.quality_page import render_quality_page

logger = logging.getLogger("frontend.ui")
# 每次页面重新执行都会读取最新配置，密钥不进入页面控件。
settings = get_settings()
configure_logging(settings.log_level)
history = get_session_history(st.session_state)
if st.session_state.get("current_page") == "聊天":
    st.session_state.current_page = "知识库问答"


st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide" if st.session_state.get("current_page") in {"知识库查询", "解析检查"} else "centered",
)
page = st.sidebar.radio(
    "页面", ["知识库问答", "知识库查询", "导入任务", "解析检查"], key="current_page"
)
st.title({"知识库问答": "📚 RAG 知识库问答", "知识库查询": "📋 知识库查询", "导入任务": "📥 文档导入任务", "解析检查": "🔎 PDF 解析检查"}[page])
st.caption("上传文档后提问。知识库由 Chroma 保存到本地，聊天记录保存在当前会话内存。")

# 解析检查仅依赖文档资产；即使向量服务初始化失败，也能复核已保存的解析结果。
if page == "解析检查":
    render_quality_page(settings)
    st.stop()

with st.sidebar:
    st.header("设置")
    backend = st.selectbox(
        "检索模式",
        ["离线演示（字符匹配）", "中文语义向量"],
        index=1 if settings.api_key else 0,
    )
    top_k = st.slider("检索片段数", 1, 12, 8)
    if backend == "离线演示（字符匹配）":
        st.caption("离线模式使用本地字符匹配并展示检索原文；PDF 使用本地规则上下文。")
    else:
        st.caption("语义检索和回答生成使用已配置的模型服务，相关资料和最近聊天会发送给该服务。")
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
    settings.chat_model,
    settings.pdf_directory,
    settings.pdf_target_size,
    settings.pdf_context_mode,
    settings.pdf_image_ocr,
    settings.pdf_ocr_language,
    settings.pdf_ocr_dpi,
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
    st.write(f"知识库：{rag.store.count()} 个片段")
    if st.button("清空聊天"):
        history.clear()
        st.rerun()
    if st.button("清空知识库和聊天"):
        rag.store.clear()
        print_step("清空知识库", 当前片段数=rag.store.count())
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

if backend == "离线演示（字符匹配）":
    st.info("当前为演示模式，知识库问题展示检索原文；问候和助手介绍可直接回答。")
for message_index, message in enumerate(history.messages()):
    render_message(message["role"], message["content"], message.get("hits", []))
    if message.get("downloads"):
        render_downloads(rag.store, message["downloads"], f"chat_download_{message_index}")

if question := st.chat_input("围绕已导入文档提问"):
    with trace_operation("聊天", 问题字符数=len(question), 检索模式=backend):
        debug_step("用户问题", lambda: {"问题": question})
        render_message("user", question)
        client = None
        try:
            downloads = []
            downloading = is_download_request(question)
            if downloading:
                downloads, reply = resolve_downloads(question, rag.store, history, rag.search)
                print_step("下载路由", 候选文档数=len(downloads))
            else:
                reply = direct_reply(question)
            retrieval = reply is None
            hits = []
            print_step("问题处理", 处理方式="文件下载" if downloading else ("知识库检索" if retrieval else "固定说明"))
            if reply is None:
                with st.spinner("正在检索知识库…"):
                    retrieval_question = history.retrieval_question(question)
                    if retrieval_question != question:
                        print_step(
                            "追问流转",
                            历史问题数=len(history.recent_questions()),
                            检索问题字符数=len(retrieval_question),
                        )
                        debug_step("追问内容", lambda: {"最近问题": history.recent_questions(), "新检索问题": retrieval_question})
                    hits = rag.search(retrieval_question, top_k)
                client = create_client(settings) if hits and rag.settings.api_key else None
                with st.spinner("正在生成回答…"):
                    reply = answer(question, hits, history, client, settings.chat_model)
            elif not downloading:
                print_step("固定说明", 流转="跳过模型调用与知识库检索")
            render_message("assistant", reply, hits)
            if downloads:
                render_downloads(rag.store, downloads, f"chat_download_{len(history) + 1}")
            history.add_turn(question, reply, hits, retrieval=retrieval, downloads=downloads)
            print_step(
                "页面接收最终结果",
                回答字符数=len(reply),
                来源数量=len(hits),
                历史消息数=len(history),
            )
            debug_step("完整回答", lambda: {"回答": reply})
        except Exception:
            print_step("聊天异常", level=logging.ERROR, 结果="未保存本轮聊天，请查看服务端异常日志")
            logger.exception("生成知识库回答失败")
            st.error(
                "回答未完成，本轮未保存。请重新发送问题；若持续失败，请联系管理员。"
            )
        finally:
            if client:
                client.close()
                print_step("聊天客户端已关闭")
