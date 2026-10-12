"""多文件后台向量解析任务页面。"""

import logging
from hashlib import sha256

import streamlit as st

from app.rag.documents import loaders
from app.ingestion import IngestionManager, get_ingestion_manager
from app.schemas import IngestionJob, Settings
from app.storage import KnowledgeStore

logger = logging.getLogger(__name__)


STAGE_LABELS = {
    "queued": "等待处理",
    "loading": "读取文档",
    "splitting": "切分片段",
    "embedding": "生成向量",
    "indexing": "写入知识库",
    "completed": "已完成",
    "duplicate": "重复内容",
    "failed": "失败",
    "cancelled": "已取消",
}


def _format_time(job: IngestionJob) -> str:
    """以本地页面时区显示任务提交时间。"""
    return job.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _render_jobs(manager: IngestionManager) -> None:
    jobs = manager.list_jobs(100)
    if not jobs:
        st.info("还没有导入任务。上传 TXT 或 Markdown 文档开始处理。")
        return

    completed = sum(job.status == "completed" for job in jobs)
    active = sum(job.status in {"queued", "processing"} for job in jobs)
    failed = sum(job.status == "failed" for job in jobs)
    skipped = sum(job.status == "skipped" for job in jobs)
    columns = st.columns(4)
    columns[0].metric("处理中 / 排队", active)
    columns[1].metric("已完成", completed)
    columns[2].metric("失败", failed)
    columns[3].metric("重复跳过", skipped)

    st.caption("每个文件是一条独立任务；同批次文件分别显示状态、处理进度和结果。")
    for job in jobs:
        title = f"{job.filename} · {STAGE_LABELS.get(job.stage, job.stage)}"
        with st.expander(title, expanded=job.status in {"processing", "failed"}):
            st.caption(
                f"批次 {job.batch_id[:8]} · {_format_time(job)} · "
                f"{job.file_size / 1024:.1f} KB · "
                f"{'离线演示' if job.embedding_mode == 'offline' else '中文语义向量'} · "
                f"第 {job.attempt} 次尝试"
            )
            st.progress(
                job.progress,
                text=f"{job.progress}% · {STAGE_LABELS.get(job.stage, job.stage)}",
            )
            if job.status == "completed":
                st.success(f"已写入 {job.chunk_count} 个片段，可在知识库问答页提问。")
            elif job.status == "skipped":
                st.info(job.result_message or "内容已存在，跳过重复导入。")
            elif job.status == "failed":
                st.error(job.error_message or "导入失败")
            if job.status in {"failed", "queued"}:
                label, action = ("重试", manager.retry) if job.status == "failed" else ("取消", manager.cancel)
                if st.button(label, key=f"{job.status}_{job.id}"):
                    try:
                        action(job.id)
                    except ValueError as exc:
                        st.warning(str(exc))
                    else:
                        st.rerun()


@st.fragment(run_every="2s")
def render_ingestion_page(
    settings: Settings, store: KnowledgeStore, offline: bool = False
) -> None:
    """提交多文件导入，并自动刷新后台任务状态。"""
    # 用独立的代数生成上传组件 key；提交成功后递增即可显示一个空的上传组件。
    uploader_generation = st.session_state.setdefault("ingestion_uploader_generation", 0)
    manager = get_ingestion_manager(settings)
    st.caption(
        "当前版本支持 UTF-8 TXT 和 Markdown。可一次选择多个文件；提交后向量解析在后台逐文件执行。"
    )
    uploads = st.file_uploader(
        "选择一个或多个 TXT / Markdown 文档",
        type=loaders.extensions,
        accept_multiple_files=True,
        key=f"ingestion_files_{uploader_generation}",
        help=f"每次最多 {settings.ingestion_max_files} 个文件；单文件上限 "
        f"{settings.ingestion_max_file_bytes // (1024 * 1024)} MB，整批上限 "
        f"{settings.ingestion_max_batch_bytes // (1024 * 1024)} MB。",
    )
    selected_uploads = []
    if uploads:
        signature = (
            getattr(store, "collection_name", type(store).__name__),
            tuple(
                (getattr(upload, "file_id", index), upload.name, upload.size)
                for index, upload in enumerate(uploads)
            ),
        )
        preview = st.session_state.get("_ingestion_preview")
        if not preview or preview["signature"] != signature:
            items = []
            first_by_hash: dict[str, str] = {}
            for upload in uploads:
                content_hash = sha256(upload.getvalue()).hexdigest()
                if content_hash in first_by_hash:
                    kind = "batch_duplicate"
                    detail = f"与本批文件“{first_by_hash[content_hash]}”内容相同"
                else:
                    first_by_hash[content_hash] = upload.name
                    existing_source = store.find_content_hash(content_hash)
                    if existing_source:
                        kind = "duplicate"
                        detail = f"与知识库中的“{existing_source}”内容相同，后台会检查完整性"
                    elif store.has_source(upload.name):
                        kind = "update"
                        detail = "知识库已有同名文件，成功导入后会替换旧内容"
                    else:
                        kind = "new"
                        detail = "新文件"
                items.append({"kind": kind, "detail": detail})
            preview = {"signature": signature, "items": items}
            st.session_state["_ingestion_preview"] = preview

        flagged = False
        with st.container(border=True):
            st.markdown("**导入检查与提交选择**")
            st.caption("相同内容会自动跳过；同名文件内容有变化时会在新内容写入成功后替换旧片段。可取消勾选不想提交的文件。")
            for index, (upload, item) in enumerate(zip(uploads, preview["items"])):
                kind = item["kind"]
                label = f"提交：{upload.name} — {item['detail']}"
                include = st.checkbox(
                    label,
                    value=kind not in {"duplicate", "batch_duplicate"},
                    key=f"ingestion_include_{index}_{getattr(upload, 'file_id', upload.name)}",
                )
                if kind in {"duplicate", "batch_duplicate"}:
                    flagged = True
                if include:
                    selected_uploads.append(upload)
            if flagged:
                st.warning("发现内容重复项。请检查上方提示，并取消勾选不需要提交的文件；后台还会再次校验。")

    if st.button("提交后台解析", type="primary", disabled=not selected_uploads):
        try:
            files = [(upload.name, upload.getvalue()) for upload in selected_uploads]
            jobs = manager.submit_batch(files, offline=offline, settings=settings)
            st.session_state["last_ingestion_batch"] = jobs[0].batch_id
            st.success(f"已提交 {len(jobs)} 个文件，后台开始处理。")
            st.session_state["ingestion_uploader_generation"] = uploader_generation + 1
            st.session_state.pop("_ingestion_preview", None)
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("后台导入任务提交失败")
            st.error("任务提交失败，请查看服务端日志后重试。")

    st.divider()
    st.subheader("导入任务")
    _render_jobs(manager)
