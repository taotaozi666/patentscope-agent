"""持久化多文件导入队列及单个后台 Worker。"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
from time import monotonic
from contextlib import closing
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from openai import RateLimitError

from app.embeddings import create_embeddings
from app.rag import RAGService
from app.schemas import IngestionJob, Settings

logger = logging.getLogger(__name__)


class JobNotFoundError(LookupError):
    """任务 ID 不存在。"""


class IngestionManager:
    """每份文件独立追踪和重试；任务记录及待处理文件在本机持久保存。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.directory = Path(settings.ingestion_directory)
        self.uploads = self.directory / "uploads"
        self.database = self.directory / "jobs.sqlite3"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._started = False
        self._thread: threading.Thread | None = None
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    embedding_mode TEXT NOT NULL DEFAULT 'semantic',
                    upload_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    result_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(ingestion_jobs)")
            }
            if "result_message" not in columns:
                connection.execute(
                    "ALTER TABLE ingestion_jobs ADD COLUMN result_message TEXT"
                )
            # 进程退出时正在执行的任务会在下次启动时重新排队。
            connection.execute(
                "UPDATE ingestion_jobs SET status='queued', stage='queued', progress=0, "
                "updated_at=? WHERE status='processing'",
                (self._now(),),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _record(row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            id=row["id"], batch_id=row["batch_id"], filename=row["filename"],
            file_size=row["file_size"], embedding_mode=row["embedding_mode"],
            status=row["status"], stage=row["stage"],
            progress=row["progress"], attempt=row["attempt"],
            chunk_count=row["chunk_count"], error_message=row["error_message"],
            result_message=row["result_message"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def start(self) -> None:
        """启动唯一 Worker，并恢复进程重启前未完成的任务。"""
        with self._lock:
            if self._started:
                return
            self._started = True
            with closing(self._connect()) as connection, connection:
                pending = connection.execute(
                    "SELECT id FROM ingestion_jobs WHERE status='queued' ORDER BY created_at"
                ).fetchall()
            for row in pending:
                self._queue.put(row["id"])
            self._thread = threading.Thread(
                target=self._work, name="document-ingestion", daemon=True
            )
            self._thread.start()

    def submit_batch(
        self, files: Sequence[tuple[str, bytes]], offline: bool = False
    ) -> list[IngestionJob]:
        """一次接收多份文件，为每份文件创建独立可追踪的任务。"""
        if not files:
            raise ValueError("请至少选择一个文件")
        if len(files) > self.settings.ingestion_max_files:
            raise ValueError(f"一次最多上传 {self.settings.ingestion_max_files} 个文件")
        if sum(len(content) for _, content in files) > self.settings.ingestion_max_batch_bytes:
            limit = self.settings.ingestion_max_batch_bytes // (1024 * 1024)
            raise ValueError(f"本批文件总大小不能超过 {limit} MB")
        prepared: list[tuple[str, bytes, str, str]] = []
        for original_name, content in files:
            filename = PureWindowsPath(original_name.replace("/", "\\")).name
            suffix = Path(filename).suffix.lower()
            if suffix not in {".txt", ".md"}:
                raise ValueError(f"{filename}：当前上传页仅支持 TXT 和 Markdown 文件")
            if not content:
                raise ValueError(f"{filename}：文件内容为空")
            if len(content) > self.settings.ingestion_max_file_bytes:
                limit = self.settings.ingestion_max_file_bytes // (1024 * 1024)
                raise ValueError(f"{filename}：超过单文件 {limit} MB 大小限制")
            job_id = uuid4().hex
            prepared.append((filename, content, job_id, str(self.uploads / f"{job_id}.upload")))

        # 先启动 Worker 并恢复旧任务，避免新任务被启动扫描和显式入队重复加入。
        self.start()
        batch_id, created_at = uuid4().hex, self._now()
        written: list[Path] = []
        try:
            for _, content, _, stored_path in prepared:
                path = Path(stored_path)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, path)
                written.append(path)
            with closing(self._connect()) as connection, connection:
                connection.executemany(
                    """INSERT INTO ingestion_jobs
                    (id,batch_id,filename,file_size,embedding_mode,upload_path,status,stage,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,'queued','queued',?,?)""",
                    [(job_id, batch_id, filename, len(content), "offline" if offline else "semantic", stored, created_at, created_at)
                     for filename, content, job_id, stored in prepared],
                )
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
                path.with_suffix(".tmp").unlink(missing_ok=True)
            raise
        for _, _, job_id, _ in prepared:
            self._queue.put(job_id)
        return [self.get(job_id) for _, _, job_id, _ in prepared]

    def _get_row(self, job_id: str) -> sqlite3.Row | None:
        with closing(self._connect()) as connection, connection:
            return connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)
            ).fetchone()

    def get(self, job_id: str) -> IngestionJob:
        row = self._get_row(job_id)
        if row is None:
            raise JobNotFoundError("找不到该导入任务")
        return self._record(row)

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def retry(self, job_id: str) -> IngestionJob:
        with closing(self._connect()) as connection, connection:
            result = connection.execute(
                """UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,
                chunk_count=0,error_message=NULL,result_message=NULL,updated_at=?
                WHERE id=? AND status='failed' AND upload_path IS NOT NULL""",
                (self._now(), job_id),
            )
        if result.rowcount == 0:
            job = self.get(job_id)
            raise ValueError("只有失败且仍保留上传文件的任务可以重试")
        self.start()
        self._queue.put(job_id)
        return self.get(job_id)

    def cancel(self, job_id: str) -> IngestionJob:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT upload_path FROM ingestion_jobs WHERE id=? AND status='queued'",
                (job_id,),
            ).fetchone()
            if row is None:
                job = self.get(job_id)
                raise ValueError("只有尚未开始的排队任务可以取消")
            connection.execute(
                "UPDATE ingestion_jobs SET status='cancelled',stage='cancelled',updated_at=? WHERE id=?",
                (self._now(), job_id),
            )
        Path(row["upload_path"]).unlink(missing_ok=True)
        return self.get(job_id)

    def _update(self, job_id: str, **changes: object) -> None:
        assignments = ",".join(f"{key}=?" for key in changes)
        values = [*changes.values(), self._now(), job_id]
        with closing(self._connect()) as connection, connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments},updated_at=? WHERE id=?",
                values,
            )

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._process(job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        row = self._get_row(job_id)
        if row is None or row["status"] != "queued":
            return
        path = Path(row["upload_path"])
        if not path.is_file():
            self._update(job_id, status="failed", stage="failed", error_message="待处理文件不存在，请重新上传")
            return
        self._update(
            job_id, status="processing", stage="loading", progress=5,
            attempt=row["attempt"] + 1, error_message=None,
        )
        try:
            offline = row["embedding_mode"] == "offline"
            embedder = create_embeddings(self.settings, offline=offline)
            service = RAGService(embedder, settings=self.settings)
            source = row["filename"]
            last_saved = {"stage": "", "progress": -1, "time": 0.0}

            def progress(stage: str, completed: int, total: int) -> None:
                if stage == "splitting":
                    value = 15
                elif stage == "embedding":
                    value = 25 if total == 0 else 25 + int(55 * completed / total)
                elif stage == "indexing":
                    value = 90
                elif stage == "completed":
                    value = 100
                elif stage == "duplicate":
                    value = 100
                else:
                    value = 10
                now = monotonic()
                stage_changed = stage != last_saved["stage"]
                progress_changed = value >= last_saved["progress"] + 1
                time_elapsed = now - last_saved["time"] >= 2.0
                finished = value >= 100
                if not (stage_changed or progress_changed or time_elapsed or finished):
                    return
                self._update(
                    job_id, stage=stage, progress=value,
                    chunk_count=completed if stage in {"indexing", "completed"} else 0,
                )
                last_saved.update(stage=stage, progress=value, time=now)

            count = service.add_file(source, path.read_bytes(), progress)
            if count == 0:
                self._update(
                    job_id,
                    status="skipped",
                    stage="duplicate",
                    progress=100,
                    chunk_count=0,
                    error_message=None,
                    result_message="知识库中已有相同内容和解析配置，已跳过重复导入。",
                )
                path.unlink(missing_ok=True)
                return
            self._update(
                job_id, status="completed", stage="completed", progress=100,
                chunk_count=count, error_message=None, result_message=None,
            )
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.exception("后台文档导入失败（任务 %s）", job_id)
            if isinstance(exc, UnicodeDecodeError):
                message = "文件不是有效的 UTF-8 编码，请转换后重试"
            elif isinstance(exc, RateLimitError):
                message = (
                    "千问向量接口触发限流或 Token 配额不足。请等待约一分钟后重试；"
                    "如果仍失败，将 .env 中的 EMBEDDING_CONCURRENCY 调低为 1，"
                    "并检查百炼控制台的模型配额。"
                )
            elif isinstance(exc, ValueError):
                message = str(exc)
            else:
                message = f"处理失败（{type(exc).__name__}），可重试或检查服务端日志"
            self._update(
                job_id, status="failed", stage="failed",
                error_message=message,
            )

