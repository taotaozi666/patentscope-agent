"""以 SQLite 为唯一队列的持久化后台文档导入。"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from time import monotonic
from uuid import uuid4

from openai import RateLimitError

from app.rag.documents import loaders
from app.rag.embeddings import create_embeddings
from app.rag import RAGService
from app.schemas import IngestionJob, Settings

logger = logging.getLogger(__name__)
_managers: dict[str, IngestionManager] = {}
_managers_lock = threading.Lock()


def get_ingestion_manager(settings: Settings) -> IngestionManager:
    """同一任务库在本进程只启动一个 Worker，配置变化不创建第二个实例。"""
    key = os.path.normcase(str(Path(settings.ingestion_directory).resolve()))
    with _managers_lock:
        manager = _managers.get(key)
        if manager is None:
            manager = IngestionManager(settings)
            _managers[key] = manager
        manager.settings = settings
        manager.start()
        return manager


class JobNotFoundError(LookupError):
    """任务 ID 不存在。"""


class IngestionManager:
    """每个文件独立追踪；SQLite 原子领取任务，重试保留原执行配置。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.directory = Path(settings.ingestion_directory).resolve()
        self.uploads = self.directory / "uploads"
        self.database = self.directory / "jobs.sqlite3"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        with closing(connection), connection:
            yield connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY, batch_id TEXT NOT NULL,
                    filename TEXT NOT NULL, file_size INTEGER NOT NULL,
                    embedding_mode TEXT NOT NULL DEFAULT 'semantic',
                    upload_path TEXT NOT NULL, status TEXT NOT NULL,
                    stage TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT, result_message TEXT,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(ingestion_jobs)")
            }
            for name, definition in {
                "result_message": "TEXT",
                "settings_json": "TEXT NOT NULL DEFAULT '{}'",
            }.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE ingestion_jobs ADD COLUMN {name} {definition}")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ingestion_queue ON ingestion_jobs(status,created_at)"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _snapshot(settings: Settings) -> str:
        values = asdict(settings)
        values.pop("api_key")  # 密钥只存在当前进程配置中，不写入任务数据库。
        return json.dumps(values, ensure_ascii=False)

    @staticmethod
    def _record(row: sqlite3.Row) -> IngestionJob:
        values = {field.name: row[field.name] for field in fields(IngestionJob)}
        for name in ("created_at", "updated_at"):
            values[name] = datetime.fromisoformat(values[name])
        return IngestionJob(**values)

    def start(self) -> None:
        """只在启动或线程退出后恢复任务，不在创建管理器时重置运行中任务。"""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            with self._connect() as connection:
                connection.execute(
                    "UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,updated_at=? "
                    "WHERE status='processing'", (self._now(),),
                )
                connection.execute(
                    "UPDATE ingestion_jobs SET settings_json=? WHERE settings_json='{}'",
                    (self._snapshot(self.settings),),
                )
            self._stop.clear()
            self._thread = threading.Thread(target=self._work, name="document-ingestion", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """处理完当前任务后停止，供进程清理及测试使用。"""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def submit_batch(
        self, files: Sequence[tuple[str, bytes]], offline: bool = False,
        settings: Settings | None = None,
    ) -> list[IngestionJob]:
        settings = settings or self.settings
        if not files:
            raise ValueError("请至少选择一个文件")
        if len(files) > settings.ingestion_max_files:
            raise ValueError(f"一次最多上传 {settings.ingestion_max_files} 个文件")
        if sum(len(content) for _, content in files) > settings.ingestion_max_batch_bytes:
            raise ValueError(f"本批文件总大小不能超过 {settings.ingestion_max_batch_bytes // (1024 * 1024)} MB")
        prepared = []
        for original_name, content in files:
            filename = PureWindowsPath(original_name.replace("/", "\\")).name
            if Path(filename).suffix.lower().lstrip(".") not in loaders.extensions:
                raise ValueError(f"{filename}：暂不支持该文件格式")
            if not content:
                raise ValueError(f"{filename}：文件内容为空")
            if len(content) > settings.ingestion_max_file_bytes:
                raise ValueError(f"{filename}：超过单文件 {settings.ingestion_max_file_bytes // (1024 * 1024)} MB 大小限制")
            job_id = uuid4().hex
            prepared.append((filename, content, job_id, self.uploads / f"{job_id}.upload"))

        self.start()
        batch_id, created_at = uuid4().hex, self._now()
        try:
            for _, content, _, path in prepared:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, path)
            with self._connect() as connection:
                connection.executemany(
                    """INSERT INTO ingestion_jobs
                    (id,batch_id,filename,file_size,embedding_mode,upload_path,status,stage,
                     settings_json,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,'queued','queued',?,?,?)""",
                    [(job_id, batch_id, name, len(content), "offline" if offline else "semantic",
                      str(path), self._snapshot(settings), created_at, created_at)
                     for name, content, job_id, path in prepared],
                )
        except Exception:
            for _, _, _, path in prepared:
                path.unlink(missing_ok=True)
                path.with_suffix(".tmp").unlink(missing_ok=True)
            raise
        self._wake.set()
        return [self.get(job_id) for _, _, job_id, _ in prepared]

    def _get_row(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()

    def get(self, job_id: str) -> IngestionJob:
        row = self._get_row(job_id)
        if row is None:
            raise JobNotFoundError("找不到该导入任务")
        return self._record(row)

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def retry(self, job_id: str) -> IngestionJob:
        with self._connect() as connection:
            result = connection.execute(
                """UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,
                chunk_count=0,error_message=NULL,result_message=NULL,updated_at=?
                WHERE id=? AND status='failed'""", (self._now(), job_id),
            )
        if not result.rowcount:
            self.get(job_id)
            raise ValueError("只有失败任务可以重试")
        self.start()
        self._wake.set()
        return self.get(job_id)

    def cancel(self, job_id: str) -> IngestionJob:
        with self._connect() as connection:
            row = connection.execute(
                "UPDATE ingestion_jobs SET status='cancelled',stage='cancelled',updated_at=? "
                "WHERE id=? AND status='queued' RETURNING upload_path",
                (self._now(), job_id),
            ).fetchone()
        if row is None:
            self.get(job_id)
            raise ValueError("只有尚未开始的排队任务可以取消")
        self._remove_upload(Path(row["upload_path"]))
        return self.get(job_id)

    def _update(self, job_id: str, **changes: object) -> None:
        assignments = ",".join(f"{key}=?" for key in changes)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments},updated_at=? WHERE id=?",
                [*changes.values(), self._now(), job_id],
            )

    def _work(self) -> None:
        failure: tuple[str, Exception] | None = None
        while not self._stop.is_set():
            job_id = None
            try:
                if failure is not None:
                    self._fail(*failure)
                    failure = None
                with self._connect() as connection:
                    row = connection.execute(
                        "SELECT id FROM ingestion_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
                    ).fetchone()
                if row is not None:
                    job_id = row["id"]
                    self._process(job_id)
                    continue
            except Exception as exc:
                logger.exception("导入 Worker 异常（任务 %s），继续处理后续任务", job_id)
                if job_id is not None:
                    failure = (job_id, exc)
            self._wake.wait(timeout=1)
            self._wake.clear()

    def _process(self, job_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "UPDATE ingestion_jobs SET status='processing',stage='loading',progress=5,"
                "attempt=attempt+1,error_message=NULL,updated_at=? "
                "WHERE id=? AND status='queued' RETURNING *", (self._now(), job_id),
            ).fetchone()
        if row is None:
            return
        try:
            settings = Settings(api_key=self.settings.api_key, **json.loads(row["settings_json"]))
            path = Path(row["upload_path"])
            if not path.is_file():
                raise ValueError("待处理文件不存在，请重新上传")
            embedder = create_embeddings(settings, offline=row["embedding_mode"] == "offline")
            service = RAGService(embedder, settings=settings)
            last_stage, last_progress, last_time = "", -1, 0.0

            def progress(stage: str, completed: int, total: int) -> None:
                nonlocal last_stage, last_progress, last_time
                value = {"splitting": 15, "indexing": 90, "completed": 100, "duplicate": 100}.get(stage, 10)
                if stage == "embedding":
                    value = 25 + (int(55 * completed / total) if total else 0)
                now = monotonic()
                if stage == last_stage and value <= last_progress and now - last_time < 2:
                    return
                self._update(job_id, stage=stage, progress=value,
                             chunk_count=completed if stage in {"indexing", "completed"} else 0)
                last_stage, last_progress, last_time = stage, value, now

            count = service.add_file(row["filename"], path.read_bytes(), progress)
            self._update(
                job_id, status="completed" if count else "skipped",
                stage="completed" if count else "duplicate", progress=100,
                chunk_count=count, error_message=None,
                result_message=None if count else "知识库中已有相同内容和解析配置，已跳过重复导入。",
            )
            self._remove_upload(path)
        except Exception as exc:
            logger.exception("后台文档导入失败（任务 %s）", job_id)
            self._fail(job_id, exc)

    def _fail(self, job_id: str, exc: Exception) -> None:
        if isinstance(exc, UnicodeDecodeError):
            message = "文件不是有效的 UTF-8 编码，请转换后重试"
        elif isinstance(exc, RateLimitError):
            message = "向量接口触发限流或配额不足，请稍后重试或检查模型配额。"
        elif isinstance(exc, ValueError):
            message = str(exc)
        else:
            message = f"处理失败（{type(exc).__name__}），可重试或检查服务端日志"
        with self._connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status='failed',stage='failed',error_message=?,updated_at=? "
                "WHERE id=? AND status IN ('queued','processing')", (message, self._now(), job_id),
            )

    @staticmethod
    def _remove_upload(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("无法清理上传文件：%s", path, exc_info=True)
