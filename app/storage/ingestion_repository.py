"""导入任务的 SQLite 持久化及原子状态流转，不负责执行解析任务。"""

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import IngestionJob, Settings


class IngestionRepository:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        with closing(connection), connection:
            yield connection

    def _initialize_database(self) -> None:
        with self.connect() as connection:
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
    def snapshot(settings: Settings) -> str:
        values = asdict(settings)
        values.pop("api_key")  # 密钥只存在当前进程配置中，不写入任务数据库。
        return json.dumps(values, ensure_ascii=False)


    @staticmethod
    def _record(row: sqlite3.Row) -> IngestionJob:
        values = {field.name: row[field.name] for field in fields(IngestionJob)}
        for name in ("created_at", "updated_at"):
            values[name] = datetime.fromisoformat(values[name])
        return IngestionJob(**values)

    def recover(self, settings: Settings) -> None:
        """Worker 启动时恢复中断任务，并为旧任务补齐执行配置。"""
        with self.connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,updated_at=? "
                "WHERE status='processing'", (self._now(),),
            )
            connection.execute(
                "UPDATE ingestion_jobs SET settings_json=? WHERE settings_json='{}'",
                (self.snapshot(settings),),
            )

    def add_batch(self, rows: Sequence[tuple[object, ...]]) -> None:
        """批次任务在同一个事务中入队；文件保存成功后才调用。"""
        with self.connect() as connection:
            connection.executemany(
                """INSERT INTO ingestion_jobs
                (id,batch_id,filename,file_size,embedding_mode,upload_path,status,stage,
                 settings_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?,'queued','queued',?,?,?)""", rows,
            )

    def get_row(self, job_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()

    def get(self, job_id: str) -> IngestionJob | None:
        row = self.get_row(job_id)
        return self._record(row) if row is not None else None

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def retry(self, job_id: str) -> bool:
        with self.connect() as connection:
            result = connection.execute(
                """UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,
                chunk_count=0,error_message=NULL,result_message=NULL,updated_at=?
                WHERE id=? AND status='failed'""", (self._now(), job_id),
            )
            return bool(result.rowcount)

    def cancel(self, job_id: str) -> Path | None:
        with self.connect() as connection:
            row = connection.execute(
                "UPDATE ingestion_jobs SET status='cancelled',stage='cancelled',updated_at=? "
                "WHERE id=? AND status='queued' RETURNING upload_path",
                (self._now(), job_id),
            ).fetchone()
        return Path(row["upload_path"]) if row is not None else None

    def update(self, job_id: str, **changes: object) -> None:
        """字段名限制为任务状态字段，字段值仍使用参数绑定。"""
        allowed = {"status", "stage", "progress", "attempt", "chunk_count", "error_message", "result_message"}
        if not changes:
            return
        if changes.keys() - allowed:
            raise ValueError("包含不允许更新的任务字段")
        assignments = ",".join(f"{key}=?" for key in changes)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE ingestion_jobs SET {assignments},updated_at=? WHERE id=?",
                [*changes.values(), self._now(), job_id],
            )

    def next_queued_id(self) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM ingestion_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
        return str(row["id"]) if row is not None else None

    def claim(self, job_id: str) -> sqlite3.Row | None:
        """条件更新并返回任务，避免取消后的任务被执行或同一任务被重复领取。"""
        with self.connect() as connection:
            return connection.execute(
                "UPDATE ingestion_jobs SET status='processing',stage='loading',progress=5,"
                "attempt=attempt+1,error_message=NULL,updated_at=? "
                "WHERE id=? AND status='queued' RETURNING *", (self._now(), job_id),
            ).fetchone()

    def fail(self, job_id: str, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status='failed',stage='failed',error_message=?,updated_at=? "
                "WHERE id=? AND status IN ('queued','processing')", (message, self._now(), job_id),
            )

