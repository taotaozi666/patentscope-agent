"""在文档资产数据库中保存检查报告和人工复核，不写入向量库。"""

import json
from .catalog import connect, initialize
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.quality import PageQuality, QualityIssue, QualityReport, QualityReview, ReviewDecision, TableQuality


class QualityRepository:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "catalog.sqlite3"
        with closing(connect(self.database)) as connection, connection:
            initialize(connection)
            connection.execute("CREATE TABLE IF NOT EXISTS quality_reports (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, checked_at TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS quality_run ON quality_reports(run_id,checked_at)")
            connection.execute("CREATE TABLE IF NOT EXISTS quality_reviews (report_id TEXT NOT NULL, target TEXT NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(report_id,target))")

    def save(self, report: QualityReport) -> None:
        with closing(connect(self.database)) as connection, connection:
            connection.execute("INSERT INTO quality_reports VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET checked_at=excluded.checked_at,payload=excluded.payload",
                               (report.id, report.run_id, report.checked_at, json.dumps(asdict(report), ensure_ascii=False)))

    def latest(self, run_id: str) -> QualityReport | None:
        with closing(connect(self.database)) as connection:
            row = connection.execute("SELECT payload FROM quality_reports WHERE run_id=? ORDER BY checked_at DESC LIMIT 1", (run_id,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["pages"] = [PageQuality(**page) for page in data["pages"]]
        data["issues"] = [QualityIssue(**issue) for issue in data["issues"]]
        data["tables"] = [TableQuality(**table) for table in data.get("tables", [])]
        return QualityReport(**data)

    def reviews(self, report_id: str) -> list[QualityReview]:
        with closing(connect(self.database)) as connection:
            rows = connection.execute("SELECT report_id,target,decision,note,updated_at FROM quality_reviews WHERE report_id=? ORDER BY updated_at DESC", (report_id,)).fetchall()
        return [QualityReview(*row) for row in rows]

    def review(self, report_id: str, target: str, decision: ReviewDecision, note: str) -> None:
        """复核绑定报告及具体目标；不覆盖自动检查结论。"""
        if decision not in {"passed", "problem", "pending"}:
            raise ValueError("无效的复核结论")
        with closing(connect(self.database)) as connection, connection:
            row = connection.execute("SELECT payload FROM quality_reports WHERE id=?", (report_id,)).fetchone()
            if row is None:
                raise ValueError("检查报告不存在")
            data = json.loads(row[0])
            allowed = {"document", *(f"page:{page['page_no']}" for page in data["pages"]),
                       *(f"issue:{issue['id']}" for issue in data["issues"]),
                       *(f"table:{table['id']}" for table in data.get("tables", [])),
                       *(f"cell:{table['id']}:{cell}" for table in data.get("tables", []) for cell in table["cell_ids"])}
            if target not in allowed:
                raise ValueError("复核目标不属于当前检查报告")
            connection.execute("INSERT INTO quality_reviews VALUES (?,?,?,?,?) ON CONFLICT(report_id,target) DO UPDATE SET decision=excluded.decision,note=excluded.note,updated_at=excluded.updated_at",
                               (report_id, target, decision, note.strip(), datetime.now(UTC).isoformat()))

    def for_chunk(self, run_id: str, chunk_id: str) -> QualityReport | None:
        """Use only a report for the exact retrieved chunk set, not another budget."""
        with closing(connect(self.database)) as connection:
            rows = connection.execute("SELECT payload FROM quality_reports WHERE run_id=? ORDER BY checked_at DESC", (run_id,)).fetchall()
        for row in rows:
            data = json.loads(row[0])
            if chunk_id in data["chunk_ids"]:
                return QualityReport(**{**data,
                    "pages": [PageQuality(**p) for p in data["pages"]],
                    "issues": [QualityIssue(**i) for i in data["issues"]],
                    "tables": [TableQuality(**t) for t in data.get("tables", [])]})
        return None
