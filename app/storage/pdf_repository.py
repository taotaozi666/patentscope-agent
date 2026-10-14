"""长期保存原 PDF、结构结果与原文；向量库可删除后重建。"""

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document

from app.schemas.pdf import ParsedPdf
from app.pdf.preview import render_source_region


class PdfRepository:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.originals = self.directory / "originals"
        self.originals.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "catalog.sqlite3"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS pdf_runs (id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS pdf_chunks (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS pdf_objects (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS pdf_contexts (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def save(self, parsed: ParsedPdf, content: bytes, chunks: list[Document]) -> None:
        """解析产物先保存，嵌入失败仍能回看原始来源。"""
        path = self.originals / f"{parsed.content_hash}.pdf"
        if not path.exists():
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        self._save_region_images(parsed, content)
        run_id = str(chunks[0].metadata["pdf_parse_id"])
        with closing(sqlite3.connect(self.database, timeout=30)) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO pdf_runs VALUES (?,?,?,?,?)",
                               (run_id, parsed.content_hash, parsed.source, json.dumps(asdict(parsed), ensure_ascii=False), "parsed"))
            connection.executemany("INSERT OR IGNORE INTO pdf_chunks VALUES (?,?,?)", [
                (doc.metadata["pdf_chunk_id"], run_id, json.dumps({"text": doc.page_content,
                 "raw_content": doc.metadata.get("_pdf_raw_content", doc.page_content),
                 "context_text": doc.metadata.get("context_text", ""),
                 "index_text": doc.metadata.get("_pdf_index_text", doc.page_content),
                 "metadata": {key: value for key, value in doc.metadata.items()
                              if key not in {"_pdf_index_text", "_pdf_raw_content"}}}, ensure_ascii=False))
                for doc in chunks
            ])
            connection.executemany("INSERT OR IGNORE INTO pdf_objects VALUES (?,?,?)",
                                   [(item.id, run_id, json.dumps(asdict(item), ensure_ascii=False)) for item in parsed.objects])

    def _save_region_images(self, parsed: ParsedPdf, content: bytes) -> None:
        regions = self.directory / "regions"
        regions.mkdir(parents=True, exist_ok=True)
        targets = {}
        for element in parsed.elements:
            image_id = element.payload.get("image_id")
            if image_id and element.kind in {"image", "table"}:
                targets[str(image_id)] = asdict(element.span)
        for item in parsed.objects:
            image_id = item.payload.get("image_id")
            if image_id and item.spans:
                targets.setdefault(str(image_id), asdict(item.spans[0]))
        for image_id, span in targets.items():
            if not re.fullmatch(r"[a-f0-9]{64}", image_id):
                continue
            path = regions / f"{image_id}.png"
            if path.exists():
                continue
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(render_source_region(content, span))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    def get_region_image(self, image_id: str) -> bytes | None:
        if not re.fullmatch(r"[a-f0-9]{64}", image_id):
            return None
        path = self.directory / "regions" / f"{image_id}.png"
        return path.read_bytes() if path.is_file() else None

    def get_object(self, object_id: str) -> dict | None:
        with closing(sqlite3.connect(self.database, timeout=30)) as connection:
            row = connection.execute("SELECT payload FROM pdf_objects WHERE id=?", (object_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_context(self, cache_id: str) -> str | None:
        with closing(sqlite3.connect(self.database, timeout=30)) as connection:
            row = connection.execute("SELECT payload FROM pdf_contexts WHERE id=?", (cache_id,)).fetchone()
        return str(json.loads(row[0])["text"]) if row else None

    def save_context(self, cache_id: str, text: str, model: str, version: str) -> None:
        with closing(sqlite3.connect(self.database, timeout=30)) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO pdf_contexts VALUES (?,?)",
                               (cache_id, json.dumps({"text": text, "model": model, "version": version}, ensure_ascii=False)))

    def get_chunk(self, chunk_id: str) -> dict | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute("SELECT payload FROM pdf_chunks WHERE id=?", (chunk_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_run(self, run_id: str) -> dict | None:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute("SELECT payload FROM pdf_runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_runs(self) -> list[dict]:
        """文档检查使用解析资产，向量化失败的文档也可查看。"""
        with closing(sqlite3.connect(self.database, timeout=30)) as connection:
            rows = connection.execute("SELECT id,source,status,content_hash FROM pdf_runs ORDER BY rowid DESC").fetchall()
        return [dict(zip(("id", "source", "status", "content_hash"), row)) for row in rows]

    def list_chunks(self, run_id: str) -> list[dict]:
        with closing(sqlite3.connect(self.database, timeout=30)) as connection:
            rows = connection.execute("SELECT payload FROM pdf_chunks WHERE run_id=? ORDER BY rowid", (run_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def read_original(self, content_hash: str) -> bytes:
        if not re.fullmatch(r"[a-f0-9]{64}", content_hash):
            raise ValueError("无效的 PDF 版本标识")
        return (self.originals / f"{content_hash}.pdf").read_bytes()

    def mark_indexed(self, run_id: str) -> None:
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("UPDATE pdf_runs SET status='indexed' WHERE id=?", (run_id,))
