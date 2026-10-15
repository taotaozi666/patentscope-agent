"""可重建的 FTS5/BM25 索引；Chroma 保存切片身份与原文的权威副本。"""

from contextlib import contextmanager
from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import threading
import unicodedata

from app.patent_identifiers import normalize_publication_number
from app.schemas import ChunkRecord

LEXICAL_VERSION = "lexical-cjk-bigram-v1"
_PARTS = re.compile(r"[A-Za-z0-9]+(?:[./_-][A-Za-z0-9]+)*|[\u3400-\u9fff]+")
_STOP = set("的了是吗多少什么请如何及与在中为该本其")


def tokens(text: str) -> list[str]:
    """中日韩汉字使用单字和双字，技术标识及小数保持整体；查询与入库同规则。"""
    terms = []
    for part in _PARTS.findall(unicodedata.normalize("NFKC", text)):
        if "\u3400" <= part[0] <= "\u9fff":
            terms.extend(c for c in part if c not in _STOP)
            terms.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            terms.append(part.casefold())
    # Encode tokens so FTS syntax and punctuation cannot reinterpret raw queries.
    return ["t" + term.encode().hex() for term in terms]


def index_fields(record: ChunkRecord) -> tuple[str, str]:
    meta = record.metadata
    headers = [str(meta.get(k) or "") for k in
               ("heading_path", "object_title", "publication_number", "table_header", "table_row_label")]
    if meta.get("doc_type_kwd") == "table":
        try:
            relations = json.loads(meta.get("table_cell_relations", "{}"))
            support = {s["cell_id"]: s["text"] for relation in relations.values() for s in relation.get("support", [])}
            headers.extend(text for text in support.values() if isinstance(text, str))
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    return record.text, "\n".join(headers)


def signature(record: ChunkRecord) -> str:
    return sha256(json.dumps(index_fields(record), ensure_ascii=False).encode()).hexdigest()


class LexicalIndex:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.lock = threading.RLock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = sqlite3.connect(":memory:", check_same_thread=False) if path is None else None
        with self.connection() as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS lexical_state (id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, built INTEGER NOT NULL, version TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO lexical_state VALUES (1,0,-1,'')")
            connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lexical_chunks USING fts5(chunk_id UNINDEXED, publication UNINDEXED, signature UNINDEXED, body, headers)")
            connection.execute("CREATE TABLE IF NOT EXISTS import_batches (id TEXT PRIMARY KEY, ready INTEGER NOT NULL)")
        self.invalidate()  # Reconcile historical collections and external edits on first use.

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            if self.memory is not None:
                connection = self.memory
            else:
                assert self.path is not None
                connection = sqlite3.connect(self.path, timeout=30)
            try:
                yield connection
            finally:
                if self.memory is None:
                    connection.close()

    def invalidate(self):
        with self.connection() as connection, connection:
            connection.execute("UPDATE lexical_state SET generation=generation+1 WHERE id=1")

    def begin_batch(self, identifier: str):
        with self.connection() as connection, connection:
            connection.execute("INSERT INTO import_batches VALUES (?,0)", (identifier,))
            connection.execute("UPDATE lexical_state SET generation=generation+1 WHERE id=1")

    def commit_batch(self, identifier: str):
        with self.connection() as connection, connection:
            connection.execute("UPDATE import_batches SET ready=1 WHERE id=?", (identifier,))
            connection.execute("UPDATE lexical_state SET generation=generation+1 WHERE id=1")

    def active_records(self, records: Iterable[ChunkRecord]) -> list[ChunkRecord]:
        records = list(records)
        ready = self._ready_batches(record.metadata.get("index_batch_id") for record in records)
        return [record for record in records
                if not record.metadata.get("index_batch_id") or record.metadata["index_batch_id"] in ready]

    def active_metadata(self, records):
        ready = self._ready_batches(meta.get("index_batch_id") for meta in records)
        return [meta for meta in records if not meta.get("index_batch_id") or meta["index_batch_id"] in ready]

    def _ready_batches(self, identifiers) -> set[str]:
        identifiers = list(dict.fromkeys(i for i in identifiers if isinstance(i, str) and i))
        ready: set[str] = set()
        with self.connection() as connection:
            for start in range(0, len(identifiers), 500):
                batch = identifiers[start:start + 500]
                rows = connection.execute("SELECT id FROM import_batches WHERE ready=1 AND id IN ("
                                          + ",".join("?" for _ in batch) + ")", batch)
                ready.update(row[0] for row in rows)
        return ready

    def recover_missing_batches(self, metadatas):
        """重建被删除的索引时，仅恢复切片数量和位置完整的批次。"""
        grouped = {}
        for meta in metadatas:
            if meta.get("index_batch_id"):
                grouped.setdefault(meta["index_batch_id"], []).append(meta)
        with self.connection() as connection, connection:
            known = {row[0] for row in connection.execute("SELECT id FROM import_batches")}
            for identifier, records in grouped.items():
                if identifier in known:
                    continue
                total = records[0].get("index_batch_total")
                complete = (type(total) is int and total > 0 and len(records) == total
                            and all(record.get("index_batch_total") == total for record in records)
                            and {r.get("index_batch_position") for r in records} == set(range(1, total + 1)))
                connection.execute("INSERT INTO import_batches VALUES (?,?)", (identifier, int(complete)))

    def synchronize(self, read_records: Callable[[], list[ChunkRecord]]) -> None:
        # Generation checks prevent a concurrent mutation from marking an old snapshot current.
        for _ in range(3):
            with self.connection() as connection:
                generation, built, version = connection.execute("SELECT generation,built,version FROM lexical_state WHERE id=1").fetchone()
            if generation == built and version == LEXICAL_VERSION:
                return
            records = read_records()
            rows = [(record.id, normalize_publication_number(str(record.metadata.get("publication_number", ""))),
                     signature(record), " ".join(tokens(body)), " ".join(tokens(headers)))
                    for record in records for body, headers in [index_fields(record)]]
            with self.connection() as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute("SELECT generation FROM lexical_state WHERE id=1").fetchone()[0]
                if current != generation:
                    continue
                connection.execute("DELETE FROM lexical_chunks")
                connection.executemany("INSERT INTO lexical_chunks VALUES (?,?,?,?,?)", rows)
                connection.execute("UPDATE lexical_state SET built=?,version=? WHERE id=1", (generation, LEXICAL_VERSION))
                return
        raise RuntimeError("关键词索引正在更新，请重试查询")

    def search(self, question: str, limit: int, publication_numbers: list[str] | None = None):
        terms = list(dict.fromkeys(tokens(question)))[:128]
        if not terms or limit <= 0:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        parameters: list[str | int] = [expression]
        clause = ""
        if publication_numbers is not None:
            numbers = list(dict.fromkeys(normalize_publication_number(n) for n in publication_numbers))
            numbers = [n for n in numbers if n]
            if not numbers:
                return []
            clause = " AND publication IN (" + ",".join("?" for _ in numbers) + ")"
            parameters.extend(numbers)
        parameters.append(limit)
        with self.connection() as connection:
            return connection.execute(
                "SELECT chunk_id,signature,bm25(lexical_chunks,0,0,0,1,2) AS score FROM lexical_chunks "
                "WHERE lexical_chunks MATCH ?" + clause + " ORDER BY score,chunk_id LIMIT ?", parameters).fetchall()
