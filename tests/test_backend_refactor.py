"""配置隔离、导入故障边界与紧凑资产兼容回归。"""

from copy import deepcopy
from contextlib import closing
from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from app.indexing.service import DocumentImportService
from app.pdf.chunking import build_chunks
from app.pdf.parser import parse_pdf
from app.pdf.profile import PdfProfile, pipeline_fingerprint, chunk_profile
from app.pdf.serialization import restore_pdf
from app.pdf.tables.evidence import read_table_cells
from app.schemas import Settings
from app.storage.pdf_repository import PdfRepository
from tests.pdf_regression.fixtures import make_pdf


class BackendRefactorTests(unittest.TestCase):
    def settings(self, **kwargs):
        return Settings(api_key="secret", chat_model="model", embedding_model="embed", **kwargs)

    def test_fingerprint_changes_only_for_relevant_settings(self):
        settings = self.settings(pdf_context_mode="model")
        fingerprint = pipeline_fingerprint(settings, "hash", ".pdf")
        for key, value in (("base_url", "https://another.invalid"), ("chat_model", "other"),
                           ("chunk_size", 700), ("pdf_table_mode", "fast"), ("pdf_ocr_dpi", 300)):
            self.assertNotEqual(fingerprint, pipeline_fingerprint(replace(settings, **{key: value}), "hash", ".pdf"))
        self.assertEqual(fingerprint, pipeline_fingerprint(replace(settings, api_key="rotated"), "hash", ".pdf"))
        rule = replace(settings, pdf_context_mode="rule")
        self.assertEqual(pipeline_fingerprint(rule, "hash", ".pdf"),
                         pipeline_fingerprint(replace(rule, base_url="other", chat_model="other"), "hash", ".pdf"))

    def test_failed_enhancement_profiles_have_distinct_chunks_and_valid_links(self):
        parsed = parse_pdf("table.pdf", make_pdf("continued_table"), ocr_images=False)
        first = build_chunks(parsed, size=40)
        second = deepcopy(first)
        settings = self.settings(pdf_context_mode="model")
        PdfProfile.from_settings(settings).bind_chunks(first)
        PdfProfile.from_settings(replace(settings, base_url="https://other.invalid")).bind_chunks(second)
        self.assertTrue(set(c.metadata["pdf_chunk_id"] for c in first).isdisjoint(
            c.metadata["pdf_chunk_id"] for c in second))
        self.assertNotEqual(chunk_profile(first[0].metadata), chunk_profile(second[0].metadata))
        ids = {c.metadata["pdf_chunk_id"] for c in first}
        for chunk in first:
            for key in ("previous_chunk_id", "next_chunk_id"):
                self.assertTrue(not chunk.metadata[key] or chunk.metadata[key] in ids)

    def test_compact_storage_restores_complete_objects_and_legacy_runs(self):
        content = make_pdf("continued_table")
        parsed = parse_pdf("table.pdf", content, ocr_images=False)
        chunks = build_chunks(parsed)
        with TemporaryDirectory() as directory:
            repo = PdfRepository(directory)
            repo.save(parsed, content, chunks)
            run_id = chunks[0].metadata["pdf_parse_id"]
            with closing(sqlite3.connect(repo.database)) as connection, connection:
                raw = json.loads(connection.execute("SELECT payload FROM pdf_runs WHERE id=?", (run_id,)).fetchone()[0])
                self.assertNotIn("objects", raw)
            run = repo.get_run(run_id)
            self.assertEqual(json.loads(json.dumps(asdict(restore_pdf(run)))), json.loads(json.dumps(asdict(parsed))))
            parent = next(obj for obj in run["objects"] if obj["kind"] == "table")
            self.assertNotIn("evidence_cells", parent["payload"])
            cells = read_table_cells(run, parent["payload"])
            self.assertEqual(len(cells), sum(len(f.cells) for f in parsed.table_fragments))
            old_payload = {**parent["payload"], "evidence_schema_version": 1, "evidence_cells": list(cells.values())}
            self.assertEqual(read_table_cells({}, old_payload), cells)
            with closing(sqlite3.connect(repo.database)) as connection, connection:
                connection.execute("UPDATE pdf_runs SET payload=? WHERE id=?", (json.dumps(asdict(parsed)), run_id))
            self.assertEqual(json.loads(json.dumps(asdict(restore_pdf(repo.get_run(run_id))))), json.loads(json.dumps(asdict(parsed))))

    def test_wrong_original_rejected_before_asset_write(self):
        parsed = parse_pdf("table.pdf", make_pdf("native_grid"), ocr_images=False)
        with TemporaryDirectory() as directory:
            repo = PdfRepository(directory)
            with self.assertRaises(ValueError):
                repo.save(parsed, b"different", build_chunks(parsed))
            self.assertEqual(repo.list_runs(), [])
            self.assertEqual(list(Path(directory, "originals").iterdir()), [])

    def test_invalid_vectors_never_reach_storage(self):
        for vectors in ([[float("nan")]], [[float("inf")]], [[]], [[1], [1, 2]], []):
            with self.subTest(vectors=vectors):
                embedder, store = Mock(), Mock()
                embedder.embed_documents.return_value = vectors
                service = DocumentImportService(embedder, store, self.settings())
                documents = [Document(page_content="a"), Document(page_content="b")] if len(vectors) == 2 else [Document(page_content="a")]
                with self.assertRaises(ValueError):
                    service.add_documents(documents)
                store.add_documents.assert_not_called()

    def test_canonical_table_rejects_foreign_fragment(self):
        parsed = parse_pdf("table.pdf", make_pdf("continued_table"), ocr_images=False)
        build_chunks(parsed)
        run = json.loads(json.dumps(asdict(parsed)))
        payload = next(o["payload"] for o in run["objects"] if o["kind"] == "table")
        run["table_fragments"][0]["cells"][0]["fragment_id"] = "foreign"
        with self.assertRaises(ValueError):
            read_table_cells(run, payload)
