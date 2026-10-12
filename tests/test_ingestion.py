"""后台导入的原子状态流转、配置快照、恢复与真实离线导入。"""

import json
import sqlite3
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import MagicMock, patch

from app.rag.embeddings import DemoEmbedder
from app.ingestion import IngestionManager, get_ingestion_manager
from app.schemas import Settings


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(ignore_cleanup_errors=True)
        self.directory = Path(self.temporary.name)
        self.settings = Settings(
            api_key="secret-never-persist", chat_model="chat", embedding_model="embedding",
            ingestion_directory=str(self.directory / "jobs"),
            chroma_directory=str(self.directory / "chroma"), chunk_size=40, chunk_overlap=5,
        )
        self.manager = IngestionManager(self.settings)

    def tearDown(self):
        self.manager.stop()
        self.temporary.cleanup()

    def submit(self, files=None, settings=None):
        with patch.object(self.manager, "start"):
            return self.manager.submit_batch(
                files or [("test.txt", b"test content")], offline=True, settings=settings
            )

    def wait_for_job(self, job_id):
        deadline = monotonic() + 5
        while monotonic() < deadline:
            job = self.manager.get(job_id)
            if job.status not in {"queued", "processing"}:
                return job
            sleep(0.01)
        self.fail("后台任务未在限定时间内结束")

    def test_same_database_reuses_worker_when_settings_change(self):
        manager = get_ingestion_manager(self.settings)
        try:
            thread = manager._thread
            job = self.submit()[0]
            manager._update(job.id, status="processing", stage="embedding")
            changed = replace(self.settings, chunk_size=80, embedding_model="new-model")
            same = get_ingestion_manager(changed)
            self.assertIs(same, manager)
            self.assertIs(same._thread, thread)
            self.assertEqual(same.get(job.id).status, "processing")
        finally:
            manager.stop()

    def test_cancelled_task_cannot_be_claimed(self):
        job = self.submit()[0]
        cancelled = self.manager.cancel(job.id)
        self.assertEqual(cancelled.status, "cancelled")
        with patch("app.ingestion.ingestion.RAGService") as factory:
            self.manager._process(job.id)
            factory.assert_not_called()
        self.assertEqual(self.manager.get(job.id).status, "cancelled")
        self.assertFalse(Path(self.manager._get_row(job.id)["upload_path"]).exists())

    def test_claimed_task_cannot_be_cancelled(self):
        job = self.submit()[0]
        service = MagicMock()
        def importing(*args):
            with self.assertRaises(ValueError):
                self.manager.cancel(job.id)
            self.assertEqual(self.manager.get(job.id).status, "processing")
            return 1
        service.add_file.side_effect = importing
        with patch("app.ingestion.ingestion.RAGService", return_value=service):
            self.manager._process(job.id)
        self.assertEqual(self.manager.get(job.id).status, "completed")
        self.assertEqual(self.manager.get(job.id).attempt, 1)

    def test_worker_survives_unhandled_error_and_transient_database_failure(self):
        first, second = self.submit([("one.txt", b"one"), ("two.txt", b"two")])
        finished = threading.Event()
        fail = self.manager._fail
        failures = 0
        def fail_temporarily(*args):
            nonlocal failures
            failures += 1
            if failures == 1:
                raise sqlite3.OperationalError("temporarily locked")
            fail(*args)
        def process(job_id):
            if job_id == first.id:
                raise RuntimeError("simulated unexpected error")
            self.manager._update(job_id, status="completed", stage="completed")
            finished.set()
        with self.assertLogs("app.ingestion", level="ERROR"):
            with patch.object(self.manager, "_process", side_effect=process), patch.object(
                self.manager, "_fail", side_effect=fail_temporarily
            ):
                self.manager.start()
                self.assertTrue(finished.wait(timeout=5))
                self.assertEqual(self.manager.get(first.id).status, "failed")
                self.assertEqual(self.manager.get(second.id).status, "completed")
                self.assertTrue(self.manager._thread.is_alive())
                self.manager.stop()

    def test_configuration_snapshot_survives_changes_and_retry(self):
        job = self.submit()[0]
        snapshot = self.manager._get_row(job.id)["settings_json"]
        self.assertNotIn(self.settings.api_key, snapshot)
        self.assertNotIn("api_key", json.loads(snapshot))
        self.manager.settings = replace(self.settings, api_key="replacement-key", chunk_size=80,
                                        embedding_model="new-model", chroma_collection="new-collection")
        service = MagicMock()
        service.add_file.side_effect = ValueError("simulated import failure")
        with self.assertLogs("app.ingestion", level="ERROR"), patch(
            "app.ingestion.ingestion.RAGService", return_value=service
        ) as factory:
            self.manager._process(job.id)
        captured = factory.call_args.kwargs["settings"]
        self.assertEqual(captured.chunk_size, 40)
        self.assertEqual(captured.embedding_model, "embedding")
        self.assertEqual(captured.chroma_collection, "knowledge")
        self.assertEqual(captured.api_key, "replacement-key")
        with patch.object(self.manager, "start"):
            self.manager.retry(job.id)
        service.add_file.side_effect = None
        service.add_file.return_value = 2
        reopened = IngestionManager(self.manager.settings)
        with patch("app.ingestion.ingestion.RAGService", return_value=service):
            reopened._process(job.id)
        result = self.manager.get(job.id)
        self.assertEqual((result.status, result.attempt, result.chunk_count), ("completed", 2, 2))
        self.assertEqual(self.manager._get_row(job.id)["settings_json"], snapshot)

    def test_legacy_schema_and_restart_recovery(self):
        job = self.submit()[0]
        self.manager._update(job.id, status="processing", stage="embedding")
        with self.manager._connect() as connection:
            connection.execute("ALTER TABLE ingestion_jobs DROP COLUMN settings_json")
            connection.execute("ALTER TABLE ingestion_jobs DROP COLUMN result_message")
        reopened = IngestionManager(self.settings)
        self.assertEqual(reopened.get(job.id).status, "processing")
        with patch("app.ingestion.ingestion.threading.Thread"):
            reopened.start()
        self.assertEqual(reopened.get(job.id).status, "queued")
        self.assertEqual(json.loads(reopened._get_row(job.id)["settings_json"])["chunk_size"], 40)

    def test_failed_submission_cleans_temporary_files(self):
        with patch.object(self.manager, "start"), patch("app.ingestion.ingestion.os.replace", side_effect=OSError):
            with self.assertRaises(OSError):
                self.manager.submit_batch([("one.txt", b"one")], offline=True)
        self.assertEqual(list(self.manager.uploads.iterdir()), [])
        self.assertEqual(self.manager.list_jobs(), [])

    def test_real_offline_worker_and_duplicate_import(self):
        first = self.manager.submit_batch([("test.txt", b"real offline content")], offline=True)[0]
        self.assertEqual(self.wait_for_job(first.id).status, "completed")
        duplicate = self.manager.submit_batch([("test.txt", b"real offline content")], offline=True)[0]
        self.assertEqual(self.wait_for_job(duplicate.id).status, "skipped")
        self.assertEqual(list(self.manager.uploads.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
