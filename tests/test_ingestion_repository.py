"""验证存储层在并发领取任务时只允许一次成功。"""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from app.schemas import Settings
from app.storage import IngestionRepository


class IngestionRepositoryTests(unittest.TestCase):
    def test_concurrent_claim_only_executes_once(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory) / "jobs.sqlite3"
            first = IngestionRepository(path)
            second = IngestionRepository(path)
            now = datetime.now(UTC).isoformat()
            snapshot = first.snapshot(Settings(api_key="", chat_model="", embedding_model=""))
            first.add_batch([("job", "batch", "test.txt", 4, "offline", "test.upload", snapshot, now, now)])
            ready = threading.Barrier(2)

            def claim(repository):
                ready.wait(timeout=5)
                return repository.claim("job") is not None

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(claim, repository) for repository in (first, second)]
                claimed = [future.result(timeout=10) for future in futures]
            self.assertEqual(sum(claimed), 1)
            job = first.get("job")
            if job is None:
                self.fail("领取后任务丢失")
            self.assertEqual((job.status, job.attempt), ("processing", 1))
