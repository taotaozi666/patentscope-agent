import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.conf.config import Config, get_settings


class ConfigTests(unittest.TestCase):
    def test_file_environment_and_reload(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'DASHSCOPE_API_KEY=test-secret\nCHAT_MODEL=" file-model "\nEMBEDDING_MODEL=embedding\n',
                encoding="utf-8",
            )
            settings = get_settings(path, {})
            self.assertEqual(settings.chat_model, "file-model")
            self.assertEqual(settings.embedding_model, "embedding")
            self.assertNotIn("test-secret", repr(settings))
            self.assertEqual(
                get_settings(path, {"CHAT_MODEL": "process-model"}).chat_model,
                "process-model",
            )
            self.assertEqual(get_settings(path, {"DASHSCOPE_API_KEY": ""}).api_key, "")
            path.write_text("CHAT_MODEL=updated", encoding="utf-8")
            self.assertEqual(get_settings(path, {}).chat_model, "updated")
            self.assertEqual(Config(path, {}).get("UNKNOWN", "fallback"), "fallback")

    def test_missing_file_and_aliases(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing.env"
            self.assertEqual(get_settings(path, {}).api_key, "")
            settings = get_settings(path, {"API_KEY": "test", "MODEL": "model"})
            self.assertEqual((settings.api_key, settings.chat_model), ("test", "model"))
            settings = get_settings(
                path, {"DASHSCOPE_BASE_URL": "https://example.com/v1"}
            )
            self.assertEqual(settings.base_url, "https://example.com/v1")
            settings = get_settings(
                path,
                {
                    "BASE_URL": "https://primary.example/v1",
                    "DASHSCOPE_BASE_URL": "https://legacy.example/v1",
                },
            )
            self.assertEqual(settings.base_url, "https://primary.example/v1")

    def test_chroma_path_and_chunk_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            settings = get_settings(
                path,
                {
                    "CHROMA_DIRECTORY": "database",
                    "CHUNK_SIZE": "800",
                    "CHUNK_OVERLAP": "80",
                },
            )
            self.assertEqual(
                Path(settings.chroma_directory), Path(directory) / "database"
            )
            self.assertEqual((settings.chunk_size, settings.chunk_overlap), (800, 80))
            with self.assertRaises(ValueError):
                get_settings(path, {"CHUNK_SIZE": "10", "CHUNK_OVERLAP": "10"})


if __name__ == "__main__":
    unittest.main()
