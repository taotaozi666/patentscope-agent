"""文档加载与扩展注册接口测试。"""

import unittest

from app.rag.documents import LoaderRegistry, UTF8DocumentLoader, loaders
from app.rag.documents import split_documents


class DocumentTests(unittest.TestCase):
    def test_supported_formats_and_bom(self):
        for source in ["文档.txt", "文档.MD"]:
            documents = loaders.load(source, "\ufeff# 标题\n\n正文".encode())
            self.assertEqual(documents[0].page_content, "# 标题\n\n正文")
            self.assertEqual(documents[0].metadata["source"], source)
        with self.assertRaises(ValueError):
            loaders.load("文档.pdf", b"content")
        with self.assertRaises(UnicodeDecodeError):
            loaders.load("文档.txt", b"\xff")

    def test_extension_registration(self):
        registry = LoaderRegistry()
        registry.register("log", UTF8DocumentLoader)
        self.assertEqual(registry.extensions, ["log"])
        self.assertEqual(
            registry.load("test.log", b"content")[0].page_content, "content"
        )

    def test_txt_hash_is_content_and_metadata_retained(self):
        documents = loaders.load("test.txt", b"# literal text")
        chunks = split_documents(documents)
        self.assertEqual(chunks[0].page_content, "# literal text")
        self.assertEqual(chunks[0].metadata["source"], "test.txt")

    def test_document_title_in_metadata_and_section_in_content(self):
        documents = loaders.load(
            "city.md",
            "# 城市\n## 云朵\n### 招领\n正文\n# 另一座城\n新正文".encode(),
        )
        chunks = split_documents(documents)
        self.assertEqual(chunks[0].page_content, "## 云朵\n### 招领\n\n正文")
        self.assertEqual(chunks[0].metadata["document_title"], "城市")
        self.assertEqual(
            chunks[0].metadata["heading_path"], "# 城市\n## 云朵\n### 招领"
        )
        self.assertEqual(chunks[0].metadata["section_path"], "## 云朵\n### 招领")
        self.assertEqual(chunks[1].page_content, "新正文")
        self.assertEqual(chunks[1].metadata["document_title"], "另一座城")
        self.assertNotIn("section_path", chunks[1].metadata)

    def test_long_title_does_not_reduce_body_budget(self):
        documents = loaders.load(
            "long.md", ("# " + "标题" * 40 + "\n" + "a" * 30).encode()
        )
        chunks = split_documents(documents, size=16, overlap=2)
        self.assertEqual([len(chunk.page_content) for chunk in chunks], [16, 16])
        self.assertTrue(
            all(chunk.metadata["document_title"] == "标题" * 40 for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
