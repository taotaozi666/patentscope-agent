"""混合召回、精确词、公开号范围、双索引恢复及失败边界。"""

from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from app.config import get_settings
from app.rag import RAGService
from app.rag.embeddings import DemoEmbedder
from app.rag.retrieval import fuse_rankings
from app.schemas import Settings
from app.storage import ChromaKnowledgeStore
from app.storage.lexical import LexicalIndex, tokens


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.embedder = DemoEmbedder()
        self.store = ChromaKnowledgeStore(self.embedder)
        self.settings = Settings(api_key="", chat_model="", embedding_model="")

    def add(self, text, source="sample.txt", **meta):
        self.store.add_documents([Document(page_content=text, metadata={"source": source, **meta})])
        return next(r for r in self.store.query(include_vectors=False).items if r.source == source)

    def test_keyword_rescues_exact_model_missing_from_vector_candidates(self):
        target = self.add("PVDF 膜材料", "target.txt")
        wrong = self.add("其他材料", "other.txt")
        with patch.object(self.store, "nearest", return_value=[(wrong, .9)]):
            hybrid = RAGService(self.embedder, self.store, self.settings).search("PVDF", 2)
            vector = RAGService(self.embedder, self.store, replace(self.settings, retrieval_mode="vector")).search("PVDF", 2)
        rescued = next(h for h in hybrid if h.source == target.source)
        self.assertEqual(set(rescued.metadata["retrieval_channels"]), {"keyword"})
        self.assertEqual([h.source for h in vector], [wrong.source])

    def test_semantic_lane_survives_without_literal_overlap(self):
        target = self.add("材料在高温下保持性能")
        with patch.object(self.store, "nearest", return_value=[(target, .9)]):
            hits = RAGService(self.embedder, self.store, self.settings).search("heat resistance", 1)
        self.assertEqual(hits[0].text, target.text)
        self.assertEqual(set(hits[0].metadata["retrieval_channels"]), {"vector"})

    def test_rrf_fuses_by_id_and_does_not_add_incompatible_raw_scores(self):
        first = self.add("first", "first.txt")
        second = self.add("second", "second.txt")
        ranked = fuse_rankings([(first, .99), (second, .1)], [(second, 999), (first, 1)])
        self.assertEqual({r.id for r, _ in ranked}, {first.id, second.id})
        self.assertAlmostEqual(ranked[0][1], 1 / 61 + 1 / 62)
        self.assertNotIn("retrieval_method", first.metadata)
        self.assertEqual(ranked[0][0].metadata["retrieval_method"], "hybrid_rrf")
        weighted = fuse_rankings([(first, .99), (second, .1)], [(second, 999), (first, 1)], vector_weight=2)
        self.assertEqual(weighted[0][0].id, first.id)

    def test_cjk_identifiers_decimal_and_raw_query_syntax(self):
        self.add("聚偏氟乙烯 PVDF 拉伸强度 35.2 MPa GB/T 1040.2", "target.txt")
        self.add("PVDF2 135.2", "other.txt")
        for question in ("PVDF", "35.2", "拉伸强度", "GB/T"):
            with self.subTest(question=question):
                self.assertEqual(self.store.keyword_search(question, 1)[0][0].source, "target.txt")
        self.assertEqual(self.store.keyword_search('" OR * : ()', 3), [])
        self.assertEqual(self.store.keyword_search("", 3), [])
        self.assertTrue(tokens("PVDF-12"))

    def test_publication_scope_on_both_lanes_and_unknown_scope(self):
        self.add("PVDF 材料", "target.txt", publication_number="CN123456789A")
        self.add("PVDF PVDF PVDF", "other.txt", publication_number="CN987654321A")
        self.assertEqual([r.source for r, _ in self.store.keyword_search("PVDF", 3, ["cn 123456789 a"])], ["target.txt"])
        self.assertEqual(self.store.keyword_search("PVDF", 3, []), [])
        rag = RAGService(self.embedder, self.store, self.settings)
        self.assertEqual({h.source for h in rag.search("CN123456789A PVDF", 3)}, {"target.txt"})
        self.assertEqual(rag.search("CN111111111A PVDF", 3), [])

    def test_table_header_support_searches_without_generated_context(self):
        import json
        self.add("35.2", doc_type_kwd="table", context_text="虚构秘密词",
                 table_cell_relations=json.dumps({"value": {"support": [
                     {"cell_id": "header", "text": "拉伸强度"}, {"cell_id": "row", "text": "实施例3"}]}}))
        self.assertTrue(self.store.keyword_search("拉伸强度", 1))
        self.assertTrue(self.store.keyword_search("实施例3", 1))
        self.assertEqual(self.store.keyword_search("虚构秘密词", 1), [])

    def test_updates_old_versions_and_clear_have_no_keyword_ghosts(self):
        from app.indexing.service import DocumentImportService
        importer = DocumentImportService(self.embedder, self.store, self.settings)
        importer.add_document("sample.txt", "PVDF 材料")
        self.assertTrue(self.store.keyword_search("PVDF", 3))
        changed = replace(self.settings, chunk_size=700)
        DocumentImportService(self.embedder, self.store, changed).add_document("sample.txt", "PVDF 材料")
        self.assertEqual(len(self.store.keyword_search("PVDF", 3)), 1)
        self.store.clear()
        self.assertEqual(self.store.keyword_search("PVDF", 3), [])
        self.assertEqual(self.store.nearest(self.embedder.embed_query("PVDF"), 3), [])

    def test_restart_legacy_backfill_shared_invalidation_and_cache_rebuild(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first = ChromaKnowledgeStore(self.embedder, directory, "hybrid_test")
            first.add_documents([Document(page_content="PVDF", metadata={"source": "a"})])
            second = ChromaKnowledgeStore(self.embedder, directory, "hybrid_test")
            self.assertTrue(second.keyword_search("PVDF", 1))
            first.add_documents([Document(page_content="PTFE", metadata={"source": "b"})])
            self.assertTrue(second.keyword_search("PTFE", 1))
            with second.lexical.connection() as connection, connection:
                connection.execute("DELETE FROM import_batches")
                connection.execute("DELETE FROM lexical_chunks")
            rebuilt = ChromaKnowledgeStore(self.embedder, directory, "hybrid_test")
            self.assertTrue(rebuilt.keyword_search("PVDF", 1))
            first.clear()
            self.assertEqual(second.keyword_search("PTFE", 1), [])

    def test_pending_import_cannot_enter_either_lane_or_hide_ready_results(self):
        ready = self.add("PVDF 材料", "ready.txt")
        pending = self.add("PVDF pending", "pending.txt")
        self.store.collection.update(ids=[pending.id], metadatas=[{"index_batch_id": "uncommitted"}])
        self.store.lexical.invalidate()
        self.assertEqual([r.id for r, _ in self.store.keyword_search("PVDF", 3)], [ready.id])
        self.assertEqual([r.id for r, _ in self.store.nearest(self.embedder.embed_query("PVDF pending"), 1)], [ready.id])

    def test_failed_batch_rolls_back_and_retry_rebuilds_lexical(self):
        self.add("PVDF old", "old.txt")
        self.assertTrue(self.store.keyword_search("PVDF", 1))
        docs = [Document(page_content=f"PTFE {i}", metadata={"source": "new.txt", "chunk": i + 1,
                "content_hash": "hash", "pipeline_fingerprint": "v1"}) for i in range(3)]
        original = self.store.collection.upsert
        calls = []
        def failing(**kwargs):
            calls.append(1)
            original(**kwargs)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
        with patch.object(self.store.client, "get_max_batch_size", return_value=2), \
             patch.object(self.store.collection, "upsert", side_effect=failing), self.assertRaises(RuntimeError):
            self.store.add_documents(docs, self.embedder.embed_documents([d.page_content for d in docs]))
        self.assertEqual(self.store.keyword_search("PTFE", 3), [])
        self.assertTrue(self.store.keyword_search("PVDF", 1))
        self.store.add_documents(docs, self.embedder.embed_documents([d.page_content for d in docs]))
        self.assertEqual(len(self.store.keyword_search("PTFE", 3)), 3)

    def test_lexical_error_falls_back_to_vector_and_preserves_raw_score(self):
        target = self.add("PVDF")
        with patch.object(self.store, "keyword_search", side_effect=sqlite3.OperationalError("unavailable")):
            hits = RAGService(self.embedder, self.store, self.settings).search("PVDF", 1)
        self.assertEqual(hits[0].text, target.text)
        self.assertNotIn("retrieval_method", hits[0].metadata)

    def test_blocked_tables_do_not_consume_final_top_k_slots(self):
        blocked = self.add("PVDF blocked", "blocked.txt", doc_type_kwd="table")
        ready = self.add("PVDF usable", "ready.txt")
        def evidence(hits, repository):
            for hit in hits:
                if hit.source == blocked.source:
                    hit.metadata["table_evidence"] = {"quality": {"usage": "blocked"}}
            return hits
        with patch.object(self.store, "nearest", return_value=[(blocked, .99), (ready, .8)]), \
             patch("app.rag.rag.expand_table_hits", side_effect=evidence):
            hits = RAGService(self.embedder, self.store, self.settings).search("PVDF", 1)
        self.assertEqual([hit.source for hit in hits], [ready.source])

    def test_rebuild_retries_if_generation_changes_and_retains_atomic_snapshot(self):
        index = LexicalIndex()
        record = self.add("PVDF")
        calls = []
        def reader():
            calls.append(1)
            if len(calls) == 1:
                index.invalidate()
            return [record]
        index.synchronize(reader)
        self.assertEqual(len(calls), 2)
        self.assertEqual(index.search("PVDF", 1)[0][0], record.id)

    def test_hybrid_configuration_validation(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            path = Path(directory, ".env")
            self.assertEqual(get_settings(path, {}).retrieval_mode, "hybrid")
            self.assertEqual(get_settings(path, {"RETRIEVAL_MODE": "vector"}).retrieval_mode, "vector")
            for key, value in (("RETRIEVAL_MODE", "bad"), ("RETRIEVAL_CANDIDATES", "0"),
                               ("RETRIEVAL_RRF_CONSTANT", "-1"), ("RETRIEVAL_KEYWORD_WEIGHT", "nan"),
                               ("RETRIEVAL_VECTOR_WEIGHT", "0")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    get_settings(path, {key: value})
