import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mortyclaw.core.harness.gateway import MortyClawGateway, _schema_for
from mortyclaw.core.research.cache import RetrievalCache
from mortyclaw.core.research.chunking import ResearchChunker
from mortyclaw.core.research.indexing import ResearchIndexer
from mortyclaw.core.research.models import ResearchDocument, ResearchSection
from mortyclaw.core.research.retrieval import ResearchRetriever, format_retrieval_result
from mortyclaw.core.research.settings import ResearchSettings
from mortyclaw.core.research.sources import ResearchSourceLoader
from mortyclaw.core.research.store import ResearchDocumentRepository
from mortyclaw.core.research.tools import (
    research_index_document,
    research_remove_document,
    research_retrieve,
    research_sync_zotero,
)
from mortyclaw.core.storage.store import RuntimeStore
from mortyclaw.core.tools.meta import get_tool_meta


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(value) for value in ids)


class FakeEmbeddings:
    def __init__(self):
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [0.1, 0.2]

    def embed_documents(self, texts):
        values = list(texts)
        self.document_calls += 1
        return [[0.1, 0.2] for _ in values]


class FakeVectorStore:
    def __init__(self, search_results=None):
        self.search_results = list(search_results or [])
        self.search_calls = 0
        self.upserted = []
        self.deleted_versions = []
        self.deleted = []

    def search(self, **kwargs):
        self.search_calls += 1
        return list(self.search_results)

    def upsert_chunks(self, chunks, vectors, *, content_hash):
        self.upserted.append((list(chunks), list(vectors), content_hash))

    def delete_old_versions(self, document_key, content_hash):
        self.deleted_versions.append((document_key, content_hash))

    def delete_version(self, document_key, content_hash):
        self.deleted_versions.append((document_key, content_hash))

    def delete_document(self, document_key):
        self.deleted.append(document_key)


class ResearchStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = RuntimeStore(str(Path(self.temp.name) / "runtime.sqlite3"))
        self.repository = ResearchDocumentRepository(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_one_research_table_is_added(self):
        with self.store._connect() as conn:
            names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'research_%'"
                ).fetchall()
            ]
        self.assertEqual(names, ["research_documents"])

    def test_repository_stores_only_document_sync_state(self):
        self.repository.upsert(
            document_key="arxiv:1234.5678",
            source="arxiv",
            source_id="1234.5678",
            title="Paper",
            uri="https://arxiv.org/abs/1234.5678",
            content_hash="abc",
            status="indexed",
            chunk_count=3,
        )
        row = self.repository.get("arxiv:1234.5678")
        self.assertEqual(row["chunk_count"], 3)
        self.assertNotIn("content", row)
        self.assertNotIn("embedding", row)


class ResearchChunkingTests(unittest.TestCase):
    def test_section_aware_chunking_keeps_title_page_and_limits_size(self):
        settings = ResearchSettings(model_cache_path=Path("."))
        chunker = ResearchChunker(
            settings,
            max_tokens=64,
            overlap_tokens=8,
            tokenizer=CharacterTokenizer(),
        )
        document = ResearchDocument(
            source="arxiv",
            source_id="1",
            title="Agent Memory",
            sections=(ResearchSection(text="第一段" * 50, heading="Methods", page=4),),
        )
        chunks = chunker.chunk(document)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.content.startswith("Agent Memory\nMethods") for chunk in chunks))
        self.assertTrue(all(chunk.page == 4 for chunk in chunks))


class ResearchIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        store = RuntimeStore(str(Path(self.temp.name) / "runtime.sqlite3"))
        self.repository = ResearchDocumentRepository(store)
        self.vector_store = FakeVectorStore()
        self.embeddings = FakeEmbeddings()
        settings = ResearchSettings(enabled=True, model_cache_path=Path(self.temp.name))
        self.indexer = ResearchIndexer(
            settings,
            repository=self.repository,
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            chunker=ResearchChunker(
                settings,
                max_tokens=64,
                overlap_tokens=8,
                tokenizer=CharacterTokenizer(),
            ),
        )
        self.document = ResearchDocument(
            source="feishu",
            source_id="doc1",
            title="研究笔记",
            sections=(ResearchSection(text="这是研究内容。" * 20),),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_document_does_not_reembed(self):
        first = self.indexer.index_document(self.document)
        second = self.indexer.index_document(self.document)
        self.assertEqual(first["status"], "indexed")
        self.assertEqual(second["status"], "unchanged")
        self.assertEqual(self.embeddings.document_calls, 1)
        self.assertEqual(len(self.vector_store.upserted), 1)


class ResearchRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.settings = ResearchSettings(enabled=True, cache_ttl_seconds=7200)
        self.embeddings = FakeEmbeddings()
        self.vector_store = FakeVectorStore([
            {
                "document_key": "zotero:A",
                "source_id": "A",
                "source": "zotero",
                "title": "Memory Paper",
                "chunk_index": 1,
                "content": "first evidence",
                "page": 2,
                "rrf_score": 0.5,
                "dense_score": 0.8,
            },
            {
                "document_key": "zotero:A",
                "source_id": "A",
                "source": "zotero",
                "title": "Memory Paper",
                "chunk_index": 2,
                "content": "second evidence",
                "page": 3,
                "rrf_score": 0.4,
                "dense_score": 0.7,
            },
        ])
        self.retriever = ResearchRetriever(
            self.settings,
            vector_store=self.vector_store,
            embeddings=self.embeddings,
        )
        self.retriever.cache = RetrievalCache(ttl_seconds=7200)

    def test_exact_repeat_hits_memory_cache(self):
        first = self.retriever.retrieve(query="agent memory", thread_id="thread-a")
        second = self.retriever.retrieve(query=" agent   memory ", thread_id="thread-a")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["retrieval_id"], second["retrieval_id"])
        self.assertEqual(self.embeddings.query_calls, 1)
        self.assertEqual(self.vector_store.search_calls, 1)

    def test_adjacent_chunks_are_merged_with_evidence_boundary(self):
        result = self.retriever.retrieve(query="memory", thread_id="thread-b")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("first evidence", result["evidence"][0]["content"])
        self.assertIn("second evidence", result["evidence"][0]["content"])
        formatted = format_retrieval_result(result)
        self.assertIn("[RETRIEVED_EVIDENCE", formatted)
        self.assertIn("[/RETRIEVED_EVIDENCE]", formatted)

    def test_disabled_retriever_does_not_load_embedding_or_query_store(self):
        retriever = ResearchRetriever(
            ResearchSettings(enabled=False),
            vector_store=self.vector_store,
            embeddings=self.embeddings,
        )
        result = retriever.retrieve(query="paper", thread_id="thread-c")
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(self.embeddings.query_calls, 0)
        self.assertEqual(self.vector_store.search_calls, 0)

    def test_top_k_is_part_of_exact_cache_key(self):
        self.retriever.retrieve(query="memory", top_k=1, thread_id="thread-d")
        self.retriever.retrieve(query="memory", top_k=2, thread_id="thread-d")
        self.assertEqual(self.embeddings.query_calls, 2)
        self.assertEqual(self.vector_store.search_calls, 2)


class ResearchSourceTests(unittest.TestCase):
    def test_zotero_markdown_listing_is_parsed(self):
        parsed = ResearchSourceLoader._parse_zotero_item_list(
            """# Recent\n\n## 1. Memory Paper\n**Item Key:** ABC123\n"
            "**Date:** 2025-02-03\n**Authors:** Alice; Bob\n"""
        )
        self.assertEqual(parsed[0]["key"], "ABC123")
        self.assertEqual(parsed[0]["title"], "Memory Paper")
        self.assertEqual(parsed[0]["authors"], ["Alice", "Bob"])


class ResearchToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_retrieval_is_low_risk_and_index_mutations_require_approval(self):
        self.assertEqual(get_tool_meta(research_retrieve).risk_level, "low")
        for tool in (research_index_document, research_remove_document, research_sync_zotero):
            meta = get_tool_meta(tool)
            self.assertEqual(meta.risk_level, "high")
            self.assertTrue(meta.requires_approval)

    def test_retrieval_schema_has_context_token(self):
        schema = _schema_for(research_retrieve)
        self.assertIn("context_token", schema["required"])

    async def test_gateway_hides_research_tools_when_disabled(self):
        gateway = MortyClawGateway()
        with patch.dict(os.environ, {"RESEARCH_RAG_ENABLED": "0"}, clear=False), patch.object(
            gateway, "_load_external_tools_quietly", return_value=[]
        ):
            await gateway.discover_tools()
        self.assertNotIn("research_retrieve", gateway._tools)

    async def test_gateway_exposes_research_tools_when_enabled(self):
        gateway = MortyClawGateway()
        with patch.dict(os.environ, {"RESEARCH_RAG_ENABLED": "1"}, clear=False), patch.object(
            gateway, "_load_external_tools_quietly", return_value=[]
        ):
            await gateway.discover_tools()
        self.assertIn("research_retrieve", gateway._tools)


if __name__ == "__main__":
    unittest.main()
