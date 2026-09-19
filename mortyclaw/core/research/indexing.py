from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .cache import get_retrieval_cache
from .chunking import ResearchChunker, normalize_text
from .embeddings import LazyDenseEmbedding
from .models import ResearchDocument
from .qdrant import ResearchVectorStore
from .settings import ResearchSettings
from .sources import ResearchSourceLoader
from .store import ResearchDocumentRepository


def document_hash(document: ResearchDocument) -> str:
    payload = {
        "source": document.source,
        "source_id": document.source_id,
        "title": document.title,
        "authors": list(document.authors),
        "year": document.year,
        "uri": document.uri,
        "sections": [
            {"heading": item.heading, "page": item.page, "text": normalize_text(item.text)}
            for item in document.sections
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResearchIndexer:
    def __init__(
        self,
        settings: ResearchSettings | None = None,
        *,
        repository: ResearchDocumentRepository | None = None,
        vector_store: ResearchVectorStore | None = None,
        embeddings: LazyDenseEmbedding | None = None,
        chunker: ResearchChunker | None = None,
        sources: ResearchSourceLoader | None = None,
    ):
        self.settings = settings or ResearchSettings.from_env()
        self.repository = repository or ResearchDocumentRepository()
        self.vector_store = vector_store or ResearchVectorStore(self.settings)
        self.embeddings = embeddings or LazyDenseEmbedding(self.settings)
        self.chunker = chunker or ResearchChunker(self.settings)
        self.sources = sources or ResearchSourceLoader()
        self.cache = get_retrieval_cache(self.settings.cache_ttl_seconds)

    def index_document(self, document: ResearchDocument, *, force: bool = False) -> dict[str, Any]:
        content_hash = document_hash(document)
        existing = self.repository.get(document.document_key)
        if (
            not force
            and existing
            and existing.get("status") == "indexed"
            and existing.get("content_hash") == content_hash
        ):
            return {
                "status": "unchanged",
                "document_key": document.document_key,
                "chunk_count": int(existing.get("chunk_count") or 0),
            }
        try:
            chunks = self.chunker.chunk(document)
            if not chunks:
                raise ValueError("文档没有可索引的文本")
            dense_vectors = self.embeddings.embed_documents(chunk.content for chunk in chunks)
            self.vector_store.upsert_chunks(chunks, dense_vectors, content_hash=content_hash)
            self.vector_store.delete_old_versions(document.document_key, content_hash)
            self.repository.upsert(
                document_key=document.document_key,
                source=document.source,
                source_id=document.source_id,
                title=document.title,
                uri=document.uri,
                content_hash=content_hash,
                status="indexed",
                chunk_count=len(chunks),
            )
            self.cache.invalidate()
            return {
                "status": "indexed",
                "document_key": document.document_key,
                "chunk_count": len(chunks),
            }
        except Exception as exc:
            try:
                self.vector_store.delete_version(document.document_key, content_hash)
            except Exception:
                # Cleanup is best effort. Preserve the original indexing error
                # and never delete the previous successfully indexed version.
                pass
            self.repository.upsert(
                document_key=document.document_key,
                source=document.source,
                source_id=document.source_id,
                title=document.title,
                uri=document.uri,
                content_hash=content_hash,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise

    def index_locator(self, source: str, locator: str, *, force: bool = False) -> dict[str, Any]:
        normalized = source.strip().lower()
        if normalized == "arxiv":
            document = self.sources.load_arxiv(locator)
        elif normalized == "feishu":
            document = self.sources.load_feishu(locator)
        else:
            raise ValueError("单文档索引只支持 feishu 或 arxiv")
        return self.index_document(document, force=force)

    def sync_zotero(
        self, *, limit: int = 1000, query: str = "", force: bool = False
    ) -> dict[str, Any]:
        documents = self.sources.load_zotero_items(limit=limit, query=query)
        result = {
            "source": "zotero",
            "query": query,
            "total": len(documents),
            "indexed": 0,
            "unchanged": 0,
            "failed": 0,
        }
        errors: list[dict[str, str]] = []
        for document in documents:
            try:
                item = self.index_document(document, force=force)
                result[item["status"]] += 1
            except Exception as exc:
                result["failed"] += 1
                errors.append({"document_key": document.document_key, "error_type": type(exc).__name__})
        result["errors"] = errors[:20]
        return result

    def remove(self, document_key: str) -> dict[str, Any]:
        self.vector_store.delete_document(document_key)
        removed = self.repository.delete(document_key)
        self.cache.invalidate()
        return {"status": "removed" if removed else "not_found", "document_key": document_key}

    def rebuild(self) -> dict[str, Any]:
        # Rebuild resets the vector collection. Source content must be re-synced explicitly,
        # so stale SQLite status is removed at the same time.
        self.vector_store.reset_collection()
        removed = self.repository.clear()
        self.cache.invalidate()
        return {"status": "rebuilt", "removed_documents": removed}
