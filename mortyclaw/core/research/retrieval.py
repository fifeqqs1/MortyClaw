from __future__ import annotations

from collections import defaultdict
import json
import uuid
from typing import Any

from ..runtime.context import get_active_thread_id
from .cache import get_retrieval_cache, retrieval_fingerprint
from .embeddings import LazyDenseEmbedding
from .qdrant import ResearchVectorStore
from .settings import ResearchSettings


_ALLOWED_SOURCES = {"zotero", "feishu", "arxiv"}


class ResearchRetriever:
    def __init__(
        self,
        settings: ResearchSettings | None = None,
        *,
        vector_store: ResearchVectorStore | None = None,
        embeddings: LazyDenseEmbedding | None = None,
    ):
        self.settings = settings or ResearchSettings.from_env()
        self.vector_store = vector_store or ResearchVectorStore(self.settings)
        self.embeddings = embeddings or LazyDenseEmbedding(self.settings)
        self.cache = get_retrieval_cache(self.settings.cache_ttl_seconds)

    def retrieve(
        self,
        *,
        query: str,
        sources: list[str] | None = None,
        document_ids: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        top_k: int | None = None,
        mode: str = "search",
        prior_retrieval_id: str | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.settings.enabled:
            return {"status": "disabled", "message": "Agentic RAG 尚未启用。"}
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {"status": "error", "error": "empty_query"}
        normalized_mode = str(mode or "search").strip().lower()
        if normalized_mode not in {"search", "expand"}:
            return {"status": "error", "error": "invalid_mode"}
        if normalized_mode == "expand" and not str(prior_retrieval_id or "").strip():
            return {"status": "error", "error": "prior_retrieval_id_required"}
        selected_sources = list(dict.fromkeys(
            str(item).strip().lower() for item in (sources or sorted(_ALLOWED_SOURCES))
            if str(item).strip().lower() in _ALLOWED_SOURCES
        ))
        if not selected_sources:
            return {"status": "error", "error": "invalid_sources"}
        limit = max(1, min(int(top_k or self.settings.default_top_k), self.settings.max_top_k))
        filters = {
            "document_ids": sorted(document_ids or []),
            "year_from": year_from,
            "year_to": year_to,
            "top_k": limit,
        }
        resolved_thread = thread_id or get_active_thread_id()
        key = retrieval_fingerprint(
            thread_id=resolved_thread,
            query=normalized_query,
            sources=selected_sources,
            filters=filters,
            index_revision=self.cache.revision,
        )
        cached = self.cache.get(resolved_thread, key)
        if cached is not None:
            return cached
        try:
            dense_vector = self.embeddings.embed_query(normalized_query)
            raw = self.vector_store.search(
                query=normalized_query,
                dense_vector=dense_vector,
                sources=selected_sources,
                document_ids=document_ids,
                year_from=year_from,
                year_to=year_to,
                limit=limit,
            )
        except Exception as exc:
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "message": "科研知识索引暂时不可用；其他 MortyClaw 功能不受影响。",
            }
        retrieval_id = f"rr_{uuid.uuid4().hex[:16]}"
        evidence = self._select_evidence(raw, limit=limit)
        result = {
            "status": "ok",
            "retrieval_id": retrieval_id,
            "cache_hit": False,
            "query": normalized_query,
            "mode": normalized_mode,
            "prior_retrieval_id": prior_retrieval_id,
            "evidence": evidence,
        }
        self.cache.put(resolved_thread, key, result)
        return result

    @staticmethod
    def _select_evidence(raw: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        per_document: dict[str, int] = defaultdict(int)
        for item in raw:
            document_key = str(item.get("document_key") or item.get("source_id") or "")
            if per_document[document_key] >= 2:
                continue
            chunk_index = int(item.get("chunk_index") or 0)
            adjacent = next(
                (
                    existing
                    for existing in selected
                    if existing["document_key"] == document_key
                    and abs(int(existing["chunk_index"]) - chunk_index) == 1
                ),
                None,
            )
            if adjacent is not None:
                content = str(item.get("content") or "").strip()
                if content and content not in adjacent["content"]:
                    adjacent["content"] = f"{adjacent['content']}\n\n{content}"
                adjacent["chunk_index"] = min(int(adjacent["chunk_index"]), chunk_index)
                continue
            evidence_id = f"ev_{len(selected) + 1}"
            selected.append(
                {
                    "evidence_id": evidence_id,
                    "document_id": str(item.get("source_id") or ""),
                    "document_key": document_key,
                    "source": str(item.get("source") or ""),
                    "title": str(item.get("title") or ""),
                    "authors": list(item.get("authors") or []),
                    "year": item.get("year"),
                    "section": str(item.get("section") or ""),
                    "page": item.get("page"),
                    "uri": str(item.get("uri") or ""),
                    "chunk_index": chunk_index,
                    "content": str(item.get("content") or "")[:1200],
                    "dense_score": item.get("dense_score"),
                    "rrf_score": item.get("rrf_score"),
                }
            )
            per_document[document_key] += 1
            if len(selected) >= limit:
                break
        return selected



def format_retrieval_result(result: dict[str, Any]) -> str:
    """Keep one structured copy of evidence inside explicit trusted-data boundaries."""
    if result.get("status") != "ok":
        return json.dumps(result, ensure_ascii=False, default=str)
    retrieval_id = str(result.get("retrieval_id") or "unknown")
    payload = json.dumps(result, ensure_ascii=False, default=str, separators=(",", ":"))
    return (
        f"[RETRIEVED_EVIDENCE id={retrieval_id}]\n"
        f"{payload}\n"
        "[/RETRIEVED_EVIDENCE]"
    )
