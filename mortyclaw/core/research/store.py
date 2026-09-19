from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..storage.store import RuntimeStore, get_runtime_store


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResearchDocumentRepository:
    """Stores only source-level sync state; chunks and vectors live in Qdrant."""

    def __init__(self, store: RuntimeStore | None = None):
        self.store = store or get_runtime_store()

    def get(self, document_key: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_documents WHERE document_key = ?", (document_key,)
            ).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        *,
        document_key: str,
        source: str,
        source_id: str,
        title: str,
        uri: str,
        content_hash: str,
        status: str,
        chunk_count: int = 0,
        error_type: str = "",
    ) -> None:
        now = _now()
        indexed_at = now if status == "indexed" else None
        with self.store._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_documents (
                    document_key, source, source_id, title, uri, content_hash,
                    status, chunk_count, indexed_at, error_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_key) DO UPDATE SET
                    source=excluded.source, source_id=excluded.source_id,
                    title=excluded.title, uri=excluded.uri,
                    content_hash=excluded.content_hash, status=excluded.status,
                    chunk_count=excluded.chunk_count, indexed_at=excluded.indexed_at,
                    error_type=excluded.error_type, updated_at=excluded.updated_at
                """,
                (
                    document_key, source, source_id, title, uri, content_hash,
                    status, max(0, int(chunk_count)), indexed_at, error_type, now,
                ),
            )
            conn.commit()

    def delete(self, document_key: str) -> bool:
        with self.store._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM research_documents WHERE document_key = ?", (document_key,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        with self.store._connect() as conn:
            cursor = conn.execute("DELETE FROM research_documents")
            conn.commit()
            return max(0, int(cursor.rowcount))

    def list(self, *, source: str = "", limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM research_documents"
        args: list[Any] = []
        if source:
            sql += " WHERE source = ?"
            args.append(source)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(int(limit), 1000)))
        with self.store._connect() as conn:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]

    def counts(self) -> dict[str, int]:
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) AS documents, COALESCE(SUM(chunk_count), 0) AS chunks "
                "FROM research_documents GROUP BY source"
            ).fetchall()
        result = {"documents": 0, "chunks": 0}
        for row in rows:
            result[str(row["source"])] = int(row["documents"])
            result["documents"] += int(row["documents"])
            result["chunks"] += int(row["chunks"])
        return result
