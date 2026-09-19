from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
import warnings

from .models import ResearchChunk
from .settings import ResearchSettings


def _bm25_document(text: str):
    from qdrant_client import models

    return models.Document(
        text=text,
        model="qdrant/bm25",
        options=models.Bm25Config(
            tokenizer=models.TokenizerType.MULTILINGUAL,
            stemmer=models.DisabledStemmerParams(type=models.NoStemmer.NONE),
            stopwords=models.StopwordsSet(languages=[], custom=[]),
            lowercase=True,
        ),
    )


class ResearchVectorStore:
    def __init__(self, settings: ResearchSettings | None = None, *, client: Any | None = None):
        self.settings = settings or ResearchSettings.from_env()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError(
                    '缺少 Agentic RAG 依赖，请执行 pip install -e ".[rag]"。'
                ) from exc
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Api key is used with an insecure connection.*")
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key or None,
                    timeout=30,
                    cloud_inference=True,
                    check_compatibility=False,
                )
        return self._client

    def health(self) -> bool:
        self.client.get_collections()
        return True

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.settings.collection):
            self.client.create_collection(
                collection_name=self.settings.collection,
                vectors_config={
                    "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        for field, schema in (
            ("source", models.PayloadSchemaType.KEYWORD),
            ("document_key", models.PayloadSchemaType.KEYWORD),
            ("source_id", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.collection,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except Exception as exc:
                # Qdrant reports an error when an equivalent index already exists.
                if "already exists" not in str(exc).lower():
                    raise

    def reset_collection(self) -> None:
        if self.client.collection_exists(self.settings.collection):
            self.client.delete_collection(self.settings.collection)
        self.ensure_collection()

    def delete_document(self, document_key: str) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.settings.collection):
            return
        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key", match=models.MatchValue(value=document_key)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_old_versions(self, document_key: str, content_hash: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key", match=models.MatchValue(value=document_key)
                        )
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="content_hash", match=models.MatchValue(value=content_hash)
                        )
                    ],
                )
            ),
            wait=True,
        )

    def delete_version(self, document_key: str, content_hash: str) -> None:
        """Remove a partially written version while retaining the last good one."""
        from qdrant_client import models

        if not self.client.collection_exists(self.settings.collection):
            return
        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key", match=models.MatchValue(value=document_key)
                        ),
                        models.FieldCondition(
                            key="content_hash", match=models.MatchValue(value=content_hash)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def upsert_chunks(
        self, chunks: list[ResearchChunk], dense_vectors: list[list[float]], *, content_hash: str
    ) -> None:
        from qdrant_client import models
        import uuid

        if len(chunks) != len(dense_vectors):
            raise ValueError("chunk 数量与 embedding 数量不一致")
        self.ensure_collection()
        indexed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        points = []
        for chunk, dense in zip(chunks, dense_vectors):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk.source}:{chunk.source_id}:{content_hash}:{chunk.chunk_index}",
                )
            )
            payload = {
                "document_key": chunk.document_key,
                "source": chunk.source,
                "source_id": chunk.source_id,
                "title": chunk.title,
                "authors": list(chunk.authors),
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": content_hash,
                "uri": chunk.uri,
                "indexed_at": indexed_at,
            }
            if chunk.page is not None:
                payload["page"] = chunk.page
            if chunk.year is not None:
                payload["year"] = chunk.year
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": dense, "bm25": _bm25_document(chunk.content)},
                    payload=payload,
                )
            )
        for start in range(0, len(points), 32):
            self.client.upsert(
                collection_name=self.settings.collection,
                points=points[start : start + 32],
                wait=True,
            )

    def search(
        self,
        *,
        query: str,
        dense_vector: list[float],
        sources: list[str],
        document_ids: list[str] | None,
        year_from: int | None,
        year_to: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        from qdrant_client import models

        query_filter = self._filter(
            sources=sources,
            document_ids=document_ids,
            year_from=year_from,
            year_to=year_to,
        )
        dense_limit = max(limit, self.settings.dense_candidates)
        sparse_limit = max(limit, self.settings.sparse_candidates)
        dense_response = self.client.query_points(
            collection_name=self.settings.collection,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=dense_limit,
            with_payload=False,
        )
        dense_scores = {str(point.id): float(point.score) for point in dense_response.points}
        response = self.client.query_points(
            collection_name=self.settings.collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=dense_limit,
                ),
                models.Prefetch(
                    query=_bm25_document(query),
                    using="bm25",
                    filter=query_filter,
                    limit=sparse_limit,
                ),
            ],
            query=models.RrfQuery(rrf=models.Rrf()),
            limit=max(limit * 4, limit),
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload.update(
                {
                    "point_id": str(point.id),
                    "dense_score": dense_scores.get(str(point.id)),
                    "rrf_score": float(point.score),
                }
            )
            results.append(payload)
        return results

    @staticmethod
    def _filter(
        *,
        sources: list[str],
        document_ids: list[str] | None,
        year_from: int | None,
        year_to: int | None,
    ):
        from qdrant_client import models

        must = []
        if sources:
            must.append(
                models.FieldCondition(key="source", match=models.MatchAny(any=list(sources)))
            )
        if document_ids:
            values = list(dict.fromkeys(str(item).strip() for item in document_ids if str(item).strip()))
            if values:
                must.append(
                    models.Filter(
                        should=[
                            models.FieldCondition(
                                key="source_id", match=models.MatchAny(any=values)
                            ),
                            models.FieldCondition(
                                key="document_key", match=models.MatchAny(any=values)
                            ),
                        ],
                    )
                )
        if year_from is not None or year_to is not None:
            must.append(
                models.FieldCondition(
                    key="year", range=models.Range(gte=year_from, lte=year_to)
                )
            )
        return models.Filter(must=must) if must else None
