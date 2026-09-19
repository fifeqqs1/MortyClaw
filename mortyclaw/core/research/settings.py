from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ..config import PROJECT_ROOT


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    return path.resolve()


@dataclass(frozen=True)
class ResearchSettings:
    enabled: bool = False
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str = ""
    qdrant_command: str = ""
    qdrant_storage_path: Path = Path("workspace/qdrant")
    qdrant_binary_dir: Path = Path("workspace/bin/qdrant")
    model_cache_path: Path = Path("workspace/models")
    collection: str = "mortyclaw_research_v1"
    embedding_model: str = "intfloat/multilingual-e5-large"
    dense_candidates: int = 30
    sparse_candidates: int = 30
    default_top_k: int = 6
    max_top_k: int = 10
    cache_ttl_seconds: int = 7200

    @classmethod
    def from_env(cls) -> "ResearchSettings":
        return cls(
            enabled=_enabled(os.getenv("RESEARCH_RAG_ENABLED", "0")),
            qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").strip().rstrip("/"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
            qdrant_command=os.getenv("QDRANT_COMMAND", "").strip(),
            qdrant_storage_path=_project_path(os.getenv("QDRANT_STORAGE_PATH", "workspace/qdrant")),
            qdrant_binary_dir=_project_path("workspace/bin/qdrant"),
            model_cache_path=_project_path("workspace/models"),
            collection=os.getenv("RESEARCH_COLLECTION", "mortyclaw_research_v1").strip()
            or "mortyclaw_research_v1",
            embedding_model=os.getenv(
                "RESEARCH_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
            ).strip()
            or "intfloat/multilingual-e5-large",
            dense_candidates=_positive_int("RESEARCH_DENSE_CANDIDATES", 30),
            sparse_candidates=_positive_int("RESEARCH_SPARSE_CANDIDATES", 30),
            default_top_k=_positive_int("RESEARCH_DEFAULT_TOP_K", 6),
            max_top_k=_positive_int("RESEARCH_MAX_TOP_K", 10),
            cache_ttl_seconds=_positive_int("RESEARCH_CACHE_TTL_SECONDS", 7200, minimum=60),
        )

    def public_status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "qdrant_url": self.qdrant_url,
            "collection": self.collection,
            "embedding_model": self.embedding_model,
            "storage_path": str(self.qdrant_storage_path),
            "api_key_configured": bool(self.qdrant_api_key),
        }
