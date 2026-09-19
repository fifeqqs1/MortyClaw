from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Iterable
import warnings

from .settings import ResearchSettings


class LazyDenseEmbedding:
    """Loads the local ONNX model only when indexing or retrieval is requested."""

    def __init__(self, settings: ResearchSettings | None = None):
        self.settings = settings or ResearchSettings.from_env()
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise RuntimeError(
                        '缺少 Agentic RAG 依赖，请执行 pip install -e ".[rag]"。'
                    ) from exc
                cache_dir = Path(self.settings.model_cache_path)
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(cache_dir))
                os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="The model .* now uses mean pooling instead of CLS embedding.*"
                    )
                    self._model = TextEmbedding(
                        model_name=self.settings.embedding_model,
                        cache_dir=str(cache_dir),
                        lazy_load=True,
                    )
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        values = list(model.query_embed([f"query: {text.strip()}"], batch_size=1))
        if not values:
            raise RuntimeError("Embedding 模型没有返回查询向量")
        return values[0].astype("float32").tolist()

    def embed_documents(self, texts: Iterable[str], *, batch_size: int = 8) -> list[list[float]]:
        prepared = [f"passage: {str(text).strip()}" for text in texts]
        if not prepared:
            return []
        model = self._get_model()
        return [
            item.astype("float32").tolist()
            for item in model.passage_embed(prepared, batch_size=max(1, batch_size))
        ]
