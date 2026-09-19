from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from .models import ResearchChunk, ResearchDocument, ResearchSection
from .settings import ResearchSettings


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_WHITESPACE = re.compile(r"[ \t\u00a0]+")


def normalize_text(text: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip() for line in str(text or "").splitlines()]
    compact: list[str] = []
    blank = False
    for line in lines:
        if line:
            compact.append(line)
            blank = False
        elif compact and not blank:
            compact.append("")
            blank = True
    return "\n".join(compact).strip()


class _Tokenizer:
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._value: Any = None
        self._lock = threading.Lock()

    def _get(self):
        if self._value is not None:
            return self._value
        with self._lock:
            if self._value is None:
                try:
                    from tokenizers import Tokenizer
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise RuntimeError(
                        '缺少 Agentic RAG 依赖，请执行 pip install -e ".[rag]"。'
                    ) from exc
                repository = self.model_name
                try:
                    from fastembed import TextEmbedding

                    metadata = next(
                        (
                            item
                            for item in TextEmbedding.list_supported_models()
                            if item.get("model") == self.model_name
                        ),
                        None,
                    )
                    repository = str((metadata or {}).get("sources", {}).get("hf") or repository)
                except ImportError:
                    pass
                tokenizer_path = hf_hub_download(
                    repo_id=repository,
                    filename="tokenizer.json",
                    cache_dir=str(Path(self.cache_dir)),
                )
                self._value = Tokenizer.from_file(tokenizer_path)
        return self._value

    def encode(self, text: str) -> list[int]:
        return list(self._get().encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self._get().decode(token_ids, skip_special_tokens=True).strip()


class ResearchChunker:
    def __init__(
        self,
        settings: ResearchSettings | None = None,
        *,
        max_tokens: int = 384,
        overlap_tokens: int = 64,
        tokenizer: Any | None = None,
    ):
        self.settings = settings or ResearchSettings.from_env()
        self.max_tokens = max(64, int(max_tokens))
        self.overlap_tokens = max(0, min(int(overlap_tokens), self.max_tokens // 2))
        self.tokenizer = tokenizer or _Tokenizer(
            self.settings.embedding_model, str(self.settings.model_cache_path)
        )

    def chunk(self, document: ResearchDocument) -> list[ResearchChunk]:
        chunks: list[ResearchChunk] = []
        for section in document.sections:
            chunks.extend(self._chunk_section(document, section, start_index=len(chunks)))
        return chunks

    def _chunk_section(
        self, document: ResearchDocument, section: ResearchSection, *, start_index: int
    ) -> list[ResearchChunk]:
        text = normalize_text(section.text)
        if not text:
            return []
        prefix = document.title.strip()
        if section.heading.strip() and section.heading.strip() != prefix:
            prefix = f"{prefix}\n{section.heading.strip()}" if prefix else section.heading.strip()
        prefix_ids = self.tokenizer.encode(prefix) if prefix else []
        budget = max(32, self.max_tokens - len(prefix_ids) - 4)
        paragraphs = [part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()]
        if not paragraphs:
            paragraphs = [text]

        bodies: list[str] = []
        current_ids: list[int] = []
        for paragraph in paragraphs:
            paragraph_ids = self.tokenizer.encode(paragraph)
            if current_ids and len(current_ids) + len(paragraph_ids) > budget:
                bodies.extend(self._split_ids(current_ids, budget))
                current_ids = current_ids[-self.overlap_tokens :] if self.overlap_tokens else []
            if len(paragraph_ids) > budget:
                if current_ids:
                    bodies.extend(self._split_ids(current_ids, budget))
                    current_ids = []
                bodies.extend(self._split_ids(paragraph_ids, budget))
            else:
                current_ids.extend(paragraph_ids)
        if current_ids:
            bodies.extend(self._split_ids(current_ids, budget))

        result: list[ResearchChunk] = []
        for offset, body in enumerate(body for body in bodies if body.strip()):
            content = f"{prefix}\n\n{body}".strip() if prefix else body.strip()
            result.append(
                ResearchChunk(
                    document_key=document.document_key,
                    source=document.source,
                    source_id=document.source_id,
                    title=document.title,
                    content=content,
                    chunk_index=start_index + offset,
                    section=section.heading,
                    page=section.page,
                    authors=document.authors,
                    year=document.year,
                    uri=document.uri,
                )
            )
        return result

    def _split_ids(self, token_ids: list[int], budget: int) -> list[str]:
        if len(token_ids) <= budget:
            return [self.tokenizer.decode(token_ids)]
        result: list[str] = []
        step = max(1, budget - self.overlap_tokens)
        for start in range(0, len(token_ids), step):
            piece = token_ids[start : start + budget]
            if not piece:
                break
            result.append(self.tokenizer.decode(piece))
            if start + budget >= len(token_ids):
                break
        return result
