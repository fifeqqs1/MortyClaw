from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResearchSource = Literal["zotero", "feishu", "arxiv"]


@dataclass(frozen=True)
class ResearchSection:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass(frozen=True)
class ResearchDocument:
    source: ResearchSource
    source_id: str
    title: str
    sections: tuple[ResearchSection, ...]
    uri: str = ""
    authors: tuple[str, ...] = field(default_factory=tuple)
    year: int | None = None

    @property
    def document_key(self) -> str:
        return f"{self.source}:{self.source_id}"


@dataclass(frozen=True)
class ResearchChunk:
    document_key: str
    source: ResearchSource
    source_id: str
    title: str
    content: str
    chunk_index: int
    section: str = ""
    page: int | None = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    year: int | None = None
    uri: str = ""
