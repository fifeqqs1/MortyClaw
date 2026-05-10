from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .snapshot import MemorySnapshot
from .store import MemoryRecord


class MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def initialize(self, session_id: str, **kwargs) -> None:
        return None

    def render_prompt_block(self, snapshot: MemorySnapshot | None) -> str:
        return ""

    def recall(self, query: str | None, context: dict[str, Any] | None = None) -> str:
        return ""

    def capture(self, turn: dict[str, Any], context: dict[str, Any] | None = None) -> list[MemoryRecord]:
        return []

    def on_pre_compact(self, messages: list[Any], state: dict[str, Any] | None = None) -> str:
        return ""

    def shutdown(self) -> None:
        return None


class BuiltinMemoryProvider(MemoryProvider):
    def __init__(
        self,
        *,
        get_memory_store_fn,
        build_memory_record_fn,
        memory_dir: str,
        default_long_term_scope: str,
        user_profile_memory_type: str,
        build_long_term_memory_prompt_fn,
    ):
        self._get_memory_store_fn = get_memory_store_fn
        self._build_memory_record_fn = build_memory_record_fn
        self._memory_dir = memory_dir
        self._default_long_term_scope = default_long_term_scope
        self._user_profile_memory_type = user_profile_memory_type
        self._build_long_term_memory_prompt_fn = build_long_term_memory_prompt_fn

    @property
    def name(self) -> str:
        return "builtin"

    def render_prompt_block(self, snapshot: MemorySnapshot | None) -> str:
        if snapshot and snapshot.profile_content.strip():
            return "【用户长期画像 (静态偏好)】\n" + snapshot.profile_content.strip()
        return ""

    def recall(self, query: str | None, context: dict[str, Any] | None = None) -> str:
        return self._build_long_term_memory_prompt_fn(query)

    def capture(self, turn: dict[str, Any], context: dict[str, Any] | None = None) -> list[MemoryRecord]:
        from ..memory_policy import extract_long_term_memory_records

        content = str(turn.get("user") or turn.get("query") or "")
        return extract_long_term_memory_records(
            content,
            build_memory_record_fn=self._build_memory_record_fn,
            default_long_term_scope=self._default_long_term_scope,
        )


class MemoryProviderManager:
    def __init__(self, providers: list[MemoryProvider] | None = None):
        self._providers = list(providers or [])

    @property
    def providers(self) -> list[MemoryProvider]:
        return list(self._providers)

    def add_provider(self, provider: MemoryProvider) -> None:
        self._providers.append(provider)

    def initialize_all(self, session_id: str, **kwargs) -> None:
        for provider in self._providers:
            provider.initialize(session_id, **kwargs)

    def render_prompt_blocks(self, snapshot: MemorySnapshot | None) -> str:
        blocks = [
            provider.render_prompt_block(snapshot)
            for provider in self._providers
        ]
        return "\n\n".join(block for block in blocks if block.strip())

    def recall_all(self, query: str | None, context: dict[str, Any] | None = None) -> str:
        blocks = [
            provider.recall(query, context=context)
            for provider in self._providers
        ]
        return "\n\n".join(block for block in blocks if block.strip())

    def on_pre_compact_all(self, messages: list[Any], state: dict[str, Any] | None = None) -> str:
        blocks = [
            provider.on_pre_compact(messages, state=state)
            for provider in self._providers
        ]
        return "\n\n".join(block for block in blocks if block.strip())

    def shutdown_all(self) -> None:
        for provider in reversed(self._providers):
            provider.shutdown()
