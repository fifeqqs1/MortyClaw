from __future__ import annotations

import os
from dataclasses import dataclass

from .store import DEFAULT_LONG_TERM_SCOPE, USER_PROFILE_MEMORY_TYPE


@dataclass(frozen=True)
class MemorySnapshot:
    session_id: str
    profile_content: str
    source: str = ""


def build_memory_snapshot(
    *,
    session_id: str,
    get_memory_store_fn,
    memory_dir: str,
    default_long_term_scope: str = DEFAULT_LONG_TERM_SCOPE,
    user_profile_memory_type: str = USER_PROFILE_MEMORY_TYPE,
) -> MemorySnapshot:
    records = get_memory_store_fn().list_memories(
        layer="long_term",
        scope=default_long_term_scope,
        memory_type=user_profile_memory_type,
        limit=1,
    )
    if records:
        content = str(records[0].get("content", "") or "").strip()
        if content:
            return MemorySnapshot(
                session_id=session_id,
                profile_content=content,
                source="memory_store",
            )

    profile_path = os.path.join(memory_dir, "user_profile.md")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read().strip()
        if content:
            return MemorySnapshot(
                session_id=session_id,
                profile_content=content,
                source="user_profile_file",
            )

    return MemorySnapshot(session_id=session_id, profile_content="", source="")
