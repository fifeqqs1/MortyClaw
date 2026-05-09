from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message
    return None


def _has_pending_execution_state(state: dict[str, Any]) -> bool:
    if state.get("pending_approval"):
        return True
    if state.get("pending_tool_calls"):
        return True
    if state.get("pending_execution_snapshot"):
        return True
    return False


def should_auto_compact(
    *,
    state: dict[str, Any],
    pressure_level: str,
    has_new_summary: bool,
    overflow_retry: bool = False,
    latest_user_message: BaseMessage | None = None,
) -> bool:
    if not has_new_summary:
        return False
    if _has_pending_execution_state(state):
        return False
    latest_user_id = str(getattr(latest_user_message, "id", "") or "").strip()
    if not latest_user_id:
        return False
    if overflow_retry:
        return True
    return str(pressure_level or "").strip().lower() == "high"


def compact_context_state(
    state: dict[str, Any],
    *,
    original_messages: list[BaseMessage],
    latest_user_message: BaseMessage,
    reason: str,
) -> tuple[dict[str, Any], list[BaseMessage]]:
    latest_user_id = str(getattr(latest_user_message, "id", "") or "").strip()
    if not latest_user_id:
        return {}, list(original_messages or [])

    remove_updates: list[RemoveMessage] = []
    seen_ids: set[str] = set()
    for message in original_messages or []:
        message_id = str(getattr(message, "id", "") or "").strip()
        if not message_id or message_id == latest_user_id or message_id in seen_ids:
            continue
        seen_ids.add(message_id)
        remove_updates.append(RemoveMessage(id=message_id))

    compact_generation = int(state.get("compact_generation", 0) or 0) + 1
    return (
        {
            "messages": remove_updates,
            "compact_generation": compact_generation,
            "last_compact_at": _utc_now_iso(),
            "last_compact_reason": str(reason or "").strip() or "auto_compact",
        },
        [latest_user_message],
    )


__all__ = [
    "compact_context_state",
    "should_auto_compact",
]
