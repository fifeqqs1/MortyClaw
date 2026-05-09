from __future__ import annotations

from copy import deepcopy

from langchain_core.messages import BaseMessage, SystemMessage


def apply_provider_prompt_cache(
    messages: list[BaseMessage],
    *,
    provider_name: str = "",
    model_name: str = "",
) -> tuple[list[BaseMessage], dict[str, bool]]:
    normalized_provider = str(provider_name or "").strip().lower()
    normalized_model = str(model_name or "").strip().lower()
    if normalized_provider != "anthropic" and "claude" not in normalized_model:
        return messages, {"provider_cache_applied": False}

    cached_messages = [deepcopy(message) for message in messages]
    if cached_messages and isinstance(cached_messages[0], SystemMessage):
        first = cached_messages[0]
        additional_kwargs = dict(getattr(first, "additional_kwargs", {}) or {})
        additional_kwargs["cache_control"] = {"type": "ephemeral"}
        first.additional_kwargs = additional_kwargs
        return cached_messages, {"provider_cache_applied": True}
    return cached_messages, {"provider_cache_applied": False}
