from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MESSAGE_TOKEN_OVERHEAD = 8
TOOL_MESSAGE_EXTRA_TOKEN_OVERHEAD = 12
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
COMPRESSED_TOOL_RESULT_PREFIX = "[compressed-tool-result]"
COMPRESSED_TOOL_RESULT_LIMIT = 480
PROTECT_HEAD_NON_SYSTEM = 3
TAIL_MIN_MESSAGES = 8
TRIMMED_TOOL_RESULT_STUB = "上下文压缩说明：原始 tool result 已被裁剪，请以 handoff summary 中的 tool_results/result_summary 为准。"


def _serialize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_serialize_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


class _FallbackEncoder:
    def encode(self, text: str) -> list[int]:
        return list(str(text or "").encode("utf-8", errors="ignore"))


@lru_cache(maxsize=64)
def _resolve_token_encoder(model_name: str):
    if tiktoken is None:  # pragma: no cover
        return _FallbackEncoder()

    normalized = str(model_name or "").strip()
    if normalized:
        try:
            return tiktoken.encoding_for_model(normalized)
        except Exception:
            pass

    for encoding_name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            continue
    return _FallbackEncoder()


def _estimate_message_tokens(message: BaseMessage, *, encoder) -> int:
    text = _serialize_content(getattr(message, "content", ""))
    estimated = len(encoder.encode(text)) + MESSAGE_TOKEN_OVERHEAD
    if isinstance(message, ToolMessage):
        estimated += TOOL_MESSAGE_EXTRA_TOKEN_OVERHEAD
    return estimated


def _estimate_messages_tokens(messages: list[BaseMessage], *, encoder) -> int:
    return sum(_estimate_message_tokens(message, encoder=encoder) for message in messages)


def estimate_messages_tokens(messages: list[BaseMessage], *, model_name: str = "") -> int:
    encoder = _resolve_token_encoder(model_name)
    return _estimate_messages_tokens(messages, encoder=encoder)


def estimate_text_tokens(text: Any, *, model_name: str = "") -> int:
    encoder = _resolve_token_encoder(model_name)
    return len(encoder.encode(_serialize_content(text)))


def classify_context_pressure(
    messages: list[BaseMessage],
    *,
    model_name: str = "",
    budget_tokens: int,
    reserve_tokens: int = 0,
    extra_texts: Sequence[Any] | None = None,
    layer2_trigger_ratio: float = 0.5,
    layer3_trigger_ratio: float = 0.7,
) -> dict[str, float | int | str]:
    message_tokens = estimate_messages_tokens(messages, model_name=model_name)
    extra_tokens = sum(
        estimate_text_tokens(text, model_name=model_name)
        for text in (extra_texts or [])
        if _serialize_content(text)
    )
    normalized_budget = max(1, int(budget_tokens))
    normalized_reserve = max(0, int(reserve_tokens))
    total_used_tokens = message_tokens + extra_tokens + normalized_reserve
    usage_ratio = total_used_tokens / normalized_budget
    medium_ratio = min(0.95, max(0.0, float(layer2_trigger_ratio)))
    high_ratio = min(0.99, max(medium_ratio, float(layer3_trigger_ratio)))
    level = "low"
    if usage_ratio >= high_ratio:
        level = "high"
    elif usage_ratio >= medium_ratio:
        level = "medium"
    return {
        "level": level,
        "message_tokens": message_tokens,
        "extra_tokens": extra_tokens,
        "reserve_tokens": normalized_reserve,
        "total_used_tokens": total_used_tokens,
        "budget_tokens": normalized_budget,
        "usage_ratio": usage_ratio,
        "layer2_trigger_ratio": medium_ratio,
        "layer3_trigger_ratio": high_ratio,
    }


def _effective_token_budget(raw_budget: int | None, reserve_tokens: int) -> int | None:
    if raw_budget is None:
        return None
    if raw_budget <= 0:
        return None
    return max(raw_budget - max(reserve_tokens, 0), 1)


def _is_persisted_tool_message(message: BaseMessage) -> bool:
    if not isinstance(message, ToolMessage):
        return False
    content = str(getattr(message, "content", "") or "")
    metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    artifact = metadata.get("mortyclaw_artifact") or {}
    return (
        PERSISTED_OUTPUT_TAG in content
        or PERSISTED_OUTPUT_CLOSING_TAG in content
        or bool(getattr(artifact, "get", lambda *_args, **_kwargs: False)("artifact_persisted", False))
    )


def _tool_call_ids_from_message(message: BaseMessage) -> list[str]:
    if not isinstance(message, AIMessage):
        return []
    raw_tool_calls = getattr(message, "tool_calls", None) or getattr(message, "additional_kwargs", {}).get("tool_calls") or []
    tool_ids: list[str] = []
    for index, tool_call in enumerate(raw_tool_calls):
        if not isinstance(tool_call, dict):
            continue
        normalized_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or f"tool-call-{index}").strip()
        if normalized_id:
            tool_ids.append(normalized_id)
    return tool_ids


def _extract_persisted_preview(content: str) -> str:
    if PERSISTED_OUTPUT_TAG not in content:
        return ""
    marker = "预览（前 "
    marker_index = content.find(marker)
    if marker_index == -1:
        return ""
    preview_start = content.find("）：", marker_index)
    if preview_start == -1:
        preview_start = content.find("):", marker_index)
    if preview_start == -1:
        return ""
    preview_body = content[preview_start + 2 :]
    closing_index = preview_body.find(PERSISTED_OUTPUT_CLOSING_TAG)
    if closing_index != -1:
        preview_body = preview_body[:closing_index]
    return preview_body.strip()


def _summarize_old_tool_result(message: ToolMessage) -> ToolMessage:
    if _is_persisted_tool_message(message):
        return message
    content = str(getattr(message, "content", "") or "").strip()
    if len(content) <= COMPRESSED_TOOL_RESULT_LIMIT:
        return message
    preview = content[:COMPRESSED_TOOL_RESULT_LIMIT]
    last_break = preview.rfind("\n")
    if last_break > COMPRESSED_TOOL_RESULT_LIMIT // 2:
        preview = preview[:last_break]
    summary = f"{COMPRESSED_TOOL_RESULT_PREFIX}\n{preview.strip()}\n..."
    return ToolMessage(
        content=summary,
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
        id=getattr(message, "id", None),
        additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
    )


def _align_tail_start(messages: list[BaseMessage], start_index: int) -> int:
    aligned = max(0, start_index)
    while aligned > 0:
        current = messages[aligned]
        previous = messages[aligned - 1]
        if isinstance(current, ToolMessage):
            aligned -= 1
            continue
        if isinstance(previous, AIMessage) and _tool_call_ids_from_message(previous):
            aligned -= 1
            continue
        break
    return aligned


def _latest_human_index(messages: list[BaseMessage]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return -1


def _prune_old_tool_results(messages: list[BaseMessage], protected_tail_start: int) -> tuple[list[BaseMessage], dict[str, int]]:
    if not messages:
        return [], {"artifact_messages_seen": 0, "tool_results_pruned": 0}
    updated = list(messages)
    seen_keys: set[tuple[str, str, str]] = set()
    artifact_messages_seen = 0
    tool_results_pruned = 0
    for index, message in enumerate(updated):
        if not isinstance(message, ToolMessage):
            continue
        if _is_persisted_tool_message(message):
            artifact_messages_seen += 1
        if index >= protected_tail_start:
            continue
        tool_name = str(getattr(message, "name", "") or "").strip()
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        content = str(getattr(message, "content", "") or "")
        preview = _extract_persisted_preview(content) if _is_persisted_tool_message(message) else content[:220]
        dedupe_key = (tool_call_id, tool_name, preview.strip())
        if dedupe_key in seen_keys:
            updated[index] = _summarize_old_tool_result(message)
            tool_results_pruned += 1
            continue
        seen_keys.add(dedupe_key)
        compressed = _summarize_old_tool_result(message)
        if compressed.content != content:
            updated[index] = compressed
            tool_results_pruned += 1
    return updated, {
        "artifact_messages_seen": artifact_messages_seen,
        "tool_results_pruned": tool_results_pruned,
    }


def _sanitize_tool_pairs(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage], int]:
    if not messages:
        return [], [], 0
    sanitized: list[BaseMessage] = []
    discarded: list[BaseMessage] = []
    repaired = 0
    available_results = {
        str(getattr(message, "tool_call_id", "") or "").strip()
        for message in messages
        if isinstance(message, ToolMessage)
    }
    for message in messages:
        if isinstance(message, AIMessage):
            sanitized.append(message)
            for tool_call_id in _tool_call_ids_from_message(message):
                if tool_call_id and tool_call_id not in available_results:
                    sanitized.append(ToolMessage(content=TRIMMED_TOOL_RESULT_STUB, tool_call_id=tool_call_id))
                    available_results.add(tool_call_id)
                    repaired += 1
            continue
        if isinstance(message, ToolMessage):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
            if not tool_call_id:
                discarded.append(message)
                repaired += 1
                continue
            if not any(tool_call_id in _tool_call_ids_from_message(candidate) for candidate in sanitized if isinstance(candidate, AIMessage)):
                discarded.append(message)
                repaired += 1
                continue
        sanitized.append(message)
    return sanitized, discarded, repaired


def _trim_messages_to_token_budget(
    messages: list[BaseMessage],
    keep_tokens: int,
    reserve_tokens: int,
    model_name: str,
) -> tuple[list[BaseMessage], list[BaseMessage], dict[str, int]]:
    effective_keep_tokens = _effective_token_budget(keep_tokens, reserve_tokens)
    if effective_keep_tokens is None:
        return messages, [], {
            "artifact_messages_seen": 0,
            "tool_results_pruned": 0,
            "tool_pairs_repaired": 0,
            "discarded_middle_count": 0,
        }

    encoder = _resolve_token_encoder(model_name)
    if _estimate_messages_tokens(messages, encoder=encoder) <= effective_keep_tokens:
        return messages, [], {
            "artifact_messages_seen": sum(1 for message in messages if _is_persisted_tool_message(message)),
            "tool_results_pruned": 0,
            "tool_pairs_repaired": 0,
            "discarded_middle_count": 0,
        }

    first_system = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
    non_system = messages[1:] if first_system is not None else list(messages)
    if not non_system:
        kept = [first_system] if first_system is not None else []
        return kept, [], {
            "artifact_messages_seen": 0,
            "tool_results_pruned": 0,
            "tool_pairs_repaired": 0,
            "discarded_middle_count": 0,
        }

    provisional_tail_start = max(len(non_system) - TAIL_MIN_MESSAGES, PROTECT_HEAD_NON_SYSTEM)
    pruned_non_system, prune_stats = _prune_old_tool_results(non_system, provisional_tail_start)
    non_system_costs = [_estimate_message_tokens(message, encoder=encoder) for message in pruned_non_system]
    total_non_system_tokens = sum(non_system_costs)
    system_tokens = _estimate_message_tokens(first_system, encoder=encoder) if first_system is not None else 0
    tail_budget = max(1, effective_keep_tokens - system_tokens)

    tail_indices: list[int] = []
    tail_tokens = 0
    for index in range(len(pruned_non_system) - 1, -1, -1):
        message_tokens = non_system_costs[index]
        if tail_indices and tail_tokens + message_tokens > tail_budget:
            break
        tail_indices.append(index)
        tail_tokens += message_tokens
    if not tail_indices and pruned_non_system:
        tail_indices.append(len(pruned_non_system) - 1)
    tail_indices = sorted(tail_indices)
    tail_start = tail_indices[0] if tail_indices else len(pruned_non_system)
    tail_start = _align_tail_start(pruned_non_system, tail_start)
    latest_human_index = _latest_human_index(pruned_non_system)
    head_count = min(PROTECT_HEAD_NON_SYSTEM, tail_start)
    while head_count > 0 and system_tokens + sum(non_system_costs[:head_count]) + sum(non_system_costs[tail_start:]) > effective_keep_tokens:
        head_count -= 1
    keep_indices = set(range(head_count))
    keep_indices.update(range(tail_start, len(pruned_non_system)))
    if latest_human_index >= 0:
        keep_indices.add(latest_human_index)
    kept_non_system = [message for index, message in enumerate(pruned_non_system) if index in keep_indices]
    discarded_middle = [message for index, message in enumerate(pruned_non_system) if index not in keep_indices]
    sanitized_kept, sanitized_discarded, repaired = _sanitize_tool_pairs(kept_non_system)
    final_kept = ([first_system] if first_system is not None else []) + sanitized_kept
    final_discarded = discarded_middle + sanitized_discarded
    return final_kept, final_discarded, {
        **prune_stats,
        "tool_pairs_repaired": repaired,
        "discarded_middle_count": len(discarded_middle),
        "total_non_system_tokens": total_non_system_tokens,
    }


def _trim_messages_to_budget(
    messages: list[BaseMessage],
    keep_messages: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    if keep_messages <= 0 or len(messages) <= keep_messages:
        return messages, []

    latest_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            latest_human_index = index
            break

    keep_indices = set(range(max(0, len(messages) - keep_messages), len(messages)))
    if latest_human_index >= 0:
        keep_indices.add(latest_human_index)

    while len(keep_indices) > keep_messages:
        removable = [index for index in sorted(keep_indices) if index != latest_human_index]
        if not removable:
            break
        keep_indices.remove(removable[0])

    kept = [message for index, message in enumerate(messages) if index in keep_indices]
    discarded = [message for index, message in enumerate(messages) if index not in keep_indices]
    return kept, discarded


def trim_context_messages(
    messages: list[BaseMessage],
    trigger_turns: int = 8,
    keep_turns: int = 4,
    *,
    trigger_messages: int | None = None,
    keep_messages: int | None = None,
    trigger_tokens: int | None = None,
    keep_tokens: int | None = None,
    reserve_tokens: int = 0,
    model_name: str = "",
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    if trigger_tokens is not None and keep_tokens is not None:
        token_messages = ([first_system] if first_system else []) + list(non_system_msgs)
        encoder = _resolve_token_encoder(model_name)
        effective_trigger_tokens = _effective_token_budget(trigger_tokens, reserve_tokens)
        message_tokens = _estimate_messages_tokens(token_messages, encoder=encoder)
        token_triggered = (
            effective_trigger_tokens is not None
            and message_tokens >= effective_trigger_tokens
        )
        if token_triggered:
            final_messages, discarded_messages, _stats = _trim_messages_to_token_budget(
                token_messages,
                keep_tokens=keep_tokens,
                reserve_tokens=reserve_tokens,
                model_name=model_name,
            )
            setattr(trim_context_messages, "_last_stats", _stats)
            return final_messages, discarded_messages
        setattr(trim_context_messages, "_last_stats", {
            "artifact_messages_seen": sum(1 for message in token_messages if _is_persisted_tool_message(message)),
            "tool_results_pruned": 0,
            "tool_pairs_repaired": 0,
            "discarded_middle_count": 0,
        })
        return token_messages, []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)
    turn_triggered = total_turns >= trigger_turns
    message_triggered = (
        trigger_messages is not None
        and trigger_messages > 0
        and len(non_system_msgs) >= trigger_messages
    )

    if turn_triggered:
        recent_turns = turns[-keep_turns:]
        discarded_turns = turns[:-keep_turns]
        kept_non_system: list[BaseMessage] = []
        for turn in recent_turns:
            kept_non_system.extend(turn)

        discarded_messages: list[BaseMessage] = []
        for turn in discarded_turns:
            discarded_messages.extend(turn)
    else:
        kept_non_system = list(non_system_msgs)
        discarded_messages = []

    if keep_messages is not None and keep_messages > 0:
        should_trim_by_budget = message_triggered or len(kept_non_system) > keep_messages
        if should_trim_by_budget:
            kept_non_system, extra_discarded = _trim_messages_to_budget(kept_non_system, keep_messages)
            discarded_messages.extend(extra_discarded)

    if not discarded_messages:
        final_messages = ([first_system] if first_system else []) + kept_non_system
        setattr(trim_context_messages, "_last_stats", {
            "artifact_messages_seen": 0,
            "tool_results_pruned": 0,
            "tool_pairs_repaired": 0,
            "discarded_middle_count": 0,
        })
        return final_messages, []

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    final_messages.extend(kept_non_system)
    setattr(trim_context_messages, "_last_stats", {
        "artifact_messages_seen": 0,
        "tool_results_pruned": 0,
        "tool_pairs_repaired": 0,
        "discarded_middle_count": len(discarded_messages),
    })
    return final_messages, discarded_messages
