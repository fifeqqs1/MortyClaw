from __future__ import annotations

import json

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

from ..context import AgentState
from ..error_policy import classify_error, deserialize_classified_error, serialize_classified_error
from ..runtime.todos import merge_tool_written_todos, normalize_todos, todos_to_plan
from ..runtime.tool_results import prepare_tool_messages_for_budget


STEP_OUTCOME_SUCCESS_CANDIDATE = "success_candidate"
STEP_OUTCOME_FAILURE = "failure"
RESPONSE_KIND_STEP_RESULT = "step_result"
RESPONSE_KIND_FINAL_ANSWER = "final_answer"


def _clone_tool_message(
    message: ToolMessage,
    *,
    content: str | None = None,
    additional_kwargs: dict | None = None,
) -> ToolMessage:
    merged_additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    if additional_kwargs:
        merged_additional_kwargs.update(additional_kwargs)
    return ToolMessage(
        content=str(content if content is not None else getattr(message, "content", "") or ""),
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
        id=getattr(message, "id", None),
        additional_kwargs=merged_additional_kwargs,
    )


def _annotate_ai_message(message: AIMessage, **metadata) -> AIMessage:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs.update({key: value for key, value in metadata.items() if value is not None})
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def _extract_classified_error(message) -> object | None:
    data = getattr(message, "additional_kwargs", {}).get("mortyclaw_error")
    return deserialize_classified_error(data)


def _annotate_tool_error(message: ToolMessage) -> ToolMessage:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    if additional_kwargs.get("mortyclaw_error"):
        return message
    classified = classify_error(message=str(getattr(message, "content", "") or ""), tool_name=getattr(message, "name", "") or "")
    if classified.kind.value not in {"tool_runtime_error", "tool_schema_error", "unsafe_tool_scope"}:
        return message
    additional_kwargs["mortyclaw_error"] = serialize_classified_error(classified)
    return _clone_tool_message(message, additional_kwargs=additional_kwargs)


def _extract_tool_json_payload(message: ToolMessage) -> dict | None:
    content = str(getattr(message, "content", "") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _project_tool_recovery_followup(state: AgentState, message: ToolMessage) -> dict:
    payload = _extract_tool_json_payload(message)
    if not payload or payload.get("ok", True):
        return {}

    if str(getattr(message, "name", "") or "") not in {"edit_project_file", "write_project_file", "apply_project_patch"}:
        return {}

    error_kind = str(payload.get("error_kind", "") or "")
    if error_kind not in {
        "OLD_TEXT_NOT_FOUND",
        "OLD_TEXT_AMBIGUOUS",
        "FILE_CHANGED_SINCE_READ",
        "PATCH_PARSE_FAILED",
        "PATCH_CONTEXT_MISMATCH",
        "PATCH_DOES_NOT_APPLY",
    }:
        return {}

    summary = str(payload.get("message", "") or "")
    recovery_hint = str(payload.get("recovery_hint", "") or "")
    current_excerpt = str(payload.get("current_excerpt", "") or "")
    if recovery_hint:
        summary += f"\n恢复建议：{recovery_hint}"
    if current_excerpt:
        summary += f"\n\n最新文件片段：\n{current_excerpt}"
    return {
        "last_error": summary[:500],
        "last_error_kind": error_kind,
        "last_recovery_action": "refresh_file_then_retry",
    }


def _tool_program_followup(_state: AgentState, message: ToolMessage) -> dict:
    if (getattr(message, "name", None) or "") != "execute_tool_program":
        return {}
    payload = _extract_tool_json_payload(message)
    if not payload:
        return {}
    status = str(payload.get("status", "") or "").strip().lower()
    if status == "needs_approval":
        return {
            "program_run_id": str(payload.get("program_run_id", "") or ""),
            "program_run_status": "awaiting_approval",
            "pending_approval": True,
            "approval_granted": False,
            "approval_prompted": False,
            "approval_reason": str(payload.get("approval_reason", "") or ""),
            "pending_tool_calls": list(payload.get("resume_tool_calls", []) or []),
            "pending_execution_snapshot": dict(payload.get("pending_execution_snapshot", {}) or {}),
            "run_status": "awaiting_step_approval",
        }
    if status == "completed":
        return {
            "program_run_id": str(payload.get("program_run_id", "") or ""),
            "program_run_status": "completed",
        }
    if status == "failed":
        return {
            "program_run_id": str(payload.get("program_run_id", "") or ""),
            "program_run_status": "failed",
            "last_error": str(payload.get("message", "") or "")[:500],
            "last_error_kind": "tool_program_failed",
            "last_recovery_action": "replan",
        }
    return {}


def _worker_tool_followup(state: AgentState, message: ToolMessage) -> dict:
    tool_name = str(getattr(message, "name", None) or "")
    if tool_name not in {"delegate_subagent", "wait_subagents", "list_subagents", "cancel_subagent"}:
        return {}
    payload = _extract_tool_json_payload(message)
    if not payload:
        return {}
    if tool_name == "delegate_subagent" and payload.get("success"):
        active_workers = list(state.get("active_workers", []) or [])
        worker_id = str(payload.get("worker_id", "") or "")
        if worker_id and worker_id not in active_workers:
            active_workers.append(worker_id)
        return {
            "active_workers": active_workers,
            "worker_waiting_on": active_workers,
        }
    if tool_name == "wait_subagents" and payload.get("success"):
        workers = [item for item in (payload.get("workers", []) or []) if isinstance(item, dict)]
        completed_worker_ids = [
            str(item.get("worker_id", "") or "")
            for item in workers
            if str(item.get("status", "") or "") in {"completed", "failed", "cancelled", "timeout"}
        ]
        active_workers = [
            worker_id for worker_id in (state.get("active_workers", []) or [])
            if worker_id not in completed_worker_ids
        ]
        return {
            "worker_results": workers,
            "active_workers": active_workers,
            "worker_waiting_on": active_workers,
        }
    if tool_name == "list_subagents" and payload.get("success"):
        workers = [item for item in (payload.get("workers", []) or []) if isinstance(item, dict)]
        waiting = [
            str(item.get("worker_id", "") or "")
            for item in workers
            if str(item.get("status", "") or "") not in {"completed", "failed", "cancelled", "timeout"}
        ]
        return {
            "active_workers": [
                str(item.get("worker_id", "") or "")
                for item in workers
                if str(item.get("worker_id", "") or "").strip()
            ],
            "worker_waiting_on": waiting,
        }
    return {}


def _sync_todo_from_tool_message(state: AgentState, message: ToolMessage) -> dict:
    if (getattr(message, "name", None) or "") != "update_todo_list":
        return {}
    tool_call_id = str(getattr(message, "tool_call_id", "") or "")
    if not tool_call_id or tool_call_id == state.get("last_todo_tool_call_id", ""):
        return {}
    try:
        payload = json.loads(str(getattr(message, "content", "") or ""))
    except json.JSONDecodeError:
        return {}
    if not payload.get("success"):
        return {}
    todo_state = payload.get("todo_state") or {}
    todo_items = merge_tool_written_todos(
        state.get("plan", []),
        state.get("todos", []),
        todo_state.get("items", []),
    )
    rebuilt_plan, next_step_index = todos_to_plan(
        state.get("plan", []),
        todo_items,
        lambda _description: state.get("risk_level", "medium"),
    )
    return {
        "plan": rebuilt_plan,
        "current_step_index": next_step_index,
        "todos": todo_items,
        "active_todos": todo_items,
        "todo_revision": int(todo_state.get("revision", state.get("todo_revision", 0)) or 0),
        "todo_needs_announcement": True,
        "last_todo_tool_call_id": tool_call_id,
    }


def _complete_autonomous_todos(state: AgentState) -> dict:
    existing_todos = normalize_todos(
        list(state.get("active_todos") or state.get("todos") or [])
    )
    if not existing_todos:
        return {}

    completed_todos: list[dict] = []
    for item in existing_todos:
        next_item = dict(item)
        if next_item.get("status") in {"pending", "in_progress"}:
            next_item["status"] = "completed"
        completed_todos.append(next_item)

    rebuilt_plan, next_step_index = todos_to_plan(
        state.get("plan", []),
        completed_todos,
        lambda _description: state.get("risk_level", "medium"),
    )
    if rebuilt_plan:
        next_step_index = len(rebuilt_plan) - 1
    else:
        next_step_index = 0

    return {
        "plan": rebuilt_plan,
        "current_step_index": next_step_index,
        "todos": completed_todos,
        "active_todos": completed_todos,
        "todo_revision": int(state.get("todo_revision", 0) or 0) + 1,
        "todo_needs_announcement": False,
    }


def _prepare_recent_tool_messages(state: AgentState, *, thread_id: str, turn_id: str) -> tuple[list, dict]:
    raw_messages = list(state.get("messages", []) or [])
    trailing_indices: list[int] = []
    for index in range(len(raw_messages) - 1, -1, -1):
        message = raw_messages[index]
        if getattr(message, "type", "") == "tool":
            trailing_indices.append(index)
            continue
        break
    if not trailing_indices:
        return raw_messages, {}

    trailing_indices.reverse()
    trailing_messages = [
        _annotate_tool_error(message)
        if isinstance(message, ToolMessage)
        else message
        for message in (raw_messages[index] for index in trailing_indices)
    ]
    prepared_messages = prepare_tool_messages_for_budget(
        trailing_messages,
        thread_id=thread_id,
        turn_id=turn_id,
    )

    replace_ops: list = []
    changed = False
    for list_index, message_index in enumerate(trailing_indices):
        original = raw_messages[message_index]
        replacement = prepared_messages[list_index]
        if replacement is not original or getattr(replacement, "content", "") != getattr(original, "content", "") or getattr(replacement, "additional_kwargs", {}) != getattr(original, "additional_kwargs", {}):
            changed = True
            raw_messages[message_index] = replacement
            if getattr(original, "id", None):
                replace_ops.append(RemoveMessage(id=original.id))
            replace_ops.append(replacement)

    updated_state: dict = {}
    for message in prepared_messages:
        if isinstance(message, ToolMessage):
            updated_state.update(_sync_todo_from_tool_message(state | updated_state, message))
            updated_state.update(_project_tool_recovery_followup(state | updated_state, message))
            updated_state.update(_tool_program_followup(state | updated_state, message))
            updated_state.update(_worker_tool_followup(state | updated_state, message))

    if changed:
        updated_state["messages"] = replace_ops
    return raw_messages, updated_state
