from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import DB_PATH, WORKER_DEFAULT_TIMEOUT_SECONDS, WORKER_MAX_CONCURRENCY
from ..logger import audit_logger, build_log_file_path
from ..runtime.path_locks import get_project_path_lock_manager
from ..runtime_context import set_active_program_run_id, set_active_thread_id, set_active_tool_scope_names, set_active_worker_id
from ..storage.runtime import (
    get_conversation_writer,
    get_session_repository,
    get_worker_run_repository,
)
from ..tools.project.common import _session_project_root


WORKER_ROLE_ALLOWED_DEFAULTS = {
    "explore": {"read_project_file", "search_project_code", "show_git_diff", "update_todo_list"},
    "verify": {"read_project_file", "search_project_code", "show_git_diff", "run_project_tests", "run_project_command", "update_todo_list"},
    "implement": {"read_project_file", "search_project_code", "show_git_diff", "edit_project_file", "write_project_file", "apply_project_patch", "run_project_tests", "run_project_command", "update_todo_list"},
}
WORKER_TOOLSETS = {
    "project_read": {"read_project_file", "search_project_code", "show_git_diff", "update_todo_list"},
    "project_write": {"edit_project_file", "write_project_file", "apply_project_patch"},
    "project_verify": {"run_project_tests", "run_project_command"},
    "research": {"tavily_web_search", "summarize_content", "arxiv_rag_ask"},
}
WORKER_TOOLSETS["project_full"] = (
    WORKER_TOOLSETS["project_read"]
    | WORKER_TOOLSETS["project_write"]
    | WORKER_TOOLSETS["project_verify"]
)
RESTRICTED_WORKER_TOOL_NAMES = {
    "delegate_subagent",
    "delegate_subagents",
    "wait_subagents",
    "list_subagents",
    "cancel_subagent",
    "cancel_subagents",
    "execute_tool_program",
    "request_tool_schema",
    "save_user_profile",
    "search_sessions",
    "schedule_task",
    "list_scheduled_tasks",
    "delete_scheduled_task",
    "modify_scheduled_task",
}
FINAL_WORKER_STATUSES = {"completed", "failed", "cancelled", "timeout"}
WORKER_RESULT_SUMMARY_CHAR_LIMIT = 1600
WORKER_RESULT_LIST_LIMIT = 12
WORKER_GOAL_PREVIEW_CHAR_LIMIT = 240
WORKER_DIRECT_BUDGETS = {
    "explore": {"llm_rounds": 6, "tool_calls": 10, "file_reads": 8},
    "verify": {"llm_rounds": 6, "tool_calls": 10, "file_reads": 8},
}


class WorkerCancelledError(Exception):
    def __init__(self, summary: dict[str, Any]):
        super().__init__("worker cancellation requested")
        self.summary = summary


class WorkerTimeoutError(Exception):
    def __init__(self, summary: dict[str, Any]):
        super().__init__("worker timed out")
        self.summary = summary


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_name_list(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or []:
        name = str(value or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _parse_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = str(record.get("metadata_json", "") or "{}")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _with_parsed_metadata(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    enriched = dict(record)
    enriched["metadata"] = _parse_record_metadata(enriched)
    return enriched


def _truncate_text(text: Any, limit: int) -> str:
    normalized = str(text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return normalized[:limit]
    return normalized[: limit - 3] + "..."


def _parse_record_json_field(record: dict[str, Any], field_name: str) -> dict[str, Any]:
    raw_value = record.get(field_name, "{}")
    if isinstance(raw_value, dict):
        return dict(raw_value)
    try:
        parsed = json.loads(str(raw_value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact_string_list(values: Any, *, limit: int = WORKER_RESULT_LIST_LIMIT, char_limit: int = 180) -> list[str]:
    if not isinstance(values, list):
        return []
    compacted: list[str] = []
    for value in values:
        text = _truncate_text(value, char_limit)
        if not text:
            continue
        compacted.append(text)
        if len(compacted) >= limit:
            break
    return compacted


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_worker_result_payload(
    *,
    worker_id: str,
    status: str,
    result_summary: dict[str, Any] | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    result = dict(result_summary or {})
    summary_text = str(result.get("summary", "") or "")
    blocking_issue = str(result.get("blocking_issue", "") or error_message or "")
    return {
        "worker_id": worker_id,
        "status": status,
        "summary": _truncate_text(summary_text, WORKER_RESULT_SUMMARY_CHAR_LIMIT),
        "summary_chars": len(summary_text),
        "summary_truncated": len(summary_text) > WORKER_RESULT_SUMMARY_CHAR_LIMIT,
        "key_files": _compact_string_list(result.get("key_files")),
        "evidence": _compact_string_list(result.get("evidence"), char_limit=260),
        "changed_files": _compact_string_list(result.get("changed_files")),
        "commands_run": _compact_string_list(result.get("commands_run")),
        "tests_run": _compact_string_list(result.get("tests_run")),
        "blocking_issue": _truncate_text(blocking_issue, 500),
        "confidence": _truncate_text(result.get("confidence", ""), 40),
        "budget_exhausted": bool(result.get("budget_exhausted")),
    }


def _partial_worker_result_from_record(
    worker_id: str,
    *,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    record = get_worker_run_repository().get_worker_run(worker_id) or {}
    existing_result = _parse_record_json_field(record, "result_summary_json") if record else {}
    partial = {
        "worker_id": worker_id,
        "status": status,
        "summary": str(existing_result.get("summary", "") or ""),
        "key_files": _compact_string_list(existing_result.get("key_files"), limit=1000, char_limit=1000),
        "evidence": _compact_string_list(existing_result.get("evidence"), limit=1000, char_limit=1000),
        "changed_files": _compact_string_list(existing_result.get("changed_files"), limit=1000, char_limit=1000),
        "commands_run": _compact_string_list(existing_result.get("commands_run"), limit=1000, char_limit=1000),
        "tests_run": _compact_string_list(existing_result.get("tests_run"), limit=1000, char_limit=1000),
        "blocking_issue": str(existing_result.get("blocking_issue", "") or error_message or ""),
        "confidence": str(existing_result.get("confidence", "") or ""),
        "budget_exhausted": bool(existing_result.get("budget_exhausted")),
        "partial_result": bool(existing_result.get("summary")),
    }
    if error_message and error_message not in partial["blocking_issue"]:
        partial["blocking_issue"] = f"{partial['blocking_issue']} | {error_message}" if partial["blocking_issue"] else error_message
    return partial


def _compact_worker_record(record: dict[str, Any] | None, *, include_result: bool = False) -> dict[str, Any] | None:
    if record is None:
        return None
    enriched = _with_parsed_metadata(record) or {}
    metadata = enriched.get("metadata", {}) if isinstance(enriched.get("metadata", {}), dict) else {}
    compact = {
        "worker_id": str(enriched.get("worker_id", "") or ""),
        "worker_thread_id": str(enriched.get("worker_thread_id", "") or ""),
        "parent_thread_id": str(enriched.get("parent_thread_id", "") or ""),
        "role": str(enriched.get("role", "") or ""),
        "status": str(enriched.get("status", "") or ""),
        "task": _truncate_text(str(enriched.get("goal", "") or ""), WORKER_GOAL_PREVIEW_CHAR_LIMIT),
        "created_at": str(enriched.get("created_at", "") or ""),
        "started_at": enriched.get("started_at"),
        "finished_at": enriched.get("finished_at"),
        "task_index": _safe_int(metadata.get("task_index", 0), 0),
        "batch_id": str(metadata.get("batch_id", "") or ""),
        "waiting": str(enriched.get("status", "") or "") not in FINAL_WORKER_STATUSES,
    }
    if include_result:
        compact.update(
            _compact_worker_result_payload(
                worker_id=compact["worker_id"],
                status=compact["status"],
                result_summary=_parse_record_json_field(enriched, "result_summary_json"),
                error_message=str(_parse_record_json_field(enriched, "error_json").get("message", "") or ""),
            )
        )
    return compact


def _tool_payload_json(message: ToolMessage) -> dict[str, Any] | None:
    content = str(getattr(message, "content", "") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "```" in stripped:
        chunks = stripped.split("```")
        candidates.extend(chunk.removeprefix("json").strip() for chunk in chunks)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _append_unique_text(target: list[str], value: Any, *, limit: int = 24) -> None:
    text = str(value or "").strip()
    if not text or text in target:
        return
    target.append(text)
    if len(target) > limit:
        del target[limit:]


def _merge_summary_payload(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    if payload.get("summary"):
        summary["summary"] = str(payload.get("summary") or "").strip()
    if payload.get("blocking_issue"):
        summary["blocking_issue"] = str(payload.get("blocking_issue") or "").strip()
    if payload.get("confidence"):
        summary["confidence"] = str(payload.get("confidence") or "").strip()
    for key in ("key_files", "evidence", "changed_files", "commands_run", "tests_run"):
        values = payload.get(key)
        if isinstance(values, list):
            for value in values:
                _append_unique_text(summary.setdefault(key, []), value)
        elif values:
            _append_unique_text(summary.setdefault(key, []), values)


def _worker_cancel_requested(worker_id: str) -> bool:
    record = get_worker_run_repository().get_worker_run(worker_id)
    if not record:
        return False
    status = str(record.get("status", "") or "")
    metadata = _parse_record_metadata(record)
    return status == "cancelling" or bool(metadata.get("cancel_requested"))


class AsyncWorkerSupervisor:
    def __init__(self, *, max_workers: int = WORKER_MAX_CONCURRENCY):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mortyclaw-worker")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def resolve_worker_tools(
        self,
        *,
        role: str,
        allowed_tools: list[str] | None,
        toolsets: list[str] | None,
        parent_tool_scope: list[str] | tuple[str, ...] | set[str] | None = None,
        legacy_allowed_only: bool = False,
    ) -> dict[str, Any]:
        normalized_role = str(role or "explore").strip().lower() or "explore"
        allowed_names = set(_normalize_name_list(allowed_tools))
        requested_toolsets = _normalize_name_list(toolsets)
        invalid_toolsets = [name for name in requested_toolsets if name not in WORKER_TOOLSETS]

        requested_names = set(allowed_names)
        if legacy_allowed_only and not requested_toolsets:
            if not requested_names:
                requested_names = set(WORKER_ROLE_ALLOWED_DEFAULTS.get(normalized_role, WORKER_ROLE_ALLOWED_DEFAULTS["explore"]))
        else:
            requested_names |= set(WORKER_ROLE_ALLOWED_DEFAULTS.get(normalized_role, WORKER_ROLE_ALLOWED_DEFAULTS["explore"]))
            for toolset_name in requested_toolsets:
                requested_names |= set(WORKER_TOOLSETS.get(toolset_name, set()))

        requested_names -= RESTRICTED_WORKER_TOOL_NAMES
        parent_scope = set(_normalize_name_list(parent_tool_scope))
        if parent_scope:
            effective_names = requested_names & parent_scope
        else:
            effective_names = requested_names
        effective_names -= RESTRICTED_WORKER_TOOL_NAMES
        return {
            "role": normalized_role,
            "requested_toolsets": requested_toolsets,
            "invalid_toolsets": invalid_toolsets,
            "requested_tools": sorted(requested_names),
            "effective_tools": sorted(effective_names),
            "parent_tool_scope": sorted(parent_scope),
        }

    def _load_toolset(self, effective_tools: list[str], *, role: str):
        from ..tools.builtins import BUILTIN_TOOLS

        allowed = set(effective_tools or []) or set(WORKER_ROLE_ALLOWED_DEFAULTS.get(role, WORKER_ROLE_ALLOWED_DEFAULTS["explore"]))
        filtered = []
        for tool in BUILTIN_TOOLS:
            tool_name = getattr(tool, "name", "")
            if tool_name in RESTRICTED_WORKER_TOOL_NAMES:
                continue
            if tool_name in allowed:
                filtered.append(tool)
        return filtered

    def submit_worker(
        self,
        *,
        parent_thread_id: str,
        parent_turn_id: str,
        role: str,
        task: str,
        allowed_tools: list[str] | None = None,
        write_scope: list[str] | None = None,
        deliverables: str = "",
        timeout_seconds: int = WORKER_DEFAULT_TIMEOUT_SECONDS,
        priority: int = 1,
        toolsets: list[str] | None = None,
        effective_tools: list[str] | None = None,
        context_brief: str = "",
        batch_id: str = "",
        task_index: int = 0,
        parent_tool_scope: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        normalized_role = str(role or "explore").strip().lower() or "explore"
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        requested_toolsets = _normalize_name_list(toolsets)
        normalized_allowed_tools = _normalize_name_list(allowed_tools)
        normalized_effective_tools = _normalize_name_list(effective_tools)
        worker_thread_id = f"{parent_thread_id}-worker-{uuid.uuid4().hex[:8]}"
        worker_record = worker_repo.create_worker_run(
            parent_thread_id=parent_thread_id,
            worker_thread_id=worker_thread_id,
            parent_turn_id=parent_turn_id,
            role=normalized_role,
            goal=str(task or ""),
            allowed_tools=normalized_allowed_tools,
            write_scope=list(write_scope or []),
            tool_budget=max(1, priority or 1),
            metadata={
                "batch_id": str(batch_id or ""),
                "task_index": int(task_index or 0),
                "context_brief": str(context_brief or "").strip(),
                "deliverables": str(deliverables or ""),
                "requested_toolsets": requested_toolsets,
                "effective_tools": normalized_effective_tools,
                "parent_tool_scope": _normalize_name_list(parent_tool_scope),
                "worker_isolation_mode": True,
                "timeout_seconds": max(5, int(timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS)),
                "priority": int(priority or 1),
                "project_root": _session_project_root(),
            },
        )
        session_repo.create_branch_session(
            parent_thread_id=parent_thread_id,
            branch_thread_id=worker_thread_id,
            branch_from_message_uid="",
            title=f"{parent_thread_id} {normalized_role} worker",
        )
        session_repo.upsert_session(
            thread_id=worker_thread_id,
            display_name=worker_thread_id,
            status="idle",
            log_file=build_log_file_path(worker_thread_id),
        )
        audit_logger.log_event(
            thread_id=parent_thread_id,
            event="worker_spawned",
            content=f"spawned worker {worker_record['worker_id']} role={normalized_role}",
            worker_id=worker_record["worker_id"],
        )
        future = self._executor.submit(
            self._run_worker,
            worker_record["worker_id"],
            parent_thread_id,
            worker_thread_id,
            normalized_role,
            str(task or ""),
            normalized_effective_tools or normalized_allowed_tools,
            list(write_scope or []),
            str(context_brief or "").strip(),
            str(deliverables or ""),
            max(5, int(timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS)),
        )
        with self._lock:
            self._futures[worker_record["worker_id"]] = future
        return worker_record

    def submit_workers_batch(
        self,
        *,
        parent_thread_id: str,
        parent_turn_id: str,
        workers: list[dict[str, Any]],
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_batch_id = str(batch_id or "").strip() or f"batch-{uuid.uuid4().hex[:10]}"
        audit_logger.log_event(
            thread_id=parent_thread_id,
            event="worker_batch_spawned",
            content=f"spawning worker batch {normalized_batch_id} count={len(workers)}",
        )
        submitted: list[dict[str, Any]] = []
        for index, worker_args in enumerate(workers):
            worker = self.submit_worker(
                parent_thread_id=parent_thread_id,
                parent_turn_id=parent_turn_id,
                role=str(worker_args.get("role", "explore") or "explore"),
                task=str(worker_args.get("task", "") or ""),
                allowed_tools=list(worker_args.get("allowed_tools", []) or []),
                toolsets=list(worker_args.get("toolsets", []) or []),
                effective_tools=list(worker_args.get("effective_tools", []) or []),
                write_scope=list(worker_args.get("write_scope", []) or []),
                context_brief=str(worker_args.get("context_brief", "") or ""),
                deliverables=str(worker_args.get("deliverables", "") or ""),
                timeout_seconds=max(5, int(worker_args.get("timeout_seconds") or WORKER_DEFAULT_TIMEOUT_SECONDS)),
                priority=max(1, int(worker_args.get("priority") or 1)),
                batch_id=normalized_batch_id,
                task_index=int(worker_args.get("task_index", index) or index),
                parent_tool_scope=list(worker_args.get("parent_tool_scope", []) or []),
            )
            submitted.append(worker)
        return {
            "batch_id": normalized_batch_id,
            "workers": submitted,
        }

    def _run_worker(
        self,
        worker_id: str,
        parent_thread_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        allowed_tools: list[str],
        write_scope: list[str],
        context_brief: str,
        deliverables: str,
        timeout_seconds: int,
    ) -> None:
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        session = session_repo.get_session(parent_thread_id) or {}
        provider = str(session.get("provider", "") or os.getenv("DEFAULT_PROVIDER", "aliyun"))
        model = str(session.get("model", "") or os.getenv("DEFAULT_MODEL", "glm-5"))
        worker_record = worker_repo.get_worker_run(worker_id) or {}
        worker_metadata = _parse_record_metadata(worker_record)
        project_root = str(worker_metadata.get("project_root", "") or "").strip()
        batch_id = str(worker_metadata.get("batch_id", "") or "")
        task_index = int(worker_metadata.get("task_index", 0) or 0)
        holder_context = None
        if role == "implement" and write_scope and project_root:
            holder_context = get_project_path_lock_manager().acquire(
                holder_id=worker_id,
                project_root=project_root,
                write_scope=write_scope,
            )
        worker_repo.update_worker_run(worker_id, status="running", started=True)
        session_repo.touch_session(worker_thread_id, status="active")
        if holder_context is None:
            self._run_worker_with_context(
                worker_id,
                parent_thread_id,
                worker_thread_id,
                role,
                task,
                context_brief,
                deliverables,
                allowed_tools,
                provider,
                model,
                project_root,
                timeout_seconds,
                batch_id,
                task_index,
            )
            return
        with holder_context:
            self._run_worker_with_context(
                worker_id,
                parent_thread_id,
                worker_thread_id,
                role,
                task,
                context_brief,
                deliverables,
                allowed_tools,
                provider,
                model,
                project_root,
                timeout_seconds,
                batch_id,
                task_index,
            )

    def _run_worker_with_context(
        self,
        worker_id: str,
        parent_thread_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        context_brief: str,
        deliverables: str,
        allowed_tools: list[str],
        provider: str,
        model: str,
        project_root: str,
        timeout_seconds: int,
        batch_id: str,
        task_index: int,
    ) -> None:
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        set_active_thread_id(worker_thread_id)
        set_active_worker_id(worker_id)
        set_active_program_run_id("")
        set_active_tool_scope_names(allowed_tools)
        try:
            result = asyncio.run(
                self._run_worker_async(
                    worker_id=worker_id,
                    worker_thread_id=worker_thread_id,
                    role=role,
                    task=task,
                    context_brief=context_brief,
                    deliverables=deliverables,
                    allowed_tools=allowed_tools,
                    provider=provider,
                    model=model,
                    project_root=project_root,
                    timeout_seconds=timeout_seconds,
                    batch_id=batch_id,
                    task_index=task_index,
                )
            )
            final_status = str(result.get("status", "") or "completed")
            worker_repo.update_worker_run(
                worker_id,
                status=final_status,
                result_summary=result,
                metadata={"project_root": project_root},
                finished=True,
            )
            session_repo.touch_session(worker_thread_id, status="idle")
            session_repo.enqueue_inbox_event(
                thread_id=parent_thread_id,
                event_type="worker_result",
                payload=_compact_worker_result_payload(
                    worker_id=worker_id,
                    status=final_status,
                    result_summary=result,
                ),
                status="pending" if final_status == "completed" else final_status,
            )
            event_name = "worker_completed" if final_status == "completed" else f"worker_{final_status}"
            audit_logger.log_event(
                thread_id=parent_thread_id,
                event=event_name,
                content=f"worker {worker_id} {final_status}",
                worker_id=worker_id,
            )
        except WorkerCancelledError as exc:
            result = dict(exc.summary)
            result["status"] = "cancelled"
            worker_repo.update_worker_run(
                worker_id,
                status="cancelled",
                result_summary=result,
                metadata={"project_root": project_root},
                finished=True,
            )
            session_repo.touch_session(worker_thread_id, status="idle")
            session_repo.enqueue_inbox_event(
                thread_id=parent_thread_id,
                event_type="worker_result",
                payload=_compact_worker_result_payload(
                    worker_id=worker_id,
                    status="cancelled",
                    result_summary=result,
                ),
                status="cancelled",
            )
            audit_logger.log_event(
                thread_id=parent_thread_id,
                event="worker_cancelled",
                content=f"worker {worker_id} cancelled",
                worker_id=worker_id,
            )
        except WorkerTimeoutError as exc:
            result = dict(exc.summary)
            result["status"] = "timeout"
            if not result.get("blocking_issue"):
                result["blocking_issue"] = f"worker timed out after {timeout_seconds}s"
            worker_repo.update_worker_run(
                worker_id,
                status="timeout",
                result_summary=result,
                error={"message": result["blocking_issue"]},
                metadata={"project_root": project_root},
                finished=True,
            )
            session_repo.touch_session(worker_thread_id, status="idle")
            session_repo.enqueue_inbox_event(
                thread_id=parent_thread_id,
                event_type="worker_result",
                payload=_compact_worker_result_payload(
                    worker_id=worker_id,
                    status="timeout",
                    result_summary=result,
                    error_message=result["blocking_issue"],
                ),
                status="timeout",
            )
            audit_logger.log_event(
                thread_id=parent_thread_id,
                event="worker_timeout",
                content=f"worker {worker_id} timed out",
                worker_id=worker_id,
            )
        except Exception as exc:
            error_payload = _partial_worker_result_from_record(worker_id, status="failed", error_message=str(exc))
            worker_repo.update_worker_run(
                worker_id,
                status="failed",
                result_summary=error_payload,
                error={"message": str(exc)},
                metadata={"project_root": project_root},
                finished=True,
            )
            session_repo.touch_session(worker_thread_id, status="idle")
            session_repo.enqueue_inbox_event(
                thread_id=parent_thread_id,
                event_type="worker_result",
                payload=_compact_worker_result_payload(
                    worker_id=worker_id,
                    status="failed",
                    result_summary=error_payload,
                    error_message=str(exc),
                ),
                status="failed",
            )
            audit_logger.log_event(
                thread_id=parent_thread_id,
                event="worker_failed",
                content=f"worker {worker_id} failed: {exc}",
                worker_id=worker_id,
            )
        finally:
            set_active_worker_id("")
            set_active_program_run_id("")
            set_active_tool_scope_names(())

    async def _run_worker_async(
        self,
        *,
        worker_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        context_brief: str,
        deliverables: str,
        allowed_tools: list[str],
        provider: str,
        model: str,
        project_root: str,
        timeout_seconds: int,
        batch_id: str = "",
        task_index: int = 0,
    ) -> dict[str, Any]:
        toolset = self._load_toolset(allowed_tools, role=role)
        summary = {
            "worker_id": worker_id,
            "status": "completed",
            "summary": "",
            "key_files": [],
            "evidence": [],
            "changed_files": [],
            "commands_run": [],
            "tests_run": [],
            "blocking_issue": "",
            "confidence": "",
            "budget_exhausted": False,
        }

        async def runner() -> dict[str, Any]:
            if role in {"explore", "verify"}:
                return await self._run_worker_direct_loop(
                    worker_id=worker_id,
                    worker_thread_id=worker_thread_id,
                    role=role,
                    task=task,
                    context_brief=context_brief,
                    deliverables=deliverables,
                    toolset=toolset,
                    provider=provider,
                    model=model,
                    project_root=project_root,
                    summary=summary,
                )
            return await self._run_worker_graph_loop(
                worker_id=worker_id,
                worker_thread_id=worker_thread_id,
                role=role,
                task=task,
                context_brief=context_brief,
                deliverables=deliverables,
                toolset=toolset,
                provider=provider,
                model=model,
                project_root=project_root,
                batch_id=batch_id,
                task_index=task_index,
                summary=summary,
            )

        try:
            return await asyncio.wait_for(runner(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            summary["status"] = "timeout"
            if not summary["blocking_issue"]:
                summary["blocking_issue"] = f"worker timed out after {timeout_seconds}s"
            raise WorkerTimeoutError(summary) from exc

    async def _run_worker_graph_loop(
        self,
        *,
        worker_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        context_brief: str,
        deliverables: str,
        toolset: list[Any],
        provider: str,
        model: str,
        project_root: str,
        batch_id: str,
        task_index: int,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_writer = get_conversation_writer()
        worker_prompt = self._build_graph_worker_prompt(
            role=role,
            task=task,
            context_brief=context_brief,
            deliverables=deliverables,
        )
        async def graph_runner() -> dict[str, Any]:
            async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
                from ..agent import create_agent_app

                app = create_agent_app(provider_name=provider, model_name=model, tools=toolset, checkpointer=memory)
                turn_id = f"worker-{uuid.uuid4()}"
                user_message = HumanMessage(content=worker_prompt, id=f"{turn_id}:user")
                conversation_writer.append_messages(
                    thread_id=worker_thread_id,
                    turn_id=turn_id,
                    messages=[user_message],
                    node_name="worker_input",
                    route="input",
                )
                inputs = {
                    "messages": [user_message],
                    "permission_mode": "auto" if role in {"implement", "verify"} else "plan",
                    "slow_execution_mode": "autonomous",
                    "current_project_path": project_root,
                    "worker_isolation_mode": True,
                    "parent_thread_id": worker_thread_id.rsplit("-worker-", 1)[0],
                    "batch_id": batch_id,
                    "task_index": task_index,
                }
                async for event in app.astream(inputs, config={"configurable": {"thread_id": worker_thread_id, "turn_id": turn_id}}, stream_mode="updates"):
                    if _worker_cancel_requested(worker_id):
                        summary["status"] = "cancelled"
                        if not summary["blocking_issue"]:
                            summary["blocking_issue"] = "worker cancellation requested"
                        raise WorkerCancelledError(summary)
                    for node_name, node_data in event.items():
                        node_messages = node_data.get("messages") if isinstance(node_data, dict) else None
                        if isinstance(node_messages, list) and node_messages:
                            conversation_writer.append_messages(
                                thread_id=worker_thread_id,
                                turn_id=turn_id,
                                messages=node_messages,
                                node_name=node_name,
                                route=str(node_data.get("route", "")),
                            )
                            for message in node_messages:
                                if isinstance(message, ToolMessage):
                                    payload = _tool_payload_json(message)
                                    tool_name = str(getattr(message, "name", "") or "")
                                    if payload and payload.get("ok", False):
                                        if payload.get("changed_paths"):
                                            for path_value in payload.get("changed_paths", []):
                                                if path_value and path_value not in summary["changed_files"]:
                                                    summary["changed_files"].append(path_value)
                                        if payload.get("path") and payload["path"] not in summary["changed_files"]:
                                            summary["changed_files"].append(payload["path"])
                                        if tool_name == "run_project_command" and payload.get("command"):
                                            summary["commands_run"].append(payload["command"])
                                        if tool_name == "run_project_tests" and payload.get("command"):
                                            summary["tests_run"].append(payload["command"])
                                    elif payload and not summary["blocking_issue"]:
                                        summary["blocking_issue"] = str(payload.get("message", "") or "")
                        if isinstance(node_data, dict):
                            final_answer = str(node_data.get("final_answer", "") or "").strip()
                            if final_answer:
                                summary["summary"] = final_answer
                    if _worker_cancel_requested(worker_id):
                        summary["status"] = "cancelled"
                        if not summary["blocking_issue"]:
                            summary["blocking_issue"] = "worker cancellation requested"
                        raise WorkerCancelledError(summary)
                return summary

        return await graph_runner()

    def _build_graph_worker_prompt(self, *, role: str, task: str, context_brief: str, deliverables: str) -> str:
        context_brief_text = str(context_brief or "").strip()
        context_brief_block = f"[Context Brief]\n{context_brief_text}\n\n" if context_brief_text else ""
        return (
            f"[Worker Role]\n{role}\n\n"
            f"{context_brief_block}"
            f"[Task]\n{task}\n\n"
            f"[Deliverables]\n{deliverables or '请聚焦当前子任务，输出结构化结果。'}\n\n"
            "约束：你是隔离的子 worker，只能完成当前子任务；不能假设父对话上下文；"
            "不允许再委派其他 worker，不允许调用 execute_tool_program，不允许写长期记忆。"
            "最终结果必须说明 summary、changed files、commands、tests、blocking issue。"
        )

    def _build_direct_worker_messages(
        self,
        *,
        role: str,
        task: str,
        context_brief: str,
        deliverables: str,
        budget: dict[str, int],
    ) -> list[Any]:
        system_prompt = (
            "你是 MortyClaw 的隔离 worker scout。你只完成当前子任务，不继承父对话全文，不写长期记忆，不再委派。"
            "优先快速定位关键文件、关键函数、调用链证据和风险；证据足够时立即停止，不做全项目漫游。"
            "只能通过已绑定工具获取信息，不得臆测，也不得要求主 agent 替你继续普通搜索。"
            "最终必须只输出一个 JSON object。"
        )
        output_shape = {
            "summary": "一句到数句结论",
            "key_files": ["相关文件路径"],
            "evidence": ["关键证据，包含文件/符号/简短原因"],
            "changed_files": [],
            "commands_run": [],
            "tests_run": [],
            "blocking_issue": "",
            "confidence": "high|medium|low",
        }
        user_prompt = (
            f"[Worker Role]\n{role}\n\n"
            f"[Context Brief]\n{str(context_brief or '').strip()}\n\n"
            f"[Task]\n{str(task or '').strip()}\n\n"
            f"[Deliverables]\n{str(deliverables or '输出当前子任务的关键文件、证据、结论和风险。').strip()}\n\n"
            "[Budget]\n"
            f"- 最多 {budget['llm_rounds']} 轮 LLM\n"
            f"- 最多 {budget['tool_calls']} 次工具调用\n"
            f"- 最多 {budget['file_reads']} 次文件读取\n\n"
            "[Output JSON Shape]\n"
            f"{json.dumps(output_shape, ensure_ascii=False)}"
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    async def _run_worker_direct_loop(
        self,
        *,
        worker_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        context_brief: str,
        deliverables: str,
        toolset: list[Any],
        provider: str,
        model: str,
        project_root: str,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        from ..provider import get_provider

        conversation_writer = get_conversation_writer()
        budget = dict(WORKER_DIRECT_BUDGETS.get(role, WORKER_DIRECT_BUDGETS["explore"]))
        turn_id = f"worker-direct-{uuid.uuid4()}"
        messages = self._build_direct_worker_messages(
            role=role,
            task=task,
            context_brief=context_brief,
            deliverables=deliverables,
            budget=budget,
        )
        conversation_writer.append_messages(
            thread_id=worker_thread_id,
            turn_id=turn_id,
            messages=messages,
            node_name="worker_direct_input",
            route="worker_direct",
        )
        llm = get_provider(provider_name=provider, model_name=model)
        llm_with_tools = llm.bind_tools(toolset)
        tool_by_name = {str(getattr(tool, "name", "") or ""): tool for tool in toolset}
        tool_call_count = 0
        file_read_count = 0
        last_ai_text = ""
        max_llm_rounds = max(1, int(budget["llm_rounds"]))
        for round_index in range(max_llm_rounds):
            if _worker_cancel_requested(worker_id):
                summary["status"] = "cancelled"
                summary["blocking_issue"] = summary["blocking_issue"] or "worker cancellation requested"
                raise WorkerCancelledError(summary)
            response = await asyncio.to_thread(llm_with_tools.invoke, messages)
            messages.append(response)
            conversation_writer.append_messages(
                thread_id=worker_thread_id,
                turn_id=turn_id,
                messages=[response],
                node_name="worker_direct_llm",
                route="worker_direct",
            )
            last_ai_text = _message_text(response).strip() or last_ai_text
            tool_calls = list(getattr(response, "tool_calls", []) or [])
            if not tool_calls:
                parsed = _extract_json_object(last_ai_text)
                if parsed:
                    _merge_summary_payload(summary, parsed)
                elif last_ai_text:
                    summary["summary"] = last_ai_text
                return summary
            if tool_call_count >= budget["tool_calls"]:
                summary["budget_exhausted"] = True
                summary["blocking_issue"] = summary["blocking_issue"] or "worker direct loop tool budget exhausted"
                break
            for tool_call in tool_calls:
                if _worker_cancel_requested(worker_id):
                    summary["status"] = "cancelled"
                    summary["blocking_issue"] = summary["blocking_issue"] or "worker cancellation requested"
                    raise WorkerCancelledError(summary)
                tool_name = str(tool_call.get("name") or "").strip()
                tool = tool_by_name.get(tool_name)
                args = dict(tool_call.get("args") or {})
                if project_root and "project_root" not in args:
                    args["project_root"] = project_root
                if tool_name == "read_project_file":
                    if file_read_count >= budget["file_reads"]:
                        summary["budget_exhausted"] = True
                        tool_result = "worker direct loop file read budget exhausted"
                    else:
                        file_read_count += 1
                        tool_result = await self._invoke_worker_tool(tool, args)
                elif tool is None:
                    tool_result = f"worker direct loop blocked unknown tool: {tool_name}"
                else:
                    tool_result = await self._invoke_worker_tool(tool, args)
                tool_call_count += 1
                if tool_name == "run_project_command" and args.get("command"):
                    _append_unique_text(summary["commands_run"], args.get("command"))
                if tool_name == "run_project_tests" and args.get("command"):
                    _append_unique_text(summary["tests_run"], args.get("command"))
                self._update_summary_from_tool_result(summary, tool_name=tool_name, args=args, result=tool_result)
                tool_message = ToolMessage(
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=str(tool_call.get("id") or f"worker-tool-{uuid.uuid4().hex[:8]}"),
                )
                messages.append(tool_message)
                conversation_writer.append_messages(
                    thread_id=worker_thread_id,
                    turn_id=turn_id,
                    messages=[tool_message],
                    node_name="worker_direct_tool",
                    route="worker_direct",
                )
                if tool_call_count >= budget["tool_calls"]:
                    summary["budget_exhausted"] = True
                    break
            if summary.get("budget_exhausted"):
                break
            if round_index == max_llm_rounds - 1:
                summary["budget_exhausted"] = True
                summary["blocking_issue"] = summary["blocking_issue"] or "worker direct loop llm round budget exhausted"
        if last_ai_text and not summary.get("summary"):
            parsed = _extract_json_object(last_ai_text)
            if parsed:
                _merge_summary_payload(summary, parsed)
            else:
                summary["summary"] = last_ai_text
        if not summary.get("summary"):
            evidence_preview = "; ".join(summary.get("evidence", [])[:3])
            summary["summary"] = evidence_preview or "worker direct loop stopped before producing a final summary."
        return summary

    async def _invoke_worker_tool(self, tool: Any, args: dict[str, Any]) -> str:
        if tool is None:
            return "worker direct loop blocked unavailable tool"
        try:
            result = await asyncio.to_thread(tool.invoke, args)
        except Exception as exc:
            return json.dumps({"ok": False, "message": str(exc), "error_kind": "TOOL_EXCEPTION"}, ensure_ascii=False)
        return str(result)

    def _update_summary_from_tool_result(self, summary: dict[str, Any], *, tool_name: str, args: dict[str, Any], result: str) -> None:
        payload = None
        text = str(result or "")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = None
        if payload and payload.get("ok") is False and not summary.get("blocking_issue"):
            summary["blocking_issue"] = str(payload.get("message") or "")
        if payload:
            for key in ("changed_paths", "changed_files"):
                for path_value in payload.get(key, []) or []:
                    _append_unique_text(summary["changed_files"], path_value)
            if payload.get("path"):
                _append_unique_text(summary["changed_files"], payload.get("path"))
        if tool_name == "read_project_file":
            filepath = str(args.get("filepath") or "")
            if filepath:
                _append_unique_text(summary["key_files"], filepath)
                _append_unique_text(summary["evidence"], f"{filepath}: read")
        elif tool_name == "search_project_code":
            query = str(args.get("query") or "")
            if query:
                _append_unique_text(summary["evidence"], f"search `{query}`: {_truncate_text(text, 240)}")
        elif tool_name == "show_git_diff":
            _append_unique_text(summary["evidence"], f"git diff: {_truncate_text(text, 240)}")

    def list_workers(self, *, parent_thread_id: str, status_filter: str = "") -> list[dict[str, Any]]:
        statuses = tuple(item.strip() for item in status_filter.split(",") if item.strip()) if status_filter else None
        records = get_worker_run_repository().list_worker_runs(parent_thread_id=parent_thread_id, statuses=statuses)
        compacted = [_compact_worker_record(record, include_result=False) for record in records]
        return [record for record in compacted if record]

    def wait_workers(
        self,
        *,
        parent_thread_id: str,
        worker_ids: list[str] | None,
        timeout_seconds: int,
        return_partial: bool,
    ) -> dict[str, Any]:
        worker_repo = get_worker_run_repository()
        deadline = time.time() + max(1, int(timeout_seconds or 1))
        requested_ids = [str(item or "").strip() for item in (worker_ids or []) if str(item or "").strip()]
        while time.time() < deadline:
            records = [
                worker_repo.get_worker_run(worker_id)
                for worker_id in requested_ids
            ] if requested_ids else worker_repo.list_worker_runs(parent_thread_id=parent_thread_id, limit=100)
            normalized = [record for record in records if record]
            final = [record for record in normalized if str(record.get("status", "") or "") in FINAL_WORKER_STATUSES]
            if normalized and len(final) == len(normalized):
                break
            time.sleep(0.2)

        records = [
            worker_repo.get_worker_run(worker_id)
            for worker_id in requested_ids
        ] if requested_ids else worker_repo.list_worker_runs(parent_thread_id=parent_thread_id, limit=100)
        normalized = [record for record in (_with_parsed_metadata(record) for record in records) if record]
        completed = [record for record in normalized if str(record.get("status", "") or "") in FINAL_WORKER_STATUSES]
        if not return_partial and normalized and len(completed) != len(normalized):
            return {
                "success": False,
                "status": "waiting",
                "workers": [],
                "message": "仍有 worker 在运行中。",
                "retry_policy": "do_not_auto_retry",
            }
        compact_workers = [
            record
            for record in (
                _compact_worker_record(record, include_result=True)
                for record in normalized
            )
            if record
        ]
        return {
            "success": True,
            "status": "completed" if normalized and len(completed) == len(normalized) else "partial",
            "workers": compact_workers,
            "completed_count": sum(1 for record in compact_workers if record.get("status") == "completed"),
            "failed_count": sum(1 for record in compact_workers if record.get("status") == "failed"),
            "timeout_count": sum(1 for record in compact_workers if record.get("status") == "timeout"),
            "cancelled_count": sum(1 for record in compact_workers if record.get("status") == "cancelled"),
            "retry_policy": "do_not_auto_retry",
            "debug_only_tools": ["list_subagents", "wait_subagents"],
            "next_action_hint": "Use the returned worker summaries directly. Do not call list_subagents/wait_subagents for normal result retrieval; only do targeted follow-up reads/searches for blocking gaps.",
        }

    def cancel_worker(self, worker_id: str, *, reason: str = "") -> dict[str, Any]:
        worker_repo = get_worker_run_repository()
        record = worker_repo.get_worker_run(worker_id)
        if record is None:
            return {"success": False, "message": f"未找到 worker：{worker_id}"}
        status = str(record.get("status", "") or "").strip()
        if status in FINAL_WORKER_STATUSES:
            return {"success": True, "worker_id": worker_id, "status": status, "message": f"worker {worker_id} 已结束。"}
        with self._lock:
            future = self._futures.get(worker_id)
        if future is not None and future.cancel():
            worker_repo.update_worker_run(
                worker_id,
                status="cancelled",
                metadata={
                    "cancelled": True,
                    "cancel_reason": str(reason or ""),
                    "cancel_requested_at": _now_iso(),
                    "cancellation_mode": "future_cancel",
                },
                finished=True,
            )
            return {"success": True, "worker_id": worker_id, "status": "cancelled", "message": f"已取消尚未开始的 worker：{worker_id}"}
        worker_repo.update_worker_run(
            worker_id,
            status="cancelling",
            metadata={
                "cancel_requested": True,
                "cancel_reason": str(reason or ""),
                "cancel_requested_at": _now_iso(),
                "cancellation_mode": "cooperative_checkpoint",
            },
        )
        audit_logger.log_event(
            thread_id=str(record.get("parent_thread_id", "") or "system_default"),
            event="worker_cancel_requested",
            content=f"worker {worker_id} cancellation requested (cooperative_checkpoint)",
            worker_id=worker_id,
        )
        return {
            "success": True,
            "worker_id": worker_id,
            "status": "cancelling",
            "message": f"worker {worker_id} 已在运行，已记录取消请求；将在下一个安全检查点协作式停止。",
        }

    def cancel_workers(self, worker_ids: list[str], *, reason: str = "") -> dict[str, Any]:
        results = [
            self.cancel_worker(worker_id, reason=reason)
            for worker_id in _normalize_name_list(worker_ids)
        ]
        return {
            "success": all(bool(item.get("success")) for item in results) if results else False,
            "results": results,
        }


_default_worker_supervisor = AsyncWorkerSupervisor()


def get_worker_supervisor() -> AsyncWorkerSupervisor:
    return _default_worker_supervisor
