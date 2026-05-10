from __future__ import annotations

import json
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field

from ...config import ENABLE_WORKER_SUBAGENTS, WORKER_DEFAULT_TIMEOUT_SECONDS
from ...logger import audit_logger
from ...runtime.worker_supervisor import get_worker_supervisor
from ...runtime_context import get_active_thread_id, get_active_tool_scope_names


WORKER_ROLES = {"explore", "verify", "implement"}


class DelegateSubagentTask(BaseModel):
    task: str = Field(description="独立子任务说明；必须边界清晰，可由 worker 单独完成。")
    role: Literal["explore", "verify", "implement"] = Field(
        default="explore",
        description="worker 角色：explore=只读分析；verify=读取并运行验证；implement=局部实现且必须声明 write_scope。",
    )
    toolsets: list[str] = Field(
        default_factory=list,
        description='建议工具集。只读分析用 ["project_read"]；验证用 ["project_read","project_verify"]；实现可用 project_full 或 project_read/project_write/project_verify。',
    )
    allowed_tools: list[str] = Field(default_factory=list, description="额外允许工具名；通常留空，系统会按 role/toolsets 和父 agent 权限裁剪。")
    write_scope: list[str] = Field(default_factory=list, description="写型 implement worker 的最小允许写入路径范围；非写任务留空。")
    context_brief: str = Field(
        description="压缩背景：说明父任务目标、当前子任务背景、关键约束、已知路径/关键词和成功标准；不要复制父对话全文。worker 不继承父对话和长期记忆，缺失信息会降低结果质量。",
    )
    deliverables: str = Field(description="该 worker 必须返回的交付物格式或检查点，例如关键文件、调用链路、结论、风险、命令/测试结果、仍需主 agent 核对的问题。")
    timeout_seconds: int | None = Field(default=None, description="该 worker 超时时间；留空使用 batch_timeout_seconds。")
    priority: int = Field(default=1, description="任务优先级，数字越大越重要；默认 1。")


class DelegateSubagentsArgs(BaseModel):
    tasks: list[DelegateSubagentTask] = Field(
        description="要委派的独立 worker 子任务。适用于互不依赖且各自有实际搜索/读取/验证成本，或会产生大量中间材料的复杂分支任务。",
    )
    batch_timeout_seconds: int = Field(default=WORKER_DEFAULT_TIMEOUT_SECONDS, description="阻塞式委派的等待超时时间。")


def _json_error(message: str, **extra: Any) -> str:
    payload = {"success": False, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _normalize_list(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_worker_ids_input(worker_ids: list[str] | tuple[str, ...] | set[str] | str | None) -> list[str]:
    if worker_ids is None:
        return []
    if isinstance(worker_ids, str):
        text = worker_ids.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return _normalize_list(tuple(str(item or "") for item in parsed))
        if isinstance(parsed, str):
            return _normalize_list((parsed,))
        if "," in text:
            return _normalize_list(tuple(item.strip().strip('"').strip("'") for item in text.split(",")))
        return _normalize_list((text,))
    if isinstance(worker_ids, (list, tuple, set)):
        return _normalize_list(tuple(str(item or "") for item in worker_ids))
    return _normalize_list((str(worker_ids or ""),))


def _normalize_worker_tasks_input(tasks: list[dict[str, Any]] | tuple[Any, ...] | str | None) -> tuple[list[Any], str]:
    if tasks is None:
        return [], ""
    if isinstance(tasks, str):
        text = tasks.strip()
        if not text:
            return [], ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return [], f"tasks 必须是数组；收到字符串但无法解析为 JSON array：{exc.msg}"
        if not isinstance(parsed, list):
            return [], "tasks JSON 字符串必须解析为 array。"
        return parsed, ""
    if isinstance(tasks, (list, tuple)):
        return list(tasks), ""
    return [], "tasks 必须是 array。"


def _normalize_worker_task(
    task_payload: dict[str, Any],
    *,
    task_index: int,
    batch_timeout_seconds: int,
    legacy_allowed_only: bool,
) -> tuple[dict[str, Any] | None, str]:
    if hasattr(task_payload, "model_dump"):
        task_payload = task_payload.model_dump()
    if not isinstance(task_payload, dict):
        return None, f"tasks[{task_index}] 必须是 object。"
    task_text = str(task_payload.get("task", "") or "").strip()
    if not task_text:
        return None, f"tasks[{task_index}].task 不能为空。"
    role = str(task_payload.get("role", "explore") or "explore").strip().lower() or "explore"
    if role not in WORKER_ROLES:
        return None, f"tasks[{task_index}] 不支持的 worker role：{role}"
    write_scope = _normalize_list(task_payload.get("write_scope") or [])
    if role == "implement" and not write_scope:
        return None, f"tasks[{task_index}] implement worker 必须显式声明 write_scope。"
    context_brief = str(task_payload.get("context_brief", "") or "").strip()
    deliverables = str(task_payload.get("deliverables", "") or "").strip()
    if not legacy_allowed_only:
        if not context_brief:
            return None, f"tasks[{task_index}].context_brief 不能为空；请说明父任务目标、当前背景、关键约束和成功标准。"
        if not deliverables:
            return None, f"tasks[{task_index}].deliverables 不能为空；请说明 worker 必须返回的交付物。"
    allowed_tools = _normalize_list(task_payload.get("allowed_tools") or [])
    toolsets = _normalize_list(task_payload.get("toolsets") or [])
    timeout_seconds = max(5, int(task_payload.get("timeout_seconds") or batch_timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS))
    priority = max(1, int(task_payload.get("priority") or 1))
    supervisor = get_worker_supervisor()
    resolution = supervisor.resolve_worker_tools(
        role=role,
        allowed_tools=allowed_tools,
        toolsets=toolsets,
        parent_tool_scope=get_active_tool_scope_names(default=()),
        legacy_allowed_only=legacy_allowed_only,
    )
    if resolution["invalid_toolsets"]:
        return None, f"tasks[{task_index}] 包含未知 worker toolsets：{', '.join(resolution['invalid_toolsets'])}"
    if not resolution["effective_tools"]:
        return None, f"tasks[{task_index}] 没有可用工具；请检查 role/toolsets/allowed_tools 与当前工具权限。"
    return {
        "task": task_text,
        "role": role,
        "allowed_tools": allowed_tools,
        "toolsets": toolsets,
        "write_scope": write_scope,
        "context_brief": context_brief,
        "deliverables": deliverables,
        "timeout_seconds": timeout_seconds,
        "priority": priority,
        "task_index": task_index,
        "requested_toolsets": resolution["requested_toolsets"],
        "effective_tools": resolution["effective_tools"],
        "parent_tool_scope": resolution["parent_tool_scope"],
    }, ""


def delegate_subagent_impl(
    *,
    task: str,
    role: str,
    allowed_tools: list[str] | None,
    toolsets: list[str] | None = None,
    write_scope: list[str] | None,
    context_brief: str = "",
    deliverables: str,
    timeout_seconds: int,
    priority: int,
) -> str:
    if not ENABLE_WORKER_SUBAGENTS:
        return _json_error("当前环境未启用 worker sub-agent。")
    normalized, error = _normalize_worker_task(
        {
            "task": task,
            "role": role,
            "allowed_tools": allowed_tools,
            "toolsets": toolsets,
            "write_scope": write_scope,
            "context_brief": context_brief,
            "deliverables": deliverables,
            "timeout_seconds": timeout_seconds,
            "priority": priority,
        },
        task_index=0,
        batch_timeout_seconds=timeout_seconds,
        legacy_allowed_only=not bool(toolsets),
    )
    if error or normalized is None:
        return _json_error(error)
    worker = get_worker_supervisor().submit_worker(
        parent_thread_id=get_active_thread_id(default="system_default"),
        parent_turn_id="",
        role=normalized["role"],
        task=normalized["task"],
        allowed_tools=normalized["allowed_tools"],
        toolsets=normalized["toolsets"],
        effective_tools=normalized["effective_tools"],
        write_scope=normalized["write_scope"],
        context_brief=normalized["context_brief"],
        deliverables=normalized["deliverables"],
        timeout_seconds=normalized["timeout_seconds"],
        priority=normalized["priority"],
        parent_tool_scope=normalized["parent_tool_scope"],
    )
    return json.dumps(
        {
            "success": True,
            "worker_id": worker["worker_id"],
            "worker_thread_id": worker["worker_thread_id"],
            "role": worker["role"],
            "status": worker["status"],
            "requested_toolsets": normalized["requested_toolsets"],
            "effective_tools": normalized["effective_tools"],
        },
        ensure_ascii=False,
    )


def delegate_subagents_impl(
    *,
    tasks: list[dict[str, Any]] | tuple[Any, ...] | str | None,
    batch_timeout_seconds: int = WORKER_DEFAULT_TIMEOUT_SECONDS,
) -> str:
    if not ENABLE_WORKER_SUBAGENTS:
        return _json_error("当前环境未启用 worker sub-agent。")
    task_items, input_error = _normalize_worker_tasks_input(tasks)
    if input_error:
        audit_logger.log_event(
            thread_id=get_active_thread_id(default="system_default"),
            event="worker_batch_rejected",
            content=input_error,
        )
        return _json_error(input_error)
    if not task_items:
        return _json_error("tasks 不能为空。")
    normalized_tasks: list[dict[str, Any]] = []
    for index, task_payload in enumerate(task_items):
        normalized, error = _normalize_worker_task(
            task_payload,
            task_index=index,
            batch_timeout_seconds=batch_timeout_seconds,
            legacy_allowed_only=False,
        )
        if error or normalized is None:
            audit_logger.log_event(
                thread_id=get_active_thread_id(default="system_default"),
                event="worker_batch_rejected",
                content=error,
            )
            return _json_error(error, rejected_task_index=index)
        normalized_tasks.append(normalized)
    batch_id = f"batch-{uuid.uuid4().hex[:10]}"
    batch = get_worker_supervisor().submit_workers_batch(
        parent_thread_id=get_active_thread_id(default="system_default"),
        parent_turn_id="",
        workers=normalized_tasks,
        batch_id=batch_id,
    )
    worker_ids = [str(item.get("worker_id", "") or "") for item in batch["workers"] if item.get("worker_id")]
    wait_result = get_worker_supervisor().wait_workers(
        parent_thread_id=get_active_thread_id(default="system_default"),
        worker_ids=worker_ids,
        timeout_seconds=max(1, int(batch_timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS)),
        return_partial=True,
    )
    workers = list(wait_result.get("workers") or [])
    status_counts = {
        "completed": sum(1 for item in workers if str(item.get("status", "") or "") == "completed"),
        "failed": sum(1 for item in workers if str(item.get("status", "") or "") == "failed"),
        "timeout": sum(1 for item in workers if str(item.get("status", "") or "") == "timeout"),
        "cancelled": sum(1 for item in workers if str(item.get("status", "") or "") == "cancelled"),
    }
    waiting_count = sum(
        1
        for item in workers
        if str(item.get("status", "") or "") not in {"completed", "failed", "timeout", "cancelled"}
    )
    if waiting_count:
        batch_status = "timeout"
    elif workers and status_counts["completed"] == len(workers):
        batch_status = "completed"
    elif status_counts["completed"]:
        batch_status = "partial"
    elif status_counts["timeout"]:
        batch_status = "timeout"
    elif status_counts["failed"] or status_counts["cancelled"]:
        batch_status = "failed"
    else:
        batch_status = str(wait_result.get("status", "") or "failed")
    return json.dumps(
        {
            "success": True,
            "status": batch_status,
            "batch_id": batch["batch_id"],
            "worker_ids": worker_ids,
            "workers": workers,
            "completed_count": status_counts["completed"],
            "failed_count": status_counts["failed"],
            "timeout_count": status_counts["timeout"] + waiting_count,
            "cancelled_count": status_counts["cancelled"],
            "retry_policy": "do_not_auto_retry",
            "debug_only_tools": ["list_subagents", "wait_subagents"],
            "next_action_hint": (
                "直接基于 workers 中的 compact summaries 汇总。不要调用 list_subagents/wait_subagents 获取常规结果；"
                "如有 failed/timeout/cancelled worker，只针对 blocking_issue 做目标文件级补查，不要自动再次委派。"
            ),
        },
        ensure_ascii=False,
    )


def list_subagents_impl(*, status_filter: str = "") -> str:
    workers = get_worker_supervisor().list_workers(
        parent_thread_id=get_active_thread_id(default="system_default"),
        status_filter=status_filter,
    )
    return json.dumps({"success": True, "workers": workers}, ensure_ascii=False)


def wait_subagents_impl(
    *,
    worker_ids: list[str] | str | None,
    timeout_seconds: int,
    return_partial: bool,
) -> str:
    result = get_worker_supervisor().wait_workers(
        parent_thread_id=get_active_thread_id(default="system_default"),
        worker_ids=_normalize_worker_ids_input(worker_ids),
        timeout_seconds=max(1, int(timeout_seconds or 1)),
        return_partial=bool(return_partial),
    )
    return json.dumps(result, ensure_ascii=False)


def cancel_subagent_impl(*, worker_id: str) -> str:
    result = get_worker_supervisor().cancel_worker(str(worker_id or "").strip())
    return json.dumps(result, ensure_ascii=False)


def cancel_subagents_impl(*, worker_ids: list[str] | str | None, reason: str = "") -> str:
    result = get_worker_supervisor().cancel_workers(_normalize_worker_ids_input(worker_ids), reason=str(reason or ""))
    return json.dumps(result, ensure_ascii=False)
