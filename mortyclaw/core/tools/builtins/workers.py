from __future__ import annotations

import json
from typing import Any

from ...config import ENABLE_WORKER_SUBAGENTS, WORKER_DEFAULT_TIMEOUT_SECONDS
from ...runtime.worker_supervisor import get_worker_supervisor
from ...runtime_context import get_active_thread_id


def delegate_subagent_impl(
    *,
    task: str,
    role: str,
    allowed_tools: list[str] | None,
    write_scope: list[str] | None,
    deliverables: str,
    timeout_seconds: int,
    priority: int,
) -> str:
    if not ENABLE_WORKER_SUBAGENTS:
        return json.dumps({"success": False, "message": "当前环境未启用 worker sub-agent。"}, ensure_ascii=False)
    normalized_role = str(role or "explore").strip().lower() or "explore"
    if normalized_role not in {"explore", "verify", "implement"}:
        return json.dumps({"success": False, "message": f"不支持的 worker role：{normalized_role}"}, ensure_ascii=False)
    if normalized_role == "implement" and not list(write_scope or []):
        return json.dumps({"success": False, "message": "implement worker 必须显式声明 write_scope。"}, ensure_ascii=False)
    worker = get_worker_supervisor().submit_worker(
        parent_thread_id=get_active_thread_id(default="system_default"),
        parent_turn_id="",
        role=normalized_role,
        task=str(task or ""),
        allowed_tools=list(allowed_tools or []),
        write_scope=list(write_scope or []),
        deliverables=str(deliverables or ""),
        timeout_seconds=max(5, int(timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS)),
        priority=max(1, int(priority or 1)),
    )
    return json.dumps(
        {
            "success": True,
            "worker_id": worker["worker_id"],
            "worker_thread_id": worker["worker_thread_id"],
            "role": worker["role"],
            "status": worker["status"],
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
    worker_ids: list[str] | None,
    timeout_seconds: int,
    return_partial: bool,
) -> str:
    result = get_worker_supervisor().wait_workers(
        parent_thread_id=get_active_thread_id(default="system_default"),
        worker_ids=list(worker_ids or []),
        timeout_seconds=max(1, int(timeout_seconds or 1)),
        return_partial=bool(return_partial),
    )
    return json.dumps(result, ensure_ascii=False)


def cancel_subagent_impl(*, worker_id: str) -> str:
    result = get_worker_supervisor().cancel_worker(str(worker_id or "").strip())
    return json.dumps(result, ensure_ascii=False)

