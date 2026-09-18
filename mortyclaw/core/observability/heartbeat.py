from __future__ import annotations

import asyncio
from datetime import datetime

from ..config import PROJECT_ROOT
from ..harness import HarnessRuntime, new_turn_request
from .audit import audit_logger
from ..storage.runtime import (
    get_session_repository,
    get_task_repository,
    local_now_str,
)


def build_heartbeat_message(description: str) -> str:
    return (
        f"【系统内部心跳触发】\n"
        f"你设定的定时任务已到期，请立即主动提醒用户或执行动作。\n"
        f"任务内容：{description}"
    )


def process_due_tasks_once(
    *,
    now: str | None = None,
    limit: int = 100,
    task_repository=None,
    session_repository=None,
) -> list[dict]:
    task_repo = task_repository or get_task_repository()
    session_repo = session_repository or get_session_repository()
    task_repo.bootstrap_legacy_tasks()

    due_time = now or local_now_str()
    triggered_tasks: list[dict] = []
    for task in task_repo.list_due_tasks(now=due_time, limit=limit):
        try:
            session_repo.upsert_session(thread_id=task["thread_id"], display_name=task["thread_id"], status="idle")
            inbox_event = session_repo.enqueue_inbox_event(
                thread_id=task["thread_id"],
                event_type="heartbeat_task",
                payload={
                    "task_id": task["task_id"],
                    "content": build_heartbeat_message(task["description"]),
                    "description": task["description"],
                    "target_time": task["target_time"],
                },
            )
            task_repo.record_task_run(
                task_id=task["task_id"],
                thread_id=task["thread_id"],
                status="triggered",
                result_summary=f"inbox_event={inbox_event['event_id']}",
            )
            updated_task = task_repo.advance_after_dispatch(task["task_id"]) or task
            audit_logger.log_event(
                thread_id=task["thread_id"],
                event="system_action",
                content=(
                    f"heartbeat queued task {task['task_id']} for session {task['thread_id']} "
                    f"(status={updated_task['status']})"
                ),
            )
            triggered_tasks.append(updated_task)
        except Exception as exc:
            task_repo.mark_task_failed(task["task_id"], error_message=str(exc))
            audit_logger.log_event(
                thread_id=task["thread_id"],
                event="system_action",
                content=f"heartbeat failed to queue task {task['task_id']}: {exc}",
            )

    return triggered_tasks


async def process_due_tasks_with_harness_once(
    *,
    now: str | None = None,
    limit: int = 100,
    task_repository=None,
    session_repository=None,
    runtime: HarnessRuntime | None = None,
) -> list[dict]:
    """Execute due tasks as scheduled Harness turns and persist their user-facing result."""
    task_repo = task_repository or get_task_repository()
    session_repo = session_repository or get_session_repository()
    task_repo.bootstrap_legacy_tasks()
    owned_runtime = runtime is None
    active_runtime = runtime or HarnessRuntime()
    triggered_tasks: list[dict] = []
    try:
        await active_runtime.start()
        for task in task_repo.list_due_tasks(now=now or local_now_str(), limit=limit):
            try:
                session_repo.upsert_session(
                    thread_id=task["thread_id"],
                    display_name=task["thread_id"],
                    status="active",
                )
                result = await active_runtime.run_turn(new_turn_request(
                    thread_id=task["thread_id"],
                    text=build_heartbeat_message(task["description"]),
                    source="scheduled",
                    workspace=PROJECT_ROOT,
                ))
                event_type = "scheduled_approval" if result.approval_batch_id else "scheduled_result"
                event = session_repo.enqueue_inbox_event(
                    thread_id=task["thread_id"],
                    event_type=event_type,
                    payload={
                        "task_id": task["task_id"],
                        "content": result.final_response,
                        "approval_batch_id": result.approval_batch_id or "",
                    },
                )
                task_repo.record_task_run(
                    task_id=task["task_id"],
                    thread_id=task["thread_id"],
                    status="approval_required" if result.approval_batch_id else "completed",
                    result_summary=f"inbox_event={event['event_id']}",
                )
                updated = task_repo.advance_after_dispatch(task["task_id"]) or task
                triggered_tasks.append(updated)
                audit_logger.log_event(
                    thread_id=task["thread_id"],
                    event="system_action",
                    content=f"scheduled Harness turn completed for task {task['task_id']} ({event_type})",
                )
            except Exception as exc:
                task_repo.mark_task_failed(task["task_id"], error_message=type(exc).__name__)
                audit_logger.log_event(
                    thread_id=task["thread_id"],
                    event="system_action",
                    content=f"scheduled Harness turn failed for task {task['task_id']}: {type(exc).__name__}",
                )
    finally:
        if owned_runtime:
            await active_runtime.close()
    return triggered_tasks


async def pacemaker_loop(check_interval: int = 10):
    runtime = HarnessRuntime()
    while True:
        try:
            await process_due_tasks_with_harness_once(runtime=runtime)
            await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            break
    await runtime.close()
