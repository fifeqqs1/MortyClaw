from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import DB_PATH, WORKER_DEFAULT_TIMEOUT_SECONDS, WORKER_MAX_CONCURRENCY
from ..logger import audit_logger, build_log_file_path
from ..runtime.path_locks import get_project_path_lock_manager
from ..runtime_context import set_active_program_run_id, set_active_thread_id, set_active_worker_id
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
RESTRICTED_WORKER_TOOL_NAMES = {
    "delegate_subagent",
    "wait_subagents",
    "list_subagents",
    "cancel_subagent",
    "execute_tool_program",
    "save_user_profile",
}


def _tool_payload_json(message: ToolMessage) -> dict[str, Any] | None:
    content = str(getattr(message, "content", "") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class AsyncWorkerSupervisor:
    def __init__(self, *, max_workers: int = WORKER_MAX_CONCURRENCY):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mortyclaw-worker")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    def _load_toolset(self, allowed_tools: list[str], *, role: str):
        from ..tools.builtins import BUILTIN_TOOLS

        allowed = set(allowed_tools or []) or set(WORKER_ROLE_ALLOWED_DEFAULTS.get(role, WORKER_ROLE_ALLOWED_DEFAULTS["explore"]))
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
        allowed_tools: list[str] | None,
        write_scope: list[str] | None,
        deliverables: str,
        timeout_seconds: int,
        priority: int,
    ) -> dict[str, Any]:
        normalized_role = str(role or "explore").strip().lower() or "explore"
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        worker_thread_id = f"{parent_thread_id}-worker-{uuid.uuid4().hex[:8]}"
        worker_record = worker_repo.create_worker_run(
            parent_thread_id=parent_thread_id,
            worker_thread_id=worker_thread_id,
            parent_turn_id=parent_turn_id,
            role=normalized_role,
            goal=str(task or ""),
            allowed_tools=list(allowed_tools or []),
            write_scope=list(write_scope or []),
            tool_budget=max(1, priority or 1),
            metadata={
                "deliverables": str(deliverables or ""),
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
            list(allowed_tools or []),
            list(write_scope or []),
            str(deliverables or ""),
            max(5, int(timeout_seconds or WORKER_DEFAULT_TIMEOUT_SECONDS)),
        )
        with self._lock:
            self._futures[worker_record["worker_id"]] = future
        return worker_record

    def _run_worker(
        self,
        worker_id: str,
        parent_thread_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        allowed_tools: list[str],
        write_scope: list[str],
        deliverables: str,
        timeout_seconds: int,
    ) -> None:
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        session = session_repo.get_session(parent_thread_id) or {}
        provider = str(session.get("provider", "") or os.getenv("DEFAULT_PROVIDER", "aliyun"))
        model = str(session.get("model", "") or os.getenv("DEFAULT_MODEL", "glm-5"))
        worker_record = worker_repo.get_worker_run(worker_id) or {}
        worker_metadata_raw = str(worker_record.get("metadata_json", "") or "{}")
        try:
            worker_metadata = json.loads(worker_metadata_raw)
        except json.JSONDecodeError:
            worker_metadata = {}
        project_root = str(worker_metadata.get("project_root", "") or "").strip()
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
                deliverables,
                allowed_tools,
                provider,
                model,
                project_root,
                timeout_seconds,
            )
            return
        with holder_context:
            self._run_worker_with_context(
                worker_id,
                parent_thread_id,
                worker_thread_id,
                role,
                task,
                deliverables,
                allowed_tools,
                provider,
                model,
                project_root,
                timeout_seconds,
            )

    def _run_worker_with_context(
        self,
        worker_id: str,
        parent_thread_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        deliverables: str,
        allowed_tools: list[str],
        provider: str,
        model: str,
        project_root: str,
        timeout_seconds: int,
    ) -> None:
        worker_repo = get_worker_run_repository()
        session_repo = get_session_repository()
        set_active_thread_id(worker_thread_id)
        set_active_worker_id(worker_id)
        set_active_program_run_id("")
        try:
            result = asyncio.run(
                self._run_worker_async(
                    worker_id=worker_id,
                    worker_thread_id=worker_thread_id,
                    role=role,
                    task=task,
                    deliverables=deliverables,
                    allowed_tools=allowed_tools,
                    provider=provider,
                    model=model,
                    project_root=project_root,
                    timeout_seconds=timeout_seconds,
                )
            )
            worker_repo.update_worker_run(
                worker_id,
                status="completed",
                result_summary=result,
                metadata={"project_root": project_root},
                finished=True,
            )
            session_repo.touch_session(worker_thread_id, status="idle")
            session_repo.enqueue_inbox_event(
                thread_id=parent_thread_id,
                event_type="worker_result",
                payload=result,
            )
            audit_logger.log_event(
                thread_id=parent_thread_id,
                event="worker_completed",
                content=f"worker {worker_id} completed",
                worker_id=worker_id,
            )
        except Exception as exc:
            error_payload = {
                "worker_id": worker_id,
                "status": "failed",
                "summary": "",
                "changed_files": [],
                "commands_run": [],
                "tests_run": [],
                "blocking_issue": str(exc),
            }
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
                payload=error_payload,
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

    async def _run_worker_async(
        self,
        *,
        worker_id: str,
        worker_thread_id: str,
        role: str,
        task: str,
        deliverables: str,
        allowed_tools: list[str],
        provider: str,
        model: str,
        project_root: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        conversation_writer = get_conversation_writer()
        worker_prompt = (
            f"[Worker Role]\n{role}\n\n"
            f"[Task]\n{task}\n\n"
            f"[Deliverables]\n{deliverables or '请聚焦当前子任务，输出结构化结果。'}\n\n"
            "约束：你是子 worker，不允许再委派其他 worker，不允许调用 execute_tool_program，不允许写长期记忆。"
        )
        toolset = self._load_toolset(allowed_tools, role=role)
        summary = {
            "worker_id": worker_id,
            "status": "completed",
            "summary": "",
            "changed_files": [],
            "commands_run": [],
            "tests_run": [],
            "blocking_issue": "",
        }

        async def runner() -> dict[str, Any]:
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
                }
                async for event in app.astream(inputs, config={"configurable": {"thread_id": worker_thread_id, "turn_id": turn_id}}, stream_mode="updates"):
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
                return summary

        return await asyncio.wait_for(runner(), timeout=timeout_seconds)

    def list_workers(self, *, parent_thread_id: str, status_filter: str = "") -> list[dict[str, Any]]:
        statuses = tuple(item.strip() for item in status_filter.split(",") if item.strip()) if status_filter else None
        return get_worker_run_repository().list_worker_runs(parent_thread_id=parent_thread_id, statuses=statuses)

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
            final = [record for record in normalized if str(record.get("status", "") or "") in {"completed", "failed", "cancelled", "timeout"}]
            if normalized and len(final) == len(normalized):
                break
            time.sleep(0.2)

        records = [
            worker_repo.get_worker_run(worker_id)
            for worker_id in requested_ids
        ] if requested_ids else worker_repo.list_worker_runs(parent_thread_id=parent_thread_id, limit=100)
        normalized = [record for record in records if record]
        completed = [record for record in normalized if str(record.get("status", "") or "") in {"completed", "failed", "cancelled", "timeout"}]
        if not return_partial and normalized and len(completed) != len(normalized):
            return {
                "success": False,
                "status": "waiting",
                "workers": [],
                "message": "仍有 worker 在运行中。",
            }
        return {
            "success": True,
            "status": "completed" if normalized and len(completed) == len(normalized) else "partial",
            "workers": normalized,
        }

    def cancel_worker(self, worker_id: str) -> dict[str, Any]:
        worker_repo = get_worker_run_repository()
        record = worker_repo.get_worker_run(worker_id)
        if record is None:
            return {"success": False, "message": f"未找到 worker：{worker_id}"}
        with self._lock:
            future = self._futures.get(worker_id)
        if future is not None and future.cancel():
            worker_repo.update_worker_run(worker_id, status="cancelled", metadata={"cancelled": True}, finished=True)
            return {"success": True, "message": f"已取消尚未开始的 worker：{worker_id}"}
        worker_repo.update_worker_run(worker_id, metadata={"cancel_requested": True})
        return {"success": False, "message": f"worker {worker_id} 已在运行，当前仅记录取消请求。"}


_default_worker_supervisor = AsyncWorkerSupervisor()


def get_worker_supervisor() -> AsyncWorkerSupervisor:
    return _default_worker_supervisor
