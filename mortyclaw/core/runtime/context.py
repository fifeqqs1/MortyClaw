from __future__ import annotations

import contextvars
import threading


_active_thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mortyclaw_active_thread_id",
    default="system_default",
)
_active_worker_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mortyclaw_active_worker_id",
    default="",
)
_active_program_run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mortyclaw_active_program_run_id",
    default="",
)
_active_tool_scope_names_var: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "mortyclaw_active_tool_scope_names",
    default=(),
)
_thread_local = threading.local()


def set_active_thread_id(thread_id: str | None) -> str:
    normalized = (thread_id or "system_default").strip() or "system_default"
    _active_thread_id_var.set(normalized)
    _thread_local.thread_id = normalized
    return normalized


def get_active_thread_id(default: str = "system_default") -> str:
    thread_id = _active_thread_id_var.get(default)
    if thread_id and thread_id != default:
        return thread_id

    local_thread_id = getattr(_thread_local, "thread_id", "")
    if local_thread_id:
        return local_thread_id
    return default


def set_active_worker_id(worker_id: str | None) -> str:
    normalized = (worker_id or "").strip()
    _active_worker_id_var.set(normalized)
    _thread_local.worker_id = normalized
    return normalized


def get_active_worker_id(default: str = "") -> str:
    worker_id = _active_worker_id_var.get(default)
    if worker_id:
        return worker_id
    return getattr(_thread_local, "worker_id", "") or default


def set_active_program_run_id(program_run_id: str | None) -> str:
    normalized = (program_run_id or "").strip()
    _active_program_run_id_var.set(normalized)
    _thread_local.program_run_id = normalized
    return normalized


def get_active_program_run_id(default: str = "") -> str:
    program_run_id = _active_program_run_id_var.get(default)
    if program_run_id:
        return program_run_id
    return getattr(_thread_local, "program_run_id", "") or default


def set_active_tool_scope_names(tool_names) -> tuple[str, ...]:
    normalized = tuple(
        sorted({
            str(name or "").strip()
            for name in (tool_names or [])
            if str(name or "").strip()
        })
    )
    _active_tool_scope_names_var.set(normalized)
    _thread_local.tool_scope_names = normalized
    return normalized


def get_active_tool_scope_names(default: tuple[str, ...] | None = None) -> tuple[str, ...]:
    fallback = tuple(default or ())
    names = _active_tool_scope_names_var.get(fallback)
    if names:
        return tuple(names)
    return tuple(getattr(_thread_local, "tool_scope_names", fallback) or fallback)
