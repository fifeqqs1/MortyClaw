try:
    from typing import Annotated, NotRequired, TypedDict
except ImportError:  # pragma: no cover
    from typing import TypedDict

    from typing_extensions import Annotated, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class WorkingMemoryState(TypedDict, total=False):
    goal: str
    plan: list[dict]
    current_step_index: int
    planner_required: bool
    route_locked: bool
    route_source: str
    route_reason: str
    route_confidence: float
    plan_source: str
    replan_reason: str
    pending_approval: bool
    approval_reason: str
    recent_tool_results: list[dict]
    last_error: str
    last_error_kind: str
    last_recovery_action: str
    repeated_failure_signature: str
    repeated_failure_count: int
    current_project_path: str
    current_mode: str
    permission_mode: str
    run_status: str
    todos: list[dict]
    active_todos: list[dict]
    slow_execution_mode: str
    execution_mode: str
    pending_tool_calls: list[dict]
    pending_execution_snapshot: dict
    execution_guard_status: str
    execution_guard_reason: str
    active_workers: list[str]
    worker_results: list[dict]
    worker_waiting_on: list[str]
    program_run_id: str
    program_run_status: str
    loaded_subdirectory_contexts: list[str]
    subdirectory_context_hints: list[dict]
    compact_generation: int
    last_compact_at: str
    last_compact_reason: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    working_memory: NotRequired[WorkingMemoryState]
    route: str
    goal: str
    complexity: str
    risk_level: str
    planner_required: bool
    route_locked: bool
    route_source: str
    route_reason: str
    route_confidence: float
    plan_source: str
    replan_reason: str
    plan: list[dict]
    current_step_index: int
    step_results: list[dict]
    pending_approval: bool
    approval_granted: bool
    approval_prompted: bool
    approval_reason: str
    permission_mode: NotRequired[str]
    permission_prompted: NotRequired[bool]
    last_error: str
    last_error_kind: str
    last_recovery_action: str
    repeated_failure_signature: NotRequired[str]
    repeated_failure_count: NotRequired[int]
    retry_count: int
    max_retries: int
    todos: NotRequired[list[dict]]
    active_todos: NotRequired[list[dict]]
    todo_revision: NotRequired[int]
    todo_needs_announcement: NotRequired[bool]
    last_todo_tool_call_id: NotRequired[str]
    pending_tool_calls: NotRequired[list[dict]]
    pending_execution_snapshot: NotRequired[dict]
    slow_execution_mode: NotRequired[str]
    execution_mode: NotRequired[str]
    execution_guard_status: NotRequired[str]
    execution_guard_reason: NotRequired[str]
    active_workers: NotRequired[list[str]]
    worker_results: NotRequired[list[dict]]
    worker_waiting_on: NotRequired[list[str]]
    program_run_id: NotRequired[str]
    program_run_status: NotRequired[str]
    loaded_subdirectory_contexts: NotRequired[list[str]]
    subdirectory_context_hints: NotRequired[list[dict]]
    compact_generation: NotRequired[int]
    last_compact_at: NotRequired[str]
    last_compact_reason: NotRequired[str]
    final_answer: str
    run_status: str
    current_project_path: NotRequired[str]


def build_working_memory_snapshot(
    state: AgentState,
    *,
    recent_tool_results_limit: int = 3,
) -> WorkingMemoryState:
    step_results = list(state.get("step_results", []) or [])
    return {
        "goal": state.get("goal", ""),
        "plan": [dict(step) for step in (state.get("plan", []) or [])],
        "current_step_index": state.get("current_step_index", 0),
        "planner_required": state.get("planner_required", False),
        "route_locked": state.get("route_locked", False),
        "route_source": state.get("route_source", ""),
        "route_reason": state.get("route_reason", ""),
        "route_confidence": state.get("route_confidence", 0.0),
        "plan_source": state.get("plan_source", ""),
        "replan_reason": state.get("replan_reason", ""),
        "pending_approval": state.get("pending_approval", False),
        "approval_reason": state.get("approval_reason", ""),
        "recent_tool_results": step_results[-recent_tool_results_limit:],
        "last_error": state.get("last_error", ""),
        "last_error_kind": state.get("last_error_kind", ""),
        "last_recovery_action": state.get("last_recovery_action", ""),
        "repeated_failure_signature": state.get("repeated_failure_signature", ""),
        "repeated_failure_count": state.get("repeated_failure_count", 0),
        "current_project_path": state.get("current_project_path", ""),
        "current_mode": state.get("route", ""),
        "permission_mode": state.get("permission_mode", ""),
        "run_status": state.get("run_status", ""),
        "todos": [dict(item) for item in (state.get("todos", []) or []) if isinstance(item, dict)],
        "active_todos": [dict(item) for item in (state.get("active_todos", state.get("todos", [])) or []) if isinstance(item, dict)],
        "slow_execution_mode": state.get("slow_execution_mode", ""),
        "execution_mode": state.get("execution_mode", ""),
        "pending_tool_calls": [dict(item) for item in (state.get("pending_tool_calls", []) or []) if isinstance(item, dict)],
        "pending_execution_snapshot": dict(state.get("pending_execution_snapshot", {}) or {}),
        "execution_guard_status": state.get("execution_guard_status", ""),
        "execution_guard_reason": state.get("execution_guard_reason", ""),
        "active_workers": [str(item) for item in (state.get("active_workers", []) or []) if str(item or "").strip()],
        "worker_results": [dict(item) for item in (state.get("worker_results", []) or []) if isinstance(item, dict)],
        "worker_waiting_on": [str(item) for item in (state.get("worker_waiting_on", []) or []) if str(item or "").strip()],
        "program_run_id": state.get("program_run_id", ""),
        "program_run_status": state.get("program_run_status", ""),
        "loaded_subdirectory_contexts": [str(item) for item in (state.get("loaded_subdirectory_contexts", []) or []) if str(item or "").strip()],
        "subdirectory_context_hints": [dict(item) for item in (state.get("subdirectory_context_hints", []) or []) if isinstance(item, dict)],
        "compact_generation": int(state.get("compact_generation", 0) or 0),
        "last_compact_at": state.get("last_compact_at", ""),
        "last_compact_reason": state.get("last_compact_reason", ""),
    }
