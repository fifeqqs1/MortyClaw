from __future__ import annotations

def make_router_node(
    *,
    with_working_memory_fn,
    get_latest_user_query_fn,
    schedule_long_term_memory_capture_fn,
    sync_session_memory_from_query_fn,
    load_session_project_path_fn,
    build_route_decision_fn,
    clear_session_todo_state_fn,
    audit_logger_instance,
):
    def router_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        latest_user_query = get_latest_user_query_fn(state.get("messages", []))
        schedule_long_term_memory_capture_fn(latest_user_query)
        session_state_updates = sync_session_memory_from_query_fn(latest_user_query, thread_id)
        if not session_state_updates.get("current_project_path"):
            existing_project_path = state.get("current_project_path") or load_session_project_path_fn(thread_id)
            if existing_project_path:
                session_state_updates["current_project_path"] = existing_project_path

        if state.get("pending_approval"):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="router resumed slow path while waiting for approval response",
            )
            return with_working_memory_fn(state, {
                "route": "slow",
                "goal": state.get("goal", ""),
                "complexity": state.get("complexity", "high_risk"),
                "risk_level": state.get("risk_level", "high"),
                "planner_required": state.get("planner_required", True),
                "route_locked": state.get("route_locked", False),
                "route_source": state.get("route_source", "resume_pending_approval"),
                "route_reason": state.get("route_reason", ""),
                "route_confidence": state.get("route_confidence", 1.0),
                "plan_source": state.get("plan_source", ""),
                "replan_reason": state.get("replan_reason", ""),
                "plan": state.get("plan", []),
                "current_step_index": state.get("current_step_index", 0),
                "step_results": state.get("step_results", []),
                "pending_approval": True,
                "approval_granted": state.get("approval_granted", False),
                "approval_prompted": state.get("approval_prompted", False),
                "approval_reason": state.get("approval_reason", ""),
                "permission_mode": state.get("permission_mode", ""),
                "permission_prompted": state.get("permission_prompted", False),
                "last_error": "",
                "last_error_kind": state.get("last_error_kind", ""),
                "last_recovery_action": state.get("last_recovery_action", ""),
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", 2) or 2,
                "todos": state.get("todos", []),
                "active_todos": state.get("active_todos", state.get("todos", [])),
                "todo_revision": state.get("todo_revision", 0),
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": state.get("last_todo_tool_call_id", ""),
                "pending_tool_calls": state.get("pending_tool_calls", []),
                "pending_execution_snapshot": state.get("pending_execution_snapshot", {}),
                "slow_execution_mode": state.get("slow_execution_mode", "autonomous"),
                "final_answer": "",
                "run_status": "awaiting_approval_response",
                "execution_guard_status": state.get("execution_guard_status", ""),
                "execution_guard_reason": state.get("execution_guard_reason", ""),
                **session_state_updates,
            })

        if (
            str(state.get("route", "") or "") == "slow"
            and not str(state.get("permission_mode", "") or "").strip().lower()
            and state.get("permission_prompted", False)
            and str(state.get("run_status", "") or "") == "waiting_user"
        ):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="router resumed slow path while waiting for execution mode selection",
            )
            return with_working_memory_fn(state, {
                "route": "slow",
                "goal": state.get("goal", latest_user_query or ""),
                "complexity": state.get("complexity", "high_risk"),
                "risk_level": state.get("risk_level", "high"),
                "planner_required": state.get("planner_required", False),
                "route_locked": state.get("route_locked", False),
                "route_source": state.get("route_source", "resume_permission_selection"),
                "route_reason": state.get("route_reason", ""),
                "route_confidence": state.get("route_confidence", 1.0),
                "plan_source": state.get("plan_source", ""),
                "replan_reason": state.get("replan_reason", ""),
                "plan": state.get("plan", []),
                "current_step_index": state.get("current_step_index", 0),
                "step_results": state.get("step_results", []),
                "pending_approval": state.get("pending_approval", False),
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": state.get("approval_reason", ""),
                "permission_mode": "",
                "permission_prompted": True,
                "last_error": "",
                "last_error_kind": state.get("last_error_kind", ""),
                "last_recovery_action": state.get("last_recovery_action", ""),
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", 2) or 2,
                "todos": state.get("todos", []),
                "active_todos": state.get("active_todos", state.get("todos", [])),
                "todo_revision": state.get("todo_revision", 0),
                "todo_needs_announcement": False,
                "last_todo_tool_call_id": state.get("last_todo_tool_call_id", ""),
                "pending_tool_calls": state.get("pending_tool_calls", []),
                "pending_execution_snapshot": state.get("pending_execution_snapshot", {}),
                "slow_execution_mode": state.get("slow_execution_mode", "autonomous"),
                "final_answer": "",
                "run_status": "awaiting_permission_mode",
                "execution_guard_status": state.get("execution_guard_status", ""),
                "execution_guard_reason": state.get("execution_guard_reason", ""),
                **session_state_updates,
            })

        route_decision = build_route_decision_fn(latest_user_query)
        if route_decision.get("route") != "slow":
            clear_session_todo_state_fn(thread_id)
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=(
                f"router selected {route_decision['route']} path | "
                f"complexity={route_decision['complexity']} | "
                f"risk={route_decision['risk_level']} | "
                f"source={route_decision.get('route_source', 'unknown')}"
            ),
        )
        return with_working_memory_fn(state, {
            **route_decision,
            "plan_source": "",
            "replan_reason": "",
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "pending_approval": False,
            "approval_granted": False,
            "approval_prompted": False,
            "approval_reason": "",
            "permission_mode": "",
            "permission_prompted": False,
            "last_error": "",
            "last_error_kind": "",
            "last_recovery_action": "",
            "retry_count": 0,
            "max_retries": state.get("max_retries", 2) or 2,
            "todos": [],
            "active_todos": [],
            "todo_revision": 0,
            "todo_needs_announcement": False,
            "last_todo_tool_call_id": "",
            "pending_tool_calls": [],
            "pending_execution_snapshot": {},
            "slow_execution_mode": str(route_decision.get("slow_execution_mode", "") or ""),
            "execution_guard_status": "",
            "execution_guard_reason": "",
            "final_answer": "",
            "run_status": "routing",
            **session_state_updates,
        })

    return router_node
