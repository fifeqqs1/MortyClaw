from __future__ import annotations

from langchain_core.messages import AIMessage


def make_approval_gate_node(
    *,
    with_working_memory_fn,
    get_latest_user_query_fn,
    build_approval_message_fn,
    build_permission_mode_message_fn,
    parse_permission_mode_response_fn,
    build_default_replan_input_fn,
    is_negative_approval_response_fn,
    is_affirmative_approval_response_fn,
    audit_logger_instance,
):
    def approval_gate_node(state, config) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id", "system_default")
        latest_user_query = get_latest_user_query_fn(state.get("messages", []))
        permission_mode = str(state.get("permission_mode", "") or "").strip().lower()
        selected_mode = parse_permission_mode_response_fn(latest_user_query)

        if not state.get("pending_approval"):
            if permission_mode:
                return with_working_memory_fn(state, {"run_status": "approved", "approval_prompted": False})

            # Backward-compatible path for old checkpoints that were already waiting for
            # an up-front permission mode selection before this behavior changed.
            if not state.get("permission_prompted", False):
                return with_working_memory_fn(state, {
                    "run_status": "approved",
                    "approval_prompted": False,
                    "permission_prompted": False,
                })

            if is_negative_approval_response_fn(latest_user_query):
                cancel_message = "已取消该 slow 任务，本次不会执行任何写入、测试或命令操作。"
                return with_working_memory_fn(state, {
                    "pending_approval": False,
                    "approval_granted": False,
                    "approval_prompted": False,
                    "approval_reason": "",
                    "permission_mode": "",
                    "permission_prompted": False,
                    "pending_tool_calls": [],
                    "pending_execution_snapshot": {},
                    "final_answer": cancel_message,
                    "run_status": "cancelled",
                    "messages": [AIMessage(content=cancel_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
                })

            if selected_mode:
                if selected_mode == "plan":
                    task_text = "\n".join(
                        part for part in [
                            str(state.get("goal", "") or "").strip(),
                            build_default_replan_input_fn(state) if state.get("plan") else "",
                        ]
                        if part
                    )
                    if (
                        any(step.get("risk_level") == "high" for step in (state.get("plan", []) or []) if isinstance(step, dict))
                        or "修改" in task_text or "修复" in task_text or "实现" in task_text
                            or "运行" in task_text or "执行" in task_text or "测试" in task_text
                            or "patch" in task_text.lower() or "write" in task_text.lower()
                    ):
                        termination_message = (
                            "已切换到 `plan` 只读模式，但当前任务包含修改文件、测试或命令执行需求，"
                            "与只读限制冲突，因此已终止。请改用 `ask` 或 `auto` 重新执行。"
                        )
                    audit_logger_instance.log_event(
                        thread_id=thread_id,
                        event="system_action",
                        content="approval gate terminated task because plan mode conflicts with destructive requirements",
                    )
                    return with_working_memory_fn(state, {
                        "pending_approval": False,
                        "approval_granted": False,
                        "approval_prompted": False,
                        "approval_reason": "",
                        "permission_mode": selected_mode,
                        "permission_prompted": False,
                        "pending_tool_calls": [],
                        "pending_execution_snapshot": {},
                        "final_answer": termination_message,
                        "run_status": "cancelled",
                        "messages": [AIMessage(content=termination_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
                    })

                return with_working_memory_fn(state, {
                    "permission_mode": selected_mode,
                    "permission_prompted": False,
                    "approval_prompted": False,
                    "run_status": "approved",
                })

            selection_message = build_permission_mode_message_fn(state)
            return with_working_memory_fn(state, {
                "permission_prompted": True,
                "final_answer": selection_message,
                "run_status": "waiting_user",
                "messages": [AIMessage(content=selection_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
            })

        if not permission_mode:
            if is_negative_approval_response_fn(latest_user_query):
                cancel_message = "已取消该高风险任务，本次不会执行任何写入或命令操作。"
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content="approval gate cancelled pending destructive batch before permission mode selection",
                )
                return with_working_memory_fn(state, {
                    "pending_approval": False,
                    "approval_granted": False,
                    "approval_prompted": False,
                    "approval_reason": "",
                    "permission_prompted": False,
                    "pending_tool_calls": [],
                    "pending_execution_snapshot": {},
                    "final_answer": cancel_message,
                    "run_status": "cancelled",
                    "messages": [AIMessage(content=cancel_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
                })

            if selected_mode == "plan":
                termination_message = (
                    "当前批次已经进入高风险执行阶段，而 `plan` 只读模式不允许继续。"
                    "本次任务已终止；如需继续执行，请重新发起并选择 `确认执行`、`ask` 或 `auto`。"
                )
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content="approval gate rejected pending destructive batch because user selected plan mode",
                )
                return with_working_memory_fn(state, {
                    "pending_approval": False,
                    "approval_granted": False,
                    "approval_prompted": False,
                    "approval_reason": "",
                    "permission_mode": "plan",
                    "permission_prompted": False,
                    "pending_tool_calls": [],
                    "pending_execution_snapshot": {},
                    "final_answer": termination_message,
                    "run_status": "cancelled",
                    "messages": [AIMessage(content=termination_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
                })

            if selected_mode in {"ask", "auto"}:
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content=f"approval gate accepted pending destructive batch and set mode={selected_mode}",
                )
                return with_working_memory_fn(state, {
                    "pending_approval": False,
                    "approval_granted": True,
                    "approval_prompted": False,
                    "approval_reason": "",
                    "permission_mode": selected_mode,
                    "permission_prompted": False,
                    "run_status": "approved",
                })

            if is_affirmative_approval_response_fn(latest_user_query):
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content="approval gate approved current destructive batch with one-time confirmation",
                )
                return with_working_memory_fn(state, {
                    "pending_approval": False,
                    "approval_granted": True,
                    "approval_prompted": False,
                    "approval_reason": "",
                    "permission_prompted": False,
                    "run_status": "approved",
                })

            approval_message = build_approval_message_fn(state)
            if not state.get("approval_prompted", False):
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content="approval gate requested first destructive-batch confirmation or mode selection",
                )
            else:
                audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content="approval gate waiting for valid destructive-batch confirmation response",
                )
            return with_working_memory_fn(state, {
                "approval_prompted": True,
                "permission_prompted": False,
                "final_answer": approval_message,
                "run_status": "waiting_user",
                "messages": [AIMessage(content=approval_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
            })

        if permission_mode == "plan":
            termination_message = (
                "当前任务处于 `plan` 只读模式，但执行过程中检测到需要修改文件、运行测试或执行命令的高风险操作，"
                "因此已终止。请改用 `确认执行`、`ask` 或 `auto` 后重试。"
            )
            return with_working_memory_fn(state, {
                "pending_approval": False,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": "",
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "final_answer": termination_message,
                "run_status": "cancelled",
                "messages": [AIMessage(content=termination_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
            })

        if permission_mode == "auto":
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="approval gate auto-approved destructive operation due to auto mode",
            )
            return with_working_memory_fn(state, {
                "pending_approval": False,
                "approval_granted": True,
                "approval_prompted": False,
                "approval_reason": "",
                "run_status": "approved",
            })

        if not state.get("approval_prompted", False):
            approval_message = build_approval_message_fn(state)
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="approval gate requested explicit confirmation",
            )
            return with_working_memory_fn(state, {
                "approval_prompted": True,
                "final_answer": approval_message,
                "run_status": "waiting_user",
                "messages": [AIMessage(content=approval_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
            })

        if is_negative_approval_response_fn(latest_user_query):
            cancel_message = "已取消该高风险任务，本次不会执行任何写入或命令操作。"
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="approval gate cancelled high-risk execution",
            )
            return with_working_memory_fn(state, {
                "pending_approval": False,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": "",
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "final_answer": cancel_message,
                "run_status": "cancelled",
                "messages": [AIMessage(content=cancel_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
            })

        if is_affirmative_approval_response_fn(latest_user_query):
            audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content="approval gate received explicit approval",
            )
            return with_working_memory_fn(state, {
                "pending_approval": False,
                "approval_granted": True,
                "approval_prompted": False,
                "approval_reason": "",
                "run_status": "approved",
            })

        approval_message = build_approval_message_fn(state)
        audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content="approval gate waiting for valid confirmation response",
        )
        return with_working_memory_fn(state, {
            "approval_prompted": True,
            "final_answer": approval_message,
            "run_status": "waiting_user",
            "messages": [AIMessage(content=approval_message, additional_kwargs={"mortyclaw_response_kind": "final_answer"})],
        })

    return approval_gate_node
