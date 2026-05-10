from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from ..config import (
    CONTEXT_COMPRESSION_BUDGET_TOKENS,
    CONTEXT_INTERACTIVE_BUDGET_TOKENS,
    CONTEXT_LAYER2_TRIGGER_RATIO,
    CONTEXT_LAYER3_TRIGGER_RATIO,
    ENABLE_DYNAMIC_CONTEXT_FOR_FAST_AGENT,
    ENABLE_DYNAMIC_CONTEXT_FOR_SLOW_AGENT,
    SUBDIRECTORY_HINT_CHAR_BUDGET,
)
from ..context import (
    AgentState,
    render_dynamic_system_context,
    render_reference_context,
)
from ..context.compact import compact_context_state, should_auto_compact
from ..context.handoff import render_handoff_summary
from ..context.window import estimate_text_tokens
from ..context.window import classify_context_pressure
from ..prompts.provider_cache import apply_provider_prompt_cache
from ..runtime.todos import hydrate_todos_from_state_or_session
from ..runtime_context import set_active_tool_scope_names
from .recovery import _extract_classified_error
from .tool_policy import REQUEST_TOOL_SCHEMA_TOOL_NAME


CONTEXT_TRIM_KEEP_TOKENS = 220000
CONTEXT_OVERFLOW_KEEP_TOKENS = 120000
CONTEXT_NON_MESSAGE_RESERVE_TOKENS = 80000
_TOOL_SCHEMA_TEXT_CACHE: dict[str, str] = {}


def _current_turn_id(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("turn_id", "turn-default")


def _resolve_llm_model_name(llm) -> str:
    for attr in ("model_name", "model"):
        value = str(getattr(llm, attr, "") or "").strip()
        if value:
            return value
    return ""


def _resolve_llm_provider_name(llm, state: AgentState) -> str:
    state_provider = str(
        state.get("provider_name")
        or state.get("provider")
        or state.get("llm_provider")
        or ""
    ).strip()
    if state_provider:
        return state_provider
    for attr in ("provider_name", "provider"):
        value = str(getattr(llm, attr, "") or "").strip()
        if value:
            return value
    return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _serialize_tool_schema_text(tools: list[BaseTool]) -> str:
    signature = _hash_text(json.dumps(sorted(getattr(tool, "name", "") for tool in tools), ensure_ascii=False))
    cached = _TOOL_SCHEMA_TEXT_CACHE.get(signature)
    if cached is not None:
        return cached
    payload = []
    for tool in tools:
        args_schema = getattr(tool, "args_schema", None)
        schema_payload = {}
        if args_schema is not None:
            schema_method = getattr(args_schema, "model_json_schema", None) or getattr(args_schema, "schema", None)
            if callable(schema_method):
                try:
                    schema_payload = schema_method()
                except Exception:
                    schema_payload = {}
        payload.append({
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
            "schema": schema_payload,
        })
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    _TOOL_SCHEMA_TEXT_CACHE[signature] = rendered
    return rendered


def _tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {
        str(getattr(tool, "name", "") or ""): tool
        for tool in tools
        if str(getattr(tool, "name", "") or "").strip()
    }


def _extract_requested_deferred_tool_names(messages: list) -> set[str]:
    requested: set[str] = set()
    for message in reversed(messages or []):
        if getattr(message, "type", "") != "tool":
            break
        if str(getattr(message, "name", "") or "") != REQUEST_TOOL_SCHEMA_TOOL_NAME:
            continue
        payload = _extract_tool_payload(message)
        if not payload:
            continue
        for name in payload.get("requested_tools", []) or []:
            normalized = str(name or "").strip()
            if normalized:
                requested.add(normalized)
        break
    return requested


def _should_use_direct_arxiv_shortcut(
    *,
    active_route: str,
    route_source: str,
    effective_user_query: str,
    should_direct_route_to_arxiv_rag_fn,
) -> bool:
    if active_route != "fast":
        return False
    if route_source not in {"arxiv_direct", "pure_paper_task"}:
        return False
    if not effective_user_query:
        return False
    return bool(should_direct_route_to_arxiv_rag_fn(effective_user_query))


def _extract_tool_payload(message) -> dict | None:
    if getattr(message, "type", "") != "tool":
        return None
    content = str(getattr(message, "content", "") or "").strip()
    if not content.startswith("{"):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_tool_failure_info(raw_messages: list) -> dict | None:
    if not raw_messages:
        return None
    latest_message = raw_messages[-1]
    payload = _extract_tool_payload(latest_message)
    if not payload or payload.get("ok", True):
        return None
    tool_name = str(getattr(latest_message, "name", "") or "").strip()
    signature = json.dumps(
        {
            "tool": tool_name,
            "error_kind": str(payload.get("error_kind", "") or "").strip(),
            "message": " ".join(str(payload.get("message", "") or "").split())[:200],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "message": latest_message,
        "payload": payload,
        "tool_name": tool_name,
        "signature": signature,
    }


def _find_matching_tool_call(raw_messages: list, tool_call_id: str) -> dict | None:
    normalized_id = str(tool_call_id or "").strip()
    if not normalized_id:
        return None
    for message in reversed(raw_messages):
        for tool_call in list(getattr(message, "tool_calls", []) or []):
            if str(tool_call.get("id") or "").strip() == normalized_id:
                return dict(tool_call)
    return None


def _step_likely_creates_file(current_plan_step: dict | None, goal_text: str = "") -> bool:
    step_text = str((current_plan_step or {}).get("description", "") or "").strip()
    intent = str((current_plan_step or {}).get("intent", "") or "").strip().lower()
    combined = f"{step_text}\n{goal_text}".lower()
    create_markers = (
        "新建",
        "创建",
        "生成",
        "写入",
        "create",
        "new file",
        "write file",
    )
    if intent == "file_write":
        return True
    return any(marker in step_text or marker in combined for marker in create_markers)


def _step_expects_runtime_output(step_text: str) -> bool:
    lowered = str(step_text or "").lower()
    output_hints = (
        "打印",
        "输出",
        "结果",
        "示例",
        "stdout",
        "print",
        "run result",
        "show output",
    )
    return any(hint in lowered for hint in output_hints)


def _build_successful_execution_step_response(
    raw_messages: list,
    current_plan_step: dict | None,
    *,
    deps: ReactNodeDependencies,
):
    if not current_plan_step:
        return None
    current_intent = str(current_plan_step.get("intent") or "")
    if current_intent not in {"shell_execute", "test_verify"}:
        return None

    step_text = str(current_plan_step.get("description") or "")
    trailing_tool_messages = []
    for message in reversed(raw_messages):
        if getattr(message, "type", "") != "tool":
            break
        trailing_tool_messages.append(message)

    for message in trailing_tool_messages:
        tool_name = str(getattr(message, "name", "") or "")
        if tool_name not in {"run_project_command", "run_project_tests"}:
            continue
        payload = _extract_tool_payload(message)
        if not payload or not payload.get("ok", False):
            continue

        stdout = str(payload.get("stdout", "") or "").strip()
        command = str(payload.get("command", "") or "").strip()
        exit_code = payload.get("exit_code", 0)
        if exit_code not in {0, None}:
            continue

        if current_intent == "shell_execute":
            if not stdout or not _step_expects_runtime_output(step_text):
                continue
            success_summary = stdout[:400]
        else:
            success_summary = stdout[:400] if stdout else (
                f"验证完成：{command or tool_name} 执行通过。"
            )

        return deps.annotate_ai_message_fn(
            AIMessage(content=success_summary),
            mortyclaw_step_outcome=deps.step_outcome_success_candidate,
            mortyclaw_response_kind=deps.response_kind_step_result,
        )

    return None


def _persist_current_todo_state(
    *,
    deps,
    thread_id: str,
    active_route: str,
    state: AgentState,
) -> None:
    has_todos = bool(state.get("todos"))
    if has_todos and deps.should_enable_todos_fn(
        active_route,
        state.get("plan", []),
        execution_mode=str(state.get("slow_execution_mode", "") or ""),
        todos=state.get("todos", []),
    ):
        todo_state = deps.build_todo_state_from_plan_fn(
            state.get("plan", []),
            int(state.get("current_step_index", 0) or 0),
            revision=int(state.get("todo_revision", 1) or 1),
            last_event="active",
        )
    elif has_todos:
        todo_state = {
            "items": list(state.get("todos", []) or []),
            "revision": int(state.get("todo_revision", 1) or 1),
            "last_event": "active",
        }
    else:
        deps.clear_session_todo_state_fn(thread_id)
        return

    todo_state["items"] = list(state.get("todos", []) or [])
    todo_state["plan_snapshot"] = [
        dict(step)
        for step in (state.get("plan", []) or [])
        if isinstance(step, dict)
    ]
    deps.save_session_todo_state_fn(thread_id, todo_state)


@dataclass(frozen=True)
class ReactNodeDependencies:
    set_active_thread_id_fn: object
    prepare_recent_tool_messages_fn: object
    build_session_memory_prompt_fn: object
    should_enable_todos_fn: object
    build_todo_state_from_plan_fn: object
    load_session_todo_state_fn: object
    save_session_todo_state_fn: object
    clear_session_todo_state_fn: object
    audit_logger_instance: object
    extract_passthrough_text_fn: object
    annotate_ai_message_fn: object
    with_working_memory_fn: object
    is_affirmative_approval_response_fn: object
    get_latest_user_query_fn: object
    get_current_plan_step_fn: object
    select_tools_for_current_step_fn: object
    select_tools_for_fast_route_fn: object
    apply_permission_mode_to_tools_fn: object
    select_tools_for_autonomous_slow_fn: object
    split_tools_for_deferred_schema_fn: object
    route_eager_tool_names_fn: object
    should_direct_route_to_arxiv_rag_fn: object
    arxiv_rag_tool: object
    extract_passthrough_payload_fn: object
    trim_context_messages_fn: object
    compact_context_messages_deterministic_fn: object
    summarize_discarded_context_fn: object
    conversation_writer: object
    build_long_term_memory_prompt_fn: object
    build_react_prompt_bundle_fn: object
    assemble_dynamic_context_fn: object
    classify_error_fn: object
    serialize_classified_error_fn: object
    normalize_tavily_tool_calls_fn: object
    enforce_slow_step_tool_scope_fn: object
    destructive_tool_calls_fn: object
    build_pending_execution_snapshot_fn: object
    build_pending_tool_approval_reason_fn: object
    looks_like_explicit_failure_text_fn: object
    complete_autonomous_todos_fn: object
    fast_path_excluded_tool_names: set[str]
    auto_mode_blocked_tool_names: set[str]
    update_subdirectory_context_fn: object
    response_kind_final_answer: str
    response_kind_step_result: str
    step_outcome_failure: str
    step_outcome_success_candidate: str
    session_memory_prompt_limit: int
    context_summary_timeout_seconds: float
    select_tools_for_structured_slow_fn: object | None = None


def run_react_agent_node(
    state: AgentState,
    config: RunnableConfig,
    llm,
    llm_with_tools,
    all_tools: list[BaseTool],
    route_mode: str,
    *,
    deps: ReactNodeDependencies,
) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "system_default")
    turn_id = _current_turn_id(config)
    deps.set_active_thread_id_fn(thread_id)
    raw_messages, preprocessing_updates = deps.prepare_recent_tool_messages_fn(
        state,
        thread_id=thread_id,
        turn_id=turn_id,
    )
    working_state = dict(state)
    for key, value in preprocessing_updates.items():
        if key != "messages":
            working_state[key] = value
    working_state.update(
        hydrate_todos_from_state_or_session(
            working_state,
            session_todo_state=deps.load_session_todo_state_fn(thread_id),
            summary_text=working_state.get("summary", ""),
            risk_fallback=lambda _description: str(working_state.get("risk_level", "medium") or "medium"),
        )
    )
    active_route = working_state.get("route", route_mode)
    worker_isolation_mode = bool(working_state.get("worker_isolation_mode", False))
    formatted_session_prompt = "" if worker_isolation_mode else deps.build_session_memory_prompt_fn(
        thread_id,
        limit=deps.session_memory_prompt_limit,
    )

    _persist_current_todo_state(
        deps=deps,
        thread_id=thread_id,
        active_route=active_route,
        state=working_state,
    )

    if raw_messages:
        recent_tool_msgs = []
        for msg in reversed(raw_messages):
            if msg.type == "tool":
                recent_tool_msgs.append(msg)
            else:
                break
        for msg in reversed(recent_tool_msgs):
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="tool_result",
                tool=msg.name,
                result_summary=msg.content[:200],
            )

    passthrough_text = deps.extract_passthrough_text_fn(raw_messages[-1]) if raw_messages else None
    if passthrough_text:
        final_message = deps.annotate_ai_message_fn(
            AIMessage(content=passthrough_text),
            mortyclaw_response_kind=deps.response_kind_final_answer,
        )
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=passthrough_text,
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": passthrough_text,
            "run_status": "done",
            "messages": [final_message],
        })

    if (
        active_route == "slow"
        and working_state.get("approval_granted")
        and working_state.get("pending_tool_calls")
    ):
        pending_tool_calls = [
            dict(tool_call)
            for tool_call in (working_state.get("pending_tool_calls", []) or [])
            if isinstance(tool_call, dict)
        ]
        if pending_tool_calls:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=f"slow agent resumed {len(pending_tool_calls)} approved tool call(s)",
            )
            resumed_message = AIMessage(content="", tool_calls=pending_tool_calls)
            return deps.with_working_memory_fn(state, {
                "route": active_route,
                "approval_granted": False,
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "run_status": "running",
                "messages": [resumed_message],
                **preprocessing_updates,
            })

    if (
        active_route == "slow"
        and preprocessing_updates.get("pending_approval")
        and not working_state.get("approval_granted")
    ):
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content="slow agent routed pending execution batch to approval gate",
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "run_status": str(preprocessing_updates.get("run_status", "awaiting_step_approval") or "awaiting_step_approval"),
            **preprocessing_updates,
        })

    slow_execution_mode = str(working_state.get("slow_execution_mode", "") or "").strip().lower()
    permission_mode = str(working_state.get("permission_mode", "") or "").strip().lower()
    latest_user_query = deps.get_latest_user_query_fn(raw_messages)
    current_plan_step = (
        deps.get_current_plan_step_fn(working_state)
        if active_route == "slow" and slow_execution_mode != "autonomous"
        else None
    )
    latest_tool_failure = _latest_tool_failure_info(raw_messages)
    failure_tracking_updates: dict[str, object] = {}
    if latest_tool_failure is None:
        failure_tracking_updates = {
            "repeated_failure_signature": "",
            "repeated_failure_count": 0,
        }
    else:
        previous_signature = str(working_state.get("repeated_failure_signature", "") or "").strip()
        previous_count = int(working_state.get("repeated_failure_count", 0) or 0)
        signature = str(latest_tool_failure["signature"] or "").strip()
        repeated_count = previous_count + 1 if previous_signature == signature else 1
        failure_tracking_updates = {
            "repeated_failure_signature": signature,
            "repeated_failure_count": repeated_count,
        }

        tool_call_id = str(getattr(latest_tool_failure["message"], "tool_call_id", "") or "").strip()
        matching_tool_call = _find_matching_tool_call(raw_messages, tool_call_id)
        matching_args = dict((matching_tool_call or {}).get("args", {}) or {})
        current_goal = str(working_state.get("goal", "") or "")

        if (
            active_route == "slow"
            and current_plan_step is not None
            and latest_tool_failure["tool_name"] == "write_project_file"
            and str(latest_tool_failure["payload"].get("error_kind", "") or "").strip() == "FILE_NOT_FOUND"
            and matching_tool_call is not None
            and not bool(matching_args.get("create_if_missing", False))
            and _step_likely_creates_file(current_plan_step, current_goal)
        ):
            repaired_tool_call = dict(matching_tool_call)
            repaired_args = dict(matching_args)
            repaired_args["create_if_missing"] = True
            repaired_tool_call["args"] = repaired_args
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    "slow agent auto-retried write_project_file with create_if_missing=true "
                    "after FILE_NOT_FOUND while creating a new file"
                ),
            )
            return deps.with_working_memory_fn(state, {
                "route": active_route,
                "run_status": "running",
                "messages": [AIMessage(content="", tool_calls=[repaired_tool_call])],
                **preprocessing_updates,
                **failure_tracking_updates,
            })

        if int(failure_tracking_updates.get("repeated_failure_count", 0) or 0) >= 3:
            failure_message = str(latest_tool_failure["payload"].get("message", "") or "").strip()
            replan_reason = (
                f"同一工具错误已连续发生 3 次：{failure_message or latest_tool_failure['tool_name']}"
            )
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    "slow agent stopped repeated identical tool failures and requested replan: "
                    f"{latest_tool_failure['tool_name']}"
                ),
            )
            return deps.with_working_memory_fn(state, {
                "route": active_route,
                "run_status": "replan_requested",
                "replan_reason": replan_reason,
                "last_error": failure_message[:200],
                "last_error_kind": str(latest_tool_failure["payload"].get("error_kind", "") or "tool_runtime_error"),
                "last_recovery_action": "replan",
                "retry_count": 0,
                **preprocessing_updates,
                **failure_tracking_updates,
            })

    execution_success_response = _build_successful_execution_step_response(
        raw_messages,
        current_plan_step,
        deps=deps,
    )
    if active_route == "slow" and execution_success_response is not None:
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=str(execution_success_response.content or ""),
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": "",
            "run_status": "review_pending",
            "todo_needs_announcement": False,
            "messages": [execution_success_response],
            **preprocessing_updates,
        })

    effective_user_query = latest_user_query
    active_llm_with_tools = llm_with_tools
    allowed_tool_names: set[str] = {getattr(tool, "name", "") for tool in all_tools}
    selected_route_tools: list[BaseTool] = list(all_tools)
    all_tools_by_name = _tools_by_name(all_tools)
    request_schema_tool = all_tools_by_name.get(REQUEST_TOOL_SCHEMA_TOOL_NAME)
    if active_route == "fast":
        fast_tools = deps.select_tools_for_fast_route_fn(
            working_state,
            all_tools,
            latest_user_query=latest_user_query,
        )
        if request_schema_tool and request_schema_tool not in fast_tools:
            fast_tools = list(fast_tools) + [request_schema_tool]
        selected_route_tools = list(fast_tools)
        allowed_tool_names = {getattr(tool, "name", "") for tool in fast_tools}

    if active_route == "slow" and current_plan_step is not None:
        effective_user_query = current_plan_step["description"]
        if deps.select_tools_for_structured_slow_fn is not None:
            structured_route_tools = deps.select_tools_for_structured_slow_fn(
                working_state,
                all_tools,
                latest_user_query=latest_user_query,
            )
        else:
            structured_route_tools = list(all_tools)
        allowed_step_tools = deps.select_tools_for_current_step_fn(
            current_plan_step,
            structured_route_tools,
            current_project_path=str(working_state.get("current_project_path", "") or ""),
        )
        allowed_step_tools = deps.apply_permission_mode_to_tools_fn(
            allowed_step_tools,
            permission_mode=permission_mode,
        )
        if request_schema_tool and request_schema_tool not in allowed_step_tools:
            allowed_step_tools = list(allowed_step_tools) + [request_schema_tool]
        selected_route_tools = list(allowed_step_tools)
        allowed_tool_names = {getattr(tool, "name", "") for tool in allowed_step_tools}
    elif active_route == "slow" and slow_execution_mode == "autonomous":
        autonomous_tools = deps.select_tools_for_autonomous_slow_fn(
            working_state,
            all_tools,
            latest_user_query=latest_user_query,
        )
        if str(working_state.get("current_project_path", "") or "").strip():
            autonomous_tools = [
                tool for tool in autonomous_tools
                if getattr(tool, "name", "") != "write_office_file"
            ] or autonomous_tools
        autonomous_tools = deps.apply_permission_mode_to_tools_fn(
            autonomous_tools,
            permission_mode=permission_mode,
        )
        if request_schema_tool and request_schema_tool not in autonomous_tools:
            autonomous_tools = list(autonomous_tools) + [request_schema_tool]
        selected_route_tools = list(autonomous_tools)
        allowed_tool_names = {getattr(tool, "name", "") for tool in autonomous_tools}
    elif active_route == "slow" and state.get("goal") and deps.is_affirmative_approval_response_fn(latest_user_query):
        effective_user_query = state["goal"]

    requested_deferred_names = _extract_requested_deferred_tool_names(raw_messages)
    unauthorized_requested_names = sorted(name for name in requested_deferred_names if name not in allowed_tool_names)
    if unauthorized_requested_names:
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="system_action",
            content=(
                "ignored unauthorized deferred tool schema request: "
                + ", ".join(unauthorized_requested_names)
            ),
        )
    expanded_tool_names = requested_deferred_names & allowed_tool_names
    eager_tool_names = deps.route_eager_tool_names_fn(
        working_state,
        active_route=active_route,
        slow_execution_mode=slow_execution_mode,
        current_plan_step=current_plan_step,
        latest_user_query=latest_user_query,
    ) & allowed_tool_names
    bound_tools, deferred_tools, expanded_tools = deps.split_tools_for_deferred_schema_fn(
        selected_route_tools,
        expanded_tool_names=expanded_tool_names,
        eager_tool_names=eager_tool_names,
    )
    if not bound_tools:
        bound_tools = bound_tools or selected_route_tools
    if bound_tools:
        active_llm_with_tools = llm.bind_tools(bound_tools)
    else:
        active_llm_with_tools = llm if hasattr(llm, "invoke") else llm_with_tools
    set_active_tool_scope_names([getattr(tool, "name", "") for tool in bound_tools])

    route_source = str(working_state.get("route_source", "") or "")
    if _should_use_direct_arxiv_shortcut(
        active_route=active_route,
        route_source=route_source,
        effective_user_query=effective_user_query,
        should_direct_route_to_arxiv_rag_fn=deps.should_direct_route_to_arxiv_rag_fn,
    ):
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="tool_call",
            tool="arxiv_rag_ask",
            args={"query": effective_user_query, "session_id": thread_id},
        )
        tool_result = deps.arxiv_rag_tool.invoke({"query": effective_user_query, "session_id": thread_id})
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="tool_result",
            tool="arxiv_rag_ask",
            result_summary=tool_result[:200],
        )

        passthrough_payload = deps.extract_passthrough_payload_fn(tool_result)
        direct_reply = tool_result
        if passthrough_payload is not None:
            display_text = passthrough_payload.get("display_text") or passthrough_payload.get("answer")
            if isinstance(display_text, str) and display_text.strip():
                direct_reply = display_text

        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=direct_reply,
        )
        final_message = deps.annotate_ai_message_fn(
            AIMessage(content=direct_reply),
            mortyclaw_response_kind=deps.response_kind_final_answer,
        )
        return deps.with_working_memory_fn(state, {
            "route": active_route,
            "final_answer": direct_reply,
            "run_status": "done",
            "messages": [final_message],
        })

    current_summary = state.get("summary", "")
    model_name = _resolve_llm_model_name(llm)
    state_updates = {"route": active_route}
    state_updates.update(failure_tracking_updates)
    for key, value in preprocessing_updates.items():
        if key != "messages":
            state_updates[key] = value
    if preprocessing_updates.get("messages"):
        state_updates["messages"] = list(preprocessing_updates["messages"])

    subdirectory_updates = {}
    if active_route == "slow":
        subdirectory_updates = deps.update_subdirectory_context_fn(
            working_state | state_updates,
            raw_messages,
            char_budget=SUBDIRECTORY_HINT_CHAR_BUDGET,
        )
        if subdirectory_updates:
            state_updates.update(subdirectory_updates)
            working_state.update(subdirectory_updates)

    if worker_isolation_mode:
        long_term_prompt = ""
    else:
        try:
            long_term_prompt = deps.build_long_term_memory_prompt_fn(latest_user_query, thread_id=thread_id)
        except TypeError:
            long_term_prompt = deps.build_long_term_memory_prompt_fn(latest_user_query)
    use_dynamic_context = (
        ENABLE_DYNAMIC_CONTEXT_FOR_SLOW_AGENT
        if active_route == "slow"
        else ENABLE_DYNAMIC_CONTEXT_FOR_FAST_AGENT
    )
    preview_dynamic_context_envelope = (
        deps.assemble_dynamic_context_fn(
            view="slow" if active_route == "slow" else "fast",
            state=working_state | state_updates,
            user_query=effective_user_query or latest_user_query or "",
            session_prompt=formatted_session_prompt or "",
            long_term_prompt=long_term_prompt,
            active_summary=current_summary,
            current_plan_step=current_plan_step,
        )
        if use_dynamic_context else None
    )
    extra_context_texts = []
    if preview_dynamic_context_envelope:
        preview_todo_block = ""
        if active_route == "slow":
            preview_todo_items = (working_state | state_updates).get("active_todos") or (working_state | state_updates).get("todos")
            if preview_todo_items:
                from ..runtime.todos import render_todo_for_prompt

                preview_todo_block = render_todo_for_prompt(preview_todo_items)
        extra_context_texts.append(
            render_dynamic_system_context(
                preview_dynamic_context_envelope,
                state=working_state | state_updates,
                active_route=active_route,
                current_plan_step=current_plan_step,
                todo_block=preview_todo_block,
                explicit_project_code_goal=False,
            )
        )
        reference_preview = render_reference_context(preview_dynamic_context_envelope)
        if reference_preview:
            extra_context_texts.append(reference_preview)
    else:
        if formatted_session_prompt:
            extra_context_texts.append(formatted_session_prompt)
        if long_term_prompt:
            extra_context_texts.append(long_term_prompt)
        if current_summary:
            extra_context_texts.append(render_handoff_summary(current_summary))
    deferred_catalog_preview = ""
    if deferred_tools:
        try:
            from ..prompts.builder import render_deferred_tool_catalog

            deferred_catalog_preview = render_deferred_tool_catalog(deferred_tools)
        except Exception:
            deferred_catalog_preview = ""
    if deferred_catalog_preview:
        extra_context_texts.append(deferred_catalog_preview)

    context_pressure = classify_context_pressure(
        raw_messages,
        model_name=model_name,
        budget_tokens=CONTEXT_INTERACTIVE_BUDGET_TOKENS,
        reserve_tokens=0,
        extra_texts=extra_context_texts,
        layer2_trigger_ratio=CONTEXT_LAYER2_TRIGGER_RATIO,
        layer3_trigger_ratio=CONTEXT_LAYER3_TRIGGER_RATIO,
    )
    pressure_level = str(context_pressure.get("level", "low") or "low")
    if pressure_level != "low":
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="context_pressure_detected",
            content=(
                f"context pressure {pressure_level}: "
                f"total={int(context_pressure.get('total_used_tokens', 0) or 0)}, "
                f"budget={int(context_pressure.get('budget_tokens', 0) or 0)}, "
                f"ratio={float(context_pressure.get('usage_ratio', 0.0) or 0.0):.3f}"
            ),
        )
    if pressure_level == "low":
        final_msgs, discarded_msgs, deterministic_result = raw_messages, [], None
    else:
        deterministic_result = deps.compact_context_messages_deterministic_fn(
            raw_messages,
            thread_id=thread_id,
            turn_id=turn_id,
            model_name=model_name,
            persist_artifacts=True,
        )
        final_msgs = list(getattr(deterministic_result, "kept_messages", raw_messages) or raw_messages)
        discarded_msgs = list(getattr(deterministic_result, "removed_messages", []) or [])
        deterministic_stats = dict(getattr(deterministic_result, "stats", {}) or {})
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="context_deterministic_compaction",
            content=(
                f"deterministic context compaction applied "
                f"(pressure={pressure_level}, "
                f"stubbed={int(deterministic_stats.get('stubbed_count', 0) or 0)}, "
                f"safely_discarded={int(deterministic_stats.get('safely_discarded_count', 0) or 0)}, "
                f"removed={int(deterministic_stats.get('removed_count', len(discarded_msgs)) or 0)}, "
                f"artifacts={int(deterministic_stats.get('artifact_count', 0) or 0)})"
            ),
        )
        if int(deterministic_stats.get("artifact_count", 0) or 0):
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="context_artifact_persisted",
                content=f"context compaction persisted {int(deterministic_stats.get('artifact_count', 0) or 0)} artifact(s)",
            )
        if active_route == "slow" and discarded_msgs:
            state_updates.setdefault("messages", [])
            state_updates["messages"].extend([RemoveMessage(id=m.id) for m in discarded_msgs if m.id])
            state_updates["messages"].extend(list(getattr(deterministic_result, "stub_messages", []) or []))

    if discarded_msgs and pressure_level == "high":
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="context_llm_summary_merge",
            content=(
                "high pressure context merge started "
                f"(discarded={len(discarded_msgs)}, stubbed={len(getattr(deterministic_result, 'stub_messages', []) or [])})"
            ),
        )
        active_summary = deps.summarize_discarded_context_fn(
            llm,
            current_summary,
            discarded_msgs + list(getattr(deterministic_result, "stub_messages", []) or []),
            thread_id,
            state=working_state | state_updates,
            timeout_seconds=deps.context_summary_timeout_seconds,
        )

        state_updates["summary"] = active_summary
        state_updates["structured_handoff"] = active_summary
        deps.conversation_writer.record_summary(
            thread_id=thread_id,
            summary=active_summary,
            summary_type="structured_handoff",
            messages=discarded_msgs,
            metadata={
                "discarded_message_count": len(discarded_msgs),
                "pressure_level": pressure_level,
                "deterministic_stats": getattr(deterministic_result, "stats", {}) if deterministic_result else {},
            },
        )
    else:
        active_summary = current_summary

    compact_reason = ""
    if discarded_msgs and pressure_level == "high":
        latest_user_message = next(
            (message for message in reversed(raw_messages) if getattr(message, "type", "") == "human"),
            None,
        )
        if should_auto_compact(
            state=working_state | state_updates,
            pressure_level=str(context_pressure.get("level", "low")),
            has_new_summary=bool(str(active_summary or "").strip()),
            latest_user_message=latest_user_message,
        ):
            compact_reason = f"auto_compact:{str(context_pressure.get('level', 'high'))}"
            compact_updates, compacted_messages = compact_context_state(
                working_state | state_updates,
                original_messages=raw_messages,
                latest_user_message=latest_user_message,
                reason=compact_reason,
            )
            if compact_updates and compacted_messages:
                state_updates.setdefault("messages", [])
                state_updates["messages"].extend(compact_updates.pop("messages", []))
                state_updates.update(compact_updates)
                final_msgs = compacted_messages
                deps.audit_logger_instance.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    content=(
                        "automatic compact applied after context trim "
                        f"(reason={compact_reason}, kept_messages={len(compacted_messages)})"
                    ),
                )
                deps.conversation_writer.record_summary(
                    thread_id=thread_id,
                    summary=active_summary,
                    summary_type="compact_reset",
                    messages=raw_messages,
                    metadata={"reason": compact_reason, "kept_message_count": len(compacted_messages)},
                )

    dynamic_context_envelope = None
    if use_dynamic_context:
        if preview_dynamic_context_envelope is not None and active_summary == current_summary:
            dynamic_context_envelope = preview_dynamic_context_envelope
        else:
            dynamic_context_envelope = deps.assemble_dynamic_context_fn(
                view="slow" if active_route == "slow" else "fast",
                state=working_state | state_updates,
                user_query=effective_user_query or latest_user_query or "",
                session_prompt=formatted_session_prompt or "",
                long_term_prompt=long_term_prompt,
                active_summary=active_summary,
                current_plan_step=current_plan_step,
            )
    prompt_state = dict(working_state)
    prompt_state.update(state_updates)
    prompt_state["_prompt_turn_id"] = turn_id
    prompt_state["provider_name"] = _resolve_llm_provider_name(llm, working_state)
    prompt_state["model_name"] = model_name
    prompt_state["_base_toolset_signature"] = _hash_text(
        json.dumps(sorted({getattr(tool, "name", "") for tool in all_tools}), ensure_ascii=False)
    )
    prompt_bundle = deps.build_react_prompt_bundle_fn(
        final_msgs,
        active_route,
        prompt_state,
        active_summary=active_summary,
        session_prompt=formatted_session_prompt or "",
        long_term_prompt=long_term_prompt,
        current_plan_step=current_plan_step,
        include_approved_goal_context=bool(
            active_route == "slow"
            and working_state.get("goal")
            and deps.is_affirmative_approval_response_fn(latest_user_query)
        ),
        dynamic_context_envelope=dynamic_context_envelope,
        deferred_tools=deferred_tools,
    )
    if isinstance(prompt_bundle, tuple):
        sys_prompt, llm_messages = prompt_bundle
        msgs_for_llm = [SystemMessage(content=sys_prompt)] + llm_messages
        prompt_hashes = {}
        prompt_token_stats = {}
        prompt_cache_stats = {}
    else:
        msgs_for_llm = prompt_bundle.final_messages()
        prompt_hashes = dict(getattr(prompt_bundle, "prompt_hashes", {}) or {})
        prompt_token_stats = dict(getattr(prompt_bundle, "token_stats", {}) or {})
        prompt_cache_stats = dict(getattr(prompt_bundle, "cache_stats", {}) or {})
    for message in msgs_for_llm:
        if isinstance(message.content, str):
            message.content = message.content.encode("utf-8", "ignore").decode("utf-8")

    tool_schema_text = _serialize_tool_schema_text(bound_tools)
    tool_group_signature = _hash_text(tool_schema_text)
    prompt_hashes["tools"] = tool_group_signature
    prompt_token_stats["tools_schema"] = estimate_text_tokens(tool_schema_text)
    prompt_token_stats["final_input"] = sum(
        estimate_text_tokens(str(getattr(message, "content", "") or ""))
        for message in msgs_for_llm
    )
    cached_msgs_for_llm, provider_cache_stats = apply_provider_prompt_cache(
        msgs_for_llm,
        provider_name=prompt_state.get("provider_name", ""),
        model_name=model_name,
    )
    msgs_for_llm = cached_msgs_for_llm

    deps.audit_logger_instance.log_event(
        thread_id=thread_id,
        event="llm_input",
        message_count=len(msgs_for_llm),
        prompt_hashes=prompt_hashes,
        token_stats=prompt_token_stats,
        cache_stats={**prompt_cache_stats, **provider_cache_stats},
        slow_execution_mode=slow_execution_mode,
        tool_group_signature=tool_group_signature,
        bound_tool_names=[getattr(tool, "name", "") for tool in bound_tools],
        deferred_tool_names=[getattr(tool, "name", "") for tool in deferred_tools],
        expanded_deferred_tool_names=[getattr(tool, "name", "") for tool in expanded_tools],
    )

    response = None
    for attempt in range(2):
        try:
            response = active_llm_with_tools.invoke(msgs_for_llm)
            break
        except Exception as exc:
            classified = deps.classify_error_fn(exc=exc, state=working_state)
            if classified.kind.value == "context_overflow" and attempt == 0:
                forced_msgs, forced_discarded = deps.trim_context_messages_fn(
                    raw_messages,
                    trigger_tokens=1,
                    keep_tokens=CONTEXT_OVERFLOW_KEEP_TOKENS,
                    reserve_tokens=CONTEXT_NON_MESSAGE_RESERVE_TOKENS,
                    model_name=model_name,
                )
                raw_forced_trim_stats = getattr(deps.trim_context_messages_fn, "_last_stats", {})
                forced_trim_stats = dict(raw_forced_trim_stats) if isinstance(raw_forced_trim_stats, dict) else {}
                if forced_discarded:
                    deps.audit_logger_instance.log_event(
                        thread_id=thread_id,
                        event="system_action",
                        content=(
                            "context overflow retry triggered forced trim "
                            f"(artifact_messages_seen={int(forced_trim_stats.get('artifact_messages_seen', 0) or 0)}, "
                            f"tool_results_pruned={int(forced_trim_stats.get('tool_results_pruned', 0) or 0)}, "
                            f"tool_pairs_repaired={int(forced_trim_stats.get('tool_pairs_repaired', 0) or 0)}, "
                            f"discarded_middle_count={int(forced_trim_stats.get('discarded_middle_count', len(forced_discarded)) or 0)})"
                        ),
                    )
                    active_summary = deps.summarize_discarded_context_fn(
                        llm,
                        active_summary,
                        forced_discarded,
                        thread_id,
                        state=working_state,
                        timeout_seconds=deps.context_summary_timeout_seconds,
                    )
                    state_updates["summary"] = active_summary
                    state_updates.setdefault("messages", [])
                    state_updates["messages"].extend([RemoveMessage(id=m.id) for m in forced_discarded if m.id])
                    latest_user_message = next(
                        (message for message in reversed(raw_messages) if getattr(message, "type", "") == "human"),
                        None,
                    )
                    if should_auto_compact(
                        state=working_state | state_updates,
                        pressure_level="high",
                        has_new_summary=bool(str(active_summary or "").strip()),
                        overflow_retry=True,
                        latest_user_message=latest_user_message,
                    ):
                        compact_reason = "auto_compact:overflow_retry"
                        compact_updates, compacted_messages = compact_context_state(
                            working_state | state_updates,
                            original_messages=raw_messages,
                            latest_user_message=latest_user_message,
                            reason=compact_reason,
                        )
                        if compact_updates and compacted_messages:
                            state_updates["messages"].extend(compact_updates.pop("messages", []))
                            state_updates.update(compact_updates)
                            forced_msgs = compacted_messages
                            deps.audit_logger_instance.log_event(
                                thread_id=thread_id,
                                event="system_action",
                                content=(
                                    "automatic compact applied after context overflow retry "
                                    f"(kept_messages={len(compacted_messages)})"
                                ),
                            )
                            deps.conversation_writer.record_summary(
                                thread_id=thread_id,
                                summary=active_summary,
                                summary_type="compact_reset",
                                messages=raw_messages,
                                metadata={"reason": compact_reason, "kept_message_count": len(compacted_messages)},
                            )
                    retry_dynamic_context_envelope = dynamic_context_envelope
                    if use_dynamic_context:
                        retry_dynamic_context_envelope = deps.assemble_dynamic_context_fn(
                            view="slow" if active_route == "slow" else "fast",
                            state=working_state | state_updates,
                            user_query=effective_user_query or latest_user_query or "",
                            session_prompt=formatted_session_prompt or "",
                            long_term_prompt=long_term_prompt,
                            active_summary=active_summary,
                            current_plan_step=current_plan_step,
                        )
                    retry_prompt_state = dict(working_state)
                    retry_prompt_state.update(state_updates)
                    retry_prompt_state["_prompt_turn_id"] = turn_id
                    retry_prompt_state["provider_name"] = _resolve_llm_provider_name(llm, working_state)
                    retry_prompt_state["model_name"] = model_name
                    retry_prompt_state["_base_toolset_signature"] = _hash_text(
                        json.dumps(sorted({getattr(tool, "name", "") for tool in all_tools}), ensure_ascii=False)
                    )
                    prompt_bundle = deps.build_react_prompt_bundle_fn(
                        forced_msgs,
                        active_route,
                        retry_prompt_state,
                        active_summary=active_summary,
                        session_prompt=formatted_session_prompt or "",
                        long_term_prompt=long_term_prompt,
                        current_plan_step=current_plan_step,
                        include_approved_goal_context=bool(
                            active_route == "slow"
                            and working_state.get("goal")
                            and deps.is_affirmative_approval_response_fn(latest_user_query)
                        ),
                        dynamic_context_envelope=retry_dynamic_context_envelope if use_dynamic_context else None,
                    )
                    if isinstance(prompt_bundle, tuple):
                        sys_prompt, llm_messages = prompt_bundle
                        msgs_for_llm = [SystemMessage(content=sys_prompt)] + llm_messages
                    else:
                        msgs_for_llm = prompt_bundle.final_messages()
                    continue
            if classified.retryable and attempt + 1 < classified.retry_policy.max_attempts:
                continue
            response = AIMessage(
                content=classified.user_visible_hint or str(exc),
                additional_kwargs={
                    "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                    "mortyclaw_step_outcome": deps.step_outcome_failure,
                },
            )
            break
    if response is None:
        classified = deps.classify_error_fn(message="", state=working_state)
        response = AIMessage(
            content=classified.user_visible_hint,
            additional_kwargs={
                "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                "mortyclaw_step_outcome": deps.step_outcome_failure,
            },
        )

    response = deps.normalize_tavily_tool_calls_fn(response, effective_user_query, thread_id)
    response = deps.enforce_slow_step_tool_scope_fn(response, current_plan_step, allowed_tool_names, thread_id)
    if active_route == "fast":
        fast_escalation_reason = ""
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if destructive_calls:
            blocked_names = ", ".join(
                sorted({str(tool_call.get("name") or "").strip() for tool_call in destructive_calls if tool_call.get("name")})
            ) or "高风险工具"
            fast_escalation_reason = (
                "fast path discovered high-risk tool intent and escalated to planner: "
                f"{blocked_names}"
            )
        else:
            metadata = getattr(response, "additional_kwargs", {}) or {}
            requested_escalation = str(metadata.get("mortyclaw_fast_escalate", "") or "").strip().lower()
            if requested_escalation == "planner":
                fast_escalation_reason = str(metadata.get("mortyclaw_fast_escalate_reason") or "").strip() or (
                    "fast path explicitly requested planner escalation"
                )

        if fast_escalation_reason:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=fast_escalation_reason,
            )
            state_updates.update({
                "route": "slow",
                "planner_required": True,
                "route_source": "fast_escalation",
                "route_reason": fast_escalation_reason,
                "goal": working_state.get("goal", "") or latest_user_query or "",
                "complexity": "high_risk" if destructive_calls else "complex",
                "risk_level": "high" if destructive_calls else str(working_state.get("risk_level", "medium") or "medium"),
                "pending_tool_calls": [],
                "pending_execution_snapshot": {},
                "pending_approval": False,
                "approval_granted": False,
                "approval_prompted": False,
                "approval_reason": "",
                "final_answer": "",
                "run_status": "planner_requested",
            })
            return deps.with_working_memory_fn(state, state_updates)
    if active_route == "slow" and response.tool_calls:
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if permission_mode == "plan" and destructive_calls:
            blocked_names = ", ".join(
                sorted({str(tool_call.get("name") or "").strip() for tool_call in destructive_calls if tool_call.get("name")})
            ) or "高风险工具"
            response = deps.annotate_ai_message_fn(
                AIMessage(content=(
                    "当前任务处于 `plan` 只读模式，但执行过程中检测到需要修改文件、运行测试或执行命令的操作，"
                    f"已终止本次任务。涉及工具：{blocked_names}。请改用 `ask` 或 `auto` 后重试。"
                )),
                mortyclaw_response_kind=deps.response_kind_final_answer,
            )
            state_updates["pending_tool_calls"] = []
            state_updates["pending_execution_snapshot"] = {}
            state_updates["pending_approval"] = False
            state_updates["approval_granted"] = False
            state_updates["approval_prompted"] = False
            state_updates["approval_reason"] = ""
            state_updates["run_status"] = "failed"
            state_updates["final_answer"] = str(response.content or "")
        elif permission_mode == "auto":
            forbidden_auto_calls = [
                tool_call for tool_call in (response.tool_calls or [])
                if str(tool_call.get("name") or "").strip() in deps.auto_mode_blocked_tool_names
            ]
            if forbidden_auto_calls:
                blocked_names = ", ".join(
                    sorted({str(tool_call.get("name") or "").strip() for tool_call in forbidden_auto_calls if tool_call.get("name")})
                ) or "受限工具"
                response = deps.annotate_ai_message_fn(
                    AIMessage(content=(
                        "当前任务处于 `auto` 模式，但检测到被禁止的原始 shell/batch 操作，"
                        f"已终止本次任务。涉及工具：{blocked_names}。"
                    )),
                    mortyclaw_response_kind=deps.response_kind_final_answer,
                )
                state_updates["pending_tool_calls"] = []
                state_updates["pending_execution_snapshot"] = {}
                state_updates["pending_approval"] = False
                state_updates["approval_granted"] = False
                state_updates["approval_prompted"] = False
                state_updates["approval_reason"] = ""
                state_updates["run_status"] = "failed"
                state_updates["final_answer"] = str(response.content or "")
    approval_staged = False
    if (
        active_route == "slow"
        and slow_execution_mode == "autonomous"
        and response.tool_calls
        and permission_mode not in {"plan", "auto"}
        and not working_state.get("approval_granted", False)
    ):
        destructive_calls = deps.destructive_tool_calls_fn(response.tool_calls)
        if destructive_calls:
            approval_staged = True
            state_updates["pending_approval"] = True
            state_updates["approval_granted"] = False
            state_updates["approval_prompted"] = False
            state_updates["approval_reason"] = deps.build_pending_tool_approval_reason_fn(response.tool_calls)
            state_updates["pending_tool_calls"] = [dict(tool_call) for tool_call in response.tool_calls]
            state_updates["pending_execution_snapshot"] = deps.build_pending_execution_snapshot_fn(
                working_state | state_updates,
                response.tool_calls,
            )
            state_updates["run_status"] = "awaiting_step_approval"
            response = AIMessage(content="")
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="system_action",
                content=(
                    "slow autonomous agent staged destructive tool call batch for approval: "
                    f"{state_updates['approval_reason']}"
                ),
            )
    if not approval_staged and not response.tool_calls and not str(response.content or "").strip():
        classified = deps.classify_error_fn(message="", state=working_state)
        response = AIMessage(
            content=classified.user_visible_hint,
            additional_kwargs={
                "mortyclaw_error": deps.serialize_classified_error_fn(classified),
                "mortyclaw_step_outcome": deps.step_outcome_failure,
            },
        )

    if response.tool_calls:
        state_updates["run_status"] = "running"
        state_updates["pending_tool_calls"] = []
        state_updates["pending_execution_snapshot"] = {}
        for tool_call in response.tool_calls:
            deps.audit_logger_instance.log_event(
                thread_id=thread_id,
                event="tool_call",
                tool=tool_call["name"],
                args=tool_call["args"],
            )
    elif response.content:
        error_payload = _extract_classified_error(response)
        if active_route == "slow":
            state_updates["final_answer"] = ""
            explicit_failure = bool(error_payload is not None or deps.looks_like_explicit_failure_text_fn(response.content))
            if explicit_failure:
                if error_payload is None:
                    classified = deps.classify_error_fn(message=str(response.content or ""), state=working_state)
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_error=deps.serialize_classified_error_fn(classified),
                    )
                    error_payload = classified
                response = deps.annotate_ai_message_fn(
                    response,
                    mortyclaw_step_outcome=deps.step_outcome_failure,
                    mortyclaw_response_kind=deps.response_kind_step_result,
                )
                state_updates["last_error"] = str(response.content or "")[:200]
                state_updates["last_error_kind"] = error_payload.kind.value
                state_updates["last_recovery_action"] = error_payload.recovery_action.value
                if slow_execution_mode == "autonomous":
                    recovery_action = str(error_payload.recovery_action.value or "")
                    retry_count = int(working_state.get("retry_count", 0) or 0)
                    max_retries = int(working_state.get("max_retries", 2) or 2)
                    state_updates["replan_reason"] = ""
                    if recovery_action in {"retry", "compress_and_retry"}:
                        if retry_count < max_retries:
                            state_updates["retry_count"] = retry_count + 1
                            state_updates["run_status"] = "retrying"
                        else:
                            state_updates["retry_count"] = 0
                            state_updates["replan_reason"] = error_payload.user_visible_hint or str(response.content or "")[:200]
                            state_updates["run_status"] = "replan_requested"
                    elif recovery_action == "replan":
                        state_updates["retry_count"] = 0
                        state_updates["replan_reason"] = error_payload.user_visible_hint or str(response.content or "")[:200]
                        state_updates["run_status"] = "replan_requested"
                    else:
                        state_updates["retry_count"] = 0
                        state_updates["final_answer"] = error_payload.user_visible_hint or str(response.content or "")
                        state_updates["run_status"] = "failed"
                else:
                    state_updates["run_status"] = "review_pending"
            else:
                if slow_execution_mode == "autonomous":
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_response_kind=deps.response_kind_final_answer,
                    )
                    state_updates.update(deps.complete_autonomous_todos_fn(working_state | state_updates))
                    state_updates["final_answer"] = response.content
                    state_updates["pending_approval"] = False
                    state_updates["approval_granted"] = False
                    state_updates["approval_prompted"] = False
                    state_updates["approval_reason"] = ""
                    state_updates["pending_tool_calls"] = []
                    state_updates["pending_execution_snapshot"] = {}
                    state_updates["retry_count"] = 0
                    state_updates["last_error"] = ""
                    state_updates["last_error_kind"] = ""
                    state_updates["last_recovery_action"] = ""
                    state_updates["replan_reason"] = ""
                    state_updates["run_status"] = "done"
                else:
                    response = deps.annotate_ai_message_fn(
                        response,
                        mortyclaw_step_outcome=deps.step_outcome_success_candidate,
                        mortyclaw_response_kind=deps.response_kind_step_result,
                    )
                    state_updates["run_status"] = "review_pending"
        else:
            response = deps.annotate_ai_message_fn(
                response,
                mortyclaw_response_kind=deps.response_kind_final_answer,
            )
            state_updates["final_answer"] = response.content
            state_updates["run_status"] = "done"
            if error_payload is not None:
                state_updates["last_error"] = str(response.content or "")[:200]
                state_updates["last_error_kind"] = error_payload.kind.value
                state_updates["last_recovery_action"] = error_payload.recovery_action.value
        deps.audit_logger_instance.log_event(
            thread_id=thread_id,
            event="ai_message",
            content=response.content,
        )

    if active_route == "slow":
        state_updates["todo_needs_announcement"] = False

    if "messages" not in state_updates:
        state_updates["messages"] = []
    state_updates["messages"].append(response)
    _persist_current_todo_state(
        deps=deps,
        thread_id=thread_id,
        active_route=active_route,
        state=working_state | state_updates,
    )

    return deps.with_working_memory_fn(state, state_updates)
