from langchain_core.tools import BaseTool

from ..context import AgentState
from ..planning import (
    looks_like_file_write_request,
    step_matches_shell_action,
    step_matches_test_action,
)


REQUEST_TOOL_SCHEMA_TOOL_NAME = "request_tool_schema"
FAST_PATH_EXCLUDED_TOOL_NAMES = {"update_todo_list"}
GENERAL_UTILITY_TOOL_NAMES = {
    "get_current_time",
    "calculator",
    "get_system_model_info",
    "restore_context_artifact",
    "search_sessions",
    "save_user_profile",
    "list_office_files",
    "read_office_file",
    "schedule_task",
    "list_scheduled_tasks",
    "delete_scheduled_task",
    "modify_scheduled_task",
}
PROJECT_FAST_UTILITY_TOOL_NAMES = {
    "get_current_time",
    "calculator",
    "get_system_model_info",
    "search_sessions",
    "save_user_profile",
}
RESEARCH_TOOL_NAMES = {
    "tavily_web_search",
    "arxiv_rag_ask",
    "summarize_content",
}
CODING_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "edit_project_file",
    "write_project_file",
    "apply_project_patch",
    "show_git_diff",
    "run_project_tests",
    "run_project_command",
    "execute_tool_program",
    "delegate_subagent",
    "delegate_subagents",
    "wait_subagents",
    "list_subagents",
    "cancel_subagent",
    "cancel_subagents",
    "update_todo_list",
}
OFFICE_TOOL_NAMES = {
    "list_office_files",
    "read_office_file",
    "write_office_file",
    "execute_office_shell",
}
FAST_PATH_PROJECT_ANALYSIS_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "tavily_web_search",
    "get_current_time",
    "calculator",
    "get_system_model_info",
}
SLOW_DESTRUCTIVE_TOOL_NAMES = {
    "edit_project_file",
    "write_project_file",
    "apply_project_patch",
    "write_office_file",
    "run_project_tests",
    "run_project_command",
    "execute_office_shell",
    "execute_tool_program",
    "delegate_subagent",
    "delegate_subagents",
}
PLAN_MODE_BLOCKED_TOOL_NAMES = set(SLOW_DESTRUCTIVE_TOOL_NAMES)
AUTO_MODE_BLOCKED_TOOL_NAMES = {"execute_office_shell"}
AUTONOMOUS_PROJECT_READ_TOOL_NAMES = {
    *GENERAL_UTILITY_TOOL_NAMES,
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "update_todo_list",
    "list_subagents",
    "wait_subagents",
    "cancel_subagent",
}
AUTONOMOUS_PROJECT_WRITE_TOOL_NAMES = {
    *GENERAL_UTILITY_TOOL_NAMES,
    *CODING_TOOL_NAMES,
}
CODING_WORKBENCH_EAGER_TOOL_NAMES = {
    "read_project_file",
    "search_project_code",
    "show_git_diff",
    "edit_project_file",
    "write_project_file",
    "apply_project_patch",
    "run_project_tests",
    "run_project_command",
    "delegate_subagent",
    "delegate_subagents",
    "wait_subagents",
    "list_subagents",
    "cancel_subagent",
    "cancel_subagents",
}
BASE_EAGER_TOOL_NAMES = {
    REQUEST_TOOL_SCHEMA_TOOL_NAME,
    "get_current_time",
    "calculator",
    "get_system_model_info",
}


def _tool_map(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {
        str(getattr(tool, "name", "") or ""): tool
        for tool in tools
        if str(getattr(tool, "name", "") or "").strip()
    }


def _select_available_tools_by_names(tools: list[BaseTool], names: set[str]) -> list[BaseTool]:
    tool_by_name = _tool_map(tools)
    return [
        tool_by_name[name]
        for name in sorted(names)
        if name in tool_by_name
    ]


def _query_needs_research(query: str) -> bool:
    lowered = str(query or "").lower()
    markers = (
        "论文",
        "paper",
        "arxiv",
        "web",
        "网页",
        "联网",
        "搜索",
        "新闻",
        "news",
        "赛程",
        "schedule",
        "比赛",
        "latest",
        "最新",
        "research",
        "browser",
        "summarize",
        "总结链接",
    )
    return any(marker in lowered for marker in markers)


def _select_tools_by_names(all_tools: list[BaseTool], allowed_names: set[str]) -> list[BaseTool]:
    selected = [
        tool for tool in all_tools
        if getattr(tool, "name", "") in allowed_names
    ]
    return selected or all_tools


def _looks_like_fast_project_analysis(
    state: AgentState,
    *,
    latest_user_query: str,
) -> bool:
    complexity = str(state.get("complexity", "") or "").strip().lower()
    if complexity != "read_only_analysis":
        return False

    current_project_path = str(state.get("current_project_path", "") or "").strip()
    if current_project_path:
        return True

    query = str(latest_user_query or "").strip().lower()
    if not query:
        return False

    if any(marker in query for marker in ("/", "\\", ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json", ".yaml", ".yml")):
        return True

    return False


def select_tools_for_fast_route(
    state: AgentState,
    all_tools: list[BaseTool],
    *,
    latest_user_query: str,
) -> list[BaseTool]:
    baseline_tools = [
        tool for tool in all_tools
        if getattr(tool, "name", "") not in FAST_PATH_EXCLUDED_TOOL_NAMES
    ]

    if _looks_like_fast_project_analysis(state, latest_user_query=latest_user_query):
        allowed_names = PROJECT_FAST_UTILITY_TOOL_NAMES | FAST_PATH_PROJECT_ANALYSIS_TOOL_NAMES
        return _select_tools_by_names(baseline_tools, allowed_names)

    allowed_names = set(GENERAL_UTILITY_TOOL_NAMES)
    if _query_needs_research(latest_user_query):
        allowed_names |= RESEARCH_TOOL_NAMES
    return _select_tools_by_names(baseline_tools, allowed_names)


def select_tools_for_structured_slow(
    state: AgentState,
    all_tools: list[BaseTool],
    *,
    latest_user_query: str,
) -> list[BaseTool]:
    current_project_path = str(state.get("current_project_path", "") or "").strip()
    allowed_names = set(GENERAL_UTILITY_TOOL_NAMES)
    if current_project_path:
        allowed_names |= CODING_TOOL_NAMES
    else:
        allowed_names |= OFFICE_TOOL_NAMES
    if _query_needs_research(str(state.get("goal", "") or latest_user_query or "")):
        allowed_names |= RESEARCH_TOOL_NAMES
    if current_project_path:
        allowed_names.discard("write_office_file")
        allowed_names.discard("execute_office_shell")
    return _select_tools_by_names(all_tools, allowed_names)


def select_tools_for_autonomous_slow(
    state: AgentState,
    all_tools: list[BaseTool],
    *,
    latest_user_query: str,
) -> list[BaseTool]:
    current_project_path = str(state.get("current_project_path", "") or "").strip()
    task_text = str(state.get("goal", "") or latest_user_query or "").strip()
    if not task_text:
        return all_tools

    if current_project_path:
        is_project_write_task = (
            str(state.get("risk_level", "") or "").strip().lower() == "high"
            or looks_like_file_write_request(task_text)
            or step_matches_test_action(task_text)
            or step_matches_shell_action(task_text)
        )
        allowed_tool_names = (
            AUTONOMOUS_PROJECT_WRITE_TOOL_NAMES
            if is_project_write_task
            else AUTONOMOUS_PROJECT_READ_TOOL_NAMES
        )
        if _query_needs_research(task_text):
            allowed_tool_names |= RESEARCH_TOOL_NAMES
        allowed_tool_names.discard("write_office_file")
        allowed_tool_names.discard("execute_office_shell")
        return _select_tools_by_names(all_tools, allowed_tool_names)

    allowed_tool_names = set(GENERAL_UTILITY_TOOL_NAMES) | OFFICE_TOOL_NAMES
    if _query_needs_research(task_text):
        allowed_tool_names |= RESEARCH_TOOL_NAMES
    return _select_tools_by_names(all_tools, allowed_tool_names)


def apply_permission_mode_to_tools(
    tools: list[BaseTool],
    *,
    permission_mode: str,
) -> list[BaseTool]:
    normalized_mode = str(permission_mode or "").strip().lower()
    if normalized_mode == "plan":
        blocked = PLAN_MODE_BLOCKED_TOOL_NAMES
    elif normalized_mode == "auto":
        blocked = AUTO_MODE_BLOCKED_TOOL_NAMES
    else:
        return tools
    filtered = [
        tool for tool in tools
        if getattr(tool, "name", "") not in blocked
    ]
    return filtered or tools


def destructive_tool_calls(tool_calls: list[dict] | None) -> list[dict]:
    return [
        dict(tool_call)
        for tool_call in (tool_calls or [])
        if isinstance(tool_call, dict)
        and str(tool_call.get("name") or "").strip() in SLOW_DESTRUCTIVE_TOOL_NAMES
    ]


def build_pending_tool_approval_reason(tool_calls: list[dict] | None) -> str:
    destructive_calls = destructive_tool_calls(tool_calls)
    if not destructive_calls:
        return ""
    tool_names: list[str] = []
    for tool_call in destructive_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        if tool_name and tool_name not in tool_names:
            tool_names.append(tool_name)
    if not tool_names:
        return ""
    return f"本轮待执行的高风险工具调用：{', '.join(tool_names)}"


def route_eager_tool_names(
    state: AgentState,
    *,
    active_route: str,
    slow_execution_mode: str,
    current_plan_step: dict | None,
    latest_user_query: str,
) -> set[str]:
    eager_names = set(BASE_EAGER_TOOL_NAMES)
    current_project_path = str(state.get("current_project_path", "") or "").strip()
    task_text = str(state.get("goal", "") or latest_user_query or "").strip()

    if (
        active_route == "slow"
        and slow_execution_mode == "autonomous"
        and current_project_path
        and (
            str(state.get("risk_level", "") or "").strip().lower() == "high"
            or looks_like_file_write_request(task_text)
            or step_matches_test_action(task_text)
            or step_matches_shell_action(task_text)
        )
    ):
        eager_names |= CODING_WORKBENCH_EAGER_TOOL_NAMES

    if active_route == "slow" and current_plan_step is not None:
        description = str(current_plan_step.get("description", "") or "")
        intent = str(current_plan_step.get("intent", "") or "").strip().lower()
        if current_project_path and (
            intent in {"implement", "edit", "shell_execute", "test_verify"}
            or looks_like_file_write_request(description)
            or step_matches_test_action(description)
            or step_matches_shell_action(description)
        ):
            eager_names |= CODING_WORKBENCH_EAGER_TOOL_NAMES

    return eager_names


def split_tools_for_deferred_schema(
    selected_tools: list[BaseTool],
    *,
    expanded_tool_names: set[str] | None = None,
    eager_tool_names: set[str] | None = None,
) -> tuple[list[BaseTool], list[BaseTool], list[BaseTool]]:
    selected_by_name = _tool_map(selected_tools)
    if REQUEST_TOOL_SCHEMA_TOOL_NAME not in selected_by_name:
        return list(selected_tools), [], []
    expanded_names = set(expanded_tool_names or set())
    eager_names = set(eager_tool_names or set()) | {REQUEST_TOOL_SCHEMA_TOOL_NAME}
    bound_names = {
        name
        for name in selected_by_name
        if name in eager_names or name in expanded_names
    }
    if REQUEST_TOOL_SCHEMA_TOOL_NAME in selected_by_name:
        bound_names.add(REQUEST_TOOL_SCHEMA_TOOL_NAME)

    bound_tools = _select_available_tools_by_names(selected_tools, bound_names)
    deferred_tools = [
        selected_by_name[name]
        for name in sorted(selected_by_name)
        if name not in {str(getattr(tool, "name", "") or "") for tool in bound_tools}
    ]
    expanded_tools = _select_available_tools_by_names(selected_tools, expanded_names)
    return bound_tools or selected_tools, deferred_tools, expanded_tools
