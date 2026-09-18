from . import (
    apply_project_patch,
    calculator,
    delete_scheduled_task,
    edit_project_file,
    execute_office_shell,
    get_current_time,
    get_system_model_info,
    list_office_files,
    list_scheduled_tasks,
    modify_scheduled_task,
    read_office_file,
    read_project_file,
    run_project_command,
    run_project_tests,
    save_user_profile,
    schedule_task,
    search_project_code,
    search_sessions,
    show_git_diff,
    tavily_web_search,
    write_office_file,
    write_project_file,
)
from ..meta import ToolMeta, attach_tool_meta


def _meta(
    name: str,
    capabilities: set[str],
    *,
    risk_level: str = "low",
    requires_approval: bool = False,
) -> ToolMeta:
    return ToolMeta.build(
        name=name,
        capabilities=capabilities,
        risk_level=risk_level,
        requires_approval=requires_approval,
    )


BUILTIN_TOOL_META = {
    "get_current_time": _meta("get_current_time", {"system_read"}),
    "calculator": _meta("calculator", {"compute"}),
    "tavily_web_search": _meta("tavily_web_search", {"web_read"}),
    "save_user_profile": _meta(
        "save_user_profile",
        {"memory_write"},
        risk_level="medium",
        requires_approval=False,
    ),
    "list_office_files": _meta("list_office_files", {"office_read"}),
    "read_office_file": _meta("read_office_file", {"office_read"}),
    "write_office_file": _meta(
        "write_office_file",
        {"office_write", "file_write"},
        risk_level="high",
        requires_approval=True,
    ),
    "execute_office_shell": _meta(
        "execute_office_shell",
        {"office_write", "shell_exec"},
        risk_level="high",
        requires_approval=True,
    ),
    "read_project_file": _meta("read_project_file", {"project_read", "file_read"}),
    "search_project_code": _meta("search_project_code", {"project_read"}),
    "edit_project_file": _meta(
        "edit_project_file",
        {"project_write", "file_write"},
        risk_level="high",
        requires_approval=True,
    ),
    "write_project_file": _meta(
        "write_project_file",
        {"project_write", "file_write"},
        risk_level="high",
        requires_approval=True,
    ),
    "apply_project_patch": _meta(
        "apply_project_patch",
        {"project_write", "file_write"},
        risk_level="high",
        requires_approval=True,
    ),
    "show_git_diff": _meta("show_git_diff", {"project_read"}),
    "run_project_tests": _meta(
        "run_project_tests",
        {"project_read", "shell_exec"},
        risk_level="high",
        requires_approval=True,
    ),
    "run_project_command": _meta(
        "run_project_command",
        {"project_write", "shell_exec"},
        risk_level="high",
        requires_approval=True,
    ),
    "get_system_model_info": _meta("get_system_model_info", {"system_read"}),
    "schedule_task": _meta("schedule_task", {"task_write"}, risk_level="medium"),
    "list_scheduled_tasks": _meta("list_scheduled_tasks", {"task_read"}),
    "delete_scheduled_task": _meta("delete_scheduled_task", {"task_write"}, risk_level="medium"),
    "modify_scheduled_task": _meta("modify_scheduled_task", {"task_write"}, risk_level="medium"),
    "search_sessions": _meta("search_sessions", {"session_read"}),
}


def _with_builtin_meta(tool):
    meta = BUILTIN_TOOL_META.get(str(getattr(tool, "name", "") or ""))
    if meta is None:
        return tool
    return attach_tool_meta(tool, meta)


BUILTIN_TOOLS = [
    get_current_time,
    calculator,
    tavily_web_search,
    save_user_profile,
    list_office_files,
    read_office_file,
    write_office_file,
    execute_office_shell,
    read_project_file,
    search_project_code,
    edit_project_file,
    write_project_file,
    apply_project_patch,
    show_git_diff,
    run_project_tests,
    run_project_command,
    get_system_model_info,
    schedule_task,
    list_scheduled_tasks,
    delete_scheduled_task,
    modify_scheduled_task,
    search_sessions,
]

BUILTIN_TOOLS = [_with_builtin_meta(tool) for tool in BUILTIN_TOOLS]
