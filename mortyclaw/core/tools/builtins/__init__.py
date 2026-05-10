import threading
import json
from typing import Any

from ..base import mortyclaw_tool
from ..office import execute_office_shell, list_office_files, read_office_file, write_office_file
from ..project_tools import (
    apply_project_patch,
    edit_project_file,
    read_project_file,
    run_project_command,
    run_project_tests,
    search_project_code,
    show_git_diff,
    write_project_file,
)
from ..summarize import summarize_content
from ..web import arxiv_rag_ask, tavily_web_search
from ...config import MEMORY_DIR, TASKS_FILE
from ...logger import build_log_file_path
from ...memory import (
    DEFAULT_LONG_TERM_SCOPE,
    USER_PROFILE_MEMORY_ID,
    USER_PROFILE_MEMORY_TYPE,
    build_memory_record,
    get_memory_store,
)
from ...runtime.todos import build_todo_state, merge_tool_written_todos, normalize_todos
from ...runtime_context import get_active_thread_id
from ...runtime.tool_results import restore_context_artifact as restore_context_artifact_impl
from ...storage.runtime import get_conversation_repository, get_session_repository, get_task_repository
from .profile import save_user_profile_impl
from .programs import execute_tool_program_impl
from .sessions import (
    ensure_session_record_impl,
    get_active_session_thread_id_impl,
    load_session_todo_state_impl,
    search_sessions_impl,
)
from .tasks import (
    delete_scheduled_task_impl,
    ensure_task_store_bootstrapped_impl,
    list_scheduled_tasks_impl,
    modify_scheduled_task_impl,
    schedule_task_impl,
)
from .system import calculator_impl, get_current_time_impl, get_system_model_info_impl
from .todo import TodoInputItem, UpdateTodoListArgs, update_todo_list_impl
from .workers import (
    cancel_subagent_impl,
    cancel_subagents_impl,
    DelegateSubagentsArgs,
    delegate_subagent_impl,
    delegate_subagents_impl,
    list_subagents_impl,
    wait_subagents_impl,
)


tasks_lock = threading.Lock()
PROFILE_PATH = f"{MEMORY_DIR}/user_profile.md"


def ensure_task_store_bootstrapped() -> None:
    ensure_task_store_bootstrapped_impl(
        get_task_repository_fn=get_task_repository,
        file_path=TASKS_FILE,
    )


def get_active_session_thread_id() -> str:
    return get_active_session_thread_id_impl(
        get_active_thread_id_fn=get_active_thread_id,
    )


def ensure_session_record(thread_id: str) -> None:
    ensure_session_record_impl(
        thread_id,
        get_session_repository_fn=get_session_repository,
        build_log_file_path_fn=build_log_file_path,
    )


def load_session_todo_state(thread_id: str) -> dict[str, Any]:
    return load_session_todo_state_impl(
        thread_id,
        get_session_repository_fn=get_session_repository,
    )


@mortyclaw_tool
def get_system_model_info() -> str:
    """
    获取当前 MortyClaw 正在运行的底层大模型（LLM）型号和提供商信息。
    当用户询问“你是基于什么模型”、“你的底层大模型是什么”、“你是GPT还是GLM”、“现在用的什么模型”等身份问题时，调用此工具。
    """
    return get_system_model_info_impl()


@mortyclaw_tool
def get_current_time() -> str:
    """
    获取当前的系统时间和日期。
    当用户询问“现在几点”、“今天星期几”、“今天几号”等与当前时间相关的问题时，调用此工具。
    """
    return get_current_time_impl()


@mortyclaw_tool
def calculator(expression: str) -> str:
    """
    一个简单的数学计算器。
    用于计算基础的数学表达式，例如: '3 * 5' 或 '100 / 4'。
    注意：参数 expression 必须是一个合法的 Python 数学表达式字符串。
    """
    return calculator_impl(expression)


@mortyclaw_tool
def request_tool_schema(tool_names: list[str], reason: str = "") -> str:
    """
    请求下一轮加载一个或多个低频工具的完整参数 schema。
    当你预计接下来会连续使用多个 deferred 工具时，请一次性把它们都放进 tool_names。
    这个工具只请求 schema，不执行目标工具。
    """
    seen: set[str] = set()
    requested_tools: list[str] = []
    for name in tool_names or []:
        normalized = str(name or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        requested_tools.append(normalized)
    return json.dumps(
        {
            "requested_tools": requested_tools,
            "reason": str(reason or ""),
            "hint": "下一轮将同时加载这些工具 schema；若需要多个 deferred 工具，请一次性批量请求。",
        },
        ensure_ascii=False,
    )


@mortyclaw_tool
def restore_context_artifact(ref_id: str) -> str:
    """
    按 ref_id 恢复被上下文裁剪替换为 stub 的原始长工具结果预览。
    仅当 stub 的 restore_hint 明确要求使用本工具，且当前回答需要查看被裁剪的完整证据时调用。
    """
    return restore_context_artifact_impl(ref_id)


@mortyclaw_tool
def save_user_profile(new_content: str) -> str:
    """
    更新用户的全局显性记忆档案。
    当你发现用户的偏好发生改变，或者有新的重要事实需要记录时：
    1.请先调用 read_user_profile 获取当前的完整档案。
    2.在你的上下文中，将新信息融入档案，并删去冲突或过时的旧信息。
    3.将修改后的一整篇完整 Markdown 文本作为 new_content 参数传入此工具。
    注意：此操作将完全覆盖旧文件！请确保传入的是完整的最新档案。
    """
    return save_user_profile_impl(
        new_content,
        get_memory_store_fn=get_memory_store,
        build_memory_record_fn=build_memory_record,
        memory_dir=MEMORY_DIR,
        profile_path=PROFILE_PATH,
        default_long_term_scope=DEFAULT_LONG_TERM_SCOPE,
        user_profile_memory_id=USER_PROFILE_MEMORY_ID,
        user_profile_memory_type=USER_PROFILE_MEMORY_TYPE,
    )


@mortyclaw_tool
def schedule_task(target_time: str, description: str, repeat: str = None, repeat_count: int = None) -> str:
    """
    为一个未来的任务设定闹钟或提醒。
    参数 target_time 必须是严格的格式："YYYY-MM-DD HH:MM:SS"（请先调用 get_current_time 获取当前时间，并在其基础上推算）。
    参数 description 是需要执行的动作或要说的话。
    """
    return schedule_task_impl(
        target_time=target_time,
        description=description,
        repeat=repeat,
        repeat_count=repeat_count,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        ensure_session_record_fn=ensure_session_record,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def list_scheduled_tasks() -> str:
    """
    查看当前所有待处理的定时任务列表。
    当用户询问“我都有哪些任务”、“查一下闹钟”、“刚才定了什么”时调用此工具。
    """
    return list_scheduled_tasks_impl(
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
    )


@mortyclaw_tool
def delete_scheduled_task(task_id: str) -> str:
    """
    根据任务 ID 取消或删除一个定时任务。
    """
    return delete_scheduled_task_impl(
        task_id=task_id,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def modify_scheduled_task(task_id: str, new_time: str = None, new_description: str = None) -> str:
    """
    修改现有定时任务的时间或内容。
    """
    return modify_scheduled_task_impl(
        task_id=task_id,
        new_time=new_time,
        new_description=new_description,
        ensure_task_store_bootstrapped_fn=ensure_task_store_bootstrapped,
        get_active_session_thread_id_fn=get_active_session_thread_id,
        get_task_repository_fn=get_task_repository,
        tasks_file=TASKS_FILE,
    )


@mortyclaw_tool
def search_sessions(
    query: str = "",
    role_filter: str = "",
    limit: int = 3,
    include_current: bool = False,
    include_tool_results: bool = True,
    summarize: bool = True,
    summary_timeout_seconds: int = 45,
) -> str:
    """
    搜索 MortyClaw 的历史会话、旧对话和以前执行过的工具结果。
    """
    def _llm_factory():
        from ...provider import get_provider
        from ...routing import get_route_classifier_model

        return get_provider(model_name=get_route_classifier_model(), temperature=0.0)

    return search_sessions_impl(
        query=query,
        role_filter=role_filter,
        limit=limit,
        include_current=include_current,
        include_tool_results=include_tool_results,
        current_thread_id=get_active_session_thread_id(),
        get_conversation_repository_fn=get_conversation_repository,
        summarize=summarize,
        summary_timeout_seconds=summary_timeout_seconds,
        llm_factory=_llm_factory,
    )


@mortyclaw_tool(args_schema=UpdateTodoListArgs)
def update_todo_list(items: list[dict[str, Any]], reason: str = "") -> str:
    """
    更新当前复杂任务的 Todo checklist。
    """
    thread_id = get_active_session_thread_id()
    session_repo = get_session_repository()
    todo_state = load_session_todo_state(thread_id)
    return update_todo_list_impl(
        items=items,
        reason=reason,
        thread_id=thread_id,
        session_repo=session_repo,
        todo_state=todo_state,
        build_todo_state_fn=build_todo_state,
        merge_tool_written_todos_fn=merge_tool_written_todos,
        normalize_todos_fn=normalize_todos,
    )


@mortyclaw_tool
def execute_tool_program(
    goal: str,
    program: str,
    tool_allowlist: list[str] | None = None,
    max_steps: int = 40,
    max_wall_time_seconds: int = 60,
    program_run_id: str = "",
    resume_approved: bool = False,
) -> str:
    """
    使用受限 DSL 编排多个项目级工具调用，适合批量搜索、读改测闭环和复杂 coding 工作流。
    """
    return execute_tool_program_impl(
        goal=goal,
        program=program,
        tool_allowlist=tool_allowlist,
        max_steps=max_steps,
        max_wall_time_seconds=max_wall_time_seconds,
        program_run_id=program_run_id,
        resume_approved=resume_approved,
    )


@mortyclaw_tool
def delegate_subagent(
    task: str,
    role: str = "explore",
    allowed_tools: list[str] | None = None,
    toolsets: list[str] | None = None,
    write_scope: list[str] | None = None,
    context_brief: str = "",
    deliverables: str = "",
    timeout_seconds: int = 180,
    priority: int = 1,
) -> str:
    """
    委派一个边界清晰的独立 worker session。优先用于复杂任务中的非阻塞 side task；多个独立分支应使用 delegate_subagents。建议提供 context_brief 和 deliverables。
    """
    return delegate_subagent_impl(
        task=task,
        role=role,
        allowed_tools=allowed_tools,
        toolsets=toolsets,
        write_scope=write_scope,
        context_brief=context_brief,
        deliverables=deliverables,
        timeout_seconds=timeout_seconds,
        priority=priority,
    )


@mortyclaw_tool(args_schema=DelegateSubagentsArgs)
def delegate_subagents(
    tasks: list[dict[str, Any]],
    batch_timeout_seconds: int = 180,
) -> str:
    """
    阻塞式委派多个互不依赖的 worker session 执行。适合复杂任务中的独立分析/验证/局部实现分支，尤其是每个分支都需要独立搜索/读文件/验证，或会产生大量中间材料、容易挤占主 agent 上下文的任务。工具会等待 worker 完成、超时或取消后返回紧凑结构化摘要；worker 的中间搜索、读取和推理过程保留在 worker session 中。每个 task 必须包含 context_brief 和 deliverables。
    """
    return delegate_subagents_impl(
        tasks=tasks,
        batch_timeout_seconds=batch_timeout_seconds,
    )


@mortyclaw_tool
def wait_subagents(
    worker_ids: list[str] | str | None = None,
    timeout_seconds: int = 30,
    return_partial: bool = False,
) -> str:
    """
    调试/兼容工具：等待一个或多个 worker 完成，并返回紧凑结构化摘要。正常使用 delegate_subagents 时不需要再调用本工具。
    """
    return wait_subagents_impl(
        worker_ids=worker_ids,
        timeout_seconds=timeout_seconds,
        return_partial=return_partial,
    )


@mortyclaw_tool
def list_subagents(status_filter: str = "") -> str:
    """
    调试/状态工具：查看当前父会话下的 worker 列表和状态；不用于获取 worker 结果正文。
    """
    return list_subagents_impl(status_filter=status_filter)


@mortyclaw_tool
def cancel_subagent(worker_id: str) -> str:
    """
    取消指定 worker。
    """
    return cancel_subagent_impl(worker_id=worker_id)


@mortyclaw_tool
def cancel_subagents(worker_ids: list[str] | str, reason: str = "") -> str:
    """
    批量请求取消 worker。worker_ids 可传列表、单个 id 或 JSON 字符串；运行中的 worker 会在下一个安全检查点协作式停止，不强杀阻塞中的 LLM/tool 调用。
    """
    return cancel_subagents_impl(worker_ids=worker_ids, reason=reason)


from .registry import BUILTIN_TOOLS  # noqa: E402


__all__ = [
    "BUILTIN_TOOLS",
    "MEMORY_DIR",
    "PROFILE_PATH",
    "TASKS_FILE",
    "TodoInputItem",
    "UpdateTodoListArgs",
    "apply_project_patch",
    "arxiv_rag_ask",
    "calculator",
    "cancel_subagent",
    "cancel_subagents",
    "delegate_subagent",
    "delegate_subagents",
    "delete_scheduled_task",
    "edit_project_file",
    "ensure_session_record",
    "ensure_task_store_bootstrapped",
    "execute_tool_program",
    "execute_office_shell",
    "get_active_session_thread_id",
    "get_active_thread_id",
    "get_conversation_repository",
    "get_current_time",
    "get_memory_store",
    "get_session_repository",
    "get_system_model_info",
    "list_subagents",
    "get_task_repository",
    "list_office_files",
    "list_scheduled_tasks",
    "load_session_todo_state",
    "modify_scheduled_task",
    "read_office_file",
    "read_project_file",
    "request_tool_schema",
    "restore_context_artifact",
    "run_project_command",
    "run_project_tests",
    "save_user_profile",
    "schedule_task",
    "search_project_code",
    "search_sessions",
    "show_git_diff",
    "summarize_content",
    "tavily_web_search",
    "tasks_lock",
    "update_todo_list",
    "wait_subagents",
    "write_office_file",
    "write_project_file",
]
