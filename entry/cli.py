import os
import typer
import questionary
import logging
import asyncio
import re
import json
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from dotenv import set_key, load_dotenv, unset_key
import sys
import socket
import secrets

from mortyclaw.core.config import TASKS_FILE
from mortyclaw.core.maintenance import (
    collect_doctor_report,
    format_bytes,
    gc_logs,
    gc_runtime,
    gc_state,
)
from mortyclaw.core.storage.runtime import (
    get_conversation_repository,
    get_session_repository,
    get_task_repository,
    get_worker_run_repository,
)

ENTRY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ENTRY_DIR) 

os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = typer.Typer(help="MortyClaw - 极客专属的赛博智能终端")
gc_app = typer.Typer(help="运行态垃圾回收与归档工具")
mcp_app = typer.Typer(help="配置、检查和禁用 MCP 服务")
harness_app = typer.Typer(help="配置和诊断 DeepSeek Harness")
gateway_app = typer.Typer(help="检查 MortyClaw MCP Gateway")
approvals_app = typer.Typer(help="查看和处理暂存审批")
research_app = typer.Typer(help="配置和管理本地 Agentic RAG 科研知识库")
research_sync_app = typer.Typer(help="增量同步研究资料来源")
research_add_app = typer.Typer(help="向研究知识库添加指定文档")
console = Console()

morty_style = questionary.Style([
    ('qmark', 'fg:#8d52ff bold'),       
    ('question', 'fg:#00ffff bold'),    
    ('answer', 'fg:#8d52ff bold'),      
    ('pointer', 'fg:#00ffff bold'),     
    ('highlighted', 'fg:#00ffff bold'), 
    ('selected', 'fg:#00ffff'),
    ('instruction', 'fg:#808080 dim'),  
])

ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
SHORT_SESSION_ID_PATTERN = re.compile(r"^session-(\d{1,5})$")
app.add_typer(gc_app, name="gc")
app.add_typer(mcp_app, name="mcp")
app.add_typer(harness_app, name="harness")
app.add_typer(gateway_app, name="gateway")
app.add_typer(approvals_app, name="approvals")
app.add_typer(research_app, name="research")
research_app.add_typer(research_sync_app, name="sync")
research_app.add_typer(research_add_app, name="add")


def _is_transient_test_thread_id(thread_id: str | None) -> bool:
    normalized = (thread_id or "").strip().lower()
    return normalized.startswith("test_")


def _generate_thread_id(session_repository=None) -> str:
    session_repo = session_repository or get_session_repository()
    used_numbers = set()
    for session in session_repo.list_sessions(limit=10000):
        match = SHORT_SESSION_ID_PATTERN.match(session.get("thread_id", ""))
        if match:
            used_numbers.add(int(match.group(1)))

    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"session-{next_number}"


def _resolve_default_thread_id(session_repository=None) -> str:
    session_repo = session_repository or get_session_repository()
    for session in session_repo.list_sessions(limit=100):
        latest_thread_id = (session or {}).get("thread_id", "").strip()
        if latest_thread_id and not _is_transient_test_thread_id(latest_thread_id):
            return latest_thread_id
    return _generate_thread_id(session_repo)


def _print_doctor_report(report: dict) -> None:
    db_lines = []
    for label, info in report["databases"].items():
        table_summary = ", ".join(
            f"{table}={count}" for table, count in info["table_counts"].items()
        )
        db_lines.append(
            f"- {label}: {format_bytes(info['size_bytes'])} | {table_summary} | {info['path']}"
        )

    log_lines = [
        f"- logs: {format_bytes(report['logs']['size_bytes'])} | files={report['logs']['file_count']} | {report['logs']['path']}"
    ]
    for item in report["logs"]["largest_files"]:
        log_lines.append(
            f"  - {os.path.basename(item['path'])}: {format_bytes(item['size_bytes'])}"
        )

    console.print(
        Panel(
            "[bold #00ffff]Databases[/bold #00ffff]\n"
            + "\n".join(db_lines)
            + "\n\n[bold #00ffff]Logs[/bold #00ffff]\n"
            + "\n".join(log_lines),
            title="[bold white]MortyClaw Doctor[/bold white]",
            border_style="#8d52ff",
        )
    )


def _print_gc_report(name: str, report: dict) -> None:
    mode = "dry-run" if report.get("dry_run", True) else "apply"
    lines = [f"mode={mode}"]

    if name == "logs":
        lines.append(f"candidate_count={report['candidate_count']}")
        if report.get("archive_path"):
            lines.append(f"archive={report['archive_path']}")
        lines.append(f"removed_count={report['removed_count']}")
    elif name == "runtime":
        lines.append(
            "session_inbox: "
            f"candidate={report['inbox']['candidate_count']} "
            f"deleted={report['inbox']['deleted_count']}"
        )
        lines.append(
            "task_runs: "
            f"candidate={report['task_runs']['candidate_count']} "
            f"deleted={report['task_runs']['deleted_count']}"
        )
    elif name == "state":
        lines.append(
            f"keep_latest_per_thread={report['keep_latest_per_thread']} "
            f"checkpoint_candidate_count={report['checkpoint_candidate_count']} "
            f"write_candidate_count={report['write_candidate_count']}"
        )
        if report.get("backup_path"):
            lines.append(f"backup={report['backup_path']}")
        lines.append(
            f"size_before={format_bytes(report['size_before_bytes'])} "
            f"size_after={format_bytes(report['size_after_bytes'])}"
        )

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold white]MortyClaw GC · {name}[/bold white]",
            border_style="#00ffff" if not report.get("dry_run", True) else "#8d52ff",
        )
    )

    if name == "logs" and report.get("candidates"):
        for item in report["candidates"][:10]:
            console.print(
                f"[dim]- {item['path']} | reasons={','.join(item['reasons'])} | size={format_bytes(item['size_bytes'])}[/dim]"
            )
    if name == "runtime":
        for item in report["inbox"]["candidates"][:10]:
            console.print(
                f"[dim]- inbox {item['event_id']} | thread={item['thread_id']} | delivered_at={item['delivered_at']}[/dim]"
            )
        for item in report["task_runs"]["candidates"][:10]:
            console.print(
                f"[dim]- task_run {item['run_id']} | task={item['task_id']} | triggered_at={item['triggered_at']}[/dim]"
            )
    if name == "state":
        for thread_key, summary in sorted(report["threads"].items()):
            if summary["checkpoint_prunable"] > 0:
                console.print(
                    f"[dim]- {thread_key} | checkpoints={summary['checkpoint_total']} | "
                    f"prunable={summary['checkpoint_prunable']} | writes={summary['write_prunable']}[/dim]"
                )


def _format_session_lineage(item: dict) -> str:
    parent = str(item.get("parent_thread_id", "") or "").strip()
    root = str(item.get("lineage_root_thread_id", "") or "").strip()
    if parent:
        suffix = f" | parent={parent}"
        if root and root != parent:
            suffix += f" | root={root}"
        return suffix
    if root and root != str(item.get("thread_id", "") or "").strip():
        return f" | root={root}"
    return ""


def _decode_json_field(raw_value: str, default):
    try:
        return json.loads(raw_value or "")
    except Exception:
        return default

@app.command("config")
def config_wizard():
    """配置 DeepSeek Harness（第一版仅支持 DeepSeek）。"""
    configure_harness()


def _show_boot_error():
    console.print(Panel(
        "[bold #00ffff]MortyClaw未完成配置![/bold #00ffff]\n\n"
        "[#8d52ff]检测到 API Key、模型或Baseurl。请重新执行以下命令完成配置：[/#8d52ff]\n"
        "[bold #00ffff]mortyclaw config[/bold #00ffff]",
        title="[bold #8d52ff]⚠️ Boot Sequence Failed[/bold #8d52ff]",
        border_style="#8d52ff"
    ))


@app.command("run")
def run_agent(
    thread_id: str | None = typer.Option(None, "--thread-id", help="指定要运行的会话 thread_id"),
    new_session: bool = typer.Option(False, "--new", "--new-session", help="创建一个新的短编号会话 ID，例如 session-1"),
    branch_from: str | None = typer.Option(None, "--branch-from", help="从指定历史会话创建一个语义分支会话"),
):
    load_dotenv(ENV_PATH)
    from mortyclaw.core.harness.settings import HarnessSettings
    settings = HarnessSettings.from_env()
    try:
        settings.validate()
    except RuntimeError:
        _show_boot_error()
        raise typer.Exit(code=1)
    provider = "deepseek-official"
    model = settings.model

    session_repository = get_session_repository()
    resolved_thread_id = thread_id
    if new_session and not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    if branch_from and not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    resolved_thread_id = (resolved_thread_id or _resolve_default_thread_id(session_repository)).strip()
    if not resolved_thread_id:
        resolved_thread_id = _generate_thread_id(session_repository)
    if branch_from:
        session_repository.create_branch_session(
            parent_thread_id=branch_from,
            branch_thread_id=resolved_thread_id,
            provider=provider,
            model=model,
        )
        console.print(
            f"[dim #8d52ff]已创建会话分支：{resolved_thread_id} <- {branch_from}。"
            "当前分支保留 lineage 元数据，运行状态仍由新的 thread_id 独立恢复。[/dim #8d52ff]"
        )

    import entry.main as mortyclaw_main
    mortyclaw_main.main(thread_id=resolved_thread_id)


def _ensure_env_file() -> None:
    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, "a", encoding="utf-8").close()


@mcp_app.command("configure")
def configure_mcp_service(
    service: str = typer.Argument(..., help="要配置的服务：zotero 或 arxiv"),
):
    """启用并验证 Zotero 或 Arxiv MCP。"""
    normalized = service.strip().lower()
    if normalized not in {"zotero", "arxiv"}:
        console.print("[bold red]服务只支持 zotero 或 arxiv。[/bold red]")
        raise typer.Exit(code=2)

    _ensure_env_file()
    if normalized == "zotero":
        values = {
            "ZOTERO_MCP_ENABLED": "1",
            "ZOTERO_LOCAL": "true",
            "ZOTERO_MCP_TOOLSETS": "none",
        }
        try:
            with socket.create_connection(("127.0.0.1", 23119), timeout=1):
                pass
        except OSError:
            console.print(
                "[yellow]没有检测到 Zotero Local API。请启动 Zotero，并在“设置 → 高级 → "
                "其他”中启用“允许此计算机上的其他应用与 Zotero 通信”。[/yellow]"
            )
    else:
        values = {
            "ARXIV_MCP_ENABLED": "1",
            "ARXIV_MCP_STORAGE_PATH": "workspace/arxiv-papers",
        }

    for key, value in values.items():
        set_key(ENV_PATH, key, value)
        os.environ[key] = value

    from mortyclaw.core.integrations import load_mcp_tools

    try:
        with Status(f"[bold #8d52ff]正在连接 {normalized} MCP...[/bold #8d52ff]", spinner="dots"):
            tools = load_mcp_tools(strict=True, service=normalized)
    except Exception as exc:
        console.print(
            f"[bold red]{normalized} MCP 已启用，但连接验证失败。[/bold red]\n"
            f"[dim]错误类型：{type(exc).__name__}。请先执行 pip install -e \".[research-mcp]\"。[/dim]"
        )
        raise typer.Exit(code=1)
    console.print(f"[bold green]{normalized} MCP 已连接，共加载 {len(tools)} 个工具。[/bold green]")


@mcp_app.command("disable")
def disable_mcp_service(
    service: str = typer.Argument(..., help="要禁用的服务：zotero 或 arxiv"),
):
    normalized = service.strip().lower()
    key_by_service = {"zotero": "ZOTERO_MCP_ENABLED", "arxiv": "ARXIV_MCP_ENABLED"}
    if normalized not in key_by_service:
        console.print("[bold red]服务只支持 zotero 或 arxiv。[/bold red]")
        raise typer.Exit(code=2)
    _ensure_env_file()
    set_key(ENV_PATH, key_by_service[normalized], "0")
    os.environ[key_by_service[normalized]] = "0"
    console.print(f"[bold green]已禁用 {normalized} MCP。[/bold green]")


@mcp_app.command("status")
def show_mcp_status(
    no_probe: bool = typer.Option(False, "--no-probe", help="只检查配置，不连接 Server"),
):
    """显示 MCP 服务状态，错误信息会自动脱敏。"""
    from mortyclaw.core.integrations import MCPManager

    rows = []
    for status in MCPManager().statuses(probe=not no_probe):
        state = "connected" if status.connected else ("disabled" if not status.enabled else "unavailable")
        rows.append(
            f"{status.name}: {state}, tools={status.tool_count}, "
            f"command={status.command or '-'}, error={status.error or '-'}"
        )
    console.print(Panel("\n".join(rows), title="MCP Status", border_style="#00ffff"))


@app.command("feishu-config")
def configure_feishu(
    identity: str = typer.Option(
        "app",
        "--identity",
        help="鉴权身份：app（应用身份）或 user（用户身份）",
    ),
    tools: str = typer.Option(
        "preset.light",
        "--tools",
        help="飞书 MCP 工具预设或逗号分隔的工具名",
    ),
    domain: str = typer.Option(
        "https://open.feishu.cn",
        "--domain",
        help="飞书开放平台域名；Lark 国际版使用 https://open.larksuite.com",
    ),
):
    """配置并验证飞书官方 MCP。"""
    from mortyclaw.core.integrations import (
        FeishuMCPSettings,
        feishu_oauth_redirect_urls,
        load_mcp_tools,
        run_feishu_oauth_login,
    )

    normalized_identity = identity.strip().lower()
    if normalized_identity not in {"app", "user"}:
        console.print("[bold red]--identity 只支持 app 或 user。[/bold red]")
        raise typer.Exit(code=2)

    load_dotenv(ENV_PATH)
    app_id = questionary.text(
        "输入飞书应用 App ID:",
        default=os.getenv("FEISHU_APP_ID", ""),
        style=morty_style,
    ).ask()
    if not app_id:
        console.print("[dim #8d52ff]未填写 App ID，配置已取消。[/dim #8d52ff]")
        return

    app_secret = questionary.password(
        "输入飞书应用 App Secret:",
        style=morty_style,
    ).ask()
    if app_secret is None:
        console.print("[dim #8d52ff]配置已取消。[/dim #8d52ff]")
        return
    if not app_secret:
        app_secret = os.getenv("FEISHU_APP_SECRET", "")
    if not app_secret:
        console.print("[dim #8d52ff]未填写 App Secret，配置已取消。[/dim #8d52ff]")
        return

    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, "a", encoding="utf-8").close()

    token_mode = "user_access_token" if normalized_identity == "user" else "tenant_access_token"
    oauth = normalized_identity == "user"
    values = {
        "FEISHU_MCP_ENABLED": "1",
        "FEISHU_APP_ID": app_id.strip(),
        "FEISHU_APP_SECRET": app_secret.strip(),
        "FEISHU_MCP_TOOLS": tools.strip() or "preset.light",
        "FEISHU_MCP_DOMAIN": domain.strip() or "https://open.feishu.cn",
        "FEISHU_MCP_LANGUAGE": "zh",
        "FEISHU_MCP_TOKEN_MODE": token_mode,
        "FEISHU_MCP_OAUTH": "1" if oauth else "0",
    }
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)
    for key, value in values.items():
        set_key(ENV_PATH, key, value)
        os.environ[key] = value

    settings = FeishuMCPSettings.from_env()
    try:
        if oauth:
            callback, wrapped_callback = feishu_oauth_redirect_urls()
            console.print(
                "[bold #00ffff]即将打开浏览器，请完成飞书用户授权。[/bold #00ffff]\n"
                "[dim]飞书应用后台需同时配置以下 OAuth 2.0 重定向 URL：[/dim]\n"
                f"[cyan]{callback}[/cyan]\n"
                f"[cyan]{wrapped_callback}[/cyan]"
            )
            run_feishu_oauth_login(settings)
        with Status(
            "[bold #8d52ff]正在连接飞书 MCP 并读取工具列表...[/bold #8d52ff]",
            spinner="dots",
            spinner_style="#00ffff",
        ):
            loaded_tools = load_mcp_tools(strict=True, service="feishu")
    except Exception as exc:
        console.print(
            "[bold red]飞书 MCP 配置已保存，但连接验证失败。[/bold red]\n"
            f"[dim]{exc}[/dim]"
        )
        raise typer.Exit(code=1)

    names = ", ".join(tool.name for tool in loaded_tools[:8])
    if len(loaded_tools) > 8:
        names += ", ..."
    console.print(
        Panel(
            f"已连接飞书 MCP，共加载 {len(loaded_tools)} 个工具。\n"
            f"身份模式：{token_mode}\n"
            f"工具：{names or '(无)'}\n\n"
            "重新执行 mortyclaw run 后即可在对话中使用飞书。",
            title="[bold white]Feishu MCP Connected[/bold white]",
            border_style="#00ffff",
        )
    )


@app.command("feishu-bot")
def run_feishu_bot():
    """启动飞书长连接机器人，将收到的消息交给 MortyClaw 回复。"""
    from mortyclaw.core.integrations import FeishuBotSettings, serve_feishu_bot

    load_dotenv(ENV_PATH)
    from mortyclaw.core.harness.settings import HarnessSettings
    harness_settings = HarnessSettings.from_env()
    try:
        harness_settings.validate()
    except RuntimeError:
        _show_boot_error()
        raise typer.Exit(code=1)
    settings = FeishuBotSettings.from_env()
    try:
        settings.validate()
    except RuntimeError as exc:
        console.print(f"[bold red]飞书机器人配置不完整：[/bold red] {exc}")
        raise typer.Exit(code=1)

    def show_ready() -> None:
        console.print(
            Panel(
                "飞书长连接已建立，MortyClaw 正在等待消息。\n"
                "私聊会直接回复；群聊默认需要先 @机器人。\n"
                "发送 /reset 可清空当前飞书会话上下文。\n\n"
                "按 Ctrl+C 可安全停止。",
                title="[bold white]Feishu Bot Online[/bold white]",
                border_style="#00ffff",
            )
        )

    try:
        asyncio.run(
            serve_feishu_bot(
                settings=settings,
                ready_callback=show_ready,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[dim #8d52ff]飞书机器人已安全停止。[/dim #8d52ff]")
    except Exception as exc:
        console.print(
            "[bold red]飞书机器人启动失败。[/bold red]\n"
            f"[dim]{exc}[/dim]\n\n"
            "请确认飞书开放平台已启用长连接事件订阅，并订阅 im.message.receive_v1。"
        )
        raise typer.Exit(code=1)

@app.command("monitor")
def run_monitor(
    thread_id: str | None = typer.Option(None, "--thread-id", help="指定要监控的会话 thread_id"),
    latest: bool = typer.Option(False, "--latest", help="自动选择最近活跃的会话"),
    list_sessions: bool = typer.Option(False, "--list-sessions", help="列出当前已知会话"),
):
    try:
        import entry.monitor as mortyclaw_monitor
        mortyclaw_monitor.main(thread_id=thread_id, latest=latest, list_sessions=list_sessions)
    except ImportError as e:
        console.print(f"[bold red]启动失败：找不到监视器模块！[/bold red]\n[dim]请确保 monitor.py 和 cli.py 在同一目录下。\n报错信息: {e}[/dim]")


@app.command("heartbeat")
def run_heartbeat(
    interval: int = typer.Option(10, "--interval", min=1, help="轮询检查间隔，单位秒"),
    once: bool = typer.Option(False, "--once", help="只执行一次到期任务扫描"),
):
    from mortyclaw.core.heartbeat import pacemaker_loop, process_due_tasks_with_harness_once

    if once:
        triggered = asyncio.run(process_due_tasks_with_harness_once())
        console.print(f"[bold #00ffff]本次心跳共执行 {len(triggered)} 个到期任务。[/bold #00ffff]")
        return

    console.print(f"[bold #00ffff]Heartbeat 已启动[/bold #00ffff] [dim](interval={interval}s，Ctrl+C 停止)[/dim]")
    try:
        asyncio.run(pacemaker_loop(check_interval=interval))
    except KeyboardInterrupt:
        console.print("[dim #8d52ff]Heartbeat 已停止。[/dim #8d52ff]")


@app.command("sessions")
def list_sessions(limit: int = typer.Option(20, "--limit", min=1, help="最多展示多少个会话")):
    sessions = get_session_repository().list_sessions(limit=limit)
    if not sessions:
        console.print("[dim]当前还没有会话记录。[/dim]")
        return

    lines = []
    for item in sessions:
        lines.append(
            f"- {item['thread_id']} | status={item['status']} | model={item['model'] or 'unknown'} | last_active={item['last_active_at']}"
            f"{_format_session_lineage(item)}"
        )
    console.print("[bold #00ffff]已记录会话：[/bold #00ffff]\n" + "\n".join(lines))


@app.command("workers")
def list_workers(
    parent_thread_id: str = typer.Option("", "--parent-thread-id", help="仅查看指定父会话下的 worker"),
    status_filter: str = typer.Option("", "--status-filter", help="限制状态，例如 pending,running,completed"),
    limit: int = typer.Option(20, "--limit", min=1, max=100, help="最多展示多少条 worker 记录"),
):
    statuses = tuple(item.strip() for item in status_filter.split(",") if item.strip()) or None
    workers = get_worker_run_repository().list_worker_runs(
        parent_thread_id=parent_thread_id.strip(),
        statuses=statuses,
        limit=limit,
    )
    normalized = []
    for item in workers:
        normalized.append({
            "worker_id": item["worker_id"],
            "status": item.get("status", ""),
            "role": item.get("role", ""),
            "parent_thread_id": item.get("parent_thread_id", ""),
            "worker_thread_id": item.get("worker_thread_id", ""),
            "goal": item.get("goal", ""),
            "write_scope": _decode_json_field(item.get("write_scope_json", "[]"), []),
            "allowed_tools": _decode_json_field(item.get("allowed_tools_json", "[]"), []),
            "created_at": item.get("created_at", ""),
            "started_at": item.get("started_at"),
            "finished_at": item.get("finished_at"),
            "result_summary": _decode_json_field(item.get("result_summary_json", "{}"), {}),
            "error": _decode_json_field(item.get("error_json", "{}"), {}),
        })
    console.print(json.dumps({"success": True, "count": len(normalized), "workers": normalized}, ensure_ascii=False, indent=2))


@app.command("session-search")
def session_search(
    query: str = typer.Argument("", help="要搜索的历史关键词；留空则列出最近会话"),
    role_filter: str = typer.Option("", "--role-filter", help="限制角色，例如 user,assistant,tool"),
    limit: int = typer.Option(3, "--limit", min=1, max=5, help="最多返回多少个会话"),
    include_current: bool = typer.Option(False, "--include-current", help="是否包含当前/指定会话"),
    current_thread_id: str | None = typer.Option(None, "--current-thread-id", help="用于排除当前 lineage 的 thread_id"),
    include_tool_results: bool = typer.Option(True, "--tool-results/--no-tool-results", help="是否搜索和展示工具结果"),
):
    roles = [role.strip() for role in role_filter.split(",") if role.strip()]
    results = get_conversation_repository().search_sessions(
        query=query,
        role_filter=roles or None,
        limit=limit,
        include_current=include_current,
        current_thread_id=current_thread_id,
        include_tool_results=include_tool_results,
    )
    console.print(json.dumps({
        "success": True,
        "query": query,
        "mode": "recent" if not query.strip() else "search",
        "count": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2))


@app.command("session-show")
def session_show(
    thread_id: str = typer.Argument(..., help="要展示的会话 thread_id"),
    limit: int = typer.Option(80, "--limit", min=1, help="最多展示多少条消息"),
):
    session = get_session_repository().get_session(thread_id)
    messages = get_conversation_repository().get_session_conversation(thread_id, limit=limit)
    if session is None and not messages:
        console.print(f"[dim]没有找到会话 {thread_id} 的 conversation message 记录。[/dim]")
        return

    header_lines = []
    if session is not None:
        header_lines.append(
            f"thread_id={session['thread_id']} | status={session['status']} | model={session.get('model') or 'unknown'}"
        )
        lineage_hint = _format_session_lineage(session)
        if lineage_hint:
            header_lines.append(lineage_hint.removeprefix(" | "))

    worker_runs = get_worker_run_repository().list_worker_runs(parent_thread_id=thread_id, limit=8)
    if worker_runs:
        header_lines.append("workers:")
        for item in worker_runs[:5]:
            header_lines.append(
                f"  - {item['worker_id']} | role={item.get('role','')} | status={item.get('status','')} | worker_thread={item.get('worker_thread_id','')}"
            )

    lines = []
    for message in messages:
        tool = f" tool={message['tool_name']}" if message.get("tool_name") else ""
        preview = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if len(preview) > 220:
            preview = preview[:219] + "…"
        lines.append(
            f"#{message['seq']:04d} {message['created_at']} {message['role']}{tool}: {preview}"
        )
    output_parts = []
    if header_lines:
        output_parts.append("[bold #00ffff]Session Overview[/bold #00ffff]\n" + "\n".join(header_lines))
    if lines:
        output_parts.append("[bold #00ffff]Conversation Messages[/bold #00ffff]\n" + "\n".join(lines))
    console.print("\n\n".join(output_parts))


@app.command("doctor")
def run_doctor():
    _print_doctor_report(collect_doctor_report())


@gc_app.command("logs")
def run_gc_logs(
    apply: bool = typer.Option(False, "--apply", help="真正执行归档；默认仅 dry-run 预览"),
):
    _print_gc_report("logs", gc_logs(apply=apply))


@gc_app.command("runtime")
def run_gc_runtime(
    apply: bool = typer.Option(False, "--apply", help="真正执行清理；默认仅 dry-run 预览"),
):
    _print_gc_report("runtime", gc_runtime(apply=apply))


@gc_app.command("state")
def run_gc_state(
    apply: bool = typer.Option(False, "--apply", help="真正执行 checkpoint 裁剪；默认仅 dry-run 预览"),
):
    _print_gc_report("state", gc_state(apply=apply))


@app.command("migrate-tasks")
def migrate_tasks(
    source_path: str = typer.Option(TASKS_FILE, "--source", help="旧 tasks.json 的路径"),
    default_thread_id: str = typer.Option("local_geek_master", "--default-thread-id", help="旧任务默认归属的会话 ID"),
    force: bool = typer.Option(False, "--force", help="已存在同 ID 任务时允许覆盖"),
):
    result = get_task_repository().import_legacy_tasks(
        file_path=source_path,
        default_thread_id=default_thread_id,
        overwrite=force,
    )
    console.print(
        f"[bold #00ffff]任务迁移完成[/bold #00ffff] [dim](imported={result['imported']}, skipped={result['skipped']})[/dim]"
    )


@research_app.command("install")
def research_install():
    """下载并校验官方 Qdrant 1.19.1 Windows x64 发行包。"""
    from mortyclaw.core.research.sidecar import QdrantSidecar

    with Status("[bold #8d52ff]正在安装 Qdrant 1.19.1...[/bold #8d52ff]", spinner="dots"):
        path = QdrantSidecar().install()
    console.print(f"[bold green]Qdrant 已安装。[/bold green] [dim]{path}[/dim]")


@research_app.command("start")
def research_start():
    """在后台隐藏启动本地 Qdrant。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research.sidecar import QdrantSidecar

    result = QdrantSidecar().start()
    console.print(json.dumps(result, ensure_ascii=False, indent=2))


@research_app.command("stop")
def research_stop():
    """停止由 MortyClaw 启动的本地 Qdrant。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research.sidecar import QdrantSidecar

    console.print(json.dumps(QdrantSidecar().stop(), ensure_ascii=False, indent=2))


@research_app.command("configure")
def research_configure():
    """启用 Agentic RAG、启动 Qdrant 并创建混合检索 collection。"""
    _ensure_env_file()
    load_dotenv(ENV_PATH, override=True)
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or secrets.token_urlsafe(32)
    values = {
        "RESEARCH_RAG_ENABLED": "1",
        "QDRANT_URL": os.getenv("QDRANT_URL", "http://127.0.0.1:6333").strip()
        or "http://127.0.0.1:6333",
        "QDRANT_API_KEY": api_key,
        "QDRANT_STORAGE_PATH": os.getenv("QDRANT_STORAGE_PATH", "workspace/qdrant").strip()
        or "workspace/qdrant",
        "RESEARCH_COLLECTION": os.getenv("RESEARCH_COLLECTION", "mortyclaw_research_v1").strip()
        or "mortyclaw_research_v1",
        "RESEARCH_EMBEDDING_MODEL": "intfloat/multilingual-e5-large",
    }
    for key, value in values.items():
        set_key(ENV_PATH, key, value)
        os.environ[key] = value
    from mortyclaw.core.research.qdrant import ResearchVectorStore
    from mortyclaw.core.research.embeddings import LazyDenseEmbedding
    from mortyclaw.core.research.settings import ResearchSettings
    from mortyclaw.core.research.sidecar import QdrantSidecar

    settings = ResearchSettings.from_env()
    sidecar = QdrantSidecar(settings)
    if sidecar.executable() is None:
        with Status("[bold #8d52ff]正在安装 Qdrant...[/bold #8d52ff]", spinner="dots"):
            sidecar.install()
    if not sidecar.is_healthy():
        with Status("[bold #8d52ff]正在启动 Qdrant...[/bold #8d52ff]", spinner="dots"):
            sidecar.start()
    with Status("[bold #8d52ff]正在初始化混合检索 collection...[/bold #8d52ff]", spinner="dots"):
        ResearchVectorStore(settings).ensure_collection()
    with Status("[bold #8d52ff]正在下载并验证本地多语言 Embedding...[/bold #8d52ff]", spinner="dots"):
        vector = LazyDenseEmbedding(settings).embed_query("MortyClaw 科研知识检索配置检查")
    if len(vector) != 1024:
        raise RuntimeError(f"Embedding 维度错误：期望 1024，实际 {len(vector)}")
    console.print(
        Panel(
            "Agentic RAG 已启用。Qdrant API Key 已保存到 Git 忽略的 .env。\n"
            "Qdrant collection 与本地 Embedding 模型均已就绪。",
            title="Research RAG Configured",
            border_style="#00ffff",
        )
    )


@research_app.command("status")
def research_status():
    """显示脱敏后的 Qdrant、模型和知识库状态。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research.settings import ResearchSettings
    from mortyclaw.core.research.sidecar import QdrantSidecar
    from mortyclaw.core.research.store import ResearchDocumentRepository

    settings = ResearchSettings.from_env()
    status = QdrantSidecar(settings).status()
    status.update(settings.public_status())
    status.update(ResearchDocumentRepository().counts())
    console.print(json.dumps(status, ensure_ascii=False, indent=2))


@research_app.command("doctor")
def research_doctor():
    """验证 Qdrant、Dense Embedding、BM25 与 RRF 混合查询。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research.embeddings import LazyDenseEmbedding
    from mortyclaw.core.research.qdrant import ResearchVectorStore
    from mortyclaw.core.research.settings import ResearchSettings

    settings = ResearchSettings.from_env()
    if not settings.enabled:
        console.print("[bold red]Agentic RAG 尚未启用，请先执行 mortyclaw research configure。[/bold red]")
        raise typer.Exit(code=1)
    store = ResearchVectorStore(settings)
    store.health()
    store.ensure_collection()
    with Status("[bold #8d52ff]正在加载并验证本地多语言 Embedding...[/bold #8d52ff]", spinner="dots"):
        vector = LazyDenseEmbedding(settings).embed_query("MortyClaw 科研知识检索连接检查")
    store.search(
        query="MortyClaw 科研知识检索连接检查",
        dense_vector=vector,
        sources=["zotero", "feishu", "arxiv"],
        document_ids=None,
        year_from=None,
        year_to=None,
        limit=1,
    )
    console.print(
        f"[bold green]Research doctor passed.[/bold green] "
        f"qdrant=connected dense_dimensions={len(vector)} bm25=connected rrf=connected"
    )


@research_sync_app.command("zotero")
def research_sync_zotero_cli(
    limit: int = typer.Option(1000, "--limit", min=1, max=10000),
    query: str = typer.Option("", "--query", help="只同步匹配主题、标题或作者的条目"),
    force: bool = typer.Option(False, "--force"),
):
    """增量同步本机 Zotero 中可读取全文的条目。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research import ResearchIndexer

    result = ResearchIndexer().sync_zotero(limit=limit, query=query, force=force)
    console.print(json.dumps(result, ensure_ascii=False, indent=2))


@research_add_app.command("feishu")
def research_add_feishu(document_url: str, force: bool = typer.Option(False, "--force")):
    """添加一份明确指定的飞书文档或知识库节点。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research import ResearchIndexer

    console.print(json.dumps(
        ResearchIndexer().index_locator("feishu", document_url, force=force),
        ensure_ascii=False,
        indent=2,
    ))


@research_add_app.command("arxiv")
def research_add_arxiv(locator: str, force: bool = typer.Option(False, "--force")):
    """添加 arXiv ID、已下载论文或本地 PDF。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research import ResearchIndexer

    console.print(json.dumps(
        ResearchIndexer().index_locator("arxiv", locator, force=force),
        ensure_ascii=False,
        indent=2,
    ))


@research_app.command("list")
def research_list(
    source: str = typer.Option("", "--source"),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
):
    """列出科研知识库文档同步状态。"""
    from mortyclaw.core.research.store import ResearchDocumentRepository

    rows = ResearchDocumentRepository().list(source=source.strip().lower(), limit=limit)
    console.print(json.dumps(rows, ensure_ascii=False, indent=2))


@research_app.command("remove")
def research_remove(document_key: str):
    """从本地索引移除文档，不删除原始资料。"""
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research import ResearchIndexer

    console.print(json.dumps(ResearchIndexer().remove(document_key), ensure_ascii=False, indent=2))


@research_app.command("rebuild")
def research_rebuild(
    yes: bool = typer.Option(False, "--yes", help="确认清空本地研究索引状态"),
):
    """清空 Qdrant collection，之后需要重新同步来源。"""
    if not yes:
        console.print("[bold red]该操作会清空本地研究索引，请增加 --yes。[/bold red]")
        raise typer.Exit(code=2)
    load_dotenv(ENV_PATH, override=True)
    from mortyclaw.core.research import ResearchIndexer

    console.print(json.dumps(ResearchIndexer().rebuild(), ensure_ascii=False, indent=2))

@harness_app.command("configure")
def configure_harness():
    """保存 DeepSeek 凭据和 Harness 参数，并执行启动握手。"""
    _ensure_env_file()
    load_dotenv(ENV_PATH)
    existing = os.getenv("DEEPSEEK_API_KEY", "")
    if not existing and "deepseek" in os.getenv("OPENAI_API_BASE", "").lower():
        existing = os.getenv("OPENAI_API_KEY", "")
    api_key = questionary.password(
        "输入 DEEPSEEK_API_KEY" + ("（直接回车保留现有值）" if existing else "") + ":",
        style=morty_style,
    ).ask()
    if api_key is None:
        return
    api_key = api_key.strip() or existing
    if not api_key:
        console.print("[bold red]API Key 不能为空。[/bold red]")
        raise typer.Exit(code=2)
    base_url = questionary.text(
        "自定义 DeepSeek 兼容 Base URL（官方地址直接回车）:", style=morty_style
    ).ask()
    if base_url is None:
        return
    base_url = base_url.strip()
    if base_url.rstrip("/") == "https://api.deepseek.com":
        base_url = ""
    previous_env = {
        key: os.environ.get(key)
        for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "MORTYCLAW_HARNESS_MODEL")
    }
    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["MORTYCLAW_HARNESS_MODEL"] = "deepseek-v4-flash"
    if base_url:
        os.environ["DEEPSEEK_BASE_URL"] = base_url
    else:
        os.environ.pop("DEEPSEEK_BASE_URL", None)
    console.print("[cyan]正在执行 DeepSeek 模型与 Harness 握手…[/cyan]")
    try:
        asyncio.run(_doctor_harness(model_probe=True))
    except Exception as exc:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        console.print(f"[bold red]握手失败：[/bold red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
    set_key(ENV_PATH, "DEEPSEEK_API_KEY", api_key)
    set_key(ENV_PATH, "MORTYCLAW_HARNESS_MODEL", "deepseek-v4-flash")
    set_key(ENV_PATH, "MORTYCLAW_HARNESS_PROFILE", "mortyclaw-sdk")
    if base_url:
        set_key(ENV_PATH, "DEEPSEEK_BASE_URL", base_url)
    else:
        unset_key(ENV_PATH, "DEEPSEEK_BASE_URL")
    console.print("[bold green]DeepSeek Harness 配置成功。[/bold green]")


@harness_app.command("status")
def harness_status():
    """显示脱敏后的 Harness 配置和已安装版本。"""
    from importlib.metadata import PackageNotFoundError, version
    from mortyclaw.core.harness.settings import HarnessSettings
    settings = HarnessSettings.from_env()
    try:
        sdk = version("deepseek-harness-sdk")
    except PackageNotFoundError:
        sdk = "not-installed"
    try:
        runtime = version("deepseek-harness-runtime-bin")
    except PackageNotFoundError:
        runtime = "not-installed"
    async def probe() -> tuple[bool, int, str]:
        from mortyclaw.core.harness import HarnessRuntime
        instance = HarnessRuntime()
        try:
            await instance.start()
            return True, instance.gateway.tool_count, ""
        except Exception as exc:
            return False, 0, type(exc).__name__
        finally:
            await instance.close()
    connected, tool_count, error = asyncio.run(probe())
    data = settings.public_status()
    lines = [
        f"sdk={sdk}", f"runtime={runtime}", f"connected={connected}",
        f"tool_count={tool_count}", f"error={error or '-'}",
    ] + [f"{key}={value}" for key, value in data.items()]
    console.print("\n".join(lines))


async def _doctor_harness(*, model_probe: bool = False) -> None:
    from mortyclaw.core.config import PROJECT_ROOT
    from mortyclaw.core.harness import HarnessRuntime, new_turn_request
    runtime = HarnessRuntime()
    try:
        await runtime.start()
        console.print(f"gateway=connected tools={runtime.gateway.tool_count}")
        if model_probe:
            result = await runtime.run_turn(new_turn_request(
                thread_id="harness-doctor",
                text="这是连接检查。只回复 OK。",
                source="cli",
                workspace=PROJECT_ROOT,
            ))
            console.print(f"model=connected finish_reason={result.finish_reason}")
    finally:
        await runtime.close()


@harness_app.command("doctor")
def harness_doctor():
    """启动 SDK、runtime 和 Gateway，验证完整本地链路。"""
    asyncio.run(_doctor_harness(model_probe=True))


@gateway_app.command("status")
def gateway_status():
    """发现并统计 Gateway 工具，不显示 URL 或 Token。"""
    async def probe():
        from mortyclaw.core.harness.gateway import MortyClawGateway
        gateway = MortyClawGateway()
        try:
            await gateway.start()
            return gateway.tool_count, gateway.error
        finally:
            await gateway.close()
    count, error = asyncio.run(probe())
    console.print(f"connected={not bool(error)} tool_count={count} error={error or '-'}")


@approvals_app.command("list")
def approvals_list(status: str = typer.Option("pending", "--status")):
    from mortyclaw.core.harness.storage import HarnessStore
    rows = HarnessStore().list_batches(status=status)
    console.print("\n".join(
        f"{row['batch_id']} | {row['status']} | {row['thread_id']} | expires={row['expires_at']}"
        for row in rows
    ) or "没有匹配的审批批次。")


@approvals_app.command("reject")
def approvals_reject(batch_id: str):
    async def reject():
        from mortyclaw.core.harness import HarnessRuntime
        runtime = HarnessRuntime()
        try:
            return await runtime.resolve_approval(batch_id, approved=False)
        finally:
            await runtime.close()
    result = asyncio.run(reject())
    console.print(result.final_response)


@approvals_app.command("approve")
def approvals_approve(batch_id: str):
    async def execute():
        from mortyclaw.core.harness import HarnessRuntime
        runtime = HarnessRuntime()
        try:
            return await runtime.resolve_approval(batch_id, approved=True)
        finally:
            await runtime.close()
    result = asyncio.run(execute())
    console.print(result.final_response)


def main():
    app()

if __name__ == "__main__":
    main()
