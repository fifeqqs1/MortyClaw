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

from mortyclaw.core.provider import get_provider
from mortyclaw.core.provider import (
    get_compatible_provider_api_key_env_vars,
    get_compatible_provider_base_url_env_vars,
)
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
    get_tool_program_run_repository,
    get_worker_run_repository,
)
from langchain_core.messages import HumanMessage

ENTRY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ENTRY_DIR) 

os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = typer.Typer(help="MortyClaw - 极客专属的赛博智能终端")
gc_app = typer.Typer(help="运行态垃圾回收与归档工具")
mcp_app = typer.Typer(help="配置、检查和禁用 MCP 服务")
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


def _compatible_provider_primary_api_key(provider: str) -> str:
    return get_compatible_provider_api_key_env_vars(provider)[0]


def _compatible_provider_primary_base_key(provider: str) -> str:
    return get_compatible_provider_base_url_env_vars(provider)[0]


def _has_configured_compatible_provider_key(provider: str) -> bool:
    return any(os.getenv(env_key) for env_key in get_compatible_provider_api_key_env_vars(provider))


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
    console.clear()
    console.print(Panel(
        "😈 Welcome to [bold #8d52ff]MortyClaw[/bold #8d52ff]...\n\n☁️[dim] 请完成模型配置，我们将把密钥安全固化在本地。[/dim]",
        title="[bold white]✦  MortyClaw Config[/bold white]",
        border_style="#8d52ff"
    ))
    provider_raw = questionary.select(
        "选择你的模型提供商 (Provider):",
        choices=["openai", "anthropic", "aliyun (openai compatible)","tencent (openai compatible)", "z.ai (openai compatible)", "other (openai compatible)", "ollama"],
        style=morty_style,
        instruction="(按上下键选择，回车确认)"
    ).ask()

    if not provider_raw:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    provider = provider_raw.split(" ")[0].strip()
    is_openai_compatible = "openai" in provider_raw.lower()

    model_name = questionary.text(
        "输入指定的模型型号 (如 gpt-4o-mini, qwen-max, glm-4 等):",
        style=morty_style
    ).ask()

    if model_name is None:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    api_key = ""
    env_key = ""
    if provider != "ollama":
        if is_openai_compatible or provider in ["aliyun", "dashscope", "z.ai", "tencent", "other"]:
            env_key = _compatible_provider_primary_api_key(provider)
        elif provider == "anthropic":
            env_key = "ANTHROPIC_API_KEY"

        api_key = questionary.password(
            f"输入你的 {env_key} (对应 {provider_raw}):",
            style=morty_style
        ).ask()

        if api_key is None:
            console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
            return

    base_url = ""
    if provider in ["openai", "anthropic"]:
        base_url = questionary.text(
            f"输入 {provider} 代理 Base URL (直连请直接回车跳过):",
            style=morty_style
        ).ask()
    elif provider == "ollama":
        base_url = questionary.text(
            "输入 Ollama Base URL (默认 http://localhost:11434，直接回车跳过):",
            style=morty_style
        ).ask()
    else:
        base_url = questionary.text(
            "输入兼容 Base URL (不填直接回车将使用官方默认地址):",
            style=morty_style
        ).ask()

    if base_url is None:
        console.print("[dim #8d52ff]✦   录入中断，MortyClaw 配置已取消。[/dim #8d52ff]")
        return

    console.print("\n[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")

    with Status(f"[bold #8d52ff]正在连接 {provider.upper()} 引擎并发送探测包...[/bold #8d52ff]", spinner="dots", spinner_style="#00ffff"):
        try:
            if env_key and api_key:
                os.environ[env_key] = api_key
            if base_url:
                if is_openai_compatible or provider in ["aliyun", "dashscope", "z.ai", "tencent", "other"]:
                    os.environ[_compatible_provider_primary_base_key(provider)] = base_url
                else:
                    os.environ[f"{provider.upper()}_BASE_URL"] = base_url

            llm = get_provider(provider_name=provider, model_name=model_name)
            response = llm.invoke([HumanMessage(content="回复我'收到'。")])

            console.print(" [bold #00ffff][ 配置成功!][/bold #00ffff]")
            
        except Exception as e:

            console.print(f" [bold #8d52ff][ 配置失败!][/bold #8d52ff]  无法连接到模型，请检查 Key、Base URL、模型型号 或 网络！\n[dim]错误信息: {str(e)}[/dim]")
            return


    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, 'w').close()

    logging.getLogger("dotenv.main").setLevel(logging.ERROR)

    unset_key(ENV_PATH, "OPENAI_API_BASE")
    unset_key(ENV_PATH, "ALIYUN_BASE_URL")
    unset_key(ENV_PATH, "DASHSCOPE_BASE_URL")
    unset_key(ENV_PATH, "ZAI_BASE_URL")
    unset_key(ENV_PATH, "TENCENT_BASE_URL")
    unset_key(ENV_PATH, "ANTHROPIC_BASE_URL")
    unset_key(ENV_PATH, "OLLAMA_BASE_URL")

    if env_key and api_key:
        set_key(ENV_PATH, env_key, api_key)
        
    if base_url:
        if is_openai_compatible or provider in ["aliyun", "dashscope", "z.ai", "tencent", "other"]:
            set_key(ENV_PATH, _compatible_provider_primary_base_key(provider), base_url)
        else:
            set_key(ENV_PATH, f"{provider.upper()}_BASE_URL", base_url)
    
    set_key(ENV_PATH, "DEFAULT_PROVIDER", provider)
    set_key(ENV_PATH, "DEFAULT_MODEL", model_name)

    local_windows_launcher = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "mortyclaw.exe")
    launch_command = (
        r".\.venv\Scripts\mortyclaw.exe run"
        if os.name == "nt" and os.path.exists(local_windows_launcher)
        else "mortyclaw run"
    )
    console.print(Panel(
        f"配置已保存至 [#8d52ff]{ENV_PATH}[/#8d52ff]\n"
        f"当前默认提供商: [#8d52ff]{provider}[/#8d52ff] | 模型: [#8d52ff]{model_name}[/#8d52ff]\n\n"
        f"👉 输入 [bold #00ffff]{launch_command}[/bold #00ffff] 即可启动系统！",
        border_style="#00ffff"
    ))

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
    provider = os.getenv("DEFAULT_PROVIDER")
    model = os.getenv("DEFAULT_MODEL")
    if not provider or not model:
        _show_boot_error()
        raise typer.Exit()
    if provider != "ollama":
        if provider in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
            if not _has_configured_compatible_provider_key(provider):
                _show_boot_error()
                raise typer.Exit()
                
        elif provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                _show_boot_error()
                raise typer.Exit()

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
    provider = os.getenv("DEFAULT_PROVIDER", "").strip()
    model = os.getenv("DEFAULT_MODEL", "").strip()
    if not provider or not model:
        _show_boot_error()
        raise typer.Exit(code=1)
    if provider != "ollama":
        if provider in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
            if not _has_configured_compatible_provider_key(provider):
                _show_boot_error()
                raise typer.Exit(code=1)
        elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
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
                provider=provider,
                model=model,
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
    from mortyclaw.core.heartbeat import pacemaker_loop, process_due_tasks_once

    if once:
        triggered = process_due_tasks_once()
        console.print(f"[bold #00ffff]本次心跳共投递 {len(triggered)} 个到期任务。[/bold #00ffff]")
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


@app.command("program-runs")
def list_program_runs(
    thread_id: str = typer.Option("", "--thread-id", help="仅查看指定会话下的程序化执行记录"),
    status_filter: str = typer.Option("", "--status-filter", help="限制状态，例如 awaiting_approval,completed"),
    limit: int = typer.Option(20, "--limit", min=1, max=100, help="最多展示多少条程序运行记录"),
):
    statuses = tuple(item.strip() for item in status_filter.split(",") if item.strip()) or None
    runs = get_tool_program_run_repository().list_program_runs(
        thread_id=thread_id.strip(),
        statuses=statuses,
        limit=limit,
    )
    normalized = []
    for item in runs:
        normalized.append({
            "program_run_id": item["program_run_id"],
            "thread_id": item.get("thread_id", ""),
            "status": item.get("status", ""),
            "pc": item.get("pc", 0),
            "created_at": item.get("created_at", ""),
            "updated_at": item.get("updated_at", ""),
            "finished_at": item.get("finished_at"),
            "metadata": _decode_json_field(item.get("metadata_json", "{}"), {}),
            "result_summary": _decode_json_field(item.get("result_summary_json", "{}"), {}),
        })
    console.print(json.dumps({"success": True, "count": len(normalized), "program_runs": normalized}, ensure_ascii=False, indent=2))


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

    program_runs = get_tool_program_run_repository().list_program_runs(thread_id=thread_id, limit=8)
    if program_runs:
        header_lines.append("program_runs:")
        for item in program_runs[:5]:
            header_lines.append(
                f"  - {item['program_run_id']} | status={item.get('status','')} | pc={item.get('pc', 0)} | updated_at={item.get('updated_at','')}"
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

def main():
    app()

if __name__ == "__main__":
    main()
