from __future__ import annotations

import asyncio
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Iterable, TypeVar

from langchain_core.tools import BaseTool

from ..tools.meta import ToolMeta, attach_tool_meta


_T = TypeVar("_T")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _install_quiet_stdio_transport() -> None:
    """Keep third-party MCP stderr out of user logs and status output."""
    import langchain_mcp_adapters.sessions as adapter_sessions
    if getattr(adapter_sessions, "_mortyclaw_quiet_stdio", False):
        return
    from mcp.client.stdio import stdio_client as mcp_stdio_client

    @asynccontextmanager
    async def quiet_stdio_client(server):
        with open(os.devnull, "w", encoding="utf-8") as sink:
            async with mcp_stdio_client(server, errlog=sink) as streams:
                yield streams

    adapter_sessions.stdio_client = quiet_stdio_client
    adapter_sessions._mortyclaw_quiet_stdio = True

ZOTERO_READ_PREFIXES = (
    "search_", "get_", "list_", "read_", "find_", "export_", "retrieve_",
)
ZOTERO_READ_TOOLS = {
    "advanced_search",
    "synthesize_annotations",
}
ZOTERO_DISABLED_TOOLS = {
    # zotero-mcp implements this through its optional ChromaDB index. MortyClaw
    # has one shared Qdrant research index instead, exposed as research_retrieve.
    "semantic_search",
}
ARXIV_WRITE_TOOLS = {
    "download_paper",
    "watch_topic",
    "unwatch_topic",
    "check_alerts",
    "reindex",
}


def _env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_command(explicit: str, executable: str) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise RuntimeError(f"找不到配置的命令：{explicit}")

    names = [executable]
    if os.name == "nt" and not executable.lower().endswith(".exe"):
        names.insert(0, f"{executable}.exe")
    script_dir = "Scripts" if os.name == "nt" else "bin"
    for name in names:
        local = PROJECT_ROOT / ".venv" / script_dir / name
        if local.exists():
            return str(local)
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise RuntimeError(
        f"找不到 {executable}。请执行 pip install -e \".[research-mcp]\" 安装研究 MCP 依赖。"
    )


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    enabled: bool
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    read_only: bool = False

    def connection(self) -> dict[str, Any]:
        if not self.command:
            raise RuntimeError(f"{self.name} MCP 缺少启动命令")
        return {
            "transport": "stdio",
            "command": self.command,
            "args": list(self.args),
            "env": self.env,
        }


@dataclass(frozen=True)
class MCPServiceStatus:
    name: str
    enabled: bool
    command: str = ""
    connected: bool = False
    tool_count: int = 0
    error: str = ""


def build_zotero_config() -> MCPServerConfig:
    enabled = _env_flag("ZOTERO_MCP_ENABLED")
    command = ""
    if enabled:
        command = _resolve_command(os.getenv("ZOTERO_MCP_COMMAND", "").strip(), "zotero-mcp")
    # First release is deliberately local and read-only. No parent credentials
    # are forwarded to this child process.
    child_env = {
        "ZOTERO_LOCAL": "true",
        "ZOTERO_MCP_TOOLSETS": os.getenv("ZOTERO_MCP_TOOLSETS", "none").strip() or "none",
    }
    return MCPServerConfig(
        name="zotero",
        enabled=enabled,
        command=command,
        args=("serve", "--transport", "stdio"),
        env=child_env,
        read_only=True,
    )


def build_arxiv_config() -> MCPServerConfig:
    enabled = _env_flag("ARXIV_MCP_ENABLED", True)
    command = ""
    if enabled:
        command = _resolve_command(os.getenv("ARXIV_MCP_COMMAND", "").strip(), "arxiv-mcp-server")
    storage = os.getenv("ARXIV_MCP_STORAGE_PATH", "workspace/arxiv-papers").strip()
    storage_path = Path(storage).expanduser()
    if not storage_path.is_absolute():
        storage_path = PROJECT_ROOT / storage_path
    return MCPServerConfig(
        name="arxiv",
        enabled=enabled,
        command=command,
        args=("--storage-path", str(storage_path.resolve())),
        env={},
    )


def _bare_tool_name(service: str, name: str) -> str:
    prefix = f"{service}_"
    while name.startswith(prefix):
        name = name[len(prefix):]
    return name


def _canonical_tool_name(service: str, name: str) -> str:
    return f"{service}_{_bare_tool_name(service, name)}"


def _tool_meta(service: str, tool: BaseTool) -> ToolMeta | None:
    name = str(getattr(tool, "name", "") or "")
    bare = _bare_tool_name(service, name).lower()
    if service == "zotero":
        if bare in ZOTERO_DISABLED_TOOLS:
            return None
        if not (bare.startswith(ZOTERO_READ_PREFIXES) or bare in ZOTERO_READ_TOOLS):
            return None
        return ToolMeta.build(
            name=name,
            capabilities={"zotero_read", "external_read", "research_read"},
            risk_level="low",
        )
    if service == "arxiv":
        if bare in ARXIV_WRITE_TOOLS:
            return ToolMeta.build(
                name=name,
                capabilities={"arxiv_write", "external_write", "research_write"},
                risk_level="high",
                requires_approval=True,
            )
        read_prefixes = (
            "search_", "semantic_", "get_", "list_", "read_", "export_", "citation_",
        )
        if bare.startswith(read_prefixes):
            return ToolMeta.build(
                name=name,
                capabilities={"arxiv_read", "external_read", "research_read"},
                risk_level="low",
            )
        return ToolMeta.build(
            name=name,
            capabilities={"arxiv_unknown", "external_write"},
            risk_level="high",
            requires_approval=True,
        )
    from .feishu_mcp import feishu_tool_meta

    return feishu_tool_meta(tool)


async def _discover_tools(config: MCPServerConfig) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _install_quiet_stdio_transport()

    client = MultiServerMCPClient(
        {config.name: config.connection()},
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    discovered = await client.get_tools()
    tools: list[BaseTool] = []
    for tool in discovered:
        tool.name = _canonical_tool_name(config.name, str(getattr(tool, "name", "") or ""))
        meta = _tool_meta(config.name, tool)
        if meta is not None:
            tools.append(attach_tool_meta(tool, meta))
    return tools


def _run_awaitable_blocking(awaitable: Awaitable[_T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-manager") as pool:
        return pool.submit(asyncio.run, awaitable).result()


class MCPManager:
    def __init__(self, configs: Iterable[MCPServerConfig] | None = None):
        self._configs = list(configs) if configs is not None else self.configs_from_env()

    @staticmethod
    def configs_from_env() -> list[MCPServerConfig]:
        configs: list[MCPServerConfig] = []
        from .feishu_mcp import FeishuMCPSettings, build_feishu_mcp_connection

        feishu = FeishuMCPSettings.from_env()
        if feishu.enabled:
            connection = build_feishu_mcp_connection(feishu)
            configs.append(MCPServerConfig(
                name="feishu",
                enabled=True,
                command=connection["command"],
                args=tuple(connection["args"]),
                env=connection["env"],
            ))
        for builder in (build_zotero_config, build_arxiv_config):
            try:
                configs.append(builder())
            except Exception as exc:
                name = "zotero" if builder is build_zotero_config else "arxiv"
                configs.append(MCPServerConfig(name=name, enabled=True, command=""))
                warnings.warn(f"{name} MCP 配置不可用：{type(exc).__name__}")
        return configs

    def load_tools(self, *, strict: bool = False, service: str | None = None) -> list[BaseTool]:
        loaded: list[BaseTool] = []
        for config in self._configs:
            if not config.enabled or (service and config.name != service):
                continue
            try:
                loaded.extend(_run_awaitable_blocking(_discover_tools(config)))
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"{config.name} MCP 连接失败：{type(exc).__name__}") from exc
                warnings.warn(f"{config.name} MCP 暂时不可用：{type(exc).__name__}")
        return loaded

    def statuses(self, *, probe: bool = True) -> list[MCPServiceStatus]:
        statuses: list[MCPServiceStatus] = []
        known = {config.name: config for config in self._configs}
        for name in ("feishu", "zotero", "arxiv"):
            config = known.get(name)
            if config is None:
                statuses.append(MCPServiceStatus(name=name, enabled=False))
                continue
            if not config.enabled:
                statuses.append(MCPServiceStatus(name=name, enabled=False, command=config.command))
                continue
            if not config.command:
                statuses.append(MCPServiceStatus(name=name, enabled=True, error="command_not_found"))
                continue
            if not probe:
                statuses.append(MCPServiceStatus(name=name, enabled=True, command=config.command))
                continue
            try:
                tools = _run_awaitable_blocking(_discover_tools(config))
                statuses.append(MCPServiceStatus(
                    name=name, enabled=True, command=config.command,
                    connected=True, tool_count=len(tools),
                ))
            except Exception as exc:
                statuses.append(MCPServiceStatus(
                    name=name, enabled=True, command=config.command,
                    error=type(exc).__name__,
                ))
        return statuses


def load_mcp_tools(*, strict: bool = False, service: str | None = None) -> list[BaseTool]:
    return MCPManager().load_tools(strict=strict, service=service)


def is_mcp_tool(tool_or_name: BaseTool | str | None, service: str | None = None) -> bool:
    name = tool_or_name if isinstance(tool_or_name, str) else getattr(tool_or_name, "name", "")
    normalized = str(name or "")
    services = (service,) if service else ("feishu", "zotero", "arxiv")
    return any(normalized.startswith(f"{item}_") for item in services)


__all__ = [
    "ARXIV_WRITE_TOOLS", "MCPManager", "MCPServerConfig", "MCPServiceStatus",
    "build_arxiv_config", "build_zotero_config", "is_mcp_tool", "load_mcp_tools",
]
