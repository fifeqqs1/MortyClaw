from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, TypeVar

from langchain_core.tools import BaseTool

from ..tools.meta import ToolMeta, attach_tool_meta


_T = TypeVar("_T")
FEISHU_TOOL_PREFIX = "feishu_"
_WRITE_ACTIONS = {
    "add",
    "batchcreate",
    "batchdelete",
    "batchupdate",
    "create",
    "delete",
    "import",
    "patch",
    "remove",
    "reply",
    "send",
    "subscribe",
    "update",
}
_READ_ACTIONS = {
    "download",
    "get",
    "list",
    "raw",
    "rawcontent",
    "read",
    "search",
}


def feishu_oauth_redirect_urls(
    host: str = "localhost",
    port: int = 3000,
) -> tuple[str, str]:
    """Return every redirect URL used by lark-mcp's local OAuth login flow."""
    callback = f"http://{host}:{port}/callback"
    # lark-mcp 0.5.1 wraps the local callback so it can forward OAuth clients
    # after Feishu returns. Feishu validates this full URL, including its query.
    wrapped_callback = f"{callback}?redirect_uri={callback}"
    return callback, wrapped_callback


def _env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FeishuMCPSettings:
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    tools: str = "preset.light"
    domain: str = "https://open.feishu.cn"
    token_mode: str = "tenant_access_token"
    user_access_token: str = ""
    oauth: bool = False
    language: str = "zh"
    command: str = ""

    @classmethod
    def from_env(cls) -> "FeishuMCPSettings":
        return cls(
            enabled=_env_flag("FEISHU_MCP_ENABLED"),
            app_id=os.getenv("FEISHU_APP_ID", "").strip(),
            app_secret=os.getenv("FEISHU_APP_SECRET", "").strip(),
            tools=os.getenv("FEISHU_MCP_TOOLS", "preset.light").strip() or "preset.light",
            domain=os.getenv("FEISHU_MCP_DOMAIN", "https://open.feishu.cn").strip()
            or "https://open.feishu.cn",
            token_mode=os.getenv("FEISHU_MCP_TOKEN_MODE", "tenant_access_token").strip()
            or "tenant_access_token",
            user_access_token=os.getenv("FEISHU_USER_ACCESS_TOKEN", "").strip(),
            oauth=_env_flag("FEISHU_MCP_OAUTH"),
            language=os.getenv("FEISHU_MCP_LANGUAGE", "zh").strip() or "zh",
            command=os.getenv("FEISHU_MCP_COMMAND", "").strip(),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.app_id or not self.app_secret:
            raise RuntimeError(
                "飞书 MCP 已启用，但缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET。"
                "请先执行 mortyclaw feishu-config。"
            )
        if self.token_mode not in {"auto", "tenant_access_token", "user_access_token"}:
            raise RuntimeError(
                "FEISHU_MCP_TOKEN_MODE 只支持 auto、tenant_access_token 或 user_access_token。"
            )
        if self.language not in {"zh", "en"}:
            raise RuntimeError("FEISHU_MCP_LANGUAGE 只支持 zh 或 en。")


def _resolve_npx_command(explicit_command: str = "") -> str:
    if explicit_command:
        return explicit_command
    resolved = shutil.which("npx") or shutil.which("npx.cmd")
    if not resolved:
        raise RuntimeError(
            "没有找到 npx。飞书官方 MCP 需要 Node.js 20+，请先安装 Node.js LTS。"
        )
    return resolved


def build_feishu_mcp_connection(settings: FeishuMCPSettings) -> dict[str, Any]:
    settings.validate()
    child_env = dict(os.environ)
    child_env.update(
        {
            "APP_ID": settings.app_id,
            "APP_SECRET": settings.app_secret,
            "LARK_TOOLS": settings.tools,
            "LARK_DOMAIN": settings.domain,
            "LARK_TOKEN_MODE": settings.token_mode,
        }
    )
    if settings.user_access_token:
        child_env["USER_ACCESS_TOKEN"] = settings.user_access_token

    args = [
        "-y",
        "@larksuiteoapi/lark-mcp",
        "mcp",
        "--language",
        settings.language,
        "--tool-name-case",
        "snake",
    ]
    if settings.oauth:
        args.append("--oauth")

    return {
        "transport": "stdio",
        "command": _resolve_npx_command(settings.command),
        "args": args,
        "env": child_env,
    }


def is_feishu_tool(tool_or_name: BaseTool | str | None) -> bool:
    if isinstance(tool_or_name, str):
        name = tool_or_name
    else:
        name = str(getattr(tool_or_name, "name", "") or "")
    return name.startswith(FEISHU_TOOL_PREFIX)


def feishu_tool_meta(tool: BaseTool) -> ToolMeta:
    name = str(getattr(tool, "name", "") or "").strip()
    # Official tool names may retain camelCase operation names even when snake
    # naming is requested (for example appTableRecord_batchCreate/getNode).
    action = name.rsplit("_", 1)[-1].lower()
    is_write = any(marker in action for marker in _WRITE_ACTIONS)
    is_read = any(marker in action for marker in _READ_ACTIONS)

    if is_read and not is_write:
        return ToolMeta.build(
            name=name,
            capabilities={"feishu_read", "external_read"},
            risk_level="low",
            allowed_routes={"fast", "slow"},
        )

    return ToolMeta.build(
        name=name,
        capabilities={"feishu_write", "external_write"},
        risk_level="high",
        allowed_routes={"slow"},
        requires_approval=True,
    )


async def _load_feishu_mcp_tools_async(settings: FeishuMCPSettings) -> list[BaseTool]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError(
            "缺少 langchain-mcp-adapters，请重新执行 pip install -e . 安装项目依赖。"
        ) from exc

    client = MultiServerMCPClient(
        {"feishu": build_feishu_mcp_connection(settings)},
        tool_name_prefix=True,
        handle_tool_errors=True,
    )
    tools = await client.get_tools()
    return [attach_tool_meta(tool, feishu_tool_meta(tool)) for tool in tools]


# Backward-compatible private alias for existing callers and tests.
_feishu_tool_meta = feishu_tool_meta


def _run_awaitable_blocking(awaitable: Awaitable[_T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # create_agent_app is synchronous and is also called from MortyClaw's async CLI.
    # Run discovery on an isolated loop so startup remains compatible with both callers.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="feishu-mcp-loader") as pool:
        return pool.submit(asyncio.run, awaitable).result()


def load_feishu_mcp_tools(
    settings: FeishuMCPSettings | None = None,
    *,
    strict: bool = True,
) -> list[BaseTool]:
    resolved = settings or FeishuMCPSettings.from_env()
    if not resolved.enabled:
        return []
    try:
        resolved.validate()
        return _run_awaitable_blocking(_load_feishu_mcp_tools_async(resolved))
    except Exception as exc:
        if strict:
            raise
        warnings.warn(f"飞书 MCP 暂时不可用，MortyClaw 将在不加载飞书工具的情况下启动：{exc}")
        return []


def run_feishu_oauth_login(settings: FeishuMCPSettings | None = None) -> None:
    resolved = settings or FeishuMCPSettings.from_env()
    resolved.validate()
    child_env = build_feishu_mcp_connection(resolved)["env"]
    command = _resolve_npx_command(resolved.command)
    args = [
        "-y",
        "@larksuiteoapi/lark-mcp",
        "login",
        "--domain",
        resolved.domain,
    ]
    subprocess.run([command, *args], env=child_env, check=True)
