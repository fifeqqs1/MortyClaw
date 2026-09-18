from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import io
import json
import logging
import os
import secrets
import socket
import threading
from dataclasses import dataclass
from typing import Any

import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..integrations.mcp_manager import MCPManager
from ..runtime.context import set_active_thread_id
from ..tools.builtins.registry import BUILTIN_TOOLS
from ..tools.meta import ToolMeta, get_tool_meta
from .storage import HarnessStore

logger = logging.getLogger(__name__)

_EXCLUDED_BUILTINS = {
    "tavily_web_search",
}

_PUBLIC_NAMES = {
    "get_current_time": "task_get_current_time",
    "calculator": "project_calculator",
    "get_system_model_info": "project_get_system_model_info",
    "save_user_profile": "memory_save_user_profile",
    "search_sessions": "memory_search_sessions",
    "schedule_task": "task_schedule",
    "list_scheduled_tasks": "task_list",
    "delete_scheduled_task": "task_delete",
    "modify_scheduled_task": "task_modify",
    "list_office_files": "project_list_office_files",
    "read_office_file": "project_read_office_file",
    "write_office_file": "project_write_office_file",
    "execute_office_shell": "project_execute_office_shell",
    "read_project_file": "project_read_file",
    "search_project_code": "project_search_code",
    "edit_project_file": "project_edit_file",
    "write_project_file": "project_write_file",
    "apply_project_patch": "project_apply_patch",
    "show_git_diff": "project_show_git_diff",
    "run_project_tests": "project_run_tests",
    "run_project_command": "project_run_command",
}

_SIDE_EFFECT_CAPABILITIES = {
    "file_write", "file_delete", "shell_exec", "program_exec", "task_write",
    "memory_write", "external_write", "office_write", "project_write",
}


@dataclass
class GatewayTool:
    public_name: str
    tool: Any
    meta: ToolMeta


class _BearerMiddleware:
    def __init__(self, app: Any, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        headers = dict(scope.get("headers") or [])
        if headers.get(b"authorization", b"") != self.expected:
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _schema_for(tool: Any) -> dict[str, Any]:
    schema: dict[str, Any]
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None and hasattr(args_schema, "model_json_schema"):
        schema = args_schema.model_json_schema()
    else:
        schema = dict(getattr(tool, "args", {}) or {})
        if "type" not in schema:
            schema = {"type": "object", "properties": schema}
    schema = json.loads(json.dumps(schema, default=str))
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})["context_token"] = {
        "type": "string", "description": "MortyClaw 为当前轮次签发的短期上下文令牌。"
    }
    required = list(schema.get("required") or [])
    if "context_token" not in required:
        required.append("context_token")
    schema["required"] = required
    return schema


def _is_high_risk(meta: ToolMeta) -> bool:
    return (
        meta.requires_approval
        or meta.risk_level != "low"
        or not meta.capabilities.isdisjoint(_SIDE_EFFECT_CAPABILITIES)
    )


class MortyClawGateway:
    """Authenticated loopback MCP server exposing governed MortyClaw tools."""

    def __init__(self, *, store: HarnessStore | None = None):
        self.store = store or HarnessStore()
        self.token = secrets.token_urlsafe(36)
        self.host = "127.0.0.1"
        self.port = 0
        self.url = ""
        self._tools: dict[str, GatewayTool] = {}
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._ready = threading.Event()
        self._error = ""
        self.approval_ttl_seconds = max(
            60, int(os.getenv("MORTYCLAW_APPROVAL_TTL_SECONDS", "900") or 900)
        )

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def error(self) -> str:
        return self._error

    async def discover_tools(self) -> None:
        tools = [tool for tool in BUILTIN_TOOLS if str(getattr(tool, "name", "")) not in _EXCLUDED_BUILTINS]
        external = await asyncio.to_thread(self._load_external_tools_quietly)
        tools.extend(external)
        catalog: dict[str, GatewayTool] = {}
        for tool in tools:
            original = str(getattr(tool, "name", "") or "")
            if not original:
                continue
            public = _PUBLIC_NAMES.get(original, original)
            catalog[public] = GatewayTool(public, tool, get_tool_meta(tool))
        self._tools = catalog

    @staticmethod
    def _load_external_tools_quietly() -> list[Any]:
        # Third-party stdio servers may print banners or environment diagnostics.
        # The Gateway exposes only sanitized service state through MortyClaw commands.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return MCPManager().load_tools()

    async def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._error = ""
        self._server = None
        await self.discover_tools()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, 0))
            self.port = int(sock.getsockname()[1])
        self.url = f"http://{self.host}:{self.port}/mcp"
        self._thread = threading.Thread(target=self._run_server, name="mortyclaw-mcp-gateway", daemon=True)
        self._thread.start()
        if not await asyncio.to_thread(self._ready.wait, 10):
            raise RuntimeError("MCP Gateway 启动超时")
        if self._error:
            raise RuntimeError(f"MCP Gateway 启动失败：{self._error}")
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            try:
                _reader, writer = await asyncio.open_connection(self.host, self.port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("MCP Gateway 未开始监听")
                await asyncio.sleep(0.05)

    def _run_server(self) -> None:
        try:
            server: Server = Server("mortyclaw")

            @server.list_tools()
            async def list_tools() -> list[types.Tool]:
                return [
                    types.Tool(
                        name=item.public_name,
                        description=str(getattr(item.tool, "description", "") or ""),
                        inputSchema=_schema_for(item.tool),
                    )
                    for item in self._tools.values()
                ]

            @server.call_tool(validate_input=True)
            async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                return await self.call(name, arguments)

            manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)
            endpoint = _BearerMiddleware(StreamableHTTPASGIApp(manager), self.token)
            app = Starlette(
                routes=[Route("/mcp", endpoint=endpoint, methods=["GET", "POST", "DELETE"])],
                lifespan=lambda _app: manager.run(),
            )
            config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", access_log=False)
            self._server = uvicorn.Server(config)
            self._ready.set()
            asyncio.run(self._server.serve())
        except Exception as exc:  # pragma: no cover - startup diagnostics
            self._error = type(exc).__name__
            self._ready.set()
            logger.error("MortyClaw MCP Gateway failed: %s", type(exc).__name__)

    async def close(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            await asyncio.to_thread(self._thread.join, 5)
        self._server = None
        self._thread = None
        self._ready.clear()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        item = self._tools.get(name)
        if item is None:
            return {"error": "unknown_tool", "tool": name}
        clean_args = dict(arguments or {})
        context_token = str(clean_args.pop("context_token", "") or "")
        context = self.store.validate_context_token(context_token)
        if not context:
            return {"error": "invalid_context_token"}
        if self._contains_configured_secret(clean_args):
            return {"error": "sensitive_argument_rejected"}
        if _is_high_risk(item.meta):
            reason = f"{item.meta.risk_level}:{','.join(sorted(item.meta.capabilities)) or 'unknown'}"
            batch_id, operation_id, created = self.store.stage_approval(
                context=context,
                tool_name=name,
                arguments=clean_args,
                risk_reason=reason,
                ttl_seconds=self.approval_ttl_seconds,
            )
            return {
                "status": "approval_required", "batch_id": batch_id,
                "operation_id": operation_id, "deduplicated": not created,
                "message": "该操作已暂存，批准前不会执行。",
            }
        return await self._execute(item, clean_args, context)

    def _contains_configured_secret(self, arguments: dict[str, Any]) -> bool:
        serialized = json.dumps(arguments, ensure_ascii=False, default=str)
        sensitive_values = [self.token]
        for key, value in os.environ.items():
            upper = key.upper()
            if any(marker in upper for marker in ("API_KEY", "APP_SECRET", "ACCESS_TOKEN", "PASSWORD")):
                sensitive_values.append(value)
        return any(secret and len(secret) >= 6 and secret in serialized for secret in sensitive_values)

    async def _execute(self, item: GatewayTool, arguments: dict[str, Any], context: dict[str, str]) -> dict[str, Any]:
        set_active_thread_id(context["thread_id"])
        try:
            value = await item.tool.ainvoke(arguments)
        except AttributeError:
            value = await asyncio.to_thread(item.tool.invoke, arguments)
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(text) > 16000:
            text = text[:16000] + "\n[结果已截断]"
        return {"status": "ok", "tool": item.public_name, "result": text}

    async def execute_approved(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.get_batch(batch_id)
        if not batch or batch["status"] != "pending":
            raise ValueError("审批批次不存在或已处理")
        if batch["expires_at"] <= datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"):
            self.store.resolve_batch(batch_id, "expired")
            raise ValueError("审批批次已过期")
        context = {key: str(batch[key]) for key in ("thread_id", "turn_id", "source", "workspace")}
        results: list[dict[str, Any]] = []
        failed = False
        for operation in batch["operations"]:
            if failed:
                self.store.update_operation(operation["operation_id"], "cancelled")
                continue
            item = self._tools.get(operation["tool_name"])
            if item is None:
                result = {"status": "error", "error": "tool_unavailable"}
                self.store.update_operation(operation["operation_id"], "failed", result=result, error_type="tool_unavailable")
                failed = True
            else:
                try:
                    result = await self._execute(item, json.loads(operation["arguments_json"]), context)
                    self.store.update_operation(operation["operation_id"], "completed", result=result)
                except Exception as exc:
                    result = {"status": "error", "error": type(exc).__name__}
                    self.store.update_operation(operation["operation_id"], "failed", result=result, error_type=type(exc).__name__)
                    failed = True
            results.append({"operation_id": operation["operation_id"], **result})
        self.store.resolve_batch(batch_id, "failed" if failed else "completed")
        return {"batch_id": batch_id, "status": "failed" if failed else "completed", "results": results}
