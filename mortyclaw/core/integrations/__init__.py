"""External service integrations for MortyClaw."""

from .feishu_mcp import (
    FeishuMCPSettings,
    build_feishu_mcp_connection,
    feishu_oauth_redirect_urls,
    is_feishu_tool,
    load_feishu_mcp_tools,
    run_feishu_oauth_login,
)
from .mcp_manager import (
    MCPManager,
    MCPServerConfig,
    MCPServiceStatus,
    build_arxiv_config,
    build_zotero_config,
    is_mcp_tool,
    load_mcp_tools,
)

__all__ = [
    "FeishuMCPSettings",
    "build_feishu_mcp_connection",
    "feishu_oauth_redirect_urls",
    "is_feishu_tool",
    "load_feishu_mcp_tools",
    "run_feishu_oauth_login",
    "FeishuBotSettings",
    "MortyClawFeishuRuntime",
    "build_feishu_channel",
    "feishu_thread_id",
    "serve_feishu_bot",
    "MCPManager",
    "MCPServerConfig",
    "MCPServiceStatus",
    "build_arxiv_config",
    "build_zotero_config",
    "is_mcp_tool",
    "load_mcp_tools",
]


def __getattr__(name: str):
    # Keep bot imports lazy: the bot depends on HarnessRuntime, whose gateway
    # imports the MCP manager from this package.
    if name in {
        "FeishuBotSettings", "MortyClawFeishuRuntime", "build_feishu_channel",
        "feishu_thread_id", "serve_feishu_bot",
    }:
        from . import feishu_bot
        return getattr(feishu_bot, name)
    raise AttributeError(name)
