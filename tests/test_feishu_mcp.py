import os
import unittest
from unittest.mock import patch

from langchain_core.tools import tool

from mortyclaw.core.integrations.feishu_mcp import (
    FeishuMCPSettings,
    _feishu_tool_meta,
    build_feishu_mcp_connection,
    feishu_oauth_redirect_urls,
    load_feishu_mcp_tools,
)
from mortyclaw.core.tools.meta import attach_tool_meta


@tool("calculator")
def fake_calculator(expression: str) -> str:
    """Evaluate an expression."""
    return expression


@tool("feishu_im_v1_message_list")
def feishu_message_list(page_size: int = 20) -> str:
    """List Feishu messages."""
    return str(page_size)


@tool("feishu_im_v1_message_create")
def feishu_message_create(content: str) -> str:
    """Create a Feishu message."""
    return content


@tool("feishu_docx_v1_document_rawContent")
def feishu_document_raw_content(document_id: str) -> str:
    """Read raw Feishu document content."""
    return document_id


@tool("feishu_wiki_v2_space_getNode")
def feishu_wiki_get_node(token: str) -> str:
    """Read a Feishu wiki node."""
    return token


class TestFeishuMCP(unittest.TestCase):
    def test_oauth_redirect_urls_include_lark_mcp_wrapped_callback(self):
        callback, wrapped_callback = feishu_oauth_redirect_urls()

        self.assertEqual(callback, "http://localhost:3000/callback")
        self.assertEqual(
            wrapped_callback,
            "http://localhost:3000/callback?redirect_uri=http://localhost:3000/callback",
        )

    def test_disabled_loader_does_not_require_optional_runtime(self):
        self.assertEqual(load_feishu_mcp_tools(FeishuMCPSettings(enabled=False)), [])

    def test_connection_passes_secrets_via_child_environment(self):
        settings = FeishuMCPSettings(
            enabled=True,
            app_id="cli_test",
            app_secret="secret_test",
            command="npx.cmd",
        )
        connection = build_feishu_mcp_connection(settings)

        self.assertEqual(connection["transport"], "stdio")
        self.assertEqual(connection["env"]["APP_ID"], "cli_test")
        self.assertEqual(connection["env"]["APP_SECRET"], "secret_test")
        self.assertNotIn("cli_test", connection["args"])
        self.assertNotIn("secret_test", connection["args"])

    def test_read_and_write_tools_receive_different_risk_metadata(self):
        read_meta = _feishu_tool_meta(feishu_message_list)
        write_meta = _feishu_tool_meta(feishu_message_create)
        raw_content_meta = _feishu_tool_meta(feishu_document_raw_content)
        get_node_meta = _feishu_tool_meta(feishu_wiki_get_node)

        self.assertEqual(read_meta.risk_level, "low")
        self.assertFalse(read_meta.requires_approval)
        self.assertEqual(write_meta.risk_level, "high")
        self.assertTrue(write_meta.requires_approval)
        self.assertEqual(raw_content_meta.risk_level, "low")
        self.assertEqual(get_node_meta.risk_level, "low")

        write_tool = attach_tool_meta(feishu_message_create, write_meta)

    @patch.dict(os.environ, {"FEISHU_MCP_ENABLED": "0"}, clear=False)
    def test_settings_default_to_disabled(self):
        self.assertFalse(FeishuMCPSettings.from_env().enabled)


if __name__ == "__main__":
    unittest.main()
