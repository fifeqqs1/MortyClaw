import os
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.tools import tool

from mortyclaw.core.agent.tool_policy import select_tools_for_fast_route
from mortyclaw.core.integrations.mcp_manager import (
    MCPManager,
    MCPServerConfig,
    _canonical_tool_name,
    _tool_meta,
    build_arxiv_config,
    build_zotero_config,
)
from mortyclaw.core.tools.meta import attach_tool_meta, get_tool_meta


@tool("zotero_search_items")
def zotero_search_items(query: str) -> str:
    """Search a Zotero library."""
    return query


@tool("zotero_create_item")
def zotero_create_item(title: str) -> str:
    """Create a Zotero item."""
    return title


@tool("arxiv_search_papers")
def arxiv_search_papers(query: str) -> str:
    """Search arXiv."""
    return query


@tool("arxiv_download_paper")
def arxiv_download_paper(paper_id: str) -> str:
    """Download an arXiv paper."""
    return paper_id


@tool("arxiv_semantic_search")
def arxiv_semantic_search(query: str) -> str:
    """Semantically search cached arXiv papers."""
    return query


class MCPManagerTests(unittest.TestCase):
    @patch.dict(os.environ, {"ZOTERO_MCP_ENABLED": "0"}, clear=False)
    def test_zotero_disabled_does_not_resolve_command(self):
        self.assertFalse(build_zotero_config().enabled)

    @patch.dict(os.environ, {"ARXIV_MCP_ENABLED": "0"}, clear=False)
    def test_arxiv_disabled_does_not_resolve_command(self):
        self.assertFalse(build_arxiv_config().enabled)

    def test_zotero_write_tools_are_filtered(self):
        self.assertIsNotNone(_tool_meta("zotero", zotero_search_items))
        self.assertIsNone(_tool_meta("zotero", zotero_create_item))

    def test_duplicate_server_prefix_is_collapsed(self):
        self.assertEqual(
            _canonical_tool_name("zotero", "zotero_zotero_search_items"),
            "zotero_search_items",
        )

    def test_arxiv_reads_are_safe_and_state_changes_need_approval(self):
        read_meta = _tool_meta("arxiv", arxiv_search_papers)
        write_meta = _tool_meta("arxiv", arxiv_download_paper)
        self.assertEqual(read_meta.risk_level, "low")
        self.assertFalse(read_meta.requires_approval)
        self.assertEqual(_tool_meta("arxiv", arxiv_semantic_search).risk_level, "low")
        self.assertEqual(write_meta.risk_level, "high")
        self.assertTrue(write_meta.requires_approval)

    def test_query_routing_separates_zotero_and_arxiv(self):
        zotero = attach_tool_meta(zotero_search_items, _tool_meta("zotero", zotero_search_items))
        arxiv = attach_tool_meta(arxiv_search_papers, _tool_meta("arxiv", arxiv_search_papers))
        tools = [zotero, arxiv]
        zotero_selected = select_tools_for_fast_route({}, tools, latest_user_query="搜索我 Zotero 中收藏的论文")
        arxiv_selected = select_tools_for_fast_route({}, tools, latest_user_query="搜索 Arxiv 上的 Transformer 论文")
        self.assertIn("zotero_search_items", {item.name for item in zotero_selected})
        self.assertNotIn("zotero_search_items", {item.name for item in arxiv_selected})
        self.assertIn("arxiv_search_papers", {item.name for item in arxiv_selected})

    @patch("mortyclaw.core.integrations.mcp_manager._discover_tools", new_callable=AsyncMock)
    def test_service_failure_is_isolated(self, discover):
        discover.side_effect = [RuntimeError("secret details"), [arxiv_search_papers]]
        manager = MCPManager([
            MCPServerConfig(name="zotero", enabled=True, command="zotero"),
            MCPServerConfig(name="arxiv", enabled=True, command="arxiv"),
        ])
        with self.assertWarns(Warning):
            tools = manager.load_tools(strict=False)
        self.assertEqual(tools, [arxiv_search_papers])

    def test_metadata_attached_to_arxiv_tool(self):
        meta = _tool_meta("arxiv", arxiv_search_papers)
        attached = attach_tool_meta(arxiv_search_papers, meta)
        self.assertIn("arxiv_read", get_tool_meta(attached).capabilities)


if __name__ == "__main__":
    unittest.main()
