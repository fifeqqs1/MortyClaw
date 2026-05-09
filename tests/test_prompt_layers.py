import os
import sys
import unittest

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.agent.tool_policy import (
    select_tools_for_autonomous_slow,
    select_tools_for_fast_route,
    select_tools_for_structured_slow,
)
from mortyclaw.core.planning import select_tools_for_current_step
from mortyclaw.core.prompt_builder import build_react_prompt_bundle
from mortyclaw.core.prompts.provider_cache import apply_provider_prompt_cache
from mortyclaw.core.tools.base import tool


class TestPromptLayers(unittest.TestCase):
    def test_base_prompt_excludes_runtime_state_and_reference_content(self):
        bundle = build_react_prompt_bundle(
            [HumanMessage(content="请继续修复项目")],
            "slow",
            {
                "goal": "在 /tmp/demo/main.py 中修复 bug 并运行测试",
                "permission_mode": "auto",
                "risk_level": "high",
                "current_project_path": "/tmp/demo",
                "slow_execution_mode": "autonomous",
            },
            active_summary="之前已经定位到 parser 分支。",
            session_prompt="- 当前项目路径：/tmp/demo\n- 代码策略：本轮只分析，不修改任何代码。",
            long_term_prompt="【用户长期画像 (静态偏好)】\n- 喜欢简洁回答",
            current_plan_step=None,
            include_approved_goal_context=False,
        )

        self.assertNotIn("/tmp/demo/main.py", bundle.base_system_prompt)
        self.assertNotIn("permission_mode", bundle.base_system_prompt)
        self.assertNotIn("structured-handoff", bundle.base_system_prompt)
        self.assertIn("permission_mode=auto", bundle.trusted_turn_context)
        self.assertIn("current_project_path=/tmp/demo", bundle.trusted_turn_context)
        self.assertTrue(bundle.reference_messages)

    def test_bundle_emits_two_system_layers_and_short_goal_continuation(self):
        bundle = build_react_prompt_bundle(
            [HumanMessage(content="批准，继续执行")],
            "slow",
            {
                "goal": "在 /tmp/demo/main.py 中实现 stream=True 并验证输出",
                "permission_mode": "auto",
                "risk_level": "high",
                "current_project_path": "/tmp/demo",
                "slow_execution_mode": "autonomous",
            },
            active_summary="",
            session_prompt="",
            long_term_prompt="",
            current_plan_step=None,
            include_approved_goal_context=True,
        )

        final_messages = bundle.final_messages()
        self.assertIsInstance(final_messages[0], SystemMessage)
        self.assertIsInstance(final_messages[1], SystemMessage)
        self.assertEqual(
            final_messages[-1].content,
            "继续执行已批准任务。以 trusted_turn_context 中的 goal、permission_mode 和 active todo 为准。",
        )

    def test_prompt_bundle_records_hashes_and_token_stats(self):
        bundle = build_react_prompt_bundle(
            [HumanMessage(content="分析一下项目结构")],
            "fast",
            {"goal": "分析一下项目结构"},
            active_summary="",
            session_prompt="",
            long_term_prompt="",
            current_plan_step=None,
            include_approved_goal_context=False,
        )

        self.assertIn("base", bundle.prompt_hashes)
        self.assertIn("trusted_turn", bundle.prompt_hashes)
        self.assertIn("reference", bundle.prompt_hashes)
        self.assertIn("base", bundle.token_stats)
        self.assertIn("conversation", bundle.token_stats)
        self.assertIn("goal_continuation", bundle.token_stats)

    def test_provider_cache_marks_first_system_message_for_anthropic(self):
        messages, stats = apply_provider_prompt_cache(
            [SystemMessage(content="base"), HumanMessage(content="hi")],
            provider_name="anthropic",
            model_name="claude-3-7-sonnet",
        )
        self.assertTrue(stats["provider_cache_applied"])
        self.assertEqual(messages[0].additional_kwargs.get("cache_control"), {"type": "ephemeral"})

    def test_route_aware_tool_grouping_keeps_project_autonomous_in_coding_group(self):
        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """read"""
            return ""

        @tool
        def write_project_file(filepath: str = "", content: str = "", project_root: str = "") -> str:
            """write"""
            return ""

        @tool
        def execute_office_shell(command: str = "") -> str:
            """shell"""
            return ""

        @tool
        def tavily_web_search(query: str = "") -> str:
            """search"""
            return ""

        tools = [read_project_file, write_project_file, execute_office_shell, tavily_web_search]
        selected = select_tools_for_autonomous_slow(
            {
                "current_project_path": "/tmp/demo",
                "goal": "在当前项目中修复 bug 并运行测试",
                "risk_level": "high",
            },
            tools,
            latest_user_query="在当前项目中修复 bug 并运行测试",
        )
        selected_names = {tool.name for tool in selected}
        self.assertIn("read_project_file", selected_names)
        self.assertIn("write_project_file", selected_names)
        self.assertNotIn("execute_office_shell", selected_names)

    def test_structured_route_group_is_filtered_against_step_scope(self):
        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """read"""
            return ""

        @tool
        def write_project_file(filepath: str = "", content: str = "", project_root: str = "") -> str:
            """write"""
            return ""

        @tool
        def tavily_web_search(query: str = "") -> str:
            """search"""
            return ""

        tools = [read_project_file, write_project_file, tavily_web_search]
        route_tools = select_tools_for_structured_slow(
            {"current_project_path": "/tmp/demo", "goal": "分析项目结构并总结"},
            tools,
            latest_user_query="分析项目结构并总结",
        )
        step_tools = select_tools_for_current_step(
            {"description": "分析这个项目的模块关系", "intent": "analyze"},
            route_tools,
            current_project_path="/tmp/demo",
        )
        selected_names = {tool.name for tool in step_tools}
        self.assertIn("read_project_file", selected_names)
        self.assertNotIn("write_project_file", selected_names)

    def test_fast_route_only_adds_research_for_explicit_research_query(self):
        @tool
        def get_system_model_info() -> str:
            """sys"""
            return ""

        @tool
        def list_office_files() -> str:
            """files"""
            return ""

        @tool
        def tavily_web_search(query: str = "") -> str:
            """search"""
            return ""

        tools = [get_system_model_info, list_office_files, tavily_web_search]
        baseline = select_tools_for_fast_route({}, tools, latest_user_query="列出当前目录文件")
        research = select_tools_for_fast_route({}, tools, latest_user_query="搜索今天的最新新闻")
        self.assertNotIn("tavily_web_search", {tool.name for tool in baseline})
        self.assertIn("tavily_web_search", {tool.name for tool in research})


if __name__ == "__main__":
    unittest.main()
