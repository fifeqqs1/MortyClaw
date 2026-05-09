import unittest
import os
import sys
import json
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mortyclaw.core.context import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


class TestAgent(unittest.TestCase):
    def _apply_message_updates(self, state, updates):
        merged_messages = list(state.get("messages", []) or [])
        for message in list(updates.get("messages", []) or []):
            if message.__class__.__name__ == "RemoveMessage":
                merged_messages = [
                    existing
                    for existing in merged_messages
                    if str(getattr(existing, "id", "") or "") != str(getattr(message, "id", "") or "")
                ]
            else:
                merged_messages.append(message)
        return {
            **state,
            **updates,
            "messages": merged_messages,
        }

    def _make_react_node_test_deps(
        self,
        *,
        trim_context_messages_fn,
        summarize_discarded_context_fn=None,
        **overrides,
    ):
        from mortyclaw.core.agent.react_node import ReactNodeDependencies

        def annotate(message, **kwargs):
            return AIMessage(
                content=getattr(message, "content", ""),
                tool_calls=list(getattr(message, "tool_calls", []) or []),
                additional_kwargs={
                    **dict(getattr(message, "additional_kwargs", {}) or {}),
                    **kwargs,
                },
            )

        deps = ReactNodeDependencies(
            set_active_thread_id_fn=lambda _thread_id: None,
            prepare_recent_tool_messages_fn=lambda state, **_kwargs: (list(state.get("messages", []) or []), {}),
            build_session_memory_prompt_fn=lambda *_args, **_kwargs: "",
            should_enable_todos_fn=lambda *_args, **_kwargs: False,
            build_todo_state_from_plan_fn=lambda *_args, **_kwargs: {},
            load_session_todo_state_fn=lambda *_args, **_kwargs: {},
            save_session_todo_state_fn=lambda *_args, **_kwargs: None,
            clear_session_todo_state_fn=lambda *_args, **_kwargs: None,
            audit_logger_instance=Mock(log_event=Mock()),
            extract_passthrough_text_fn=lambda _message: None,
            annotate_ai_message_fn=annotate,
            with_working_memory_fn=lambda state, updates: {**state, **updates},
            is_affirmative_approval_response_fn=lambda _text: False,
            get_latest_user_query_fn=lambda messages: next(
                (str(message.content) for message in reversed(messages) if isinstance(message, HumanMessage)),
                "",
            ),
            get_current_plan_step_fn=lambda _state: None,
            select_tools_for_current_step_fn=lambda *_args, **_kwargs: [],
            select_tools_for_fast_route_fn=lambda *_args, **_kwargs: [],
            apply_permission_mode_to_tools_fn=lambda tools, *_args, **_kwargs: tools,
            select_tools_for_autonomous_slow_fn=lambda *_args, **_kwargs: [],
            should_direct_route_to_arxiv_rag_fn=lambda _query: False,
            arxiv_rag_tool=Mock(),
            extract_passthrough_payload_fn=lambda _payload: None,
            trim_context_messages_fn=trim_context_messages_fn,
            summarize_discarded_context_fn=summarize_discarded_context_fn or (lambda *_args, **_kwargs: "summary"),
            conversation_writer=Mock(record_summary=Mock()),
            build_long_term_memory_prompt_fn=lambda _query: "",
            build_react_prompt_bundle_fn=lambda final_msgs, *_args, **_kwargs: (
                "sys",
                [message for message in final_msgs if not isinstance(message, SystemMessage)],
            ),
            assemble_dynamic_context_fn=lambda **_kwargs: {},
            classify_error_fn=lambda exc=None, message="", state=None: __import__(
                "mortyclaw.core.error_policy", fromlist=["classify_error"]
            ).classify_error(exc=exc, message=message, state=state),
            serialize_classified_error_fn=lambda _classified: {"kind": "serialized"},
            normalize_tavily_tool_calls_fn=lambda response, *_args, **_kwargs: response,
            enforce_slow_step_tool_scope_fn=lambda response, *_args, **_kwargs: response,
            destructive_tool_calls_fn=lambda _tool_calls: [],
            build_pending_execution_snapshot_fn=lambda *_args, **_kwargs: {},
            build_pending_tool_approval_reason_fn=lambda *_args, **_kwargs: "",
            looks_like_explicit_failure_text_fn=lambda text: str(text or "").startswith("执行失败"),
            complete_autonomous_todos_fn=lambda _state: {},
            fast_path_excluded_tool_names=set(),
            auto_mode_blocked_tool_names=set(),
            update_subdirectory_context_fn=lambda *_args, **_kwargs: {},
            response_kind_final_answer="final_answer",
            response_kind_step_result="step_result",
            step_outcome_failure="failure",
            step_outcome_success_candidate="success_candidate",
            session_memory_prompt_limit=5,
            context_summary_timeout_seconds=1.0,
        )
        return replace(deps, **overrides) if overrides else deps

    def test_agent_state_initialization(self):
        """测试 AgentState 的初始化"""
        from mortyclaw.core.context import AgentState

        initial_state = AgentState(
            messages=[],
            summary=""
        )

        self.assertEqual(initial_state["messages"], [])
        self.assertEqual(initial_state["summary"], "")

    @patch('mortyclaw.core.provider.get_provider')
    @patch('mortyclaw.core.skill_loader.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_basic(self, mock_load_skills, mock_get_provider):
        """测试创建基础代理应用（带 Mock）"""
        from mortyclaw.core.agent import create_agent_app

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        try:
            app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
            self.assertIsNotNone(app)
        except Exception as e:
            # 即使出现其他错误也记录
            print(f"Unexpected error: {e}")
            raise

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_uses_env_configured_route_classifier_model(self, mock_load_skills, mock_get_provider):
        """测试 planner / route-classifier 模型优先从环境变量读取"""
        from mortyclaw.core.agent import create_agent_app

        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider
        mock_load_skills.return_value = []

        with patch.dict(os.environ, {"ROUTE_CLASSIFIER_MODEL": "qwen3.5-flash-env"}, clear=False):
            app = create_agent_app(provider_name="aliyun", model_name="qwen3.5-plus")

        self.assertIsNotNone(app)
        self.assertGreaterEqual(mock_get_provider.call_count, 2)
        planner_call = mock_get_provider.call_args_list[1]
        self.assertEqual(planner_call.kwargs["provider_name"], "aliyun")
        self.assertEqual(planner_call.kwargs["model_name"], "qwen3.5-flash-env")

    @patch('mortyclaw.core.provider.get_provider')
    @patch('mortyclaw.core.skill_loader.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_custom_tools(self, mock_load_skills, mock_get_provider):
        """测试创建带有自定义工具的代理应用（带 Mock）"""
        from mortyclaw.core.agent import create_agent_app
        from langchain_core.tools import tool

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        # 创建一个真正的 mock 工具（使用@tool 装饰器）
        @tool
        def mock_tool(test_param: str) -> str:
            """A mock tool for testing"""
            return f"mock result: {test_param}"

        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                tools=[mock_tool]
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('mortyclaw.core.provider.get_provider')
    @patch('mortyclaw.core.skill_loader.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_checkpointer(self, mock_load_skills, mock_get_provider):
        """测试创建带有检查点的代理应用（带 Mock）"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        memory_saver = MemorySaver()
        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                checkpointer=memory_saver
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_passthrough_tool_response_skips_second_llm_call(self, mock_load_skills, mock_get_provider):
        """测试直通工具结果返回后不再进入第二次 LLM 改写"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.tools.web_tools import MORTYCLAW_PASSTHROUGH_FLAG
        from langchain_core.tools import tool

        mock_load_skills.return_value = []

        @tool
        def passthrough_tool() -> str:
            """Return a passthrough response"""
            return json.dumps({
                MORTYCLAW_PASSTHROUGH_FLAG: True,
                "display_text": "这是 arxiv_rag 的最终回答"
            }, ensure_ascii=False)

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.router_call_count = 0
                self.agent_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    self.router_call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "fast",
                        "slow_execution_mode": "",
                        "reason": "需要直接执行单个工具",
                        "confidence": 0.92,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.agent_call_count += 1
                if self.agent_call_count > 1:
                    raise AssertionError("passthrough 成功时不应该进入第二次 LLM 调用")
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "passthrough_tool",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }],
                )

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[passthrough_tool],
        )

        result = app.invoke(
            {"messages": [HumanMessage(content="帮我调用一个测试工具")], "summary": ""},
            config={"configurable": {"thread_id": "test_passthrough"}},
        )

        self.assertEqual(fake_provider.llm_with_tools.agent_call_count, 1)
        self.assertEqual(result["messages"][-1].content, "这是 arxiv_rag 的最终回答")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    @patch('mortyclaw.core.agent.arxiv_rag_ask')
    def test_paper_query_routes_directly_to_arxiv_rag_without_llm_rewrite(
        self,
        mock_arxiv_rag_ask,
        mock_load_skills,
        mock_get_provider,
    ):
        """测试论文类问题直接把原始用户问题送给 arxiv_rag，不经过外层 LLM 改写"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.tools.web_tools import MORTYCLAW_PASSTHROUGH_FLAG

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                raise AssertionError("论文直连 arxiv_rag 成功时不应该触发外层 LLM")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider
        mock_arxiv_rag_ask.invoke.return_value = json.dumps({
            MORTYCLAW_PASSTHROUGH_FLAG: True,
            "display_text": "这是 arxiv_rag 的原始回答"
        }, ensure_ascii=False)

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
        )

        original_query = "推荐一篇无人机论文"
        result = app.invoke(
            {"messages": [HumanMessage(content=original_query)], "summary": ""},
            config={"configurable": {"thread_id": "test_direct_arxiv"}},
        )

        mock_arxiv_rag_ask.invoke.assert_called_once_with(
            {"query": original_query, "session_id": "test_direct_arxiv"}
        )
        self.assertEqual(fake_provider.llm_with_tools.call_count, 1)
        self.assertEqual(result["messages"][-1].content, "这是 arxiv_rag 的原始回答")

    def test_infer_tavily_topic_prefers_query_intent(self):
        """测试 Tavily topic 会根据查询意图区分 general / news"""
        from mortyclaw.core.agent import _infer_tavily_topic

        self.assertEqual(_infer_tavily_topic("2026年4月18日 LPL 英雄联盟比赛赛程"), "general")
        self.assertEqual(_infer_tavily_topic("武汉天气 2026年4月17日 今天"), "general")
        self.assertEqual(_infer_tavily_topic("今天热点新闻"), "news")
        self.assertEqual(_infer_tavily_topic("OpenAI 最新发布了什么"), "news")

    def test_build_route_decision_prefers_fast_for_simple_query(self):
        """测试简单问题默认走 fast path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("现在几点了？")
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "simple")
        self.assertEqual(decision["risk_level"], "low")

    def test_build_route_decision_routes_multi_step_query_to_slow(self):
        """测试多步骤任务会被路由到 slow path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("先查看项目结构，然后总结问题，最后给我整改建议")
        self.assertEqual(decision["route"], "slow")
        self.assertEqual(decision["complexity"], "multi_step")
        self.assertEqual(decision["risk_level"], "medium")

    def test_build_route_decision_routes_high_risk_query_to_slow(self):
        """测试高风险执行型任务会被路由到 slow path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("运行 python test.py 并修改文件后重新执行")
        self.assertEqual(decision["route"], "slow")
        self.assertEqual(decision["risk_level"], "high")
        self.assertTrue(decision["route_locked"])
        self.assertEqual(decision["slow_execution_mode"], "autonomous")

    def test_build_route_decision_routes_read_only_analysis_query_to_fast(self):
        """测试只读项目分析请求默认走 fast path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("详细分析这个项目的架构设计，并给出修改建议")
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "read_only_analysis")
        self.assertEqual(decision["risk_level"], "low")
        self.assertEqual(decision["route_source"], "rule_read_only_analysis")

    def test_build_route_decision_routes_repo_review_query_to_fast(self):
        """测试仓库 review 默认走 fast path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("帮我review这个仓库")
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "read_only_analysis")

    def test_build_route_decision_routes_project_explanation_query_to_fast(self):
        """测试项目说明类请求默认走 fast path"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("解释一下这个项目是干什么的")
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "read_only_analysis")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_fast_read_only_analysis_excludes_slow_only_todo_tool(self, mock_load_skills, mock_get_provider):
        """测试 fast 路径的只读项目分析不会暴露 slow-only 的 update_todo_list"""
        from langchain_core.tools import tool
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        @tool
        def update_todo_list(items: str = "", reason: str = "") -> str:
            """Mock slow-only todo tool."""
            return "todo updated"

        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """Mock project read tool."""
            return "project file"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="这是一个教学项目。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append([getattr(tool, "name", "") for tool in tools])
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with patch('mortyclaw.core.agent.BUILTIN_TOOLS', [update_todo_list, read_project_file]):
            app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
            result = app.invoke(
                {"messages": [HumanMessage(content="解释一下这个项目是干什么的")], "summary": ""},
                config={"configurable": {"thread_id": "test_fast_excludes_todo_tool"}},
            )

        self.assertEqual(result["run_status"], "done")
        self.assertGreaterEqual(len(fake_provider.bound_tool_names), 2)
        self.assertIn("update_todo_list", fake_provider.bound_tool_names[0])
        self.assertNotIn("update_todo_list", fake_provider.bound_tool_names[-1])
        self.assertIn("read_project_file", fake_provider.bound_tool_names[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_fast_project_analysis_prefers_project_read_tools_but_keeps_lightweight_fast_utilities(self, mock_load_skills, mock_get_provider):
        """测试带项目上下文的 fast 分析任务会收紧工具，但保留联网搜索和系统信息等轻量能力"""
        from langchain_core.tools import tool
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        @tool
        def update_todo_list(items: str = "", reason: str = "") -> str:
            """Mock slow-only todo tool."""
            return "todo updated"

        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """Mock project read tool."""
            return "project file"

        @tool
        def search_project_code(query: str = "", project_root: str = "") -> str:
            """Mock project search tool."""
            return "project search"

        @tool
        def show_git_diff(project_root: str = "") -> str:
            """Mock git diff tool."""
            return "git diff"

        @tool
        def tavily_web_search(query: str = "") -> str:
            """Mock web search tool."""
            return "web"

        @tool
        def get_system_model_info() -> str:
            """Mock system info tool."""
            return "system"

        @tool
        def get_current_time() -> str:
            """Mock time tool."""
            return "time"

        @tool
        def list_office_files(sub_dir: str = "") -> str:
            """Mock office list tool."""
            return "office list"

        @tool
        def read_office_file(filepath: str = "") -> str:
            """Mock office read tool."""
            return "office file"

        @tool
        def execute_office_shell(command: str = "") -> str:
            """Mock shell tool."""
            return "shell"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="这是一个项目分析结果。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append([getattr(tool, "name", "") for tool in tools])
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        tools = [
            update_todo_list,
            read_project_file,
            search_project_code,
            show_git_diff,
            tavily_web_search,
            get_system_model_info,
            get_current_time,
            list_office_files,
            read_office_file,
            execute_office_shell,
        ]
        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=tools,
        )
        result = app.invoke(
            {
                "messages": [HumanMessage(content="/mnt/A/hust_chp/hust_chp/book 详细分析这个项目，指出优缺点")],
                "summary": "",
            },
            config={"configurable": {"thread_id": "test_fast_project_analysis_tools"}},
        )

        self.assertEqual(result["run_status"], "done")
        self.assertGreaterEqual(len(fake_provider.bound_tool_names), 2)
        fast_bound_tools = fake_provider.bound_tool_names[-1]
        self.assertIn("read_project_file", fast_bound_tools)
        self.assertIn("search_project_code", fast_bound_tools)
        self.assertIn("show_git_diff", fast_bound_tools)
        self.assertIn("tavily_web_search", fast_bound_tools)
        self.assertIn("get_system_model_info", fast_bound_tools)
        self.assertIn("get_current_time", fast_bound_tools)
        self.assertNotIn("update_todo_list", fast_bound_tools)
        self.assertNotIn("list_office_files", fast_bound_tools)
        self.assertNotIn("read_office_file", fast_bound_tools)
        self.assertNotIn("execute_office_shell", fast_bound_tools)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_fast_general_query_keeps_non_project_fast_tools_available(self, mock_load_skills, mock_get_provider):
        """测试普通 fast 查询不会被项目分析收口误伤"""
        from langchain_core.tools import tool
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        @tool
        def list_office_files(sub_dir: str = "") -> str:
            """Mock office list tool."""
            return "office list"

        @tool
        def get_system_model_info() -> str:
            """Mock system info tool."""
            return "system"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="当前模型信息如下。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append([getattr(tool, "name", "") for tool in tools])
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[list_office_files, get_system_model_info],
        )
        result = app.invoke(
            {"messages": [HumanMessage(content="告诉我当前模型信息")], "summary": ""},
            config={"configurable": {"thread_id": "test_fast_general_tools"}},
        )

        self.assertEqual(result["run_status"], "done")
        self.assertGreaterEqual(len(fake_provider.bound_tool_names), 2)
        fast_bound_tools = fake_provider.bound_tool_names[-1]
        self.assertIn("list_office_files", fast_bound_tools)
        self.assertIn("get_system_model_info", fast_bound_tools)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_autonomous_slow_project_code_task_prefers_project_tools_over_office_write(self, mock_load_skills, mock_get_provider):
        """测试带项目路径的 autonomous slow 代码任务不会再暴露 write_office_file，而是收口到 project tools"""
        from langchain_core.tools import tool
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        @tool
        def update_todo_list(items: str = "", reason: str = "") -> str:
            """Mock todo tool."""
            return "todo updated"

        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """Mock project read tool."""
            return "project file"

        @tool
        def search_project_code(query: str = "", project_root: str = "") -> str:
            """Mock project search tool."""
            return "project search"

        @tool
        def show_git_diff(project_root: str = "") -> str:
            """Mock git diff tool."""
            return "git diff"

        @tool
        def edit_project_file(path: str = "", edits: str = "", project_root: str = "", expected_hash: str = "") -> str:
            """Mock project edit tool."""
            return "{\"ok\": true, \"message\": \"edited\"}"

        @tool
        def write_project_file(path: str = "", content: str = "", project_root: str = "", expected_hash: str = "") -> str:
            """Mock project write tool."""
            return "{\"ok\": true, \"message\": \"written\"}"

        @tool
        def apply_project_patch(patch: str = "", project_root: str = "") -> str:
            """Mock project patch tool."""
            return "patch applied"

        @tool
        def run_project_tests(command: str = "", project_root: str = "") -> str:
            """Mock project tests tool."""
            return "tests ok"

        @tool
        def run_project_command(command: str = "", project_root: str = "") -> str:
            """Mock project command tool."""
            return "{\"ok\": true, \"message\": \"command ok\"}"

        @tool
        def write_office_file(filepath: str = "", content: str = "", mode: str = "w") -> str:
            """Mock office write tool."""
            return "office write"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="已完成修改。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append([getattr(tool, "name", "") for tool in tools])
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        tools = [
            update_todo_list,
            read_project_file,
            search_project_code,
            show_git_diff,
            edit_project_file,
            write_project_file,
            apply_project_patch,
            run_project_tests,
            run_project_command,
            write_office_file,
        ]
        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=tools,
            checkpointer=MemorySaver(),
        )
        first_result = app.invoke(
            {
                "messages": [HumanMessage(content="/mnt/A/hust_chp/hust_chp/Agent/chat_agent 请优化当前 Python 命令行聊天工具，并修改 main.py 后运行验证")],
                "summary": "",
            },
            config={"configurable": {"thread_id": "test_autonomous_project_tools"}},
        )
        self.assertEqual(first_result["run_status"], "waiting_user")

        result = app.invoke(
            {"messages": [HumanMessage(content="auto")], "summary": ""},
            config={"configurable": {"thread_id": "test_autonomous_project_tools"}},
        )

        self.assertEqual(result["run_status"], "done")
        self.assertGreaterEqual(len(fake_provider.bound_tool_names), 2)
        autonomous_bound_tools = fake_provider.bound_tool_names[-1]
        self.assertIn("edit_project_file", autonomous_bound_tools)
        self.assertIn("write_project_file", autonomous_bound_tools)
        self.assertIn("apply_project_patch", autonomous_bound_tools)
        self.assertIn("run_project_tests", autonomous_bound_tools)
        self.assertIn("run_project_command", autonomous_bound_tools)
        self.assertIn("read_project_file", autonomous_bound_tools)
        self.assertNotIn("write_office_file", autonomous_bound_tools)

    def test_build_route_decision_keeps_arxiv_query_fast(self):
        """测试论文问答仍保留最快直连路径"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("推荐一篇无人机论文")
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["risk_level"], "low")
        self.assertEqual(decision["route_source"], "arxiv_direct")

    def test_build_route_decision_routes_mixed_research_query_to_slow_planner(self):
        """测试显式论文 + repo/代码检查的混合科研任务进入 slow planner"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision(
            "帮我看一下我昨天 clone 的那个 repo，它用的方法和 arxiv 上 2024 年的 Mamba 论文有什么区别，"
            "然后告诉我我的代码里有没有类似的实现"
        )
        self.assertEqual(decision["route"], "slow")
        self.assertEqual(decision["complexity"], "mixed_research")
        self.assertEqual(decision["risk_level"], "medium")
        self.assertTrue(decision["planner_required"])
        self.assertEqual(decision["route_source"], "mixed_research_task")

    def test_build_route_decision_does_not_route_plain_repo_analysis_to_mixed_research(self):
        """测试普通 repo/代码分析任务不会被误判为 mixed research"""
        from mortyclaw.core.agent import _build_route_decision

        decision = _build_route_decision("帮我看一下这个 repo 里有没有类似实现")
        self.assertNotEqual(decision["route_source"], "mixed_research_task")

    def test_build_route_decision_skips_optional_classifier_for_obvious_simple_query(self):
        """测试明显简单的问题会优先使用高置信 LLM classifier 结果"""
        from mortyclaw.core.agent import _build_route_decision

        classifier = Mock(return_value={"route": "fast", "reason": "可以直接回答", "confidence": 0.99})
        decision = _build_route_decision("现在几点了？", llm_classifier_fn=classifier)
        classifier.assert_called_once()
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "simple")
        self.assertEqual(decision["route_source"], "llm_classifier_fast")

    def test_build_route_decision_routes_ambiguous_query_via_llm_classifier(self):
        """测试边界请求优先由 LLM classifier 做路由判断"""
        from mortyclaw.core.agent import _build_route_decision

        classifier = Mock(return_value={
            "route": "slow",
            "slow_execution_mode": "structured",
            "reason": "需要显式计划来推进需求",
            "confidence": 0.93,
        })
        decision = _build_route_decision("这个需求后续该怎么推进比较合适", llm_classifier_fn=classifier)
        classifier.assert_called_once()
        self.assertEqual(decision["route"], "slow")
        self.assertEqual(decision["complexity"], "uncertain")
        self.assertEqual(decision["risk_level"], "medium")
        self.assertTrue(decision["planner_required"])
        self.assertEqual(decision["slow_execution_mode"], "structured")
        self.assertEqual(decision["route_source"], "llm_classifier_slow_structured")

    def test_build_route_decision_routes_capability_question_to_fast(self):
        """测试能力确认类问题不会误进入 slow path"""
        from mortyclaw.core.agent import _build_route_decision

        classifier = Mock(return_value={
            "route": "fast",
            "reason": "这是能力确认问题，可以直接回答",
            "confidence": 0.96,
        })
        decision = _build_route_decision("你可以进行代码修改和检查吗？", llm_classifier_fn=classifier)
        classifier.assert_called_once()
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "meta_capability")
        self.assertEqual(decision["route_source"], "llm_classifier_fast")

    def test_build_route_decision_falls_back_to_rule_for_capability_question_when_classifier_unavailable(self):
        """测试 classifier 不可用时能力确认类问题由规则兜底为 fast"""
        from mortyclaw.core.agent import _build_route_decision

        classifier = Mock(side_effect=RuntimeError("classifier unavailable"))
        decision = _build_route_decision("你可以进行代码修改和检查吗？", llm_classifier_fn=classifier)
        classifier.assert_called_once()
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "meta_capability")
        self.assertEqual(decision["route_source"], "rule_meta_capability")

    def test_build_route_decision_falls_back_to_rules_when_classifier_confidence_is_low(self):
        """测试 LLM classifier 低置信时回退规则判断"""
        from mortyclaw.core.agent import _build_route_decision

        classifier = Mock(return_value={
            "route": "fast",
            "reason": "我不太确定",
            "confidence": 0.31,
        })
        decision = _build_route_decision("帮我review这个仓库", llm_classifier_fn=classifier)
        classifier.assert_called_once()
        self.assertEqual(decision["route"], "fast")
        self.assertEqual(decision["complexity"], "read_only_analysis")
        self.assertEqual(decision["route_source"], "rule_read_only_analysis")

    def test_error_classifier_handles_common_failure_kinds(self):
        """测试统一错误分类器能识别常见 provider/tool/context 异常"""
        from mortyclaw.core.error_policy import classify_error

        timeout_error = classify_error(message="request timeout while calling provider")
        context_error = classify_error(message="prompt exceeds maximum context window")
        scope_error = classify_error(message="执行失败：系统拦截了越界工具调用。禁止工具：execute_office_shell")

        self.assertEqual(timeout_error.kind.value, "provider_timeout")
        self.assertEqual(context_error.kind.value, "context_overflow")
        self.assertEqual(scope_error.kind.value, "unsafe_tool_scope")

    def test_react_node_uses_token_budget_for_normal_trim(self):
        from mortyclaw.core.agent.react_node import (
            CONTEXT_COMPRESSION_BUDGET_TOKENS,
            CONTEXT_LAYER2_TRIGGER_RATIO,
            CONTEXT_NON_MESSAGE_RESERVE_TOKENS,
            CONTEXT_TRIM_KEEP_TOKENS,
            run_react_agent_node,
        )

        trim_mock = Mock(return_value=([HumanMessage(content="继续")], []))

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, _messages):
                return AIMessage(content="完成")

        state = {"messages": [HumanMessage(content="继续")], "summary": "S" * 120000}
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "medium"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "trim-normal"}},
                FakeLLM(),
                FakeLLMWithTools(),
                [],
                "fast",
                deps=self._make_react_node_test_deps(trim_context_messages_fn=trim_mock),
            )

        self.assertEqual(result["run_status"], "done")
        trim_kwargs = trim_mock.call_args.kwargs
        self.assertEqual(trim_kwargs["trigger_tokens"], 1)
        self.assertEqual(trim_kwargs["keep_tokens"], CONTEXT_TRIM_KEEP_TOKENS)
        self.assertEqual(trim_kwargs["reserve_tokens"], CONTEXT_NON_MESSAGE_RESERVE_TOKENS)
        self.assertEqual(trim_kwargs["model_name"], "gpt-4o-mini")
        self.assertNotIn("trigger_turns", trim_kwargs)
        self.assertNotIn("keep_turns", trim_kwargs)
        self.assertGreater(CONTEXT_COMPRESSION_BUDGET_TOKENS, 0)
        self.assertGreaterEqual(CONTEXT_LAYER2_TRIGGER_RATIO, 0.5)

    def test_react_node_uses_token_budget_for_context_overflow_retry(self):
        from mortyclaw.core.agent.react_node import (
            CONTEXT_NON_MESSAGE_RESERVE_TOKENS,
            CONTEXT_OVERFLOW_KEEP_TOKENS,
            run_react_agent_node,
        )

        trim_mock = Mock(side_effect=[
            ([HumanMessage(content="继续")], []),
            ([HumanMessage(content="继续")], [AIMessage(content="旧内容")]),
        ])
        summarize_mock = Mock(return_value="压缩后的摘要")

        class FakeLLM:
            model = "gpt-4o-mini"

        class FakeLLMWithTools:
            def __init__(self):
                self.calls = 0

            def invoke(self, _messages):
                self.calls += 1
                if self.calls == 1:
                    raise Exception("prompt exceeds maximum context window")
                return AIMessage(content="重试后成功")

        state = {"messages": [HumanMessage(content="继续")], "summary": ""}
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "medium"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "trim-overflow"}},
                FakeLLM(),
                FakeLLMWithTools(),
                [],
                "fast",
                deps=self._make_react_node_test_deps(
                    trim_context_messages_fn=trim_mock,
                    summarize_discarded_context_fn=summarize_mock,
                ),
            )

        self.assertEqual(result["run_status"], "done")
        self.assertEqual(trim_mock.call_count, 2)
        overflow_kwargs = trim_mock.call_args_list[1].kwargs
        self.assertEqual(overflow_kwargs["trigger_tokens"], 1)
        self.assertEqual(overflow_kwargs["keep_tokens"], CONTEXT_OVERFLOW_KEEP_TOKENS)
        self.assertEqual(overflow_kwargs["reserve_tokens"], CONTEXT_NON_MESSAGE_RESERVE_TOKENS)
        self.assertEqual(overflow_kwargs["model_name"], "gpt-4o-mini")
        summarize_mock.assert_called_once()

    def test_react_node_consolidates_trimmed_messages_immediately(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        discarded = [AIMessage(content="旧的执行历史")]
        trim_mock = Mock(return_value=([HumanMessage(content="继续")], discarded))
        trim_mock._last_stats = {
            "artifact_messages_seen": 1,
            "tool_results_pruned": 2,
            "tool_pairs_repaired": 1,
            "discarded_middle_count": 1,
        }
        summarize_mock = Mock(return_value="滚动压缩摘要")

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, _messages):
                return AIMessage(content="完成")

        state = {"messages": [HumanMessage(content="继续")], "summary": "已有摘要"}
        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            summarize_discarded_context_fn=summarize_mock,
            with_working_memory_fn=self._apply_message_updates,
        )
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "medium"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "trim-consolidate"}},
                FakeLLM(),
                FakeLLMWithTools(),
                [],
                "fast",
                deps=deps,
            )

        self.assertEqual(result["run_status"], "done")
        summarize_mock.assert_called_once()
        self.assertEqual(result["summary"], "滚动压缩摘要")
        deps.conversation_writer.record_summary.assert_called_once()
        audit_messages = [
            str(call.kwargs.get("content", ""))
            for call in deps.audit_logger_instance.log_event.call_args_list
            if call.kwargs.get("event") == "system_action"
        ]
        self.assertTrue(any("artifact_messages_seen=1" in content for content in audit_messages))

    def test_react_node_auto_compacts_after_high_pressure_trim(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        discarded = [AIMessage(content="旧的执行历史", id="old-ai")]
        trim_mock = Mock(return_value=([HumanMessage(content="继续", id="msg-user-keep")], discarded))
        summarize_mock = Mock(return_value="滚动压缩摘要")

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, messages):
                self.messages = list(messages)
                return AIMessage(content="完成", id="msg-ai-final")

        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            summarize_discarded_context_fn=summarize_mock,
            with_working_memory_fn=self._apply_message_updates,
        )
        state = {
            "messages": [
                HumanMessage(content="旧问题", id="msg-user-old"),
                AIMessage(content="旧回答", id="msg-ai-old"),
                HumanMessage(content="继续", id="msg-user-keep"),
            ],
            "summary": "已有摘要",
        }
        llm_with_tools = FakeLLMWithTools()
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "high"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "compact-thread"}},
                FakeLLM(),
                llm_with_tools,
                [],
                "fast",
                deps=deps,
            )

        self.assertEqual(result["compact_generation"], 1)
        self.assertEqual(result["last_compact_reason"], "auto_compact:high")
        self.assertEqual([message.content for message in result["messages"]], ["继续", "完成"])
        summary_types = [
            call.kwargs.get("summary_type")
            for call in deps.conversation_writer.record_summary.call_args_list
        ]
        self.assertIn("structured_handoff", summary_types)
        self.assertIn("compact_reset", summary_types)

    def test_react_node_overflow_retry_auto_compacts(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        trim_mock = Mock(side_effect=[
            ([HumanMessage(content="继续", id="msg-user-keep")], []),
            ([HumanMessage(content="继续", id="msg-user-keep")], [AIMessage(content="旧内容", id="old-ai")]),
        ])
        summarize_mock = Mock(return_value="压缩后的摘要")

        class FakeLLM:
            model = "gpt-4o-mini"

        class FakeLLMWithTools:
            def __init__(self):
                self.calls = 0

            def invoke(self, _messages):
                self.calls += 1
                if self.calls == 1:
                    raise Exception("prompt exceeds maximum context window")
                return AIMessage(content="重试后成功", id="msg-ai-final")

        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            summarize_discarded_context_fn=summarize_mock,
            with_working_memory_fn=self._apply_message_updates,
        )
        state = {
            "messages": [
                HumanMessage(content="旧问题", id="msg-user-old"),
                AIMessage(content="旧回答", id="msg-ai-old"),
                HumanMessage(content="继续", id="msg-user-keep"),
            ],
            "summary": "",
        }
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "medium"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "compact-overflow"}},
                FakeLLM(),
                FakeLLMWithTools(),
                [],
                "fast",
                deps=deps,
            )

        self.assertEqual(result["compact_generation"], 1)
        self.assertEqual(result["last_compact_reason"], "auto_compact:overflow_retry")
        self.assertEqual([message.content for message in result["messages"]], ["继续", "重试后成功"])

    def test_react_node_does_not_auto_compact_while_pending_approval(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        discarded = [AIMessage(content="旧的执行历史", id="old-ai")]
        trim_mock = Mock(return_value=([HumanMessage(content="继续", id="msg-user-keep")], discarded))
        summarize_mock = Mock(return_value="滚动压缩摘要")

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, _messages):
                return AIMessage(content="完成", id="msg-ai-final")

        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            summarize_discarded_context_fn=summarize_mock,
        )
        state = {
            "messages": [
                HumanMessage(content="旧问题", id="msg-user-old"),
                HumanMessage(content="继续", id="msg-user-keep"),
            ],
            "summary": "已有摘要",
            "pending_approval": True,
            "approval_reason": "等待高风险确认",
        }
        with patch("mortyclaw.core.agent.react_node.classify_context_pressure", return_value={"level": "high"}):
            result = run_react_agent_node(
                state,
                {"configurable": {"thread_id": "compact-blocked"}},
                FakeLLM(),
                FakeLLMWithTools(),
                [],
                "fast",
                deps=deps,
            )

        self.assertNotIn("compact_generation", result)
        summary_types = [
            call.kwargs.get("summary_type")
            for call in deps.conversation_writer.record_summary.call_args_list
        ]
        self.assertEqual(summary_types, ["structured_handoff"])

    def test_hydrate_todos_prefers_session_state_over_summary(self):
        from mortyclaw.core.runtime.todos import hydrate_todos_from_state_or_session

        summary_text = json.dumps({
            "version": 1,
            "todos": [{"id": "summary-1", "content": "来自 summary", "status": "pending"}],
        }, ensure_ascii=False)
        updates = hydrate_todos_from_state_or_session(
            {"summary": summary_text, "plan": []},
            session_todo_state={
                "items": [{"id": "session-1", "content": "来自 session", "status": "in_progress"}],
                "revision": 3,
            },
            summary_text=summary_text,
        )

        self.assertEqual(updates["todos"][0]["content"], "来自 session")
        self.assertEqual(updates["todo_revision"], 3)

    def test_react_node_auto_repairs_write_project_file_for_new_file_creation(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        trim_mock = Mock(return_value=([HumanMessage(content="继续")], []))
        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            get_current_plan_step_fn=lambda _state: {
                "step": 2,
                "description": "在项目路径下新建 Python 文件并写入实现",
                "intent": "file_write",
            },
        )

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_write_file",
                        "name": "write_project_file",
                        "args": {
                            "path": "median_of_sorted_arrays.py",
                            "content": "print('demo')\n",
                            "project_root": "/tmp/demo",
                        },
                    }],
                ),
                ToolMessage(
                    content=json.dumps({
                        "ok": False,
                        "error_kind": "FILE_NOT_FOUND",
                        "message": "文件不存在：median_of_sorted_arrays.py。如果你要新建文件，请将 create_if_missing 设为 true 后重试。",
                    }, ensure_ascii=False),
                    name="write_project_file",
                    tool_call_id="call_write_file",
                ),
            ],
            "route": "slow",
            "goal": "新建一个 python 文件并写入实现",
            "plan": [{"step": 2, "description": "在项目路径下新建 Python 文件并写入实现", "intent": "file_write"}],
            "current_step_index": 0,
            "slow_execution_mode": "structured",
            "run_status": "running",
            "repeated_failure_signature": "",
            "repeated_failure_count": 0,
        }

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, _messages):
                raise AssertionError("should not invoke llm when auto-repairing write_project_file")

        result = run_react_agent_node(
            state,
            {"configurable": {"thread_id": "auto-repair-create-file"}},
            FakeLLM(),
            FakeLLMWithTools(),
            [],
            "slow",
            deps=deps,
        )

        self.assertEqual(result["run_status"], "running")
        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "write_project_file")
        self.assertTrue(result["messages"][0].tool_calls[0]["args"]["create_if_missing"])
        self.assertEqual(result["repeated_failure_count"], 1)

    def test_react_node_replans_after_three_identical_tool_failures(self):
        from mortyclaw.core.agent.react_node import run_react_agent_node

        trim_mock = Mock(return_value=([HumanMessage(content="继续")], []))
        deps = self._make_react_node_test_deps(
            trim_context_messages_fn=trim_mock,
            get_current_plan_step_fn=lambda _state: {
                "step": 2,
                "description": "执行验证命令",
                "intent": "shell_execute",
            },
        )

        failure_payload = {
            "ok": False,
            "error_kind": "COMMAND_BLOCKED",
            "message": "run_project_command 失败：仅允许白名单命令。",
        }
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_run_cmd",
                        "name": "run_project_command",
                        "args": {
                            "command": "ls -la /tmp/demo",
                            "project_root": "/tmp/demo",
                        },
                    }],
                ),
                ToolMessage(
                    content=json.dumps(failure_payload, ensure_ascii=False),
                    name="run_project_command",
                    tool_call_id="call_run_cmd",
                ),
            ],
            "route": "slow",
            "goal": "运行测试并打印输出",
            "plan": [{"step": 2, "description": "执行验证命令", "intent": "shell_execute"}],
            "current_step_index": 0,
            "slow_execution_mode": "structured",
            "run_status": "running",
            "repeated_failure_signature": json.dumps(
                {
                    "tool": "run_project_command",
                    "error_kind": "COMMAND_BLOCKED",
                    "message": "run_project_command 失败：仅允许白名单命令。",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "repeated_failure_count": 2,
        }

        class FakeLLM:
            model_name = "gpt-4o-mini"

        class FakeLLMWithTools:
            def invoke(self, _messages):
                raise AssertionError("should not invoke llm after trip fuse")

        result = run_react_agent_node(
            state,
            {"configurable": {"thread_id": "repeat-failure-fuse"}},
            FakeLLM(),
            FakeLLMWithTools(),
            [],
            "slow",
            deps=deps,
        )

        self.assertEqual(result["run_status"], "replan_requested")
        self.assertIn("连续发生 3 次", result["replan_reason"])
        self.assertEqual(result["repeated_failure_count"], 3)

    def test_autonomous_slow_prompt_includes_todo_guidance(self):
        """测试 autonomous slow prompt 会显式注入 todo 驱动执行指令"""
        from mortyclaw.core.prompt_builder import build_react_prompt_bundle

        bundle = build_react_prompt_bundle(
            [HumanMessage(content="详细分析项目并继续推进")],
            "slow",
            {
                "goal": "详细分析项目并继续推进",
                "slow_execution_mode": "autonomous",
                "todos": [
                    {"id": "todo-1", "content": "扫描项目结构", "status": "completed"},
                    {"id": "todo-2", "content": "阅读核心模块", "status": "in_progress"},
                    {"id": "todo-3", "content": "整理结论", "status": "pending"},
                ],
            },
            active_summary="",
            session_prompt="",
            long_term_prompt="",
            current_plan_step=None,
            include_approved_goal_context=False,
        )

        self.assertNotIn("详细分析项目并继续推进", bundle.base_system_prompt)
        self.assertIn("autonomous slow 执行模式", bundle.dynamic_system_context)
        self.assertIn("update_todo_list", bundle.dynamic_system_context)
        self.assertIn("[当前 Todo 摘要]", bundle.dynamic_system_context)
        self.assertIn("阅读核心模块", bundle.dynamic_system_context)
        self.assertEqual(len(bundle.reference_messages), 0)
        self.assertEqual(len(bundle.conversation_messages), 1)

    def test_planner_normalize_plan_steps_supports_paper_research_intent(self):
        """测试 planner 支持 paper_research 步骤 intent 及默认校验信息"""
        from mortyclaw.core.planning import normalize_plan_steps

        steps = normalize_plan_steps(
            [{
                "description": "提炼 arxiv 上 2024 年 Mamba 论文的方法差异",
                "intent": "paper_research",
                "risk_level": "low",
            }],
            fallback_risk_level="medium",
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["intent"], "paper_research")
        self.assertIn("论文方法", steps[0]["success_criteria"])
        self.assertIn("arxiv_rag_ask", steps[0]["verification_hint"])

    def test_planner_normalize_plan_steps_overrides_analyze_for_obvious_file_write(self):
        """测试明显创建文件步骤即使被 LLM 标成 analyze，也会被纠偏成 file_write"""
        from mortyclaw.core.planning import normalize_plan_steps

        steps = normalize_plan_steps(
            [{
                "description": "创建实现二叉树最大路径和的 Python 文件",
                "intent": "analyze",
                "risk_level": "medium",
            }],
            fallback_risk_level="medium",
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["intent"], "file_write")

    def test_planner_normalize_plan_steps_overrides_analyze_for_script_execution(self):
        """测试明显运行脚本步骤即使被 LLM 标成 analyze，也会被纠偏成 shell_execute"""
        from mortyclaw.core.planning import normalize_plan_steps

        steps = normalize_plan_steps(
            [{
                "description": "运行 python max_path_sum_binary_tree.py 脚本并收集输出结果",
                "intent": "analyze",
                "risk_level": "medium",
            }],
            fallback_risk_level="medium",
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["intent"], "shell_execute")

    def test_planner_normalize_plan_steps_overrides_analyze_for_verification_step(self):
        """测试明显验证结果步骤即使被 LLM 标成 analyze，也会被纠偏成 test_verify"""
        from mortyclaw.core.planning import normalize_plan_steps

        steps = normalize_plan_steps(
            [{
                "description": "验证输出结果并确认通过或失败原因",
                "intent": "analyze",
                "risk_level": "medium",
            }],
            fallback_risk_level="medium",
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["intent"], "test_verify")

    def test_planner_normalize_plan_steps_keeps_plain_analysis_as_analyze(self):
        """测试普通项目分析步骤不会被过度纠偏成执行或验证"""
        from mortyclaw.core.planning import normalize_plan_steps

        steps = normalize_plan_steps(
            [{
                "description": "分析这个项目的训练入口和模块关系",
                "intent": "analyze",
                "risk_level": "medium",
            }],
            fallback_risk_level="medium",
        )

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["intent"], "analyze")

    def test_select_tools_for_paper_research_step_only_exposes_arxiv_tool(self):
        """测试 paper_research 步骤只开放 arxiv_rag_ask，不暴露项目读工具"""
        from langchain_core.tools import tool
        from mortyclaw.core.planning import select_tools_for_current_step

        @tool
        def arxiv_rag_ask(query: str = "", session_id: str = "") -> str:
            """Mock arxiv tool."""
            return "paper"

        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """Mock project read tool."""
            return "project"

        tools = [arxiv_rag_ask, read_project_file]
        selected = select_tools_for_current_step(
            {
                "description": "提炼 arxiv 上 2024 年 Mamba 论文的方法差异",
                "intent": "paper_research",
            },
            tools,
            current_project_path="/mnt/A/demo/repo",
        )
        selected_names = [tool.name for tool in selected]
        self.assertIn("arxiv_rag_ask", selected_names)
        self.assertNotIn("read_project_file", selected_names)

    def test_direct_arxiv_shortcut_is_disabled_for_slow_route(self):
        """测试 slow 路径即使当前步骤是论文问题，也不会触发 direct arxiv shortcut"""
        from mortyclaw.core.agent.react_node import _should_use_direct_arxiv_shortcut

        self.assertFalse(_should_use_direct_arxiv_shortcut(
            active_route="slow",
            route_source="mixed_research_task",
            effective_user_query="解释一下 arxiv 上的 Mamba 论文",
            should_direct_route_to_arxiv_rag_fn=lambda _query: True,
        ))

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_fast_project_analysis_does_not_expose_arxiv_tool(self, mock_load_skills, mock_get_provider):
        """测试普通 fast 项目分析不会额外暴露 arxiv_rag_ask，避免误用论文工具"""
        from langchain_core.tools import tool
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        @tool
        def read_project_file(filepath: str = "", project_root: str = "") -> str:
            """Mock project read tool."""
            return "project file"

        @tool
        def search_project_code(query: str = "", project_root: str = "") -> str:
            """Mock project search tool."""
            return "project search"

        @tool
        def show_git_diff(project_root: str = "") -> str:
            """Mock git diff tool."""
            return "git diff"

        @tool
        def arxiv_rag_ask(query: str = "", session_id: str = "") -> str:
            """Mock arxiv tool."""
            return "paper"

        @tool
        def tavily_web_search(query: str = "") -> str:
            """Mock web tool."""
            return "web"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="这是一个项目分析结果。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append([getattr(tool, "name", "") for tool in tools])
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[read_project_file, search_project_code, show_git_diff, arxiv_rag_ask, tavily_web_search],
        )
        result = app.invoke(
            {
                "messages": [HumanMessage(content="/mnt/A/hust_chp/hust_chp/book 详细分析这个项目，指出优缺点")],
                "summary": "",
            },
            config={"configurable": {"thread_id": "test_fast_project_analysis_no_arxiv"}},
        )

        self.assertEqual(result["run_status"], "done")
        fast_bound_tools = fake_provider.bound_tool_names[-1]
        self.assertIn("read_project_file", fast_bound_tools)
        self.assertIn("search_project_code", fast_bound_tools)
        self.assertIn("show_git_diff", fast_bound_tools)
        self.assertNotIn("arxiv_rag_ask", fast_bound_tools)

    def test_autonomous_slow_prompt_keeps_explicit_code_task_close_to_user_goal(self):
        """测试明确代码修改任务会提示模型先最小读取、再按需创建 Todo，而不是改写成泛化审查计划"""
        from mortyclaw.core.prompt_builder import build_react_prompt_bundle

        bundle = build_react_prompt_bundle(
            [HumanMessage(content="/mnt/A/demo/chat_agent 请在 main.py 中实现 stream=True 和历史保存")],
            "slow",
            {
                "goal": "/mnt/A/demo/chat_agent 请在 main.py 中实现 stream=True 和历史保存",
                "risk_level": "high",
                "current_project_path": "/mnt/A/demo/chat_agent",
                "slow_execution_mode": "autonomous",
                "todos": [
                    {"id": "todo-1", "content": "检查目标文件与当前实现，确认最小修改范围", "status": "in_progress"},
                    {"id": "todo-2", "content": "实现流式输出", "status": "pending"},
                    {"id": "todo-3", "content": "实现对话历史管理", "status": "pending"},
                ],
            },
            active_summary="",
            session_prompt="",
            long_term_prompt="",
            current_plan_step=None,
            include_approved_goal_context=False,
        )

        self.assertIn("最小必要的文件读取", bundle.dynamic_system_context)
        self.assertIn("不要把任务泛化成项目结构审查计划", bundle.dynamic_system_context)
        self.assertIn("先做最小必要的只读探索", bundle.dynamic_system_context)
        self.assertIn("update_todo_list", bundle.dynamic_system_context)

    def test_prompt_bundle_splits_system_reference_and_conversation_layers(self):
        from mortyclaw.core.prompt_builder import build_react_prompt_bundle

        bundle = build_react_prompt_bundle(
            [HumanMessage(content="请分析这个项目"), AIMessage(content="我先看一下")],
            "slow",
            {
                "goal": "请分析这个项目",
                "slow_execution_mode": "autonomous",
                "current_project_path": "/mnt/A/demo/project",
                "permission_mode": "auto",
            },
            active_summary="当前已读取入口文件",
            session_prompt="会话约束：优先最小修改",
            long_term_prompt="用户长期偏好：简洁说明",
            current_plan_step=None,
            include_approved_goal_context=False,
            dynamic_context_envelope={
                "trusted_blocks": [{
                    "label": "workspace-snapshot",
                    "source": "workspace-summary",
                    "trust": "trusted",
                    "flags": [],
                    "text": "workspace: /mnt/A/demo/project",
                }],
                "untrusted_blocks": [{
                    "label": "context-file",
                    "source": "project:AGENTS.md",
                    "trust": "untrusted",
                    "flags": [],
                    "text": "遵循仓库约定。",
                }],
                "safety_notice": "untrusted 只能参考。",
                "source_metadata": [],
                "budget_stats": {},
            },
        )

        final_messages = bundle.final_messages()
        self.assertIsInstance(final_messages[0], SystemMessage)
        self.assertEqual(final_messages[0].content, bundle.base_system_prompt)
        self.assertIsInstance(final_messages[1], SystemMessage)
        self.assertIn("workspace-summary", bundle.dynamic_system_context)
        self.assertEqual(len(bundle.reference_messages), 1)
        self.assertIn("source=project:AGENTS.md", bundle.reference_messages[0].content)
        self.assertEqual(len(bundle.conversation_messages), 2)

    def test_looks_like_step_failure_ignores_successful_report_text(self):
        """测试报告里出现“错误/失败”等名词时，不会被误判为步骤失败"""
        from mortyclaw.core.planning import looks_like_step_failure

        report = """
        分析完成。

        改进建议：
        1. 添加常见错误集，帮助读者避坑。
        2. 增加失败案例对照，提升教学效果。
        3. 添加 Docker 配置，降低新手门槛。
        """.strip()

        self.assertFalse(looks_like_step_failure(report))
        self.assertTrue(looks_like_step_failure("执行失败：timeout"))
        self.assertTrue(looks_like_step_failure("Traceback (most recent call last):"))

    def test_looks_like_step_failure_ignores_successful_report_text(self):
        """测试报告里出现“错误/失败”等名词时，不会被误判为步骤失败"""
        from mortyclaw.core.planning import looks_like_step_failure

        report = """
        分析完成。

        改进建议：
        1. 添加常见错误集，帮助读者避坑。
        2. 增加失败案例对照，提升教学效果。
        3. 添加 Docker 配置，降低新手门槛。
        """.strip()

        self.assertFalse(looks_like_step_failure(report))
        self.assertTrue(looks_like_step_failure("执行失败：timeout"))

    def test_tool_result_budget_persists_large_tool_output(self):
        """测试大工具结果会落 artifact，并只把 preview 留在消息中"""
        from mortyclaw.core.tool_result_budget import prepare_tool_messages_for_budget

        message = ToolMessage(
            content="pytest output\n" + ("line\n" * 4000),
            tool_call_id="call-big-output",
            name="run_project_tests",
            id="msg-big-output",
        )

        processed = prepare_tool_messages_for_budget(
            [message],
            thread_id="budget-thread",
            turn_id="turn-budget",
        )[0]

        self.assertIn("<persisted-output>", processed.content)
        artifact = processed.additional_kwargs["mortyclaw_artifact"]
        self.assertTrue(os.path.exists(artifact["artifact_path"]))
        os.remove(artifact["artifact_path"])

    def test_tool_result_budget_keeps_medium_read_output_when_turn_budget_allows(self):
        """测试普通源码/README 结果在总预算内不会过早落盘"""
        from mortyclaw.core.tool_result_budget import prepare_tool_messages_for_budget

        message = ToolMessage(
            content="code block\n" + ("line\n" * 2000),
            tool_call_id="call-medium-read",
            name="read_project_file",
            id="msg-medium-read",
        )

        processed = prepare_tool_messages_for_budget(
            [message],
            thread_id="budget-thread",
            turn_id="turn-keep-medium",
        )[0]

        self.assertNotIn("<persisted-output>", processed.content)

    def test_tool_result_budget_persists_when_total_turn_budget_overflows(self):
        """测试总预算超限时才对普通读文件结果做 artifact 化"""
        from mortyclaw.core.tool_result_budget import prepare_tool_messages_for_budget

        messages = [
            ToolMessage(
                content="read output\n" + ("line\n" * 2600),
                tool_call_id="call-overflow-1",
                name="read_office_file",
                id="msg-overflow-1",
            ),
            ToolMessage(
                content="read output\n" + ("line\n" * 2600),
                tool_call_id="call-overflow-2",
                name="read_project_file",
                id="msg-overflow-2",
            ),
        ]

        processed = prepare_tool_messages_for_budget(
            messages,
            thread_id="budget-thread",
            turn_id="turn-overflow",
        )

        self.assertTrue(any("<persisted-output>" in message.content for message in processed))

    def test_todo_state_roundtrip_preserves_single_in_progress(self):
        """测试 todo 写回 plan 时仍只保留一个 in_progress"""
        from mortyclaw.core.todo_state import merge_tool_written_todos, todos_to_plan

        existing_plan = [
            {"step": 1, "description": "检查入口", "status": "completed", "risk_level": "low"},
            {"step": 2, "description": "定位调用", "status": "pending", "risk_level": "medium"},
            {"step": 3, "description": "运行验证", "status": "pending", "risk_level": "medium"},
        ]
        existing_todos = [
            {"id": "step-1", "content": "检查入口", "status": "completed", "source_step": 1},
            {"id": "step-2", "content": "定位调用", "status": "in_progress", "source_step": 2},
            {"id": "step-3", "content": "运行验证", "status": "pending", "source_step": 3},
        ]
        requested = [
            {"id": "step-2", "content": "定位调用并记录关键函数", "status": "in_progress"},
            {"id": "step-3", "content": "运行验证", "status": "in_progress"},
        ]

        merged = merge_tool_written_todos(existing_plan, existing_todos, requested)
        rebuilt_plan, current_step_index = todos_to_plan(existing_plan, merged, lambda _text: "medium")

        in_progress_count = sum(1 for item in merged if item.get("status") == "in_progress")
        self.assertEqual(in_progress_count, 1)
        self.assertEqual(current_step_index, 1)
        self.assertEqual(rebuilt_plan[1]["description"], "定位调用并记录关键函数")

    def test_todo_merge_preserves_new_completed_items_and_advances_next_step(self):
        """测试工具新提交的 completed 状态不会被错误降回 in_progress/pending"""
        from mortyclaw.core.todo_state import merge_tool_written_todos, todos_to_plan

        existing_plan = [
            {"step": 1, "description": "检查目标文件", "status": "pending", "risk_level": "medium"},
            {"step": 2, "description": "实现流式输出", "status": "pending", "risk_level": "high"},
            {"step": 3, "description": "运行验证", "status": "pending", "risk_level": "high"},
        ]
        existing_todos = [
            {"id": "todo-1", "content": "检查目标文件", "status": "in_progress"},
            {"id": "todo-2", "content": "实现流式输出", "status": "pending"},
            {"id": "todo-3", "content": "运行验证", "status": "pending"},
        ]
        requested = [
            {"id": "todo-1", "content": "检查目标文件", "status": "completed"},
            {"id": "todo-2", "content": "实现流式输出", "status": "in_progress"},
            {"id": "todo-3", "content": "运行验证", "status": "pending"},
        ]

        merged = merge_tool_written_todos(existing_plan, existing_todos, requested)
        rebuilt_plan, current_step_index = todos_to_plan(existing_plan, merged, lambda _text: "medium")

        self.assertEqual(merged[0]["status"], "completed")
        self.assertEqual(merged[1]["status"], "in_progress")
        self.assertEqual(current_step_index, 1)
        self.assertEqual(rebuilt_plan[0]["status"], "completed")

    def test_autonomous_todo_is_no_longer_prebuilt_at_router_stage(self):
        """测试 autonomous slow 的 todo 不再由 router 预生成，而应在执行期通过 todo 工具创建"""
        from mortyclaw.core.runtime.todos import should_enable_todos

        self.assertFalse(
            should_enable_todos(
                "slow",
                [],
                execution_mode="autonomous",
                todos=[],
            )
        )

    def test_router_prefers_planner_for_explicit_project_tasks_with_numbered_requirements(self):
        """测试带明确项目路径和编号需求的代码任务会优先走 planner，而不是固定 autonomous todo 模板"""
        from mortyclaw.core.runtime.nodes.router import make_router_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        router_node = make_router_node(
            with_working_memory_fn=_with_working_memory,
            get_latest_user_query_fn=lambda messages: messages[-1].content if messages else "",
            schedule_long_term_memory_capture_fn=lambda _query: None,
            sync_session_memory_from_query_fn=lambda _query, _thread_id: {
                "current_project_path": "/mnt/A/hust_chp/hust_chp/Agent/chat_agent",
            },
            load_session_project_path_fn=lambda _thread_id: "",
            build_route_decision_fn=lambda query: {
                "route": "slow",
                "goal": query,
                "complexity": "complex",
                "risk_level": "medium",
                "planner_required": True,
                "route_locked": False,
                "slow_execution_mode": "structured",
                "route_source": "llm_classifier_slow_structured",
                "route_reason": "该任务需要先形成结构化计划",
                "route_confidence": 1.0,
            },
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        result = router_node(
            {
                "messages": [HumanMessage(
                    content=(
                        "/mnt/A/hust_chp/hust_chp/Agent/chat_agent 优化当前 Python 命令行聊天工具。"
                        " 1. 将非流式输出改成 stream=True 的流式输出"
                        " 2. 增加对话历史保存功能，保存到 logs/chat_时间戳.json"
                    )
                )],
                "summary": "",
            },
            {"configurable": {"thread_id": "test_router_prefers_planner_numbered_requirements"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertTrue(result["planner_required"])
        self.assertEqual(result["slow_execution_mode"], "structured")
        self.assertEqual(result["todos"], [])

    def test_router_prefers_autonomous_slow_for_explicit_small_scope_write_and_run_task(self):
        """测试目标明确、小范围的项目写入 + 运行任务直接进入 autonomous slow，且 router 不再预生成 todo"""
        from mortyclaw.core.runtime.nodes.router import make_router_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        router_node = make_router_node(
            with_working_memory_fn=_with_working_memory,
            get_latest_user_query_fn=lambda messages: messages[-1].content if messages else "",
            schedule_long_term_memory_capture_fn=lambda _query: None,
            sync_session_memory_from_query_fn=lambda _query, _thread_id: {
                "current_project_path": "/mnt/A/hust_chp/hust_chp/Project/ceshi",
            },
            load_session_project_path_fn=lambda _thread_id: "",
            build_route_decision_fn=lambda query: {
                "route": "slow",
                "goal": query,
                "complexity": "high_risk",
                "risk_level": "high",
                "planner_required": False,
                "route_locked": True,
                "slow_execution_mode": "autonomous",
                "route_source": "llm_classifier_slow_autonomous",
                "route_reason": "目标明确且范围收敛，适合直接执行",
                "route_confidence": 1.0,
            },
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        result = router_node(
            {
                "messages": [HumanMessage(
                    content=(
                        "/mnt/A/hust_chp/hust_chp/Project/ceshi 在这个目录下，新建一个python文件，"
                        "实现二叉树最大路径和，并运行脚本打印举例结果出来"
                    )
                )],
                "summary": "",
            },
            {"configurable": {"thread_id": "test_router_prefers_planner_write_and_run"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertFalse(result["planner_required"])
        self.assertEqual(result["slow_execution_mode"], "autonomous")
        self.assertEqual(result["todos"], [])

    def test_router_still_prefers_planner_for_wide_scope_write_and_run_task(self):
        """测试范围更大、显式多处修改的项目任务仍保持 planner-first"""
        from mortyclaw.core.runtime.nodes.router import make_router_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        router_node = make_router_node(
            with_working_memory_fn=_with_working_memory,
            get_latest_user_query_fn=lambda messages: messages[-1].content if messages else "",
            schedule_long_term_memory_capture_fn=lambda _query: None,
            sync_session_memory_from_query_fn=lambda _query, _thread_id: {
                "current_project_path": "/mnt/A/hust_chp/hust_chp/Project/ceshi",
            },
            load_session_project_path_fn=lambda _thread_id: "",
            build_route_decision_fn=lambda query: {
                "route": "slow",
                "goal": query,
                "complexity": "complex",
                "risk_level": "high",
                "planner_required": True,
                "route_locked": True,
                "slow_execution_mode": "structured",
                "route_source": "llm_classifier_slow_structured",
                "route_reason": "范围大且涉及多文件，适合先结构化规划",
                "route_confidence": 1.0,
            },
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        result = router_node(
            {
                "messages": [HumanMessage(
                    content=(
                        "/mnt/A/hust_chp/hust_chp/Project/ceshi 在这个目录下，批量修改多个文件里的二叉树实现，"
                        "然后运行脚本打印举例结果出来"
                    )
                )],
                "summary": "",
            },
            {"configurable": {"thread_id": "test_router_prefers_planner_wide_scope_write_and_run"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertTrue(result["planner_required"])
        self.assertEqual(result["slow_execution_mode"], "structured")
        self.assertEqual(result["todos"], [])

    def test_router_keeps_generic_project_analysis_on_autonomous_slow(self):
        """测试普通项目分析任务仍保持 autonomous slow，且 router 不预生成 todo"""
        from mortyclaw.core.runtime.nodes.router import make_router_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        router_node = make_router_node(
            with_working_memory_fn=_with_working_memory,
            get_latest_user_query_fn=lambda messages: messages[-1].content if messages else "",
            schedule_long_term_memory_capture_fn=lambda _query: None,
            sync_session_memory_from_query_fn=lambda _query, _thread_id: {
                "current_project_path": "/mnt/A/demo/repo",
            },
            load_session_project_path_fn=lambda _thread_id: "",
            build_route_decision_fn=lambda query: {
                "route": "slow",
                "goal": query,
                "complexity": "uncertain",
                "risk_level": "medium",
                "planner_required": False,
                "route_locked": False,
                "slow_execution_mode": "autonomous",
                "route_source": "llm_classifier_slow_autonomous",
                "route_reason": "项目分析目标明确，可直接边查边做",
                "route_confidence": 1.0,
            },
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        result = router_node(
            {
                "messages": [HumanMessage(content="帮我详细分析这个项目的模块边界，然后给我改进建议")],
                "summary": "",
            },
            {"configurable": {"thread_id": "test_router_keeps_autonomous_project_analysis"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertFalse(result["planner_required"])
        self.assertEqual(result["slow_execution_mode"], "autonomous")
        self.assertEqual(result["todos"], [])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_simple_query_uses_fast_agent_path(self, mock_load_skills, mock_get_provider):
        """测试简单问题会进入 fast_agent 节点"""
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.router_call_count = 0
                self.agent_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    self.router_call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "fast",
                        "slow_execution_mode": "",
                        "reason": "简单问题可直接回答",
                        "confidence": 0.98,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.agent_call_count += 1
                return AIMessage(content="当前时间是 12:00:00")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
        events = list(app.stream(
            {"messages": [HumanMessage(content="现在几点了？")], "summary": ""},
            config={"configurable": {"thread_id": "test_fast_route"}},
            stream_mode="updates",
        ))

        node_names = [next(iter(event.keys())) for event in events]
        self.assertIn("router", node_names)
        self.assertIn("fast_agent", node_names)
        self.assertNotIn("slow_agent", node_names)
        self.assertEqual(fake_provider.llm_with_tools.agent_call_count, 1)

        result = app.invoke(
            {"messages": [HumanMessage(content="现在几点了？")], "summary": ""},
            config={"configurable": {"thread_id": "test_fast_route_invoke"}},
        )
        self.assertEqual(result["route"], "fast")
        self.assertEqual(result["messages"][-1].content, "当前时间是 12:00:00")
        self.assertEqual(result["working_memory"]["current_mode"], "fast")
        self.assertEqual(result["working_memory"]["run_status"], "done")
        self.assertEqual(result["working_memory"]["goal"], "现在几点了？")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_fast_agent_escalates_high_risk_tool_intent_to_planner(self, mock_load_skills, mock_get_provider):
        """测试 fast 路径出现高风险工具意图时，会升级到 planner，而不是直接执行工具"""
        from mortyclaw.core.agent import create_agent_app
        from langchain_core.tools import tool

        mock_load_skills.return_value = []
        writes = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            writes.append(content)
            return f"写入成功: {content}"

        class FakeLLMWithTools:
            def __init__(self):
                self.fast_call_count = 0
                self.planner_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务规划器" in first_content:
                    self.planner_call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "goal": "解释时间查询并补充执行边界",
                        "reason": "fast 路径暴露出高风险写入意图，需要转入 slow 规划",
                        "confidence": 0.97,
                        "steps": [
                            {
                                "description": "先说明为什么不能在 fast 路径直接执行写入",
                                "intent": "analyze",
                                "risk_level": "medium",
                                "success_criteria": "明确升级原因",
                                "verification_hint": "提到 fast 路径检测到了高风险工具",
                                "needs_tools": True,
                            },
                            {
                                "description": "等待用户选择执行模式后再继续",
                                "intent": "report",
                                "risk_level": "medium",
                                "success_criteria": "给出后续执行入口",
                                "verification_hint": "引导到 ask/plan/auto",
                                "needs_tools": True,
                            },
                        ],
                    }, ensure_ascii=False))

                self.fast_call_count += 1
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_office_file",
                        "args": {"content": "not allowed on fast path"},
                        "id": "call_write_fast",
                        "type": "tool_call",
                    }],
                )

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file],
        )

        events = list(app.stream(
            {"messages": [HumanMessage(content="现在几点了？")], "summary": ""},
            config={"configurable": {"thread_id": "test_fast_escalation_stream"}},
            stream_mode="updates",
        ))
        node_names = [next(iter(event.keys())) for event in events]

        self.assertIn("router", node_names)
        self.assertIn("fast_agent", node_names)
        self.assertIn("planner", node_names)
        self.assertIn("approval_gate", node_names)
        self.assertNotIn("fast_tools", node_names)
        self.assertEqual(writes, [])
        self.assertGreaterEqual(fake_provider.llm_with_tools.fast_call_count, 1)
        self.assertGreaterEqual(fake_provider.llm_with_tools.planner_call_count, 1)

        result = app.invoke(
            {"messages": [HumanMessage(content="现在几点了？")], "summary": ""},
            config={"configurable": {"thread_id": "test_fast_escalation_invoke"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertTrue(result["planner_required"])
        self.assertEqual(result["plan_source"], "llm_planner")
        self.assertEqual(result["run_status"], "waiting_user")
        self.assertGreaterEqual(len(result["plan"]), 2)
        self.assertEqual(result["goal"], "解释时间查询并补充执行边界")
        self.assertEqual(writes, [])
        self.assertIn("请选择执行权限模式", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_high_risk_query_uses_slow_agent_path(self, mock_load_skills, mock_get_provider):
        """测试明显 slow 任务默认走 autonomous slow 主链路，而不是先经过 planner/reviewer"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="我会先分析任务，再进入后续编排。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini", checkpointer=MemorySaver())
        first_events = list(app.stream(
            {"messages": [HumanMessage(content="先查看项目结构，然后总结模块问题")], "summary": ""},
            config={"configurable": {"thread_id": "test_slow_route"}},
            stream_mode="updates",
        ))
        self.assertEqual([next(iter(event.keys())) for event in first_events], ["router", "approval_gate"])

        events = list(app.stream(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_slow_route"}},
            stream_mode="updates",
        ))
        node_names = [next(iter(event.keys())) for event in events]
        self.assertIn("router", node_names)
        self.assertIn("approval_gate", node_names)
        self.assertIn("slow_agent", node_names)
        self.assertIn("finalizer", node_names)
        self.assertNotIn("planner", node_names)
        self.assertNotIn("reviewer", node_names)
        self.assertNotIn("fast_agent", node_names)
        self.assertEqual(fake_provider.llm_with_tools.call_count, 2)

        first_result = app.invoke(
            {"messages": [HumanMessage(content="先查看项目结构，然后总结模块问题")], "summary": ""},
            config={"configurable": {"thread_id": "test_slow_route_invoke"}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_slow_route_invoke"}},
        )
        self.assertEqual(result["route"], "slow")
        self.assertEqual(result["risk_level"], "medium")
        self.assertEqual(result["run_status"], "done")
        self.assertEqual(result["slow_execution_mode"], "autonomous")
        self.assertEqual(result["final_answer"], "我会先分析任务，再进入后续编排。")
        self.assertEqual(len(result["step_results"]), 0)
        self.assertEqual(len(result["plan"]), 0)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_slow_path_planner_builds_multistep_plan(self, mock_load_skills, mock_get_provider):
        """测试 uncertain slow 任务仍可经过 planner 产出结构化计划"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, messages):
                self.call_count += 1
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务规划器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "goal": "评估需求推进路径",
                        "reason": "该请求适合先拆成结构化步骤",
                        "confidence": 0.96,
                        "steps": [
                            {
                                "description": "梳理需求背景",
                                "intent": "analyze",
                                "risk_level": "medium",
                                "success_criteria": "明确当前任务的边界",
                                "verification_hint": "引用用户原始问题中的关键约束",
                                "needs_tools": True,
                            },
                            {
                                "description": "给出推进建议",
                                "intent": "report",
                                "risk_level": "medium",
                                "success_criteria": "提供清晰的后续建议",
                                "verification_hint": "确保建议与背景一致",
                                "needs_tools": True,
                            },
                        ],
                    }, ensure_ascii=False))
                return AIMessage(content="计划执行完毕。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            checkpointer=MemorySaver(),
        )
        first_result = app.invoke(
            {"messages": [HumanMessage(content="这个需求后续该怎么推进比较合适")], "summary": ""},
            config={"configurable": {"thread_id": "test_planner_multi_step"}},
        )
        self.assertEqual(first_result["run_status"], "waiting_user")
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_planner_multi_step"}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertEqual(result["complexity"], "uncertain")
        self.assertEqual(result["plan_source"], "llm_planner")
        self.assertEqual(result["slow_execution_mode"], "structured")
        self.assertGreaterEqual(len(result["plan"]), 2)
        self.assertTrue(all(step["status"] == "completed" for step in result["plan"]))
        self.assertIn("复杂任务已完成", result["messages"][-1].content)
        self.assertIn("计划执行完毕。", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_high_risk_query_requests_execution_mode_before_executor(self, mock_load_skills, mock_get_provider):
        """测试高风险 slow 任务会先询问 ask/plan/auto，而不是直接进入 destructive approval"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []
        writes = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            writes.append(content)
            return f"写入成功: {content}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_office_file",
                        "args": {"content": 'print(\"hello\")'},
                        "id": "call_write",
                        "type": "tool_call",
                    }],
                )

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file],
            checkpointer=MemorySaver(),
        )
        result = app.invoke(
            {"messages": [HumanMessage(content="修改文件并运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": "test_approval_needed"}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 1)
        self.assertEqual(writes, [])
        self.assertEqual(result["permission_mode"], "")
        self.assertEqual(result["run_status"], "waiting_user")
        self.assertIn("请选择执行权限模式", result["messages"][-1].content)
        self.assertFalse(result["pending_approval"])
        self.assertEqual(result["pending_tool_calls"], [])
        self.assertEqual(result["working_memory"]["run_status"], "waiting_user")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_structured_slow_auto_mode_reply_does_not_reenter_planner(self, mock_load_skills, mock_get_provider):
        """测试 structured slow 选择 auto 后，会继续原计划执行，而不会把 auto 误当成新任务送回 planner"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, messages):
                self.call_count += 1
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务规划器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "goal": "评估需求推进路径",
                        "reason": "该请求适合先拆成结构化步骤",
                        "confidence": 0.96,
                        "steps": [
                            {
                                "description": "梳理需求背景",
                                "intent": "analyze",
                                "risk_level": "medium",
                                "success_criteria": "明确当前任务的边界",
                                "verification_hint": "引用用户原始问题中的关键约束",
                                "needs_tools": True,
                            },
                            {
                                "description": "给出推进建议",
                                "intent": "report",
                                "risk_level": "medium",
                                "success_criteria": "提供清晰的后续建议",
                                "verification_hint": "确保建议与背景一致",
                                "needs_tools": True,
                            },
                        ],
                    }, ensure_ascii=False))
                return AIMessage(content="计划执行完毕。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            checkpointer=MemorySaver(),
        )
        thread_id = "test_structured_auto_mode_reply"
        first_result = app.invoke(
            {"messages": [HumanMessage(content="这个需求后续该怎么推进比较合适")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(first_result["run_status"], "waiting_user")
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="auto")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(result["route"], "slow")
        self.assertEqual(result["plan_source"], "llm_planner")
        self.assertEqual(result["permission_mode"], "auto")
        self.assertEqual(result["slow_execution_mode"], "structured")
        self.assertGreaterEqual(len(result["plan"]), 2)
        self.assertTrue(all(step["description"] != "auto" for step in result["plan"]))
        self.assertIn("计划执行完毕。", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_session_memory_is_recalled_in_later_turns(self, mock_load_skills, mock_get_provider):
        """测试会话记忆会被同步保存，并在后续同一线程中重新注入 prompt"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.memory import MemoryStore

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.system_prompts = []

            def invoke(self, messages):
                self.system_prompts.append(
                    "\n\n".join(str(getattr(message, "content", "") or "") for message in messages)
                )
                return AIMessage(content="已读取会话记忆。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=os.path.join(temp_dir, "memory.sqlite3"))
            with patch('mortyclaw.core.agent.get_memory_store', return_value=store):
                app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
                thread_id = "session-memory-thread"

                first_result = app.invoke(
                    {"messages": [HumanMessage(content="请用中文回答，不要修改代码，项目在 /tmp/demo")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )
                second_result = app.invoke(
                    {"messages": [HumanMessage(content="继续分析一下项目结构")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )

        self.assertEqual(first_result["current_project_path"], "/tmp/demo")
        self.assertEqual(second_result["current_project_path"], "/tmp/demo")
        self.assertEqual(second_result["working_memory"]["current_project_path"], "/tmp/demo")
        self.assertIn("REFERENCE CONTEXT - NOT USER REQUEST", fake_provider.llm_with_tools.system_prompts[-1])
        self.assertIn("source=session-memory", fake_provider.llm_with_tools.system_prompts[-1])
        self.assertIn("当前项目路径：/tmp/demo", fake_provider.llm_with_tools.system_prompts[-1])
        self.assertIn("代码策略：本轮只分析，不修改任何代码。", fake_provider.llm_with_tools.system_prompts[-1])
        self.assertIn("输出偏好：请使用中文输出。", fake_provider.llm_with_tools.system_prompts[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_session_memory_normalizes_unrooted_project_path(self, mock_load_skills, mock_get_provider):
        """测试用户漏掉前导斜杠时，项目路径仍会被归一化写入状态和 prompt"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.memory import MemoryStore

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.system_prompts = []

            def invoke(self, messages):
                self.system_prompts.append(
                    "\n\n".join(str(getattr(message, "content", "") or "") for message in messages)
                )
                return AIMessage(content="已读取归一化路径。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=os.path.join(temp_dir, "memory.sqlite3"))
            project_root = tempfile.mkdtemp(dir=temp_dir)
            unrooted_path = project_root.lstrip("/")

            with patch('mortyclaw.core.agent.get_memory_store', return_value=store):
                app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
                result = app.invoke(
                    {"messages": [HumanMessage(content=f"{unrooted_path} 请分析这个项目结构")], "summary": ""},
                    config={"configurable": {"thread_id": "session-unrooted-path-thread"}},
                )

        normalized = os.path.realpath(project_root)
        self.assertEqual(result["current_project_path"], normalized)
        self.assertIn(f"当前项目路径：{normalized}", fake_provider.llm_with_tools.system_prompts[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_agent_prefers_structured_long_term_profile_snapshot(self, mock_load_skills, mock_get_provider):
        """测试 agent 优先从结构化长期记忆中加载用户画像，而不是仅依赖 Markdown 文件"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.memory import (
            DEFAULT_LONG_TERM_SCOPE,
            MemoryStore,
            USER_PROFILE_MEMORY_ID,
            USER_PROFILE_MEMORY_TYPE,
            build_memory_record,
        )

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.system_prompts = []

            def invoke(self, messages):
                self.system_prompts.append(
                    "\n\n".join(str(getattr(message, "content", "") or "") for message in messages)
                )
                return AIMessage(content="已读取长期画像。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=os.path.join(temp_dir, "memory.sqlite3"))
            store.upsert_memory(build_memory_record(
                memory_id=USER_PROFILE_MEMORY_ID,
                layer="long_term",
                scope=DEFAULT_LONG_TERM_SCOPE,
                type=USER_PROFILE_MEMORY_TYPE,
                content="# 用户画像\\n- 喜欢简洁回答",
                source_kind="manual_tool",
                source_ref="test",
            ))
            with patch('mortyclaw.core.agent.get_memory_store', return_value=store), \
                 patch('mortyclaw.core.agent.MEMORY_DIR', temp_dir):
                app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
                app.invoke(
                    {"messages": [HumanMessage(content="你记得我的偏好吗？")], "summary": ""},
                    config={"configurable": {"thread_id": "long-term-profile-thread"}},
                )

        self.assertIn("喜欢简洁回答", fake_provider.llm_with_tools.system_prompts[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_simple_query_skips_long_term_prompt_when_not_needed(self, mock_load_skills, mock_get_provider):
        """测试简单问题不会默认把长期画像塞进 prompt，避免无关延迟和噪声"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.memory import (
            DEFAULT_LONG_TERM_SCOPE,
            MemoryStore,
            USER_PROFILE_MEMORY_ID,
            USER_PROFILE_MEMORY_TYPE,
            build_memory_record,
        )

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.system_prompts = []

            def invoke(self, messages):
                self.system_prompts.append(
                    "\n\n".join(str(getattr(message, "content", "") or "") for message in messages)
                )
                return AIMessage(content="当前时间是 12:00:00")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=os.path.join(temp_dir, "memory.sqlite3"))
            store.upsert_memory(build_memory_record(
                memory_id=USER_PROFILE_MEMORY_ID,
                layer="long_term",
                scope=DEFAULT_LONG_TERM_SCOPE,
                type=USER_PROFILE_MEMORY_TYPE,
                content="# 用户画像\\n- 喜欢简洁回答",
                source_kind="manual_tool",
                source_ref="test",
            ))
            with patch('mortyclaw.core.agent.get_memory_store', return_value=store), \
                 patch('mortyclaw.core.agent.MEMORY_DIR', temp_dir):
                app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
                app.invoke(
                    {"messages": [HumanMessage(content="现在几点了？")], "summary": ""},
                    config={"configurable": {"thread_id": "skip-long-term-thread"}},
                )

        self.assertNotIn("【用户长期画像 (静态偏好)】", fake_provider.llm_with_tools.system_prompts[-1])
        self.assertNotIn("喜欢简洁回答", fake_provider.llm_with_tools.system_prompts[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_long_term_memory_note_is_captured_and_recalled(self, mock_load_skills, mock_get_provider):
        """测试显式偏好会异步写入长期记忆，并在后续记忆相关问题中被召回"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.memory import MemoryStore

        mock_load_skills.return_value = []

        class InlineAsyncWriter:
            def __init__(self, store):
                self.store = store

            def submit(self, record):
                self.store.upsert_memory(record)

            def flush(self):
                return None

        class FakeLLMWithTools:
            def __init__(self):
                self.system_prompts = []

            def invoke(self, messages):
                self.system_prompts.append(
                    "\n\n".join(str(getattr(message, "content", "") or "") for message in messages)
                )
                return AIMessage(content="我还记得。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        with tempfile.TemporaryDirectory() as temp_dir:
            store = MemoryStore(db_path=os.path.join(temp_dir, "memory.sqlite3"))
            inline_writer = InlineAsyncWriter(store)
            with patch('mortyclaw.core.agent.get_memory_store', return_value=store), \
                 patch('mortyclaw.core.agent.get_async_memory_writer', return_value=inline_writer), \
                 patch('mortyclaw.core.agent.MEMORY_DIR', temp_dir):
                app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
                thread_id = "long-term-capture-thread"
                app.invoke(
                    {"messages": [HumanMessage(content="记住我喜欢简洁回答")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )
                app.invoke(
                    {"messages": [HumanMessage(content="你还记得我的偏好吗？")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )

            note_records = store.list_memories(layer="long_term", scope="user_default", memory_type="user_preference")

        self.assertTrue(note_records)
        self.assertIn("记住我喜欢简洁回答", note_records[0]["content"])
        self.assertIn("记住我喜欢简洁回答", fake_provider.llm_with_tools.system_prompts[-1])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_high_risk_query_ask_mode_then_continues_after_approval(self, mock_load_skills, mock_get_provider):
        """测试选择 ask 后仍保持逐次审批，并在确认后恢复 destructive tool calls"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []
        writes = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            writes.append(content)
            return f"写入成功: {content}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0
                self.last_messages = []

            def invoke(self, messages):
                self.last_messages = messages
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "目标明确，适合直接执行",
                        "confidence": 0.97,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "write_office_file",
                            "args": {"content": 'print("hello world")'},
                            "id": "call_write",
                            "type": "tool_call",
                        }],
                    )
                return AIMessage(content="已根据原始任务继续执行。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file],
            checkpointer=MemorySaver(),
        )
        thread_id = "test_approval_resume"
        first_result = app.invoke(
            {"messages": [HumanMessage(content="修改文件并运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(first_result["run_status"], "waiting_user")
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)
        self.assertEqual(writes, [])

        second_result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(second_result["permission_mode"], "ask")
        self.assertTrue(second_result["pending_approval"])
        self.assertEqual(second_result["run_status"], "waiting_user")

        result = app.invoke(
            {"messages": [HumanMessage(content="确认执行")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(fake_provider.llm_with_tools.execution_call_count, 2)
        self.assertEqual(writes, ['print("hello world")'])
        self.assertFalse(result["pending_approval"])
        self.assertEqual(result["run_status"], "done")
        self.assertEqual(result["execution_guard_status"], "passed")
        self.assertIn("已根据原始任务继续执行。", result["messages"][-1].content)
        self.assertEqual(result["working_memory"]["execution_guard_status"], "passed")
        llm_input_texts = [getattr(message, "content", "") for message in fake_provider.llm_with_tools.last_messages]
        self.assertTrue(any("写入成功" in str(text) for text in llm_input_texts))

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_resume_execution_replans_when_target_file_changed(self, mock_load_skills, mock_get_provider):
        """测试审批后恢复执行前若目标文件已变化，会被 execution_guard 拦下并重新规划"""
        from mortyclaw.core.agent import create_agent_app
        from mortyclaw.core.tools.project.common import _file_hash
        from mortyclaw.core.tools.project.fs import write_project_file
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self, project_root: str, expected_hash: str):
                self.project_root = project_root
                self.expected_hash = expected_hash
                self.call_count = 0
                self.execution_call_count = 0
                self.planner_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "目标明确，适合直接执行",
                        "confidence": 0.98,
                    }, ensure_ascii=False))
                if "你是 MortyClaw 的任务规划器" in first_content:
                    self.call_count += 1
                    self.planner_call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "goal": "重新读取最新文件并重新规划修改步骤",
                        "reason": "目标文件在恢复执行前已经变化，需要重新规划",
                        "confidence": 0.98,
                        "steps": [
                            {
                                "description": "重新读取目标文件最新内容",
                                "intent": "read",
                                "risk_level": "medium",
                                "success_criteria": "拿到最新文件内容",
                                "verification_hint": "确认读取的是变更后的版本",
                                "needs_tools": True,
                            },
                            {
                                "description": "基于最新内容重新规划写入步骤",
                                "intent": "report",
                                "risk_level": "medium",
                                "success_criteria": "说明新的修改入口",
                                "verification_hint": "解释为什么原审批失效",
                                "needs_tools": True,
                            },
                        ],
                    }, ensure_ascii=False))

                self.call_count += 1
                self.execution_call_count += 1
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_project_file",
                        "args": {
                            "path": "main.py",
                            "content": "print('updated by agent')\n",
                            "project_root": self.project_root,
                            "expected_hash": self.expected_hash,
                        },
                        "id": "call_project_write",
                        "type": "tool_call",
                    }],
                )

        class FakeProvider:
            def __init__(self, project_root: str, expected_hash: str):
                self.llm_with_tools = FakeLLMWithTools(project_root, expected_hash)

            def bind_tools(self, _tools):
                return self.llm_with_tools

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "main.py")
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write("print('before')\n")
            initial_hash = _file_hash(target_path)

            fake_provider = FakeProvider(temp_dir, initial_hash)
            mock_get_provider.return_value = fake_provider

            with patch("mortyclaw.core.agent._sync_session_memory_from_query", return_value={"current_project_path": temp_dir}):
                app = create_agent_app(
                    provider_name="openai",
                    model_name="gpt-4o-mini",
                    tools=[write_project_file],
                    checkpointer=MemorySaver(),
                )
                thread_id = "test_resume_replan_on_drift"

                first_result = app.invoke(
                    {"messages": [HumanMessage(content="修改 main.py 并运行 python test.py")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )
                self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

                second_result = app.invoke(
                    {"messages": [HumanMessage(content="ask")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )
                self.assertTrue(second_result["pending_approval"])
                self.assertTrue(second_result["pending_execution_snapshot"])

                with open(target_path, "w", encoding="utf-8") as handle:
                    handle.write("print('changed externally')\n")

                result = app.invoke(
                    {"messages": [HumanMessage(content="确认执行")], "summary": ""},
                    config={"configurable": {"thread_id": thread_id}},
                )

        self.assertEqual(fake_provider.llm_with_tools.execution_call_count, 1)
        self.assertEqual(fake_provider.llm_with_tools.planner_call_count, 1)
        self.assertEqual(result["execution_guard_status"], "replan_requested")
        self.assertIn("目标文件内容已变化", result["execution_guard_reason"])
        self.assertEqual(result["run_status"], "waiting_user")
        self.assertEqual(result["plan_source"], "llm_planner")
        self.assertEqual(result["working_memory"]["execution_guard_status"], "replan_requested")
        self.assertIn("请选择执行权限模式", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_high_risk_query_plan_mode_terminates_destructive_task(self, mock_load_skills, mock_get_provider):
        """测试选择 plan 后，涉及写入/测试/命令的 slow 任务会直接终止"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            return f"写入成功: {content}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="不应该真正开始执行。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file],
            checkpointer=MemorySaver(),
        )
        thread_id = "test_plan_mode_termination"
        first_result = app.invoke(
            {"messages": [HumanMessage(content="修改文件并运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="plan")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 1)
        self.assertEqual(result["permission_mode"], "plan")
        self.assertEqual(result["run_status"], "cancelled")
        self.assertIn("只读模式", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_high_risk_query_auto_mode_skips_second_approval(self, mock_load_skills, mock_get_provider):
        """测试选择 auto 后，允许的 destructive tool 会直接执行，不再二次审批"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []
        writes = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            writes.append(content)
            return f"写入成功: {content}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0
                self.last_messages = []

            def invoke(self, messages):
                self.last_messages = messages
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    self.call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "目标明确，适合直接执行",
                        "confidence": 0.97,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "write_office_file",
                            "args": {"content": 'print(\"auto mode\")'},
                            "id": "call_write_auto",
                            "type": "tool_call",
                        }],
                    )
                return AIMessage(content="auto 模式已完成。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file],
            checkpointer=MemorySaver(),
        )
        thread_id = "test_auto_mode_execution"
        first_result = app.invoke(
            {"messages": [HumanMessage(content="修改文件并运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="auto")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 3)
        self.assertEqual(writes, ['print("auto mode")'])
        self.assertFalse(result["pending_approval"])
        self.assertEqual(result["permission_mode"], "auto")
        self.assertEqual(result["run_status"], "done")
        self.assertIn("auto 模式已完成。", result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_each_high_risk_step_requires_fresh_approval(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 在连续 destructive tool 批次之间会再次进入审批"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            return f"写入成功: {content}"

        @tool
        def execute_office_shell(command: str) -> str:
            """Mock shell tool."""
            return f"shell结果: {command}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    self.call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "线性执行即可推进",
                        "confidence": 0.96,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "write_office_file",
                            "args": {"content": 'print("hello world")'},
                            "id": "call_write",
                            "type": "tool_call",
                        }],
                    )
                if self.execution_call_count == 2:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "execute_office_shell",
                            "args": {"command": "python test.py"},
                            "id": "call_shell",
                            "type": "tool_call",
                        }],
                    )
                if self.execution_call_count == 3:
                    return AIMessage(content="全部完成：hello world")
                raise AssertionError("不应该出现额外的 LLM 调用")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file, execute_office_shell],
            checkpointer=MemorySaver(),
        )
        thread_id = "test_step_level_approval"

        first_result = app.invoke(
            {"messages": [HumanMessage(content="先创建 test.py，然后运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(fake_provider.llm_with_tools.call_count, 1)
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        mode_result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(fake_provider.llm_with_tools.call_count, 2)
        self.assertTrue(mode_result["pending_approval"])
        self.assertTrue(mode_result["approval_prompted"])

        second_result = app.invoke(
            {"messages": [HumanMessage(content="确认执行")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(fake_provider.llm_with_tools.call_count, 3)
        self.assertTrue(second_result["pending_approval"])
        self.assertTrue(second_result["approval_prompted"])
        self.assertEqual(second_result["run_status"], "waiting_user")
        self.assertIn("execute_office_shell", second_result["messages"][-1].content)

        final_result = app.invoke(
            {"messages": [HumanMessage(content="确认执行")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertEqual(fake_provider.llm_with_tools.call_count, 4)
        self.assertFalse(final_result["pending_approval"])
        self.assertFalse(final_result["approval_prompted"])
        self.assertEqual(final_result["run_status"], "done")
        self.assertIn("全部完成：hello world", final_result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_low_risk_step_cannot_execute_future_shell_tool(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 就算连续提出 destructive tool，也会在第二批执行前再次停在审批"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

        mock_load_skills.return_value = []
        executed_commands = []

        @tool
        def write_office_file(content: str) -> str:
            """Mock write tool."""
            return f"写入成功: {content}"

        @tool
        def execute_office_shell(command: str) -> str:
            """Mock shell tool."""
            executed_commands.append(command)
            return f"shell结果: {command}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    self.call_count += 1
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "步骤线性清楚，适合直接执行",
                        "confidence": 0.96,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "write_office_file",
                            "args": {"content": 'print("hello world")'},
                            "id": "call_write",
                            "type": "tool_call",
                        }],
                    )
                if self.execution_call_count == 2:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "execute_office_shell",
                            "args": {"command": "python test.py"},
                            "id": "call_shell_too_early",
                            "type": "tool_call",
                        }],
                    )
                raise AssertionError("不应该出现额外的 LLM 调用")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[write_office_file, execute_office_shell],
            checkpointer=MemorySaver(),
        )
        thread_id = "test_block_future_shell"

        first_result = app.invoke(
            {"messages": [HumanMessage(content="先创建 test.py，然后确认代码里有 print，最后运行 python test.py")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)
        self.assertEqual(fake_provider.llm_with_tools.call_count, 1)

        mode_result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )
        self.assertTrue(mode_result["pending_approval"])
        self.assertEqual(fake_provider.llm_with_tools.call_count, 2)

        second_result = app.invoke(
            {"messages": [HumanMessage(content="确认执行")], "summary": ""},
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 3)
        self.assertEqual(executed_commands, [])
        self.assertTrue(second_result["pending_approval"])
        self.assertTrue(second_result["approval_prompted"])
        self.assertEqual(second_result["run_status"], "waiting_user")
        self.assertIn("execute_office_shell", second_result["messages"][-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_reviewer_retries_failed_step_before_success(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 会在可重试失败后直接在 slow_agent 内重试"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                if self.call_count == 1:
                    return AIMessage(content="执行失败：timeout")
                return AIMessage(content="步骤执行成功。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini", checkpointer=MemorySaver())
        first_result = app.invoke(
            {"messages": [HumanMessage(content="分步骤检查项目结构")], "summary": ""},
            config={"configurable": {"thread_id": "test_reviewer_retry"}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_reviewer_retry"}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 2)
        self.assertEqual(result["run_status"], "done")
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(result["final_answer"], "步骤执行成功。")
        self.assertEqual(len(result["step_results"]), 0)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_reviewer_replans_after_retry_budget_exhausted(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 在重试预算耗尽后会回到 planner 重规划"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                if self.call_count <= 5:
                    return AIMessage(content=f"执行失败：timeout #{self.call_count}")
                return AIMessage(content="重规划后执行成功。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini", checkpointer=MemorySaver())
        first_result = app.invoke(
            {"messages": [HumanMessage(content="分步骤检查项目结构")], "summary": "", "max_retries": 2},
            config={"configurable": {"thread_id": "test_reviewer_replan"}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": "", "max_retries": 2},
            config={"configurable": {"thread_id": "test_reviewer_replan"}},
        )

        self.assertGreaterEqual(fake_provider.llm_with_tools.call_count, 4)
        self.assertEqual(result["run_status"], "done")
        self.assertTrue(result["plan_source"])
        self.assertIn("重规划后执行成功。", result["messages"][-1].content)
        self.assertTrue(
            any(item.get("outcome") == "failed" for item in result.get("step_results", []))
            or result.get("plan_source") in {"rule_fallback", "llm_planner", "rule_first"}
        )

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_reviewer_does_not_retry_successful_analysis_report(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 的成功分析报告不会被误判成失败并触发重试"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "slow",
                        "slow_execution_mode": "autonomous",
                        "reason": "目标明确但需要持续执行",
                        "confidence": 0.93,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                return AIMessage(
                    content=(
                        "分析完成。\n"
                        "建议补充常见错误集，并增加失败案例对照，方便读者练习。"
                    )
                )

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini", checkpointer=MemorySaver())
        first_result = app.invoke(
            {"messages": [HumanMessage(content="详细分析这个项目，然后给改进建议")], "summary": ""},
            config={"configurable": {"thread_id": "test_success_report_no_retry"}},
        )
        self.assertIn("请选择执行权限模式", first_result["messages"][-1].content)

        result = app.invoke(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_success_report_no_retry"}},
        )

        self.assertEqual(fake_provider.llm_with_tools.execution_call_count, 1)
        self.assertEqual(result["run_status"], "done")
        self.assertEqual(result["slow_execution_mode"], "autonomous")
        self.assertEqual(result["retry_count"], 0)

    def test_reviewer_treats_missing_write_tool_report_as_failure(self):
        """测试 reviewer 遇到“没有可用的写文件工具”类回复时会判失败而不是 completed"""
        from mortyclaw.core.errors.policy import classify_error, serialize_classified_error
        from mortyclaw.core.runtime.nodes.reviewer import make_reviewer_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        def _get_current_plan_step(state):
            plan = state.get("plan", [])
            index = state.get("current_step_index", 0)
            if 0 <= index < len(plan):
                return plan[index]
            return None

        def _update_plan_step(plan, index, *, status):
            updated = [dict(step) for step in plan]
            if 0 <= index < len(updated):
                updated[index]["status"] = status
            return updated

        reviewer_node = make_reviewer_node(
            with_working_memory_fn=_with_working_memory,
            get_current_plan_step_fn=_get_current_plan_step,
            looks_like_step_failure_fn=lambda content: "没有可用的写文件工具" in str(content or ""),
            update_plan_step_fn=_update_plan_step,
            step_requires_approval_fn=lambda _step: False,
            build_approval_reason_fn=lambda _step: "",
            classify_error_fn=classify_error,
            serialize_classified_error_fn=serialize_classified_error,
            plan_to_todos_fn=lambda _plan, _index: [],
            should_enable_todos_fn=lambda _route, _plan: False,
            build_todo_state_from_plan_fn=lambda _plan, _index, revision, last_event: {
                "revision": revision,
                "last_event": last_event,
            },
            save_session_todo_state_fn=lambda _thread_id, _todo_state: None,
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        state = {
            "messages": [
                AIMessage(
                    content="当前我只能用项目读取/检索工具，没有可用的写文件工具，所以无法实际创建文件。",
                    additional_kwargs={"mortyclaw_step_outcome": "success_candidate"},
                )
            ],
            "plan": [{
                "step": 1,
                "description": "创建实现二叉树最大路径和的 Python 文件",
                "intent": "file_write",
                "status": "in_progress",
                "verification_hint": "检查文件是否真的写入。",
            }],
            "current_step_index": 0,
            "retry_count": 0,
            "max_retries": 0,
            "run_status": "running",
            "step_results": [],
        }

        result = reviewer_node(state, {"configurable": {"thread_id": "test_reviewer_write_failure"}})

        self.assertEqual(result["run_status"], "replan_requested")
        self.assertEqual(result["plan"][0]["status"], "failed")
        self.assertEqual(result["step_results"][0]["outcome"], "failed")
        self.assertIn("没有可用的写文件工具", result["last_error"])

    def test_successful_shell_execution_tool_result_short_circuits_to_step_success(self):
        """测试 shell_execute 步骤遇到成功的 run_project_command 结果时，会直接形成成功 step 结论"""
        from mortyclaw.core.agent.react_node import _build_successful_execution_step_response
        from mortyclaw.core.agent.recovery import STEP_OUTCOME_SUCCESS_CANDIDATE, RESPONSE_KIND_STEP_RESULT

        def _annotate_ai_message(message, **metadata):
            additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
            additional_kwargs.update(metadata)
            return message.model_copy(update={"additional_kwargs": additional_kwargs})

        deps = type("Deps", (), {
            "annotate_ai_message_fn": staticmethod(_annotate_ai_message),
            "step_outcome_success_candidate": STEP_OUTCOME_SUCCESS_CANDIDATE,
            "response_kind_step_result": RESPONSE_KIND_STEP_RESULT,
        })()

        tool_message = ToolMessage(
            content=json.dumps({
                "ok": True,
                "command": "python min_window_substring.py",
                "exit_code": 0,
                "stdout": "示例1的最短覆盖子串为: BANC\n",
            }, ensure_ascii=False),
            name="run_project_command",
            tool_call_id="call_run_demo",
        )
        response = _build_successful_execution_step_response(
            [tool_message],
            {"intent": "shell_execute", "description": "运行程序打印示例结果"},
            deps=deps,
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.content, "示例1的最短覆盖子串为: BANC")
        self.assertEqual(response.additional_kwargs["mortyclaw_step_outcome"], STEP_OUTCOME_SUCCESS_CANDIDATE)
        self.assertEqual(response.additional_kwargs["mortyclaw_response_kind"], RESPONSE_KIND_STEP_RESULT)

    def test_reviewer_preserves_success_when_unknown_error_follows_successful_execution(self):
        """测试系统级 unknown 异常不会覆盖已成功的 shell_execute 步骤结果"""
        from mortyclaw.core.errors.policy import (
            ClassifiedError,
            ErrorKind,
            RecoveryAction,
            RetryPolicy,
            classify_error,
            serialize_classified_error,
        )
        from mortyclaw.core.runtime.nodes.reviewer import make_reviewer_node

        class _NoopAuditLogger:
            def log_event(self, **_kwargs):
                return None

        def _with_working_memory(state, updates):
            merged = dict(state)
            merged.update(updates)
            return merged

        def _get_current_plan_step(state):
            plan = state.get("plan", [])
            index = state.get("current_step_index", 0)
            if 0 <= index < len(plan):
                return plan[index]
            return None

        def _update_plan_step(plan, index, *, status):
            updated = [dict(step) for step in plan]
            if 0 <= index < len(updated):
                updated[index]["status"] = status
            return updated

        reviewer_node = make_reviewer_node(
            with_working_memory_fn=_with_working_memory,
            get_current_plan_step_fn=_get_current_plan_step,
            looks_like_step_failure_fn=lambda _content: False,
            update_plan_step_fn=_update_plan_step,
            step_requires_approval_fn=lambda _step: False,
            build_approval_reason_fn=lambda _step: "",
            classify_error_fn=classify_error,
            serialize_classified_error_fn=serialize_classified_error,
            plan_to_todos_fn=lambda _plan, _index: [],
            should_enable_todos_fn=lambda _route, _plan: False,
            build_todo_state_from_plan_fn=lambda _plan, _index, revision, last_event: {
                "revision": revision,
                "last_event": last_event,
            },
            save_session_todo_state_fn=lambda _thread_id, _todo_state: None,
            clear_session_todo_state_fn=lambda _thread_id: None,
            audit_logger_instance=_NoopAuditLogger(),
        )

        unknown_error = ClassifiedError(
            kind=ErrorKind.UNKNOWN,
            recovery_action=RecoveryAction.REPLAN,
            retry_policy=RetryPolicy(retryable=False, max_attempts=0),
            message="context summarization timed out",
            user_visible_hint="遇到未分类异常，我会尝试重新规划当前步骤。",
        )
        state = {
            "messages": [
                ToolMessage(
                    content=json.dumps({
                        "ok": True,
                        "command": "python min_window_substring.py",
                        "exit_code": 0,
                        "stdout": "示例1的最短覆盖子串为: BANC\n",
                    }, ensure_ascii=False),
                    name="run_project_command",
                    tool_call_id="call_run_demo",
                ),
                AIMessage(
                    content="遇到未分类异常，我会尝试重新规划当前步骤。",
                    additional_kwargs={
                        "mortyclaw_error": serialize_classified_error(unknown_error),
                        "mortyclaw_step_outcome": "failure",
                    },
                ),
            ],
            "plan": [{
                "step": 4,
                "description": "运行程序打印示例结果",
                "intent": "shell_execute",
                "status": "in_progress",
                "verification_hint": "确认输出与示例一致。",
            }],
            "current_step_index": 0,
            "retry_count": 0,
            "max_retries": 0,
            "run_status": "review_pending",
            "step_results": [],
        }

        result = reviewer_node(state, {"configurable": {"thread_id": "test_reviewer_preserves_success_after_execution"}})

        self.assertEqual(result["run_status"], "done")
        self.assertEqual(result["plan"][0]["status"], "completed")
        self.assertEqual(result["step_results"][0]["outcome"], "completed")
        self.assertIn("BANC", result["step_results"][0]["result_summary"])

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_autonomous_slow_text_is_marked_as_final_answer_and_preserved_by_finalizer(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 的成功文本会直接作为 final_answer，并且 finalizer 不再重复发一次相同消息"""
        from mortyclaw.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                return AIMessage(content="分析完成。\n建议补充常见错误集，并增加失败案例对照。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini", checkpointer=MemorySaver())
        first_events = list(app.stream(
            {"messages": [HumanMessage(content="详细分析这个项目，然后给改进建议")], "summary": ""},
            config={"configurable": {"thread_id": "test_step_result_stream"}},
            stream_mode="updates",
        ))
        self.assertEqual([next(iter(event.keys())) for event in first_events], ["router", "approval_gate"])
        events = list(app.stream(
            {"messages": [HumanMessage(content="ask")], "summary": ""},
            config={"configurable": {"thread_id": "test_step_result_stream"}},
            stream_mode="updates",
        ))

        slow_agent_messages = [
            event["slow_agent"]["messages"][-1]
            for event in events
            if "slow_agent" in event and event["slow_agent"].get("messages")
        ]
        self.assertTrue(slow_agent_messages)
        self.assertTrue(any(
            msg.additional_kwargs.get("mortyclaw_response_kind") == "final_answer"
            for msg in slow_agent_messages
        ))

        finalizer_events = [
            event["finalizer"]
            for event in events
            if "finalizer" in event
        ]
        self.assertTrue(finalizer_events)
        self.assertFalse(any(event.get("messages") for event in finalizer_events))
        self.assertEqual(finalizer_events[-1].get("final_answer"), slow_agent_messages[-1].content)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.agent.BUILTIN_TOOLS', [])
    def test_autonomous_slow_success_marks_remaining_todos_completed(self, mock_load_skills, mock_get_provider):
        """测试 autonomous slow 成功完成后，会把残留的 todo/plan 收尾成 completed，避免 UI 卡在第一条"""
        from mortyclaw.core.agent import _complete_autonomous_todos

        mock_load_skills.return_value = []
        result = _complete_autonomous_todos({
            "risk_level": "high",
            "todos": [
                {"id": "todo-1", "content": "检查项目结构并定位入口", "status": "in_progress"},
                {"id": "todo-2", "content": "实现流式输出", "status": "pending"},
                {"id": "todo-3", "content": "运行验证", "status": "pending"},
            ],
            "plan": [
                {"step": 1, "description": "检查项目结构并定位入口", "status": "pending", "risk_level": "medium", "intent": "read"},
                {"step": 2, "description": "实现流式输出", "status": "pending", "risk_level": "high", "intent": "code_edit"},
                {"step": 3, "description": "运行验证", "status": "pending", "risk_level": "high", "intent": "test_verify"},
            ],
            "todo_revision": 2,
        })

        self.assertTrue(result["todos"])
        self.assertTrue(all(item.get("status") == "completed" for item in result["todos"]))
        self.assertTrue(all(step.get("status") == "completed" for step in result["plan"]))
        self.assertEqual(result["current_step_index"], 2)

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_tavily_schedule_query_is_adjusted_to_general(self, mock_load_skills, mock_get_provider):
        """测试赛程类查询会把 Tavily topic 校正为 general"""
        from mortyclaw.core.agent import create_agent_app
        from langchain_core.tools import tool

        mock_load_skills.return_value = []

        @tool
        def tavily_web_search(
            query: str,
            topic: str = "general",
            max_results: int = 5,
            include_answer: bool = True,
        ) -> str:
            """Mock Tavily search tool."""
            return f"query={query};topic={topic}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "fast",
                        "slow_execution_mode": "",
                        "reason": "搜索型问题适合 fast 工具调用",
                        "confidence": 0.95,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "tavily_web_search",
                            "args": {
                                "query": "2026年4月18日 LPL 英雄联盟比赛赛程",
                                "topic": "news",
                            },
                            "id": "call_schedule",
                            "type": "tool_call",
                        }],
                    )

                tool_messages = [m for m in messages if getattr(m, "type", "") == "tool"]
                return AIMessage(content=tool_messages[-1].content)

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[tavily_web_search],
        )

        with patch("mortyclaw.core.routing._get_local_now", return_value=datetime(2026, 4, 26, 10, 0, 0)):
            result = app.invoke(
                {"messages": [HumanMessage(content="你查一下明天英雄联盟比赛LPL赛区有哪些比赛")], "summary": ""},
                config={"configurable": {"thread_id": "test_tavily_general"}},
            )

        self.assertEqual(fake_provider.llm_with_tools.execution_call_count, 2)
        self.assertEqual(result["route"], "fast")
        self.assertEqual(result["messages"][-1].content, "query=2026-04-27 LPL 英雄联盟比赛赛程;topic=general")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    @patch('mortyclaw.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_planner_can_downgrade_uncertain_task_to_fast(self, mock_load_skills, mock_get_provider):
        """测试不确定任务进入 planner 后，可以语义降级为 fast"""
        from mortyclaw.core.agent import create_agent_app

        mock_load_skills.return_value = []

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0

            def invoke(self, _messages):
                self.call_count += 1
                if self.call_count == 1:
                    return AIMessage(content=json.dumps({
                        "route": "fast",
                        "goal": "解释需求推进方式",
                        "reason": "这是直接建议型问题，不需要 slow 编排",
                        "confidence": 0.92,
                        "steps": [],
                    }, ensure_ascii=False))
                return AIMessage(content="直接给出推进建议。")

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
        result = app.invoke(
            {"messages": [HumanMessage(content="这个需求后续该怎么推进比较合适")], "summary": ""},
            config={"configurable": {"thread_id": "test_planner_fast_downgrade"}},
        )

        self.assertEqual(fake_provider.llm_with_tools.call_count, 2)
        self.assertEqual(result["route"], "fast")
        self.assertFalse(result["planner_required"])
        self.assertEqual(result["messages"][-1].content, "直接给出推进建议。")

    @patch('mortyclaw.core.agent.get_provider')
    @patch('mortyclaw.core.agent.load_dynamic_skills')
    def test_tavily_news_query_is_adjusted_to_news(self, mock_load_skills, mock_get_provider):
        """测试新闻类查询会把 Tavily topic 校正为 news"""
        from mortyclaw.core.agent import create_agent_app
        from langchain_core.tools import tool

        mock_load_skills.return_value = []

        @tool
        def tavily_web_search(
            query: str,
            topic: str = "general",
            max_results: int = 5,
            include_answer: bool = True,
        ) -> str:
            """Mock Tavily search tool."""
            return f"query={query};topic={topic}"

        class FakeLLMWithTools:
            def __init__(self):
                self.call_count = 0
                self.execution_call_count = 0

            def invoke(self, messages):
                first_content = str(messages[0].content) if messages else ""
                if "你是 MortyClaw 的任务路由器" in first_content:
                    return AIMessage(content=json.dumps({
                        "route": "fast",
                        "slow_execution_mode": "",
                        "reason": "新闻查询适合 fast 工具调用",
                        "confidence": 0.95,
                    }, ensure_ascii=False))
                self.call_count += 1
                self.execution_call_count += 1
                if self.execution_call_count == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "tavily_web_search",
                            "args": {
                                "query": "今天热点新闻",
                            },
                            "id": "call_news",
                            "type": "tool_call",
                        }],
                    )

                tool_messages = [m for m in messages if getattr(m, "type", "") == "tool"]
                return AIMessage(content=tool_messages[-1].content)

        class FakeProvider:
            def __init__(self):
                self.llm_with_tools = FakeLLMWithTools()

            def bind_tools(self, _tools):
                return self.llm_with_tools

        fake_provider = FakeProvider()
        mock_get_provider.return_value = fake_provider

        app = create_agent_app(
            provider_name="openai",
            model_name="gpt-4o-mini",
            tools=[tavily_web_search],
        )

        with patch("mortyclaw.core.routing._get_local_now", return_value=datetime(2026, 4, 26, 10, 0, 0)):
            result = app.invoke(
                {"messages": [HumanMessage(content="今天热点新闻")], "summary": ""},
                config={"configurable": {"thread_id": "test_tavily_news"}},
            )

        self.assertEqual(fake_provider.llm_with_tools.execution_call_count, 2)
        self.assertEqual(result["messages"][-1].content, "query=2026-04-26 热点新闻;topic=news")


if __name__ == '__main__':
    unittest.main()
