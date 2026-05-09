import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from entry.cli import _generate_thread_id, _resolve_default_thread_id
from entry.main import (
    _count_rendered_output_lines,
    _effective_plan_render_state,
    _generate_next_runtime_thread_id,
    _normalize_plan_items,
    _resolve_initial_thread_id,
    _render_current_plan_block,
    _render_stream_node_message,
    _run_live_output_operation,
    _should_pre_render_live_plan,
    _should_render_message_below_live_plan,
    _should_suspend_prompt_for_live_output,
)


class FakeSessionRepository:
    def __init__(self, thread_ids):
        self.thread_ids = thread_ids

    def list_sessions(self, limit=10000):
        return [{"thread_id": thread_id} for thread_id in self.thread_ids[:limit]]

    def get_latest_session(self):
        if not self.thread_ids:
            return None
        return {"thread_id": self.thread_ids[0]}


class CliSessionIdTests(unittest.TestCase):
    def test_new_session_id_starts_with_session_1(self):
        repo = FakeSessionRepository([])

        self.assertEqual(_generate_thread_id(repo), "session-1")

    def test_new_session_id_uses_next_available_short_number(self):
        repo = FakeSessionRepository(["local_geek_master", "session-1", "session-2"])

        self.assertEqual(_generate_thread_id(repo), "session-3")

    def test_legacy_timestamp_session_ids_do_not_force_huge_numbers(self):
        repo = FakeSessionRepository(["session-20260421-195235", "session-1"])

        self.assertEqual(_generate_thread_id(repo), "session-2")

    def test_default_thread_id_prefers_latest_session(self):
        repo = FakeSessionRepository(["session-9", "session-3"])

        self.assertEqual(_resolve_default_thread_id(repo), "session-9")

    def test_default_thread_id_skips_transient_test_sessions(self):
        repo = FakeSessionRepository(["test_slow_route", "session-9", "session-3"])

        self.assertEqual(_resolve_default_thread_id(repo), "session-9")

    def test_default_thread_id_creates_new_short_id_when_repo_empty(self):
        repo = FakeSessionRepository([])

        self.assertEqual(_resolve_default_thread_id(repo), "session-1")


class CliRuntimeThreadSelectionTests(unittest.TestCase):
    def test_runtime_thread_id_prefers_latest_session(self):
        repo = FakeSessionRepository(["session-7", "session-2"])

        self.assertEqual(_resolve_initial_thread_id(repo), "session-7")

    def test_runtime_thread_id_skips_transient_test_sessions(self):
        repo = FakeSessionRepository(["test_planner_multi_step", "session-7", "session-2"])

        self.assertEqual(_resolve_initial_thread_id(repo), "session-7")

    def test_runtime_thread_id_generates_short_id_when_repo_empty(self):
        repo = FakeSessionRepository([])

        self.assertEqual(_resolve_initial_thread_id(repo), "session-1")

    def test_runtime_new_thread_id_uses_next_available_short_number(self):
        repo = FakeSessionRepository(["session-1", "session-3", "local_geek_master"])

        self.assertEqual(_generate_next_runtime_thread_id(repo), "session-2")


class CliPlanRenderTests(unittest.TestCase):
    def test_count_rendered_output_lines_matches_terminal_cursor_progress(self):
        self.assertEqual(_count_rendered_output_lines("hello"), 1)
        self.assertEqual(_count_rendered_output_lines("hello", end=""), 1)
        self.assertEqual(_count_rendered_output_lines("a\nb"), 2)
        self.assertEqual(_count_rendered_output_lines("a\nb", end=""), 2)
        self.assertEqual(_count_rendered_output_lines("", end="\n"), 1)

    def test_render_current_plan_block_prefers_todos(self):
        node_data = {
            "todos": [
                {"id": "step-1", "content": "读取关键文件", "status": "in_progress"},
                {"id": "step-2", "content": "分析模块实现", "status": "pending"},
                {"id": "step-3", "content": "对比设计差异", "status": "pending"},
                {"id": "step-4", "content": "总结修改建议", "status": "pending"},
            ]
        }

        rendered = _render_current_plan_block(node_data, frame="⠋")

        self.assertIn("Current Plan", rendered)
        self.assertIn("⠋", rendered)
        self.assertIn("1. 读取关键文件", rendered)
        self.assertIn("4. 总结修改建议", rendered)

    def test_render_current_plan_block_truncates_long_items_to_avoid_soft_wrap(self):
        node_data = {
            "todos": [
                {
                    "id": "step-1",
                    "content": "/mnt/A/hust_chp/hust_chp/Project/code_ceshi 现在你需要在这个路径下面新建一个python文件 实现给定两个大小分别为 m 和 n 的正序数组中位数 并打印输出结果",
                    "status": "in_progress",
                },
                {
                    "id": "step-2",
                    "content": "运行测试并输出验证结果",
                    "status": "pending",
                },
            ]
        }

        with patch("entry.main._terminal_columns", return_value=48):
            rendered = _render_current_plan_block(node_data, frame="⠋")

        self.assertIn("Current Plan", rendered)
        self.assertIn("…", rendered)
        self.assertNotIn("打印输出结果", rendered)

    def test_render_current_plan_block_falls_back_to_working_memory_todos(self):
        node_data = {
            "messages": [
                AIMessage(
                    content="待审批高风险工具调用：write_office_file",
                    additional_kwargs={"mortyclaw_response_kind": "final_answer"},
                )
            ],
            "working_memory": {
                "todos": [
                    {"content": "检查项目结构并定位入口", "status": "in_progress"},
                    {"content": "实现流式输出", "status": "pending"},
                    {"content": "实现历史保存与加载", "status": "pending"},
                ],
                "pending_approval": True,
            },
        }

        render_state = _effective_plan_render_state(node_data)
        rendered = _render_current_plan_block(node_data, frame="⠋")

        self.assertTrue(render_state.get("pending_approval"))
        self.assertIn("Current Plan", rendered)
        self.assertIn("1. 检查项目结构并定位入口", rendered)
        self.assertIn("↳ waiting approval", rendered)

    def test_normalize_plan_items_falls_back_to_plan_state(self):
        node_data = {
            "plan": [
                {"step": 1, "description": "读取关键文件", "status": "completed"},
                {"step": 2, "description": "分析模块实现", "status": "pending"},
                {"step": 3, "description": "对比设计差异", "status": "pending"},
            ],
            "current_step_index": 1,
            "pending_approval": True,
        }

        items = _normalize_plan_items(node_data)
        rendered = _render_current_plan_block(node_data, frame="⠙")

        self.assertEqual(items[0]["status"], "completed")
        self.assertEqual(items[1]["status"], "in_progress")
        self.assertIn("waiting approval", rendered)
        self.assertIn("分析模块实现", rendered)

    def test_render_stream_node_message_hides_slow_step_result_body(self):
        node_data = {
            "current_step_index": 0,
            "messages": [
                AIMessage(
                    content="分析完成。\n这里是一大段中间分析报告。",
                    additional_kwargs={
                        "mortyclaw_response_kind": "step_result",
                        "mortyclaw_step_outcome": "success_candidate",
                    },
                )
            ],
        }

        rendered = _render_stream_node_message("slow_agent", node_data)

        self.assertIn("步骤 1 结果已生成，正在审查", rendered)
        self.assertNotIn("这里是一大段中间分析报告", rendered)

    def test_render_stream_node_message_prefers_finalizer_final_answer(self):
        node_data = {
            "messages": [
                AIMessage(
                    content="复杂任务已完成。\n这是最终总结。",
                    additional_kwargs={"mortyclaw_response_kind": "final_answer"},
                )
            ]
        }

        rendered = _render_stream_node_message("finalizer", node_data)

        self.assertIn("复杂任务已完成", rendered)
        self.assertIn("这是最终总结", rendered)

    def test_approval_gate_message_renders_below_live_plan(self):
        node_data = {
            "messages": [
                AIMessage(
                    content="待审批高风险工具调用：write_office_file",
                    additional_kwargs={"mortyclaw_response_kind": "final_answer"},
                )
            ]
        }

        self.assertTrue(_should_render_message_below_live_plan("approval_gate", node_data))
        self.assertFalse(_should_render_message_below_live_plan("finalizer", node_data))

    def test_slow_final_answer_renders_below_live_plan_when_plan_exists(self):
        node_data = {
            "messages": [
                AIMessage(
                    content="已改好，验证通过。",
                    additional_kwargs={"mortyclaw_response_kind": "final_answer"},
                )
            ],
            "working_memory": {
                "todos": [
                    {"content": "检查项目结构并定位入口", "status": "completed"},
                    {"content": "实现流式输出与基础 Agent 能力", "status": "completed"},
                    {"content": "实现历史保存与加载", "status": "completed"},
                ],
                "run_status": "done",
            },
        }

        self.assertTrue(_should_render_message_below_live_plan("slow_agent", node_data))

    def test_pre_render_live_plan_when_message_should_stay_below_and_plan_not_yet_drawn(self):
        self.assertTrue(_should_pre_render_live_plan(
            rendered_line_count=0,
            has_live_content=True,
            keep_live_block_above=True,
        ))
        self.assertFalse(_should_pre_render_live_plan(
            rendered_line_count=2,
            has_live_content=True,
            keep_live_block_above=True,
        ))
        self.assertFalse(_should_pre_render_live_plan(
            rendered_line_count=0,
            has_live_content=False,
            keep_live_block_above=True,
        ))


class CliLiveOutputTests(unittest.IsolatedAsyncioTestCase):
    def test_should_suspend_prompt_for_live_output_only_when_app_is_running(self):
        self.assertFalse(_should_suspend_prompt_for_live_output(None))
        self.assertFalse(_should_suspend_prompt_for_live_output(SimpleNamespace(is_running=False)))
        self.assertTrue(_should_suspend_prompt_for_live_output(SimpleNamespace(is_running=True)))

    async def test_run_live_output_operation_uses_run_in_terminal_for_active_prompt(self):
        calls = []

        def operation():
            calls.append("operation")
            return "done"

        async def fake_run_in_terminal(func, render_cli_done=False):
            calls.append(("run_in_terminal", render_cli_done))
            return func()

        result = await _run_live_output_operation(
            operation,
            app_instance=SimpleNamespace(is_running=True),
            run_in_terminal_fn=fake_run_in_terminal,
        )

        self.assertEqual(result, "done")
        self.assertEqual(calls, [("run_in_terminal", False), "operation"])

    async def test_run_live_output_operation_runs_inline_without_prompt(self):
        calls = []

        def operation():
            calls.append("operation")
            return "inline"

        async def fake_run_in_terminal(func, render_cli_done=False):
            calls.append(("run_in_terminal", render_cli_done))
            return func()

        result = await _run_live_output_operation(
            operation,
            app_instance=SimpleNamespace(is_running=False),
            run_in_terminal_fn=fake_run_in_terminal,
        )

        self.assertEqual(result, "inline")
        self.assertEqual(calls, ["operation"])


if __name__ == "__main__":
    unittest.main()
