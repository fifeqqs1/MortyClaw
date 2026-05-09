import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.approval import build_approval_message, parse_permission_mode_response


class TestApprovalMessage(unittest.TestCase):
    def test_permission_mode_prompt_is_shown_before_slow_execution(self):
        state = {
            "goal": "修改 main.py 并运行测试",
            "permission_mode": "",
            "pending_tool_calls": [],
        }

        rendered = build_approval_message(state)

        self.assertIn("请选择执行权限模式", rendered)
        self.assertIn("`ask`", rendered)
        self.assertIn("`plan`", rendered)
        self.assertIn("`auto`", rendered)

    def test_parse_permission_mode_response_supports_english_and_chinese_aliases(self):
        self.assertEqual(parse_permission_mode_response("ask"), "ask")
        self.assertEqual(parse_permission_mode_response("只读"), "plan")
        self.assertEqual(parse_permission_mode_response("auto"), "auto")

    def test_pending_tool_approval_message_does_not_repeat_checklist(self):
        state = {
            "plan": [
                {"step": 1, "description": "检查项目结构并定位当前 CLI 聊天工具入口与相关代码", "status": "in_progress", "risk_level": "medium"},
                {"step": 2, "description": "实现流式输出与基础 Agent 能力的最小改造", "status": "pending", "risk_level": "high"},
            ],
            "current_step_index": 0,
            "approval_reason": "本轮待执行的高风险工具调用：write_office_file",
            "pending_tool_calls": [
                {"name": "write_office_file", "args": {"filepath": "main.py"}},
            ],
            "todos": [
                {"content": "检查项目结构并定位当前 CLI 聊天工具入口与相关代码", "status": "in_progress"},
                {"content": "实现流式输出与基础 Agent 能力的最小改造", "status": "pending"},
            ],
        }

        rendered = build_approval_message(state)

        self.assertIn("待审批高风险工具调用：write_office_file", rendered)
        self.assertIn("当前步骤 1/2：检查项目结构并定位当前 CLI 聊天工具入口与相关代码", rendered)
        self.assertIn("本轮待执行的高风险工具：write_office_file", rendered)
        self.assertNotIn("当前 Checklist", rendered)

    def test_step_level_approval_message_keeps_step_title_without_pending_tools(self):
        state = {
            "plan": [
                {"step": 1, "description": "修改配置文件", "status": "in_progress", "risk_level": "high"},
            ],
            "current_step_index": 0,
            "approval_reason": "步骤 1 包含高风险操作：修改配置文件",
            "pending_tool_calls": [],
        }

        rendered = build_approval_message(state)

        self.assertIn("待执行步骤 1/1：修改配置文件", rendered)
        self.assertNotIn("待审批高风险工具调用", rendered)

    def test_pending_tool_approval_message_uses_todos_when_plan_missing(self):
        state = {
            "plan": [],
            "approval_reason": "本轮待执行的高风险工具调用：write_office_file",
            "pending_tool_calls": [
                {"name": "write_office_file", "args": {"filepath": "main.py"}},
            ],
            "todos": [
                {"content": "检查项目结构并定位命令行聊天工具入口与当前实现", "status": "in_progress"},
                {"content": "实现流式输出与基础 Agent 能力的最小改造", "status": "pending"},
                {"content": "实现本地对话历史保存与启动时加载", "status": "pending"},
                {"content": "查看 diff 并运行验证测试", "status": "pending"},
            ],
        }

        rendered = build_approval_message(state)

        self.assertIn("待审批高风险工具调用：write_office_file", rendered)
        self.assertIn("当前步骤 1/4：检查项目结构并定位命令行聊天工具入口与当前实现", rendered)
        self.assertNotIn("暂无计划", rendered)


if __name__ == "__main__":
    unittest.main()
