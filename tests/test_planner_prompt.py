import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.planning import build_llm_planner_prompt


class TestPlannerPrompt(unittest.TestCase):
    def test_planner_prompt_only_keeps_brief_dynamic_context(self):
        prompt = build_llm_planner_prompt(
            query="请修复当前项目里的脚本并运行验证",
            state={
                "goal": "修复 hello.py 并验证输出",
                "replan_reason": "上一次执行 python hello.py 时报 SyntaxError",
                "current_project_path": "/tmp/demo-project",
            },
            route_decision={
                "route": "slow",
                "risk_level": "medium",
                "route_locked": False,
                "route_source": "planner_first_uncertain",
                "route_reason": "complex task",
            },
            current_project_path="/tmp/demo-project",
            available_tool_names=["read_project_file", "write_project_file", "run_project_command"],
            dynamic_context_text=(
                "<context source=\"session-memory\">用户偏好：最小修改</context>\n"
                "<context source=\"workspace-summary\">项目结构复杂，最近讨论过多个模块</context>\n"
                "trust=\"untrusted\" subdir:a active_todos=3"
            ),
        )

        self.assertIn("简要动态上下文", prompt)
        self.assertIn("当前项目路径：/tmp/demo-project", prompt)
        self.assertIn("当前 goal：修复 hello.py 并验证输出", prompt)
        self.assertIn("最近 1 条失败原因：上一次执行 python hello.py 时报 SyntaxError", prompt)
        self.assertNotIn("session-memory", prompt)
        self.assertNotIn("workspace-summary", prompt)
        self.assertNotIn("trust=\"untrusted\"", prompt)
        self.assertNotIn("active_todos=3", prompt)


if __name__ == "__main__":
    unittest.main()
