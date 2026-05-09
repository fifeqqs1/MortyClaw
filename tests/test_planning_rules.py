import importlib.util
import os
import sys
import types
import unittest


def _load_rules_module():
    module_name = "mortyclaw.core.planning.rules_under_test"
    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "mortyclaw",
            "core",
            "planning",
            "rules.py",
        )
    )

    for package_name in ("mortyclaw", "mortyclaw.core", "mortyclaw.core.planning"):
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

    routing_module = types.ModuleType("mortyclaw.core.routing")
    routing_module.SLOW_PATH_HIGH_RISK_HINTS = ("删除", "drop", "remove", "rm -rf")
    routing_module.contains_query_hint = (
        lambda text, lowered, hints: any(hint in text or hint in lowered for hint in hints)
    )
    sys.modules["mortyclaw.core.routing"] = routing_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RULES = _load_rules_module()


class PlanningRulesTests(unittest.TestCase):
    def test_explicit_parallel_keyword_still_delegates(self):
        text = "并行检查 frontend 和 backend 的日志并分别总结问题"
        intent = RULES.infer_step_intent(text)
        self.assertEqual(RULES.infer_execution_mode(text, intent), "delegated")

    def test_independent_workstreams_delegate_without_parallel_keyword(self):
        text = "修复 api 和 ui 两个互不冲突的模块问题"
        intent = RULES.infer_step_intent(text)
        self.assertEqual(intent, "code_edit")
        self.assertEqual(RULES.infer_execution_mode(text, intent), "delegated")

    def test_two_target_analysis_structure_delegates(self):
        text = "分析 frontend 和 backend 的日志并总结主要问题"
        intent = RULES.infer_step_intent(text)
        self.assertEqual(intent, "analyze")
        self.assertEqual(RULES.infer_execution_mode(text, intent), "delegated")

    def test_mechanical_batch_edit_stays_programmatic(self):
        text = "遍历所有命中文件并批量修改后运行测试"
        intent = RULES.infer_step_intent(text)
        self.assertEqual(RULES.infer_execution_mode(text, intent), "programmatic")

    def test_simple_read_and_summary_stays_structured(self):
        text = "读取配置文件并总结当前设置"
        intent = RULES.infer_step_intent(text)
        self.assertEqual(RULES.infer_execution_mode(text, intent), "structured")

    def test_fallback_intent_recognizes_implement_python_tool_as_code_edit(self):
        text = "实现 text_stats.py 命令行工具"
        self.assertEqual(RULES.infer_step_intent(text), "code_edit")

    def test_normalize_intent_prefers_llm_code_edit_for_implementation_task(self):
        text = "实现 text_stats.py 命令行工具"
        self.assertEqual(RULES.normalize_intent("code_edit", text), "code_edit")

    def test_normalize_intent_corrects_obvious_conflict_from_analyze_to_code_edit(self):
        text = "创建 demo.py 并写入示例代码"
        self.assertEqual(RULES.normalize_intent("analyze", text), "code_edit")

    def test_normalize_plan_steps_prefers_llm_execution_mode(self):
        steps = RULES.normalize_plan_steps(
            [
                {
                    "description": "并行检查 frontend 和 backend 的日志并分别总结问题",
                    "intent": "analyze",
                    "execution_mode": "structured",
                }
            ],
            fallback_risk_level="medium",
        )
        self.assertEqual(steps[0]["execution_mode"], "structured")

    def test_normalize_plan_steps_falls_back_when_llm_execution_mode_missing(self):
        steps = RULES.normalize_plan_steps(
            [
                {
                    "description": "分析 frontend 和 backend 的日志并总结主要问题",
                    "intent": "analyze",
                }
            ],
            fallback_risk_level="medium",
        )
        self.assertEqual(steps[0]["execution_mode"], "delegated")

    def test_normalize_plan_steps_corrects_unsafe_mechanical_delegation(self):
        steps = RULES.normalize_plan_steps(
            [
                {
                    "description": "遍历所有命中文件并批量修改后运行测试",
                    "intent": "code_edit",
                    "execution_mode": "delegated",
                }
            ],
            fallback_risk_level="high",
        )
        self.assertEqual(steps[0]["execution_mode"], "programmatic")

    def test_normalize_plan_steps_prefers_llm_intent_when_valid(self):
        steps = RULES.normalize_plan_steps(
            [
                {
                    "description": "实现 text_stats.py 命令行工具",
                    "intent": "code_edit",
                    "execution_mode": "structured",
                }
            ],
            fallback_risk_level="high",
        )
        self.assertEqual(steps[0]["intent"], "code_edit")


if __name__ == "__main__":
    unittest.main()
