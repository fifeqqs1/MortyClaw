import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.tools.project_tools import (
    apply_project_patch,
    edit_project_file,
    read_project_file,
    run_project_command,
    run_project_tests,
    search_project_code,
    show_git_diff,
    write_project_file,
)
from mortyclaw.core.tools.builtins import BUILTIN_TOOLS
from mortyclaw.core.planning import build_execution_plan, select_tools_for_current_step
from mortyclaw.core.routing import build_route_decision


def _write_file(root: str, relative_path: str, content: str) -> None:
    path = os.path.join(root, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(content).lstrip())


def _file_hash(path: str) -> str:
    import hashlib

    with open(path, "r", encoding="utf-8") as handle:
        return hashlib.sha256(handle.read().encode("utf-8")).hexdigest()


class TestProjectTools(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        _write_file(self.root, "pkg/__init__.py", "")
        _write_file(
            self.root,
            "pkg/train.py",
            """
            import argparse
            from pkg.model import train_model


            class Trainer:
                def fit(self):
                    return train()


            def load_data(path):
                with open(path, encoding="utf-8") as handle:
                    return handle.read()


            def train():
                data = load_data("data.txt")
                return train_model(data)


            def main():
                train()


            if __name__ == "__main__":
                main()
            """,
        )
        _write_file(
            self.root,
            "pkg/model.py",
            """
            def train_model(data):
                return data


            def helper():
                return train_model("x")
            """,
        )
        _write_file(
            self.root,
            "tests/test_basic.py",
            """
            import unittest

            from pkg.model import train_model


            class TestBasic(unittest.TestCase):
                def test_train_model(self):
                    self.assertEqual(train_model("x"), "x")
            """,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_project_tools_registered(self):
        tool_names = {tool.name for tool in BUILTIN_TOOLS}
        self.assertIn("read_project_file", tool_names)
        self.assertIn("search_project_code", tool_names)
        self.assertIn("edit_project_file", tool_names)
        self.assertIn("write_project_file", tool_names)
        self.assertIn("apply_project_patch", tool_names)
        self.assertIn("show_git_diff", tool_names)
        self.assertIn("run_project_tests", tool_names)
        self.assertIn("run_project_command", tool_names)

    def test_code_modification_routes_to_slow_path_and_gets_project_tools(self):
        decision = build_route_decision("请修复这个项目里的训练入口 bug，并运行测试验证")
        self.assertEqual(decision["route"], "slow")
        self.assertEqual(decision["risk_level"], "high")

        plan = build_execution_plan(decision["goal"], decision["risk_level"])
        selected_tools = select_tools_for_current_step(plan[0], BUILTIN_TOOLS)
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("read_project_file", selected_names)
        self.assertIn("search_project_code", selected_names)
        self.assertIn("show_git_diff", selected_names)
        self.assertIn("edit_project_file", selected_names)
        self.assertIn("write_project_file", selected_names)
        self.assertIn("apply_project_patch", selected_names)
        self.assertIn("run_project_tests", selected_names)
        self.assertIn("run_project_command", selected_names)

    def test_read_only_project_analysis_prefers_project_tools(self):
        step = {
            "step": 1,
            "description": "分析这个项目的训练入口和模块关系",
            "status": "pending",
            "risk_level": "low",
            "intent": "analyze",
        }

        selected_tools = select_tools_for_current_step(
            step,
            BUILTIN_TOOLS,
            current_project_path=self.root,
        )
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("read_project_file", selected_names)
        self.assertIn("search_project_code", selected_names)
        self.assertIn("show_git_diff", selected_names)
        self.assertNotIn("read_office_file", selected_names)
        self.assertNotIn("list_office_files", selected_names)

    def test_read_only_analysis_without_project_path_keeps_office_fallback(self):
        step = {
            "step": 1,
            "description": "分析这个项目的训练入口和模块关系",
            "status": "pending",
            "risk_level": "low",
            "intent": "analyze",
        }

        selected_tools = select_tools_for_current_step(step, BUILTIN_TOOLS)
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("read_office_file", selected_names)
        self.assertIn("list_office_files", selected_names)

    def test_file_write_step_with_project_path_exposes_write_project_file(self):
        step = {
            "step": 1,
            "description": "在这个目录下新建一个 Python 文件",
            "status": "pending",
            "risk_level": "high",
            "intent": "file_write",
        }

        selected_tools = select_tools_for_current_step(
            step,
            BUILTIN_TOOLS,
            current_project_path=self.root,
        )
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("write_project_file", selected_names)

    def test_file_write_step_without_project_path_exposes_write_office_file(self):
        step = {
            "step": 1,
            "description": "新建一个 Python 文件并写入示例代码",
            "status": "pending",
            "risk_level": "high",
            "intent": "file_write",
        }

        selected_tools = select_tools_for_current_step(step, BUILTIN_TOOLS)
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("write_office_file", selected_names)

    def test_shell_execute_step_exposes_project_execution_tools(self):
        step = {
            "step": 1,
            "description": "运行 python pkg/model.py 脚本并查看输出",
            "status": "pending",
            "risk_level": "high",
            "intent": "shell_execute",
        }

        selected_tools = select_tools_for_current_step(
            step,
            BUILTIN_TOOLS,
            current_project_path=self.root,
        )
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("run_project_command", selected_names)

    def test_test_verify_step_exposes_run_project_tests(self):
        step = {
            "step": 1,
            "description": "验证输出结果并确认通过或失败原因",
            "status": "pending",
            "risk_level": "high",
            "intent": "test_verify",
        }

        selected_tools = select_tools_for_current_step(
            step,
            BUILTIN_TOOLS,
            current_project_path=self.root,
        )
        selected_names = {tool.name for tool in selected_tools}

        self.assertIn("run_project_tests", selected_names)

    def test_read_project_file_with_line_numbers_and_boundaries(self):
        result = read_project_file.invoke({
            "project_root": self.root,
            "filepath": "pkg/train.py",
            "start_line": 1,
            "max_lines": 6,
        })

        self.assertIn("文件：pkg/train.py", result)
        self.assertIn("1:", result)
        self.assertIn("import argparse", result)

        escaped = read_project_file.invoke({
            "project_root": self.root,
            "filepath": "../outside.py",
        })
        self.assertIn("越权拦截", escaped)

        _write_file(self.root, ".env", "SECRET=1")
        sensitive = read_project_file.invoke({
            "project_root": self.root,
            "filepath": ".env",
        })
        self.assertIn("敏感文件", sensitive)

    def test_project_root_auto_descends_into_single_nested_repo_dir(self):
        with tempfile.TemporaryDirectory() as wrapper_root:
            nested_repo = os.path.join(wrapper_root, "learn-claude-code")
            _write_file(nested_repo, "README.md", "# Learn Claude Code")
            _write_file(nested_repo, "requirements.txt", "pytest\n")
            _write_file(
                nested_repo,
                "agents/s01_agent_loop.py",
                """
                def run():
                    return "ok"
                """,
            )

            readme_result = read_project_file.invoke({
                "project_root": wrapper_root,
                "filepath": "README.md",
            })
            self.assertIn("文件：README.md", readme_result)
            self.assertIn("Learn Claude Code", readme_result)

            search_result = search_project_code.invoke({
                "project_root": wrapper_root,
                "query": "def run",
                "mode": "text",
            })
            self.assertIn("agents/s01_agent_loop.py", search_result)

    def test_project_root_keeps_parent_when_parent_already_looks_like_repo(self):
        with tempfile.TemporaryDirectory() as wrapper_root:
            _write_file(wrapper_root, "README.md", "# Outer Repo")
            _write_file(wrapper_root, "pkg/main.py", "print('outer')\n")
            _write_file(wrapper_root, "nested/README.md", "# Inner Repo")

            readme_result = read_project_file.invoke({
                "project_root": wrapper_root,
                "filepath": "README.md",
            })
            self.assertIn("文件：README.md", readme_result)
            self.assertIn("Outer Repo", readme_result)

    def test_search_project_code_text_symbol_callers_dependencies_data_flow_entrypoints(self):
        text_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "train_model",
            "mode": "text",
        })
        self.assertIn("pkg/train.py", text_result)
        self.assertIn("pkg/model.py", text_result)

        symbol_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "train",
            "mode": "symbol",
        })
        self.assertIn("[function] train", symbol_result)
        self.assertIn("train_model", symbol_result)

        callers_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "train_model",
            "mode": "callers",
        })
        self.assertIn("pkg/train.py", callers_result)
        self.assertIn("scope=train", callers_result)

        dependency_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "pkg/train.py",
            "mode": "dependencies",
        })
        self.assertIn("模块依赖分析", dependency_result)
        self.assertIn("pkg.model", dependency_result)

        data_flow_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "pkg/train.py",
            "mode": "data_flow",
        })
        self.assertIn("load_data", data_flow_result)
        self.assertIn("训练/推理相关线索", data_flow_result)

        entrypoint_result = search_project_code.invoke({
            "project_root": self.root,
            "query": "",
            "mode": "entrypoints",
        })
        self.assertIn("pkg/train.py", entrypoint_result)
        self.assertIn("__main__", entrypoint_result)

    def test_search_project_code_index_refreshes_incrementally_and_can_fallback(self):
        initial = search_project_code.invoke({
            "project_root": self.root,
            "query": "new_helper",
            "mode": "symbol",
        })
        self.assertIn("未找到", initial)

        _write_file(
            self.root,
            "pkg/extra.py",
            """
            def new_helper():
                return "ok"
            """,
        )
        refreshed = search_project_code.invoke({
            "project_root": self.root,
            "query": "new_helper",
            "mode": "symbol",
        })
        self.assertIn("pkg/extra.py", refreshed)
        self.assertIn("new_helper", refreshed)

        os.remove(os.path.join(self.root, "pkg", "extra.py"))
        removed = search_project_code.invoke({
            "project_root": self.root,
            "query": "new_helper",
            "mode": "symbol",
        })
        self.assertIn("未找到", removed)

        fallback = search_project_code.invoke({
            "project_root": self.root,
            "query": "train_model",
            "mode": "symbol",
            "use_index": False,
        })
        self.assertIn("pkg/model.py", fallback)
        self.assertIn("train_model", fallback)

    def test_search_project_code_callers_uses_definition_then_import_chain_disambiguation(self):
        _write_file(
            self.root,
            "pkg/a.py",
            """
            def process(data):
                return data


            def helper():
                return process("a")
            """,
        )
        _write_file(
            self.root,
            "pkg/b.py",
            """
            def process(data):
                return data.upper()
            """,
        )
        _write_file(
            self.root,
            "pkg/consumer.py",
            """
            from pkg.a import process


            def run():
                return process("x")
            """,
        )
        _write_file(
            self.root,
            "pkg/other_consumer.py",
            """
            from pkg.b import process as process_b


            def run():
                return process_b("x")
            """,
        )
        _write_file(
            self.root,
            "pkg/local_only.py",
            """
            def process(data):
                return data[::-1]


            def run():
                return process("local")
            """,
        )

        indexed = search_project_code.invoke({
            "project_root": self.root,
            "query": "process",
            "mode": "callers",
        })
        self.assertIn("定义候选：pkg/a.py:1 [function] process", indexed)
        self.assertIn("pkg/consumer.py", indexed)
        self.assertIn("via=from pkg.a import process", indexed)
        self.assertIn("定义候选：pkg/b.py:1 [function] process", indexed)
        self.assertIn("pkg/other_consumer.py:5 scope=run call=process_b via=from pkg.b import process", indexed)

        fallback = search_project_code.invoke({
            "project_root": self.root,
            "query": "process",
            "mode": "callers",
            "use_index": False,
        })
        self.assertIn("定义候选：pkg/a.py:1 [function] process", fallback)
        self.assertIn("pkg/consumer.py", fallback)
        self.assertIn("via=from pkg.a import process", fallback)
        self.assertIn("定义候选：pkg/b.py:1 [function] process", fallback)
        self.assertIn("pkg/other_consumer.py:5 scope=run call=process_b via=from pkg.b import process", fallback)

    @unittest.skipIf(shutil.which("git") is None, "git command is not available")
    def test_apply_project_patch_and_show_git_diff(self):
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=self.root, check=True, capture_output=True)

        patch = """\
diff --git a/pkg/model.py b/pkg/model.py
--- a/pkg/model.py
+++ b/pkg/model.py
@@ -1,5 +1,5 @@
 def train_model(data):
-    return data
+    return data.upper()
 
 
 def helper():
"""
        result = apply_project_patch.invoke({
            "project_root": self.root,
            "patch": patch,
            "reason": "测试 patch 级修改",
        })
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "patch 已成功应用。")

        diff = show_git_diff.invoke({"project_root": self.root, "pathspec": "pkg/model.py"})
        self.assertIn("return data.upper()", diff)

        with open(os.path.join(self.root, "pkg/model.py"), encoding="utf-8") as handle:
            self.assertIn("return data.upper()", handle.read())

    def test_write_and_edit_project_file_with_version_guards(self):
        initial_hash = _file_hash(os.path.join(self.root, "pkg", "model.py"))

        edit_result = edit_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/model.py",
            "expected_hash": initial_hash,
            "edits": [{"old_text": "return data", "new_text": "return data.lower()"}],
        })
        edit_payload = json.loads(edit_result)
        self.assertTrue(edit_payload["ok"])

        stale_result = write_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/model.py",
            "expected_hash": initial_hash,
            "content": "print('stale')\n",
        })
        stale_payload = json.loads(stale_result)
        self.assertFalse(stale_payload["ok"])
        self.assertEqual(stale_payload["error_kind"], "FILE_CHANGED_SINCE_READ")
        self.assertIn("current_excerpt", stale_payload)

        write_result = write_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/new_file.py",
            "content": "print('new')\n",
            "create_if_missing": True,
        })
        write_payload = json.loads(write_result)
        self.assertTrue(write_payload["ok"])
        self.assertTrue(os.path.exists(os.path.join(self.root, "pkg", "new_file.py")))

    def test_write_project_file_missing_target_suggests_create_if_missing(self):
        result = write_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/brand_new.py",
            "content": "print('x')\n",
        })
        payload = json.loads(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_kind"], "FILE_NOT_FOUND")
        self.assertIn("create_if_missing", payload["message"])

    def test_edit_project_file_reports_not_found_and_ambiguous(self):
        not_found = edit_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/model.py",
            "edits": [{"old_text": "does_not_exist", "new_text": "x"}],
        })
        not_found_payload = json.loads(not_found)
        self.assertFalse(not_found_payload["ok"])
        self.assertEqual(not_found_payload["error_kind"], "OLD_TEXT_NOT_FOUND")

        _write_file(self.root, "pkg/dup.py", "x = 1\nx = 1\n")
        ambiguous = edit_project_file.invoke({
            "project_root": self.root,
            "path": "pkg/dup.py",
            "edits": [{"old_text": "x = 1", "new_text": "x = 2"}],
        })
        ambiguous_payload = json.loads(ambiguous)
        self.assertFalse(ambiguous_payload["ok"])
        self.assertEqual(ambiguous_payload["error_kind"], "OLD_TEXT_AMBIGUOUS")

    def test_apply_project_patch_returns_typed_error_on_bad_patch(self):
        bad_patch = "*** Begin Patch\n*** Update File: pkg/model.py\n@@\n*** End Patch\n"
        result = apply_project_patch.invoke({
            "project_root": self.root,
            "patch": bad_patch,
        })
        payload = json.loads(result)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_kind"], "PATCH_PARSE_FAILED")

    def test_run_project_tests_default_and_rejects_dangerous_command(self):
        result = run_project_tests.invoke({
            "project_root": self.root,
            "command": "",
            "timeout_seconds": 60,
        })
        payload = json.loads(result)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["message"], "验证通过。")

        rejected = run_project_tests.invoke({
            "project_root": self.root,
            "command": "rm -rf .",
        })
        rejected_payload = json.loads(rejected)
        self.assertFalse(rejected_payload["ok"])
        self.assertEqual(rejected_payload["error_kind"], "COMMAND_BLOCKED")

    def test_run_project_command_allows_whitelist_and_blocks_shell(self):
        allowed = run_project_command.invoke({
            "project_root": self.root,
            "command": f"{sys.executable} -m py_compile pkg/model.py",
        })
        allowed_payload = json.loads(allowed)
        self.assertTrue(allowed_payload["ok"])

        script_allowed = run_project_command.invoke({
            "project_root": self.root,
            "command": f"{sys.executable} pkg/model.py",
        })
        script_allowed_payload = json.loads(script_allowed)
        self.assertTrue(script_allowed_payload["ok"])

        blocked = run_project_command.invoke({
            "project_root": self.root,
            "command": "python -m py_compile pkg/model.py && echo ok",
        })
        blocked_payload = json.loads(blocked)
        self.assertFalse(blocked_payload["ok"])
        self.assertEqual(blocked_payload["error_kind"], "COMMAND_BLOCKED")

        blocked_script_flag = run_project_command.invoke({
            "project_root": self.root,
            "command": "python pkg/model.py --unsafe",
        })
        blocked_script_flag_payload = json.loads(blocked_script_flag)
        self.assertFalse(blocked_script_flag_payload["ok"])
        self.assertEqual(blocked_script_flag_payload["error_kind"], "COMMAND_BLOCKED")


if __name__ == "__main__":
    unittest.main()
