import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.runtime.execution_guard import validate_pending_execution_snapshot
from mortyclaw.core.runtime_context import set_active_thread_id
from mortyclaw.core.tools.builtins import execute_tool_program


class ExecuteToolProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.thread_id = "program-test-thread"
        set_active_thread_id(self.thread_id)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_execute_tool_program_reads_file_and_emits_result(self):
        target = os.path.join(self.root, "demo.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("print('hi')\n")

        result = execute_tool_program.invoke({
            "goal": "读取 demo.py",
            "program": (
                f"content = read_file(filepath='demo.py', project_root={self.root!r})\n"
                "emit_result(content)"
            ),
        })
        payload = json.loads(result)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "completed")
        self.assertIn("文件：demo.py", payload["stdout"])
        self.assertEqual(payload["trace_count"], 1)

    def test_execute_tool_program_pauses_and_resumes_destructive_call(self):
        target = os.path.join(self.root, "demo.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("print('before')\n")

        paused = execute_tool_program.invoke({
            "goal": "改写 demo.py",
            "program": (
                f"write_file(path='demo.py', content='print(1)\\n', project_root={self.root!r})"
            ),
        })
        paused_payload = json.loads(paused)

        self.assertTrue(paused_payload["ok"])
        self.assertEqual(paused_payload["status"], "needs_approval")
        self.assertEqual(paused_payload["pending_execution_snapshot"]["kind"], "tool_program")

        validation = validate_pending_execution_snapshot({
            "goal": "改写 demo.py",
            "pending_execution_snapshot": paused_payload["pending_execution_snapshot"],
        })
        self.assertTrue(validation["ok"])

        resumed = execute_tool_program.invoke(paused_payload["resume_tool_calls"][0]["args"])
        resumed_payload = json.loads(resumed)

        self.assertTrue(resumed_payload["ok"])
        self.assertEqual(resumed_payload["status"], "completed")
        with open(target, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "print(1)\n")

    def test_execute_tool_program_rejects_unsafe_ast(self):
        result = execute_tool_program.invoke({
            "goal": "不要运行导入",
            "program": "import os\nemit_result('x')",
        })
        payload = json.loads(result)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "invalid")
        self.assertIn("程序校验失败", payload["message"])


if __name__ == "__main__":
    unittest.main()
