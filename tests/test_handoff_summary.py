import os
import sys
import unittest

from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.context.handoff import (
    build_discarded_context_payload,
    build_handoff_summary_prompt,
    merge_handoff_summary,
    parse_handoff_summary,
)


class TestHandoffSummary(unittest.TestCase):
    def test_build_handoff_summary_prompt_mentions_persisted_output(self):
        prompt = build_handoff_summary_prompt("", [], state={})
        self.assertIn("persisted-output", prompt)
        self.assertIn("artifact", prompt)

    def test_discarded_context_payload_includes_artifact_metadata(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-1", "name": "run_project_tests", "args": {"command": "pytest"}}],
            ),
            ToolMessage(
                content=(
                    "<persisted-output>\n"
                    "完整输出已保存到：/tmp/test-artifact.txt\n"
                    "预览（前 20 字符）：\nFAILED test_demo\n"
                    "</persisted-output>"
                ),
                tool_call_id="call-1",
                name="run_project_tests",
                additional_kwargs={
                    "mortyclaw_artifact": {
                        "artifact_persisted": True,
                        "artifact_path": "/tmp/test-artifact.txt",
                        "artifact_size": 18000,
                        "preview_chars": 20,
                    }
                },
            ),
        ]

        payload = build_discarded_context_payload(messages)
        tool_events = [event for event in payload if event.get("role") == "tool"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["artifact_path"], "/tmp/test-artifact.txt")
        self.assertTrue(tool_events[0]["artifact_persisted"])

    def test_merge_handoff_summary_keeps_artifact_facts(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call-2", "name": "run_project_command", "args": {"command": "python test.py"}}],
            ),
            ToolMessage(
                content=(
                    "<persisted-output>\n"
                    "完整输出已保存到：/tmp/cmd-artifact.txt\n"
                    "预览（前 18 字符）：\nTraceback: failed\n"
                    "</persisted-output>"
                ),
                tool_call_id="call-2",
                name="run_project_command",
                additional_kwargs={
                    "mortyclaw_artifact": {
                        "artifact_persisted": True,
                        "artifact_path": "/tmp/cmd-artifact.txt",
                        "artifact_size": 26000,
                        "preview_chars": 18,
                    }
                },
            ),
        ]

        summary_text = merge_handoff_summary("", messages, state={})
        summary = parse_handoff_summary(summary_text)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["tool_results"][0]["artifact_path"], "/tmp/cmd-artifact.txt")
        self.assertTrue(summary["tool_results"][0]["artifact_persisted"])
        self.assertTrue(any("/tmp/cmd-artifact.txt" in note for note in summary["context_notes"]))

    def test_merge_handoff_summary_keeps_context_stub_artifact_ref_and_metadata(self):
        stub_payload = {
            "kind": "context_tool_stub",
            "tools": [{
                "tool_call_id": "call-stub",
                "tool_name": "run_project_tests",
                "args_summary": "{\"command\":\"pytest\"}",
                "result_summary": "FAILED tests/test_demo.py",
                "artifact_ref": "ctx_call_stub",
                "artifact_path": "/tmp/context-artifact.txt",
                "status": "stubbed",
                "command": "pytest",
            }],
        }
        messages = [
            AIMessage(
                content="[compacted-tool-interaction]\n{}",
                additional_kwargs={"mortyclaw_context_stub": stub_payload},
            )
        ]

        summary_text = merge_handoff_summary("", messages, state={"goal": "修测试"})
        summary = parse_handoff_summary(summary_text)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["compression_count"], 1)
        self.assertTrue(summary.get("updated_at"))
        self.assertEqual(summary["source_message_range"]["message_count"], 1)
        self.assertEqual(summary["tool_results"][0]["artifact_ref"], "ctx_call_stub")
        self.assertEqual(summary["tool_results"][0]["artifact_path"], "/tmp/context-artifact.txt")
        self.assertTrue(summary["tool_results"][0]["artifact_persisted"])

    def test_merge_handoff_summary_increments_compression_count(self):
        previous = '{"version":1,"compression_count":2,"tool_results":[]}'
        summary_text = merge_handoff_summary(previous, [], state={"goal": "继续"})
        summary = parse_handoff_summary(summary_text)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["compression_count"], 3)
