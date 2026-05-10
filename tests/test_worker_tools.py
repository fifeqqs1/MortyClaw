import json
import os
import sys
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.runtime_context import set_active_thread_id
from mortyclaw.core.tools.builtins import delegate_subagents
from mortyclaw.core.tools.builtins.workers import (
    cancel_subagent_impl,
    cancel_subagents_impl,
    delegate_subagent_impl,
    delegate_subagents_impl,
    list_subagents_impl,
    wait_subagents_impl,
)
from mortyclaw.core.runtime.worker_supervisor import AsyncWorkerSupervisor, _compact_worker_record


class _FakeSupervisor:
    def __init__(self):
        self.submitted = []
        self.wait_calls = []
        self.cancel_calls = []

    def submit_worker(self, **kwargs):
        self.submitted.append(kwargs)
        return {
            "worker_id": "worker-1",
            "worker_thread_id": "thread-worker-1",
            "role": kwargs["role"],
            "status": "pending",
        }

    def submit_workers_batch(self, **kwargs):
        workers = []
        for index, worker_args in enumerate(kwargs["workers"]):
            workers.append({
                "worker_id": f"worker-{index + 1}",
                "worker_thread_id": f"thread-worker-{index + 1}",
                "role": worker_args["role"],
                "status": "pending",
            })
        self.submitted.extend(kwargs["workers"])
        return {"batch_id": kwargs["batch_id"], "workers": workers}

    def resolve_worker_tools(self, **kwargs):
        toolsets = list(kwargs.get("toolsets") or [])
        if "bad_toolset" in toolsets:
            return {
                "requested_toolsets": toolsets,
                "invalid_toolsets": ["bad_toolset"],
                "effective_tools": [],
                "parent_tool_scope": [],
            }
        effective = set(kwargs.get("allowed_tools") or [])
        effective.update({"read_project_file", "search_project_code", "show_git_diff", "update_todo_list"})
        if "project_write" in toolsets:
            effective.update({"edit_project_file", "write_project_file", "apply_project_patch"})
        if "project_verify" in toolsets:
            effective.update({"run_project_tests", "run_project_command"})
        return {
            "requested_toolsets": toolsets,
            "invalid_toolsets": [],
            "effective_tools": sorted(effective),
            "parent_tool_scope": [],
        }

    def list_workers(self, **kwargs):
        return [{"worker_id": "worker-1", "status": "running"}]

    def wait_workers(self, **kwargs):
        self.wait_calls.append(kwargs)
        workers = [
            {
                "worker_id": worker_id,
                "status": "completed",
                "summary": f"{worker_id} done",
                "changed_files": [],
                "commands_run": [],
                "tests_run": [],
                "blocking_issue": "",
                "summary_truncated": False,
            }
            for worker_id in kwargs.get("worker_ids", [])
        ] or [{"worker_id": "worker-1", "status": "completed"}]
        return {
            "success": True,
            "status": "completed",
            "workers": workers,
            "completed_count": len(workers),
            "failed_count": 0,
            "timeout_count": 0,
            "cancelled_count": 0,
            "retry_policy": "do_not_auto_retry",
        }

    def cancel_worker(self, worker_id: str, *, reason: str = ""):
        return {"success": True, "message": f"cancelled {worker_id}"}

    def cancel_workers(self, worker_ids, *, reason: str = ""):
        self.cancel_calls.append({"worker_ids": list(worker_ids), "reason": reason})
        return {
            "success": True,
            "results": [{"success": True, "worker_id": worker_id, "status": "cancelling"} for worker_id in worker_ids],
        }


class WorkerToolTests(unittest.TestCase):
    def setUp(self):
        set_active_thread_id("parent-thread")
        self.supervisor = _FakeSupervisor()

    def test_delegate_subagent_requires_write_scope_for_implement(self):
        payload = json.loads(delegate_subagent_impl(
            task="修改文件",
            role="implement",
            allowed_tools=[],
            write_scope=[],
            context_brief="",
            deliverables="",
            timeout_seconds=30,
            priority=1,
        ))
        self.assertFalse(payload["success"])
        self.assertIn("write_scope", payload["message"])

    def test_delegate_subagent_calls_supervisor(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagent_impl(
                task="分析入口",
                role="explore",
                allowed_tools=["read_project_file"],
                toolsets=None,
                write_scope=[],
                context_brief="背景：只需要分析入口，不要修改文件。",
                deliverables="总结入口",
                timeout_seconds=30,
                priority=2,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["worker_id"], "worker-1")
        self.assertEqual(self.supervisor.submitted[0]["role"], "explore")
        self.assertIn("只需要分析入口", self.supervisor.submitted[0]["context_brief"])
        self.assertIn("read_project_file", payload["effective_tools"])

    def test_delegate_subagents_blocks_and_returns_summaries(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks=[
                    {
                        "task": "分析入口",
                        "role": "explore",
                        "toolsets": ["project_read"],
                        "context_brief": "背景：定位入口链路。",
                        "deliverables": "返回入口文件和调用链。",
                    },
                    {
                        "task": "跑测试",
                        "role": "verify",
                        "toolsets": ["project_verify"],
                        "context_brief": "背景：验证 worker 工具链。",
                        "deliverables": "返回测试命令和结果。",
                    },
                ],
                batch_timeout_seconds=30,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(len(payload["worker_ids"]), 2)
        self.assertEqual(self.supervisor.submitted[1]["role"], "verify")
        self.assertIn("验证 worker 工具链", self.supervisor.submitted[1]["context_brief"])
        self.assertEqual(self.supervisor.wait_calls[0]["worker_ids"], ["worker-1", "worker-2"])
        self.assertEqual(payload["completed_count"], 2)
        self.assertEqual(payload["retry_policy"], "do_not_auto_retry")
        self.assertEqual(payload["debug_only_tools"], ["list_subagents", "wait_subagents"])
        self.assertIn("不要调用 list_subagents/wait_subagents", payload["next_action_hint"])
        self.assertIn("summary", payload["workers"][1])

    def test_delegate_subagents_schema_exposes_task_fields(self):
        schema = delegate_subagents.args_schema.model_json_schema()
        task_def = schema["$defs"]["DelegateSubagentTask"]
        properties = task_def["properties"]

        for field_name in ("task", "role", "toolsets", "allowed_tools", "write_scope", "context_brief", "deliverables"):
            self.assertIn(field_name, properties)
        self.assertEqual(properties["role"]["enum"], ["explore", "verify", "implement"])
        self.assertIn("父任务目标", properties["context_brief"]["description"])
        self.assertIn("交付物", properties["deliverables"]["description"])

    def test_delegate_subagents_requires_context_brief_and_deliverables(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            missing_brief = json.loads(delegate_subagents_impl(
                tasks=[{"task": "分析入口", "role": "explore", "toolsets": ["project_read"], "deliverables": "返回结论。"}],
                batch_timeout_seconds=30,
            ))
            missing_deliverables = json.loads(delegate_subagents_impl(
                tasks=[{"task": "分析入口", "role": "explore", "toolsets": ["project_read"], "context_brief": "背景：定位入口。"}],
                batch_timeout_seconds=30,
            ))

        self.assertFalse(missing_brief["success"])
        self.assertIn("context_brief", missing_brief["message"])
        self.assertFalse(missing_deliverables["success"])
        self.assertIn("deliverables", missing_deliverables["message"])
        self.assertEqual(self.supervisor.submitted, [])

    def test_delegate_subagents_invoke_accepts_structured_schema_payload(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents.invoke({
                "tasks": [
                    {
                        "task": "分析配置模块",
                        "role": "explore",
                        "toolsets": ["project_read"],
                        "context_brief": "父任务是分析三个独立机制；当前 worker 只负责配置加载。",
                        "deliverables": "返回关键文件、调用链和维护风险。",
                    }
                ],
                "batch_timeout_seconds": 30,
            }))

        self.assertTrue(payload["success"])
        self.assertEqual(self.supervisor.submitted[0]["task"], "分析配置模块")
        self.assertIn("配置加载", self.supervisor.submitted[0]["context_brief"])

    def test_delegate_subagents_impl_accepts_json_string_tasks(self):
        tasks = json.dumps([
            {
                "task": "分析配置模块",
                "role": "explore",
                "toolsets": ["project_read"],
                "context_brief": "父任务是分析多个独立机制；当前 worker 只负责配置加载。",
                "deliverables": "返回关键文件、调用链和维护风险。",
            }
        ], ensure_ascii=False)

        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks=tasks,
                batch_timeout_seconds=30,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(self.supervisor.submitted[0]["task"], "分析配置模块")
        self.assertIn("配置加载", self.supervisor.submitted[0]["context_brief"])

    def test_delegate_subagents_impl_rejects_invalid_json_string_tasks(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks='[{"task": "broken"',
                batch_timeout_seconds=30,
            ))

        self.assertFalse(payload["success"])
        self.assertIn("无法解析为 JSON array", payload["message"])
        self.assertEqual(self.supervisor.submitted, [])

    def test_delegate_subagents_rejects_batch_all_or_nothing(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks=[
                    {
                        "task": "分析入口",
                        "role": "explore",
                        "context_brief": "背景：只读分析。",
                        "deliverables": "返回入口。",
                    },
                    {"task": "修改文件", "role": "implement", "write_scope": []},
                ],
                batch_timeout_seconds=30,
            ))

        self.assertFalse(payload["success"])
        self.assertEqual(self.supervisor.submitted, [])

    def test_delegate_subagents_partial_result_counts_statuses(self):
        def partial_wait(**kwargs):
            self.supervisor.wait_calls.append(kwargs)
            return {
                "success": True,
                "status": "partial",
                "workers": [
                    {"worker_id": "worker-1", "status": "completed", "summary": "done"},
                    {"worker_id": "worker-2", "status": "timeout", "summary": "", "blocking_issue": "slow"},
                ],
                "completed_count": 1,
                "failed_count": 0,
                "timeout_count": 1,
                "cancelled_count": 0,
                "retry_policy": "do_not_auto_retry",
            }

        self.supervisor.wait_workers = partial_wait
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks=[
                    {
                        "task": "分析入口",
                        "role": "explore",
                        "context_brief": "背景：只读分析。",
                        "deliverables": "返回入口。",
                    },
                    {
                        "task": "分析配置",
                        "role": "explore",
                        "context_brief": "背景：只读分析。",
                        "deliverables": "返回配置链路。",
                    },
                ],
                batch_timeout_seconds=30,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["completed_count"], 1)
        self.assertEqual(payload["timeout_count"], 1)
        self.assertEqual(payload["retry_policy"], "do_not_auto_retry")
        self.assertIn("目标文件级补查", payload["next_action_hint"])

    def test_delegate_subagents_failed_result_guides_targeted_followup(self):
        def failed_wait(**kwargs):
            return {
                "success": True,
                "status": "completed",
                "workers": [
                    {
                        "worker_id": "worker-1",
                        "status": "failed",
                        "summary": "已定位配置入口，但回写失败。",
                        "blocking_issue": "inbox write failed",
                    }
                ],
                "completed_count": 0,
                "failed_count": 1,
                "timeout_count": 0,
                "cancelled_count": 0,
                "retry_policy": "do_not_auto_retry",
            }

        self.supervisor.wait_workers = failed_wait
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            payload = json.loads(delegate_subagents_impl(
                tasks=[
                    {
                        "task": "分析配置",
                        "role": "explore",
                        "context_brief": "背景：只读分析配置链路。",
                        "deliverables": "返回配置文件和调用链。",
                    }
                ],
                batch_timeout_seconds=30,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["failed_count"], 1)
        self.assertEqual(payload["debug_only_tools"], ["list_subagents", "wait_subagents"])
        self.assertIn("不要调用 list_subagents/wait_subagents", payload["next_action_hint"])
        self.assertIn("不要自动再次委派", payload["next_action_hint"])

    def test_worker_tool_resolution_crops_parent_scope_and_blocks_tools(self):
        supervisor = AsyncWorkerSupervisor(max_workers=1)
        resolution = supervisor.resolve_worker_tools(
            role="implement",
            allowed_tools=["delegate_subagent", "write_project_file"],
            toolsets=["project_full", "research"],
            parent_tool_scope=["read_project_file", "write_project_file", "delegate_subagent"],
        )

        self.assertEqual(resolution["invalid_toolsets"], [])
        self.assertEqual(resolution["effective_tools"], ["read_project_file", "write_project_file"])
        self.assertNotIn("delegate_subagent", resolution["effective_tools"])

    def test_list_wait_cancel_workers(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            listed = json.loads(list_subagents_impl(status_filter="running"))
            waited = json.loads(wait_subagents_impl(worker_ids=["worker-1"], timeout_seconds=1, return_partial=False))
            cancelled = json.loads(cancel_subagent_impl(worker_id="worker-1"))
            cancelled_batch = json.loads(cancel_subagents_impl(worker_ids=["worker-1", "worker-2"], reason="测试取消"))

        self.assertTrue(listed["success"])
        self.assertEqual(listed["workers"][0]["worker_id"], "worker-1")
        self.assertTrue(waited["success"])
        self.assertEqual(waited["workers"][0]["status"], "completed")
        self.assertTrue(cancelled["success"])
        self.assertTrue(cancelled_batch["success"])
        self.assertEqual(cancelled_batch["results"][0]["status"], "cancelling")

    def test_wait_subagents_accepts_json_string_and_single_id(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            json_waited = json.loads(wait_subagents_impl(
                worker_ids='["worker-1", "worker-2"]',
                timeout_seconds=1,
                return_partial=False,
            ))
            single_waited = json.loads(wait_subagents_impl(
                worker_ids="worker-3",
                timeout_seconds=1,
                return_partial=False,
            ))
            quoted_single_waited = json.loads(wait_subagents_impl(
                worker_ids='"worker-4"',
                timeout_seconds=1,
                return_partial=False,
            ))

        self.assertTrue(json_waited["success"])
        self.assertTrue(single_waited["success"])
        self.assertTrue(quoted_single_waited["success"])
        self.assertEqual(self.supervisor.wait_calls[0]["worker_ids"], ["worker-1", "worker-2"])
        self.assertEqual(self.supervisor.wait_calls[1]["worker_ids"], ["worker-3"])
        self.assertEqual(self.supervisor.wait_calls[2]["worker_ids"], ["worker-4"])

    def test_cancel_subagents_accepts_json_string_and_single_id(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            json_cancelled = json.loads(cancel_subagents_impl(
                worker_ids='["worker-1", "worker-2"]',
                reason="测试取消",
            ))
            single_cancelled = json.loads(cancel_subagents_impl(
                worker_ids="worker-3",
                reason="单个取消",
            ))

        self.assertTrue(json_cancelled["success"])
        self.assertTrue(single_cancelled["success"])
        self.assertEqual(self.supervisor.cancel_calls[0]["worker_ids"], ["worker-1", "worker-2"])
        self.assertEqual(self.supervisor.cancel_calls[1]["worker_ids"], ["worker-3"])

    def test_compact_worker_record_omits_large_raw_payloads(self):
        long_summary = "配置模块分析。" * 300
        record = {
            "worker_id": "worker-1",
            "worker_thread_id": "thread-worker-1",
            "parent_thread_id": "parent-thread",
            "role": "explore",
            "goal": "分析配置加载与运行参数。",
            "status": "completed",
            "allowed_tools_json": json.dumps(["read_project_file"] * 30),
            "write_scope_json": "[]",
            "tool_budget": 1,
            "result_summary_json": json.dumps({
                "worker_id": "worker-1",
                "status": "completed",
                "summary": long_summary,
                "changed_files": [f"file-{index}.py" for index in range(20)],
                "commands_run": ["pytest tests/test_config.py"],
                "tests_run": [],
                "blocking_issue": "",
            }, ensure_ascii=False),
            "error_json": "{}",
            "metadata_json": json.dumps({
                "batch_id": "batch-1",
                "task_index": 2,
                "context_brief": "很长背景" * 200,
                "deliverables": "很长交付物" * 200,
                "effective_tools": ["read_project_file", "search_project_code"],
            }, ensure_ascii=False),
            "created_at": "2026-05-10T00:00:00Z",
            "started_at": "2026-05-10T00:00:01Z",
            "finished_at": "2026-05-10T00:00:02Z",
        }

        status_only = _compact_worker_record(record, include_result=False)
        with_result = _compact_worker_record(record, include_result=True)

        self.assertNotIn("result_summary_json", status_only)
        self.assertNotIn("metadata_json", status_only)
        self.assertNotIn("summary", status_only)
        self.assertEqual(status_only["worker_id"], "worker-1")
        self.assertEqual(with_result["summary_chars"], len(long_summary))
        self.assertTrue(with_result["summary_truncated"])
        self.assertLess(len(with_result["summary"]), len(long_summary))
        self.assertLessEqual(len(with_result["changed_files"]), 12)

    def test_completed_worker_inbox_status_is_pending_and_failure_preserves_partial_summary(self):
        inbox_events = []
        updates = []

        class FakeWorkerRepo:
            def __init__(self):
                self.record = {
                    "worker_id": "worker-1",
                    "status": "running",
                    "result_summary_json": json.dumps({
                        "summary": "已有 worker 摘要",
                        "changed_files": ["src/config.py"],
                        "commands_run": [],
                        "tests_run": [],
                        "blocking_issue": "",
                    }, ensure_ascii=False),
                    "error_json": "{}",
                    "metadata_json": "{}",
                }

            def get_worker_run(self, worker_id):
                return dict(self.record)

            def update_worker_run(self, worker_id, **kwargs):
                updates.append(kwargs)
                if kwargs.get("result_summary") is not None:
                    self.record["result_summary_json"] = json.dumps(kwargs["result_summary"], ensure_ascii=False)
                if kwargs.get("error") is not None:
                    self.record["error_json"] = json.dumps(kwargs["error"], ensure_ascii=False)
                if kwargs.get("status") is not None:
                    self.record["status"] = kwargs["status"]
                return dict(self.record)

        class FakeSessionRepo:
            def touch_session(self, *args, **kwargs):
                return None

            def enqueue_inbox_event(self, **kwargs):
                inbox_events.append(kwargs)
                if kwargs.get("status") is None:
                    raise AssertionError("status must not be None")
                return {"event_id": "event-1", **kwargs}

        supervisor = AsyncWorkerSupervisor(max_workers=1)
        with (
            patch("mortyclaw.core.runtime.worker_supervisor.get_worker_run_repository", return_value=FakeWorkerRepo()),
            patch("mortyclaw.core.runtime.worker_supervisor.get_session_repository", return_value=FakeSessionRepo()),
            patch.object(supervisor, "_run_worker_async", return_value={
                "worker_id": "worker-1",
                "status": "completed",
                "summary": "完成摘要",
                "changed_files": [],
                "commands_run": [],
                "tests_run": [],
                "blocking_issue": "",
            }),
        ):
            supervisor._run_worker_with_context(
                "worker-1",
                "parent-thread",
                "worker-thread",
                "explore",
                "task",
                "brief",
                "deliverables",
                ["read_project_file"],
                "provider",
                "model",
                "/project",
                30,
                "batch-1",
                0,
            )

        self.assertEqual(inbox_events[0]["status"], "pending")

        inbox_events.clear()
        updates.clear()
        with (
            patch("mortyclaw.core.runtime.worker_supervisor.get_worker_run_repository", return_value=FakeWorkerRepo()),
            patch("mortyclaw.core.runtime.worker_supervisor.get_session_repository", return_value=FakeSessionRepo()),
            patch.object(supervisor, "_run_worker_async", side_effect=RuntimeError("回写失败")),
        ):
            supervisor._run_worker_with_context(
                "worker-1",
                "parent-thread",
                "worker-thread",
                "explore",
                "task",
                "brief",
                "deliverables",
                ["read_project_file"],
                "provider",
                "model",
                "/project",
                30,
                "batch-1",
                0,
            )

        failed_update = next(item for item in updates if item.get("status") == "failed")
        self.assertIn("已有 worker 摘要", failed_update["result_summary"]["summary"])
        self.assertTrue(failed_update["result_summary"]["partial_result"])
        self.assertIn("回写失败", failed_update["result_summary"]["blocking_issue"])

    def test_explore_worker_uses_direct_loop_provider_and_base_tool(self):
        tool_invocations = []

        @tool
        def read_project_file(filepath: str, project_root: str = "", start_line: int = 1, max_lines: int = 240) -> str:
            """Read a project file through the existing tool wrapper."""
            tool_invocations.append({
                "filepath": filepath,
                "project_root": project_root,
                "start_line": start_line,
                "max_lines": max_lines,
            })
            return "文件：src/config.py\n1: PROVIDER = 'demo'"

        class FakeBoundLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "read_project_file",
                            "args": {"filepath": "src/config.py"},
                            "id": "call-read-config",
                            "type": "tool_call",
                        }],
                    )
                return AIMessage(content=json.dumps({
                    "summary": "配置入口在 src/config.py。",
                    "key_files": ["src/config.py"],
                    "evidence": ["src/config.py: PROVIDER"],
                    "changed_files": [],
                    "commands_run": [],
                    "tests_run": [],
                    "blocking_issue": "",
                    "confidence": "high",
                }, ensure_ascii=False))

        class FakeLLM:
            def __init__(self):
                self.bound_tools = None
                self.bound = FakeBoundLLM()

            def bind_tools(self, tools):
                self.bound_tools = tools
                return self.bound

        fake_llm = FakeLLM()
        supervisor = AsyncWorkerSupervisor(max_workers=1)
        with (
            patch("mortyclaw.core.provider.get_provider", return_value=fake_llm),
            patch.object(supervisor, "_load_toolset", return_value=[read_project_file]),
            patch.object(supervisor, "_run_worker_graph_loop", side_effect=AssertionError("explore must not use graph loop")),
        ):
            result = asyncio_run(supervisor._run_worker_async(
                worker_id="worker-1",
                worker_thread_id="worker-thread",
                role="explore",
                task="分析配置",
                context_brief="父任务：分析配置。",
                deliverables="返回关键文件。",
                allowed_tools=["read_project_file"],
                provider="aliyun",
                model="glm-5",
                project_root="/project",
                timeout_seconds=30,
            ))

        self.assertEqual(result["status"], "completed")
        self.assertIn("src/config.py", result["summary"])
        self.assertEqual(result["key_files"], ["src/config.py"])
        self.assertEqual(tool_invocations[0]["project_root"], "/project")
        self.assertEqual(fake_llm.bound_tools[0].name, "read_project_file")

    def test_implement_worker_uses_graph_loop(self):
        supervisor = AsyncWorkerSupervisor(max_workers=1)

        async def fake_graph_loop(**kwargs):
            return {
                "worker_id": kwargs["worker_id"],
                "status": "completed",
                "summary": "graph path",
                "changed_files": [],
                "commands_run": [],
                "tests_run": [],
                "blocking_issue": "",
            }

        with (
            patch.object(supervisor, "_run_worker_direct_loop", side_effect=AssertionError("implement must not use direct loop")),
            patch.object(supervisor, "_run_worker_graph_loop", side_effect=fake_graph_loop) as graph_mock,
        ):
            result = asyncio_run(supervisor._run_worker_async(
                worker_id="worker-impl",
                worker_thread_id="worker-thread",
                role="implement",
                task="修改配置",
                context_brief="背景",
                deliverables="改文件",
                allowed_tools=["read_project_file"],
                provider="aliyun",
                model="glm-5",
                project_root="/project",
                timeout_seconds=30,
            ))

        self.assertEqual(result["summary"], "graph path")
        self.assertTrue(graph_mock.called)

    def test_direct_loop_budget_exhaustion_returns_partial_summary(self):
        @tool
        def read_project_file(filepath: str, project_root: str = "") -> str:
            """Read a project file through the existing tool wrapper."""
            return f"文件：{filepath}\ncontent"

        class LoopingBoundLLM:
            def invoke(self, messages):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "read_project_file",
                        "args": {"filepath": "src/a.py"},
                        "id": "call-read",
                        "type": "tool_call",
                    }],
                )

        class FakeLLM:
            def bind_tools(self, tools):
                return LoopingBoundLLM()

        supervisor = AsyncWorkerSupervisor(max_workers=1)
        with (
            patch("mortyclaw.core.provider.get_provider", return_value=FakeLLM()),
            patch.object(supervisor, "_load_toolset", return_value=[read_project_file]),
        ):
            result = asyncio_run(supervisor._run_worker_async(
                worker_id="worker-budget",
                worker_thread_id="worker-thread",
                role="explore",
                task="一直读",
                context_brief="背景",
                deliverables="返回已有证据",
                allowed_tools=["read_project_file"],
                provider="aliyun",
                model="glm-5",
                project_root="/project",
                timeout_seconds=30,
            ))

        self.assertTrue(result["budget_exhausted"])
        self.assertIn("src/a.py", result["summary"])


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
