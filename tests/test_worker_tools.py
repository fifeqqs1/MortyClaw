import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.runtime_context import set_active_thread_id
from mortyclaw.core.tools.builtins.workers import (
    cancel_subagent_impl,
    delegate_subagent_impl,
    list_subagents_impl,
    wait_subagents_impl,
)


class _FakeSupervisor:
    def __init__(self):
        self.submitted = []

    def submit_worker(self, **kwargs):
        self.submitted.append(kwargs)
        return {
            "worker_id": "worker-1",
            "worker_thread_id": "thread-worker-1",
            "role": kwargs["role"],
            "status": "pending",
        }

    def list_workers(self, **kwargs):
        return [{"worker_id": "worker-1", "status": "running"}]

    def wait_workers(self, **kwargs):
        return {
            "success": True,
            "status": "completed",
            "workers": [{"worker_id": "worker-1", "status": "completed"}],
        }

    def cancel_worker(self, worker_id: str):
        return {"success": True, "message": f"cancelled {worker_id}"}


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
                write_scope=[],
                deliverables="总结入口",
                timeout_seconds=30,
                priority=2,
            ))

        self.assertTrue(payload["success"])
        self.assertEqual(payload["worker_id"], "worker-1")
        self.assertEqual(self.supervisor.submitted[0]["role"], "explore")

    def test_list_wait_cancel_workers(self):
        with patch("mortyclaw.core.tools.builtins.workers.get_worker_supervisor", return_value=self.supervisor):
            listed = json.loads(list_subagents_impl(status_filter="running"))
            waited = json.loads(wait_subagents_impl(worker_ids=["worker-1"], timeout_seconds=1, return_partial=False))
            cancelled = json.loads(cancel_subagent_impl(worker_id="worker-1"))

        self.assertTrue(listed["success"])
        self.assertEqual(listed["workers"][0]["worker_id"], "worker-1")
        self.assertTrue(waited["success"])
        self.assertEqual(waited["workers"][0]["status"], "completed")
        self.assertTrue(cancelled["success"])


if __name__ == "__main__":
    unittest.main()
