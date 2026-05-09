import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mortyclaw.core.heartbeat import process_due_tasks_once
from mortyclaw.core.runtime_store import get_session_repository, get_task_repository


class TestRuntimeTaskFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "runtime.sqlite3")
        self.legacy_tasks_path = os.path.join(self.temp_dir.name, "tasks.json")
        self.task_repo = get_task_repository(db_path=self.db_path)
        self.session_repo = get_session_repository(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_import_legacy_tasks_into_sqlite(self):
        legacy_tasks = [
            {
                "id": "legacy-task-1",
                "target_time": "2026-04-20 08:00:00",
                "description": "从旧 JSON 导入的任务",
                "repeat": "daily",
                "repeat_count": 3,
            }
        ]
        with open(self.legacy_tasks_path, "w", encoding="utf-8") as f:
            json.dump(legacy_tasks, f, ensure_ascii=False, indent=2)

        result = self.task_repo.import_legacy_tasks(
            file_path=self.legacy_tasks_path,
            default_thread_id="migrated-thread",
        )

        self.assertEqual(result["imported"], 1)
        imported_tasks = self.task_repo.list_tasks(thread_id="migrated-thread", statuses=("scheduled",))
        self.assertEqual(len(imported_tasks), 1)
        self.assertEqual(imported_tasks[0]["task_id"], "legacy-task-1")
        self.assertEqual(imported_tasks[0]["remaining_runs"], 3)

        with open(self.legacy_tasks_path, "r", encoding="utf-8") as f:
            synced_payload = json.load(f)
        self.assertEqual(synced_payload[0]["thread_id"], "migrated-thread")

    def test_due_task_is_queued_into_session_inbox(self):
        task = self.task_repo.create_task(
            target_time="2026-04-20 09:00:00",
            description="提醒我开会",
            repeat=None,
            repeat_count=None,
            thread_id="thread-alpha",
        )
        self.session_repo.upsert_session(thread_id="thread-alpha", display_name="thread-alpha", status="idle")

        triggered = process_due_tasks_once(
            now="2026-04-20 09:00:01",
            task_repository=self.task_repo,
            session_repository=self.session_repo,
        )

        self.assertEqual(len(triggered), 1)
        updated = self.task_repo.get_task(task["task_id"])
        self.assertEqual(updated["status"], "completed")

        inbox_events = self.session_repo.list_pending_inbox_events("thread-alpha")
        self.assertEqual(len(inbox_events), 1)
        payload = json.loads(inbox_events[0]["payload"])
        self.assertEqual(payload["task_id"], task["task_id"])
        self.assertIn("系统内部心跳触发", payload["content"])

    def test_repeating_task_is_rescheduled_after_trigger(self):
        task = self.task_repo.create_task(
            target_time="2026-04-20 10:00:00",
            description="每天喝水",
            repeat="daily",
            repeat_count=3,
            thread_id="thread-repeat",
        )

        process_due_tasks_once(
            now="2026-04-20 10:00:01",
            task_repository=self.task_repo,
            session_repository=self.session_repo,
        )

        updated = self.task_repo.get_task(task["task_id"])
        self.assertEqual(updated["status"], "scheduled")
        self.assertEqual(updated["remaining_runs"], 2)
        self.assertEqual(updated["target_time"], "2026-04-21 10:00:00")

    def test_runtime_storage_survives_new_repository_instances(self):
        self.task_repo.create_task(
            target_time="2026-04-20 11:00:00",
            description="重启后仍应存在",
            repeat=None,
            repeat_count=None,
            thread_id="thread-restart",
        )
        self.session_repo.enqueue_inbox_event(
            thread_id="thread-restart",
            event_type="heartbeat_task",
            payload={"content": "待恢复的 inbox 事件"},
        )

        fresh_task_repo = get_task_repository(db_path=self.db_path)
        fresh_session_repo = get_session_repository(db_path=self.db_path)

        tasks = fresh_task_repo.list_tasks(thread_id="thread-restart", statuses=("scheduled",))
        inbox_events = fresh_session_repo.list_pending_inbox_events("thread-restart")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["description"], "重启后仍应存在")
        self.assertEqual(len(inbox_events), 1)
        self.assertEqual(json.loads(inbox_events[0]["payload"])["content"], "待恢复的 inbox 事件")

    def test_multi_session_tasks_are_isolated(self):
        self.task_repo.create_task(
            target_time="2026-04-20 12:00:00",
            description="会话 A 的任务",
            repeat=None,
            repeat_count=None,
            thread_id="thread-a",
        )
        self.task_repo.create_task(
            target_time="2026-04-20 13:00:00",
            description="会话 B 的任务",
            repeat=None,
            repeat_count=None,
            thread_id="thread-b",
        )

        tasks_a = self.task_repo.list_tasks(thread_id="thread-a", statuses=("scheduled",))
        tasks_b = self.task_repo.list_tasks(thread_id="thread-b", statuses=("scheduled",))

        self.assertEqual([task["description"] for task in tasks_a], ["会话 A 的任务"])
        self.assertEqual([task["description"] for task in tasks_b], ["会话 B 的任务"])


class TestMonitorSessionSelection(unittest.TestCase):
    def test_monitor_prefers_explicit_thread_id(self):
        import entry.monitor as monitor

        self.assertEqual(
            monitor.resolve_monitor_thread_id("explicit-thread", latest=False),
            "explicit-thread",
        )

    def test_monitor_can_pick_latest_session(self):
        import entry.monitor as monitor

        class FakeRepo:
            def get_latest_session(self):
                return {"thread_id": "latest-thread"}

        with patch("entry.monitor.get_session_repository", return_value=FakeRepo()):
            self.assertEqual(
                monitor.resolve_monitor_thread_id(latest=True),
                "latest-thread",
            )


if __name__ == "__main__":
    unittest.main()
