import asyncio
import os
import sqlite3
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from mortyclaw.core.code_index import _connect as connect_code_index
from mortyclaw.core.maintenance import (
    collect_doctor_report,
    gc_logs,
    gc_runtime,
    gc_state,
)
from mortyclaw.core.memory import MemoryStore, build_memory_record
from mortyclaw.core.runtime_store import get_session_repository, get_task_repository


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_db_path = os.path.join(self.temp_dir.name, "state.sqlite3")
        self.runtime_db_path = os.path.join(self.temp_dir.name, "runtime.sqlite3")
        self.memory_db_path = os.path.join(self.temp_dir.name, "memory.sqlite3")
        self.code_index_db_path = os.path.join(self.temp_dir.name, "code_index.sqlite3")
        self.logs_dir = os.path.join(self.temp_dir.name, "logs")
        self.logs_archive_dir = os.path.join(self.temp_dir.name, "logs_archive")
        self.backups_dir = os.path.join(self.temp_dir.name, "backups")
        os.makedirs(self.logs_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def _seed_state_db(self):
        async with AsyncSqliteSaver.from_conn_string(self.state_db_path) as saver:
            for thread_id, total in (("thread-a", 5), ("thread-b", 2)):
                config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                for index in range(total):
                    checkpoint = empty_checkpoint()
                    checkpoint["id"] = f"{thread_id}-{index:04d}"
                    checkpoint["channel_values"] = {"messages": [f"{thread_id}-message-{index}"]}
                    config = await saver.aput(config, checkpoint, {"step": index}, {})
                    await saver.aput_writes(
                        config,
                        [("messages", f"{thread_id}-write-{index}")],
                        task_id=f"{thread_id}-task-{index}",
                    )

    def test_collect_doctor_report_summarizes_databases_and_logs(self):
        asyncio.run(self._seed_state_db())

        task_repo = get_task_repository(db_path=self.runtime_db_path)
        session_repo = get_session_repository(db_path=self.runtime_db_path)
        task_repo.create_task(
            target_time="2026-04-20 12:00:00",
            description="doctor 测试任务",
            repeat=None,
            repeat_count=None,
            thread_id="doctor-thread",
        )
        session_repo.upsert_session(thread_id="doctor-thread", display_name="doctor-thread")

        memory_store = MemoryStore(db_path=self.memory_db_path)
        memory_store.upsert_memory(
            build_memory_record(
                layer="long_term",
                scope="user_default",
                type="user_preference",
                subject="response_language",
                content="以后用中文回答",
                source_kind="test",
            )
        )

        with connect_code_index(self.code_index_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO files (
                    project_root, file_path, mtime_ns, size, indexed_at, status,
                    error, entrypoint_score, entrypoint_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("/tmp/project", "main.py", 1, 10, 1.0, "ok", "", 0, "[]"),
            )
            conn.commit()

        with open(os.path.join(self.logs_dir, "doctor-thread.jsonl"), "w", encoding="utf-8") as handle:
            handle.write('{"event":"ai_message"}\n')

        report = collect_doctor_report(
            state_db_path=self.state_db_path,
            runtime_db_path=self.runtime_db_path,
            memory_db_path=self.memory_db_path,
            code_index_db_path=self.code_index_db_path,
            logs_dir=self.logs_dir,
        )

        self.assertEqual(report["databases"]["state"]["table_counts"]["checkpoints"], 7)
        self.assertEqual(report["databases"]["runtime"]["table_counts"]["sessions"], 1)
        self.assertEqual(report["databases"]["runtime"]["table_counts"]["tasks"], 1)
        self.assertEqual(report["databases"]["memory"]["table_counts"]["memory_records"], 1)
        self.assertEqual(report["databases"]["code_index"]["table_counts"]["files"], 1)
        self.assertEqual(report["logs"]["file_count"], 1)

    def test_gc_logs_archives_old_or_oversized_logs(self):
        old_log = os.path.join(self.logs_dir, "old.jsonl")
        large_log = os.path.join(self.logs_dir, "large.jsonl")
        fresh_log = os.path.join(self.logs_dir, "fresh.jsonl")

        with open(old_log, "w", encoding="utf-8") as handle:
            handle.write("old\n")
        with open(large_log, "w", encoding="utf-8") as handle:
            handle.write("x" * 300)
        with open(fresh_log, "w", encoding="utf-8") as handle:
            handle.write("fresh\n")

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
        os.utime(old_log, (old_ts, old_ts))

        dry_run = gc_logs(
            log_dir=self.logs_dir,
            archive_dir=self.logs_archive_dir,
            max_age_days=14,
            max_size_bytes=128,
            apply=False,
        )
        self.assertEqual(dry_run["candidate_count"], 2)
        self.assertTrue(os.path.exists(old_log))
        self.assertTrue(os.path.exists(large_log))

        applied = gc_logs(
            log_dir=self.logs_dir,
            archive_dir=self.logs_archive_dir,
            max_age_days=14,
            max_size_bytes=128,
            apply=True,
        )
        self.assertEqual(applied["archived_count"], 2)
        self.assertTrue(os.path.exists(applied["archive_path"]))
        self.assertFalse(os.path.exists(old_log))
        self.assertFalse(os.path.exists(large_log))
        self.assertTrue(os.path.exists(fresh_log))

        with tarfile.open(applied["archive_path"], "r:gz") as archive:
            self.assertEqual(sorted(archive.getnames()), ["large.jsonl", "old.jsonl"])

    def test_gc_runtime_prunes_old_delivered_inbox_and_old_task_runs(self):
        task_repo = get_task_repository(db_path=self.runtime_db_path)
        session_repo = get_session_repository(db_path=self.runtime_db_path)

        task = task_repo.create_task(
            target_time="2026-04-20 12:00:00",
            description="runtime gc 测试",
            repeat=None,
            repeat_count=None,
            thread_id="runtime-thread",
        )

        old_event = session_repo.enqueue_inbox_event(
            thread_id="runtime-thread",
            event_type="heartbeat_task",
            payload={"content": "old delivered"},
        )
        session_repo.mark_inbox_event_delivered(old_event["event_id"])
        session_repo.enqueue_inbox_event(
            thread_id="runtime-thread",
            event_type="heartbeat_task",
            payload={"content": "pending should stay"},
        )

        now_dt = datetime(2026, 4, 24, tzinfo=timezone.utc)
        old_delivered_at = (now_dt - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with sqlite3.connect(self.runtime_db_path) as conn:
            conn.execute(
                "UPDATE session_inbox SET delivered_at = ? WHERE event_id = ?",
                (old_delivered_at, old_event["event_id"]),
            )
            conn.commit()

        for index in range(25):
            triggered_at = (now_dt - timedelta(days=40 + index)).strftime("%Y-%m-%dT%H:%M:%SZ")
            task_repo.record_task_run(
                task_id=task["task_id"],
                thread_id="runtime-thread",
                status="completed",
                triggered_at=triggered_at,
                finished_at=triggered_at,
                result_summary=f"run-{index}",
            )

        report = gc_runtime(
            runtime_db_path=self.runtime_db_path,
            inbox_retention_days=7,
            task_run_retention_days=30,
            keep_recent_task_runs=20,
            apply=False,
            now=now_dt,
        )
        self.assertEqual(report["inbox"]["candidate_count"], 1)
        self.assertEqual(report["task_runs"]["candidate_count"], 5)

        applied = gc_runtime(
            runtime_db_path=self.runtime_db_path,
            inbox_retention_days=7,
            task_run_retention_days=30,
            keep_recent_task_runs=20,
            apply=True,
            now=now_dt,
        )
        self.assertEqual(applied["inbox"]["deleted_count"], 1)
        self.assertEqual(applied["task_runs"]["deleted_count"], 5)

        with sqlite3.connect(self.runtime_db_path) as conn:
            inbox_count = conn.execute("SELECT COUNT(*) FROM session_inbox").fetchone()[0]
            task_run_count = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
        self.assertEqual(inbox_count, 1)
        self.assertEqual(task_run_count, 20)

    def test_gc_state_keeps_latest_checkpoints_and_backup_allows_resume(self):
        asyncio.run(self._seed_state_db())

        dry_run = gc_state(
            state_db_path=self.state_db_path,
            backup_dir=self.backups_dir,
            keep_latest_per_thread=2,
            apply=False,
        )
        self.assertEqual(dry_run["checkpoint_candidate_count"], 3)
        self.assertEqual(dry_run["write_candidate_count"], 3)

        applied = gc_state(
            state_db_path=self.state_db_path,
            backup_dir=self.backups_dir,
            keep_latest_per_thread=2,
            apply=True,
        )
        self.assertEqual(applied["deleted_checkpoints"], 3)
        self.assertEqual(applied["deleted_writes"], 3)
        self.assertTrue(os.path.exists(applied["backup_path"]))

        with sqlite3.connect(self.state_db_path) as conn:
            checkpoint_count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            write_count = conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0]
        self.assertEqual(checkpoint_count, 4)
        self.assertEqual(write_count, 4)

        async def verify_latest():
            async with AsyncSqliteSaver.from_conn_string(self.state_db_path) as saver:
                latest = await saver.aget_tuple({"configurable": {"thread_id": "thread-a", "checkpoint_ns": ""}})
                return latest

        latest = asyncio.run(verify_latest())
        self.assertIsNotNone(latest)
        self.assertEqual(latest.config["configurable"]["checkpoint_id"], "thread-a-0004")
        self.assertEqual(latest.pending_writes[0][2], "thread-a-write-4")


if __name__ == "__main__":
    unittest.main()
