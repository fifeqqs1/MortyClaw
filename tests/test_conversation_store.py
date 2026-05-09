import os
import sqlite3
import tempfile
import time
import unittest
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mortyclaw.core.runtime_store import (
    get_conversation_repository,
    get_conversation_writer,
    get_session_repository,
)


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "runtime.sqlite3")
        self.repo = get_conversation_repository(db_path=self.db_path)
        self.session_repo = get_session_repository(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_migrates_old_sessions_table_without_losing_rows(self):
        legacy_db = os.path.join(self.temp_dir.name, "legacy.sqlite3")
        with sqlite3.connect(legacy_db) as conn:
            conn.execute(
                """
                CREATE TABLE sessions (
                    thread_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'idle',
                    log_file TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                INSERT INTO sessions (
                    thread_id, display_name, provider, model, status, log_file,
                    created_at, updated_at, last_active_at, metadata_json
                ) VALUES ('old-thread', 'old-thread', '', '', 'idle', '', 'now', 'now', 'now', '{}')
                """
            )
            conn.commit()

        migrated_repo = get_session_repository(db_path=legacy_db)
        session = migrated_repo.get_session("old-thread")

        self.assertEqual(session["thread_id"], "old-thread")
        self.assertIn("message_count", session)
        self.assertIn("lineage_root_thread_id", session)

    def test_append_messages_is_deduplicated_and_recent_mode_is_db_only(self):
        user_message = HumanMessage(content="上次我们用 pytest 验证工具结果", id="msg-user-1")
        self.repo.append_messages(
            thread_id="thread-a",
            turn_id="turn-1",
            messages=[user_message],
            node_name="user_input",
        )
        self.repo.append_messages(
            thread_id="thread-a",
            turn_id="turn-1",
            messages=[user_message],
            node_name="user_input",
        )

        messages = self.repo.get_session_conversation("thread-a")
        recent = self.repo.search_sessions("", limit=3)

        self.assertEqual(len(messages), 1)
        self.assertEqual(recent[0]["thread_id"], "thread-a")
        self.assertEqual(recent[0]["message_count"], 1)

    def test_tool_calls_are_linked_to_tool_results_and_searchable(self):
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "id": "call-tests-1",
                "name": "run_project_tests",
                "args": {"command": "pytest tests/test_runtime.py"},
            }],
            id="msg-ai-1",
        )
        tool_message = ToolMessage(
            content="pytest tests/test_runtime.py passed，工具结果显示全部通过",
            tool_call_id="call-tests-1",
            name="run_project_tests",
            id="msg-tool-1",
        )

        self.repo.append_messages(
            thread_id="thread-tools",
            turn_id="turn-tools",
            messages=[ai_message, tool_message],
            node_name="fast_tools",
        )

        results = self.repo.search_sessions("pytest 工具结果", limit=3)

        self.assertEqual(results[0]["thread_id"], "thread-tools")
        hit_text = " ".join(hit["content_preview"] for hit in results[0]["hits"])
        self.assertIn("pytest", hit_text)
        self.assertTrue(any(hit["tool_result_preview"] for hit in results[0]["hits"]))

    def test_search_excludes_current_session_by_default(self):
        self.repo.append_messages(
            thread_id="current-thread",
            turn_id="turn-current",
            messages=[HumanMessage(content="我们修过 session_search 的 FTS", id="msg-current")],
        )
        self.repo.append_messages(
            thread_id="other-thread",
            turn_id="turn-other",
            messages=[HumanMessage(content="以前 session_search 的 FTS 是这样做的", id="msg-other")],
        )

        results = self.repo.search_sessions(
            "session_search FTS",
            limit=3,
            current_thread_id="current-thread",
        )

        self.assertEqual([item["thread_id"] for item in results], ["other-thread"])

    def test_branch_metadata_and_compression_summary_are_persisted(self):
        self.session_repo.upsert_session(thread_id="parent-thread", display_name="parent-thread")
        branch = self.session_repo.create_branch_session(
            parent_thread_id="parent-thread",
            branch_thread_id="branch-thread",
            branch_from_message_uid="msg-parent-1",
        )
        summary = self.repo.record_conversation_summary(
            thread_id="branch-thread",
            summary="目标：继续父会话。已完成：建立分支。风险：不要复制 checkpoint。",
            messages=[HumanMessage(content="父会话上下文", id="msg-parent-1")],
        )

        self.assertEqual(branch["parent_thread_id"], "parent-thread")
        self.assertEqual(branch["branch_from_message_uid"], "msg-parent-1")
        self.assertEqual(summary["thread_id"], "branch-thread")

    def test_async_writer_enqueue_stays_lightweight(self):
        writer = get_conversation_writer(db_path=self.db_path)
        start = time.perf_counter()
        writer.append_messages(
            thread_id="async-thread",
            turn_id="turn-async",
            messages=[HumanMessage(content="普通对话只应该快速入队", id="msg-async")],
        )
        elapsed = time.perf_counter() - start
        writer.flush()

        self.assertLess(elapsed, 0.05)
        self.assertEqual(len(self.repo.get_session_conversation("async-thread")), 1)

    def test_session_todo_state_can_be_saved_loaded_and_cleared(self):
        self.session_repo.upsert_session(thread_id="todo-thread", display_name="todo-thread")
        saved = self.session_repo.save_session_todo_state(
            "todo-thread",
            {
                "items": [{"id": "step-1", "content": "检查入口", "status": "in_progress"}],
                "revision": 2,
                "updated_at": "now",
                "last_event": "planned",
            },
        )

        loaded = self.session_repo.get_session_todo_state("todo-thread")
        cleared = self.session_repo.clear_session_todo_state("todo-thread")

        self.assertEqual(saved["thread_id"], "todo-thread")
        self.assertEqual(loaded["revision"], 2)
        self.assertEqual(loaded["items"][0]["content"], "检查入口")
        self.assertEqual(self.session_repo.get_session_todo_state("todo-thread"), {})
        self.assertEqual(cleared["thread_id"], "todo-thread")

    def test_tool_result_metadata_keeps_artifact_reference(self):
        ai_message = AIMessage(
            content="",
            tool_calls=[{
                "id": "call-artifact-1",
                "name": "run_project_tests",
                "args": {"command": "pytest -q"},
            }],
            id="msg-ai-artifact",
        )
        tool_message = ToolMessage(
            content="<persisted-output>\npreview\n</persisted-output>",
            tool_call_id="call-artifact-1",
            name="run_project_tests",
            id="msg-tool-artifact",
            additional_kwargs={
                "mortyclaw_artifact": {
                    "artifact_persisted": True,
                    "artifact_path": "/tmp/runtime/artifacts/thread/turn/call-artifact-1.txt",
                    "artifact_size": 12000,
                    "preview_chars": 1200,
                }
            },
        )

        self.repo.append_messages(
            thread_id="artifact-thread",
            turn_id="turn-artifact",
            messages=[ai_message, tool_message],
            node_name="slow_tools",
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT metadata_json FROM conversation_tool_calls WHERE tool_call_id = ?",
                ("call-artifact-1",),
            ).fetchone()

        metadata = json.loads(row["metadata_json"])
        self.assertTrue(metadata["artifact"]["artifact_persisted"])
        self.assertIn("call-artifact-1.txt", metadata["artifact"]["artifact_path"])


if __name__ == "__main__":
    unittest.main()
