from __future__ import annotations

import os
import sqlite3
import threading

from ..config import RUNTIME_DB_PATH


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class RuntimeStore:
    def __init__(self, db_path: str = RUNTIME_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._lock:
            with self._connect() as conn:
                try:
                    conn.execute("PRAGMA journal_mode = WAL")
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
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
                self._ensure_column(conn, "sessions", "title", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "parent_thread_id", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "branch_from_message_uid", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "lineage_root_thread_id", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "sessions", "message_count", "INTEGER NOT NULL DEFAULT 0")
                self._ensure_column(conn, "sessions", "tool_call_count", "INTEGER NOT NULL DEFAULT 0")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        description TEXT NOT NULL,
                        target_time TEXT NOT NULL,
                        repeat TEXT,
                        repeat_count INTEGER,
                        remaining_runs INTEGER,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_run_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS task_runs (
                        run_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        triggered_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        result_summary TEXT NOT NULL DEFAULT '',
                        error_message TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_inbox (
                        event_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        delivered_at TEXT
                    )
                    """
                )
                self._ensure_column(conn, "session_inbox", "retry_count", "INTEGER NOT NULL DEFAULT 0")
                self._ensure_column(conn, "session_inbox", "error_message", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "session_inbox", "leased_by", "TEXT NOT NULL DEFAULT ''")
                self._ensure_column(conn, "session_inbox", "lease_expires_at", "TEXT")
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_thread_status_target
                    ON tasks(thread_id, status, target_time)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tasks_status_target
                    ON tasks(status, target_time)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_last_active
                    ON sessions(last_active_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_session_inbox_thread_status_created
                    ON session_inbox(thread_id, status, created_at)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_task_runs_task_triggered
                    ON task_runs(task_id, triggered_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_uid TEXT UNIQUE NOT NULL,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL DEFAULT '',
                        node_name TEXT NOT NULL DEFAULT '',
                        route TEXT NOT NULL DEFAULT '',
                        tool_call_id TEXT,
                        tool_name TEXT,
                        tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        response_metadata_json TEXT NOT NULL DEFAULT '{}',
                        usage_metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(thread_id, seq)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_tool_calls (
                        tool_call_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        assistant_message_uid TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        args_json TEXT NOT NULL DEFAULT '{}',
                        result_message_uid TEXT,
                        result_preview TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'called',
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts
                    USING fts5(
                        message_uid UNINDEXED,
                        thread_id UNINDEXED,
                        role UNINDEXED,
                        tool_name UNINDEXED,
                        search_text,
                        tokenize = 'unicode61'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                        summary_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        start_message_uid TEXT NOT NULL DEFAULT '',
                        end_message_uid TEXT NOT NULL DEFAULT '',
                        summary_type TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_seq
                    ON conversation_messages(thread_id, seq)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_created
                    ON conversation_messages(thread_id, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_tool_calls_thread_created
                    ON conversation_tool_calls(thread_id, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_tool_calls_tool_name
                    ON conversation_tool_calls(tool_name, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_summaries_thread_created
                    ON conversation_summaries(thread_id, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS worker_runs (
                        worker_id TEXT PRIMARY KEY,
                        parent_thread_id TEXT NOT NULL,
                        worker_thread_id TEXT NOT NULL,
                        parent_turn_id TEXT NOT NULL DEFAULT '',
                        role TEXT NOT NULL DEFAULT 'explore',
                        goal TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'pending',
                        allowed_tools_json TEXT NOT NULL DEFAULT '[]',
                        write_scope_json TEXT NOT NULL DEFAULT '[]',
                        tool_budget INTEGER NOT NULL DEFAULT 0,
                        result_summary_json TEXT NOT NULL DEFAULT '{}',
                        error_json TEXT NOT NULL DEFAULT '{}',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_worker_runs_parent_status_created
                    ON worker_runs(parent_thread_id, status, created_at DESC)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_worker_runs_worker_thread
                    ON worker_runs(worker_thread_id)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_program_runs (
                        program_run_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'pending',
                        source_program TEXT NOT NULL DEFAULT '',
                        normalized_ir_json TEXT NOT NULL DEFAULT '{}',
                        pc INTEGER NOT NULL DEFAULT 0,
                        locals_json TEXT NOT NULL DEFAULT '{}',
                        staged_tool_calls_json TEXT NOT NULL DEFAULT '[]',
                        stdout TEXT NOT NULL DEFAULT '',
                        result_summary_json TEXT NOT NULL DEFAULT '{}',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        finished_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_tool_program_runs_thread_status_updated
                    ON tool_program_runs(thread_id, status, updated_at DESC)
                    """
                )
                # Harness integration is additive: legacy graph/checkpoint data remains
                # available for FTS search and audit, while new turns use these tables.
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS harness_session_bindings (
                        thread_id TEXT NOT NULL,
                        generation INTEGER NOT NULL DEFAULT 0,
                        harness_session_id TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (thread_id, generation)
                    );
                    CREATE TABLE IF NOT EXISTS harness_session_leases (
                        harness_session_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS harness_context_tokens (
                        token_hash TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        workspace TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        revoked_at TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_harness_context_thread_turn
                    ON harness_context_tokens(thread_id, turn_id);
                    CREATE TABLE IF NOT EXISTS approval_batches (
                        batch_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        turn_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        workspace TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        resolved_at TEXT,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_approval_batches_thread_status
                    ON approval_batches(thread_id, status, created_at DESC);
                    CREATE TABLE IF NOT EXISTS approval_operations (
                        operation_id TEXT PRIMARY KEY,
                        batch_id TEXT NOT NULL,
                        ordinal INTEGER NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments_json TEXT NOT NULL,
                        arguments_fingerprint TEXT NOT NULL,
                        risk_reason TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_json TEXT NOT NULL DEFAULT '{}',
                        error_type TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        executed_at TEXT,
                        UNIQUE(batch_id, arguments_fingerprint),
                        FOREIGN KEY(batch_id) REFERENCES approval_batches(batch_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_approval_operations_batch_ordinal
                    ON approval_operations(batch_id, ordinal);
                    CREATE TABLE IF NOT EXISTS research_documents (
                        document_key TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        uri TEXT NOT NULL DEFAULT '',
                        content_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        indexed_at TEXT,
                        error_type TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


_default_runtime_store: RuntimeStore | None = None
_default_runtime_store_lock = threading.Lock()


def get_runtime_store(db_path: str | None = None) -> RuntimeStore:
    global _default_runtime_store
    if db_path is not None:
        return RuntimeStore(db_path=db_path)

    with _default_runtime_store_lock:
        if _default_runtime_store is None:
            _default_runtime_store = RuntimeStore()
        return _default_runtime_store
