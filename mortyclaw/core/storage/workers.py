from __future__ import annotations

import json
import uuid
from typing import Any

from .common import WorkerRunRecord, safe_json_loads, safe_json_dumps, utc_now_iso
from .store import RuntimeStore


class WorkerRunRepository:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def create_worker_run(
        self,
        *,
        parent_thread_id: str,
        worker_thread_id: str,
        parent_turn_id: str = "",
        role: str = "explore",
        goal: str = "",
        allowed_tools: list[str] | None = None,
        write_scope: list[str] | None = None,
        tool_budget: int = 0,
        metadata: dict[str, Any] | None = None,
        worker_id: str | None = None,
    ) -> WorkerRunRecord:
        now = utc_now_iso()
        record = {
            "worker_id": worker_id or str(uuid.uuid4()),
            "parent_thread_id": (parent_thread_id or "system_default").strip() or "system_default",
            "worker_thread_id": (worker_thread_id or "").strip() or str(uuid.uuid4()),
            "parent_turn_id": str(parent_turn_id or "").strip(),
            "role": str(role or "explore").strip() or "explore",
            "goal": str(goal or ""),
            "status": "pending",
            "allowed_tools_json": safe_json_dumps(list(allowed_tools or [])),
            "write_scope_json": safe_json_dumps(list(write_scope or [])),
            "tool_budget": max(0, int(tool_budget or 0)),
            "result_summary_json": "{}",
            "error_json": "{}",
            "metadata_json": safe_json_dumps(metadata or {}),
            "created_at": now,
            "started_at": None,
            "finished_at": None,
        }
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO worker_runs (
                        worker_id, parent_thread_id, worker_thread_id, parent_turn_id, role, goal,
                        status, allowed_tools_json, write_scope_json, tool_budget,
                        result_summary_json, error_json, metadata_json, created_at, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["worker_id"],
                        record["parent_thread_id"],
                        record["worker_thread_id"],
                        record["parent_turn_id"],
                        record["role"],
                        record["goal"],
                        record["status"],
                        record["allowed_tools_json"],
                        record["write_scope_json"],
                        record["tool_budget"],
                        record["result_summary_json"],
                        record["error_json"],
                        record["metadata_json"],
                        record["created_at"],
                        record["started_at"],
                        record["finished_at"],
                    ),
                )
                conn.commit()
        return record

    def get_worker_run(self, worker_id: str) -> WorkerRunRecord | None:
        with self.store._lock:
            with self.store._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM worker_runs WHERE worker_id = ?",
                    ((worker_id or "").strip(),),
                ).fetchone()
        return dict(row) if row else None

    def list_worker_runs(
        self,
        *,
        parent_thread_id: str = "",
        statuses: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[WorkerRunRecord]:
        where_parts: list[str] = []
        params: list[object] = []
        if parent_thread_id:
            where_parts.append("parent_thread_id = ?")
            params.append(parent_thread_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            where_parts.append(f"status IN ({placeholders})")
            params.extend(statuses)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.append(max(1, min(int(limit or 50), 200)))
        with self.store._lock:
            with self.store._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM worker_runs
                    {where_sql}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_worker_run(
        self,
        worker_id: str,
        *,
        status: str | None = None,
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        started: bool = False,
        finished: bool = False,
    ) -> WorkerRunRecord | None:
        existing = self.get_worker_run(worker_id)
        if existing is None:
            return None
        next_metadata = safe_json_loads(existing.get("metadata_json"))
        if isinstance(metadata, dict):
            next_metadata.update(metadata)
        started_at = existing.get("started_at")
        finished_at = existing.get("finished_at")
        now = utc_now_iso()
        if started and not started_at:
            started_at = now
        if finished:
            finished_at = now
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    UPDATE worker_runs
                    SET status = ?, result_summary_json = ?, error_json = ?, metadata_json = ?,
                        started_at = ?, finished_at = ?
                    WHERE worker_id = ?
                    """,
                    (
                        str(status or existing.get("status", "pending")),
                        safe_json_dumps(result_summary if result_summary is not None else safe_json_loads(existing.get("result_summary_json"))),
                        safe_json_dumps(error if error is not None else safe_json_loads(existing.get("error_json"))),
                        safe_json_dumps(next_metadata),
                        started_at,
                        finished_at,
                        existing["worker_id"],
                    ),
                )
                conn.commit()
        return self.get_worker_run(worker_id)

