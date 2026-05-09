from __future__ import annotations

from typing import Any

from .common import ToolProgramRunRecord, safe_json_dumps, safe_json_loads, utc_now_iso
from .store import RuntimeStore


class ToolProgramRunRepository:
    def __init__(self, store: RuntimeStore):
        self.store = store

    def upsert_program_run(
        self,
        *,
        program_run_id: str,
        thread_id: str,
        turn_id: str = "",
        status: str = "pending",
        source_program: str = "",
        normalized_ir: dict[str, Any] | None = None,
        pc: int = 0,
        locals_payload: dict[str, Any] | None = None,
        staged_tool_calls: list[dict[str, Any]] | None = None,
        stdout: str = "",
        result_summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        finished: bool = False,
    ) -> ToolProgramRunRecord:
        existing = self.get_program_run(program_run_id)
        now = utc_now_iso()
        payload = {
            "program_run_id": program_run_id,
            "thread_id": (thread_id or "system_default").strip() or "system_default",
            "turn_id": str(turn_id or ""),
            "status": str(status or "pending"),
            "source_program": str(source_program or ""),
            "normalized_ir_json": safe_json_dumps(normalized_ir if normalized_ir is not None else safe_json_loads((existing or {}).get("normalized_ir_json"))),
            "pc": max(0, int(pc or 0)),
            "locals_json": safe_json_dumps(locals_payload if locals_payload is not None else safe_json_loads((existing or {}).get("locals_json"))),
            "staged_tool_calls_json": safe_json_dumps(staged_tool_calls if staged_tool_calls is not None else safe_json_loads((existing or {}).get("staged_tool_calls_json"))),
            "stdout": str(stdout if stdout != "" else (existing or {}).get("stdout", "") or ""),
            "result_summary_json": safe_json_dumps(result_summary if result_summary is not None else safe_json_loads((existing or {}).get("result_summary_json"))),
            "metadata_json": safe_json_dumps(metadata if metadata is not None else safe_json_loads((existing or {}).get("metadata_json"))),
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
            "finished_at": now if finished else (existing or {}).get("finished_at"),
        }
        with self.store._lock:
            with self.store._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_program_runs (
                        program_run_id, thread_id, turn_id, status, source_program, normalized_ir_json,
                        pc, locals_json, staged_tool_calls_json, stdout, result_summary_json, metadata_json,
                        created_at, updated_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(program_run_id) DO UPDATE SET
                        thread_id = excluded.thread_id,
                        turn_id = excluded.turn_id,
                        status = excluded.status,
                        source_program = excluded.source_program,
                        normalized_ir_json = excluded.normalized_ir_json,
                        pc = excluded.pc,
                        locals_json = excluded.locals_json,
                        staged_tool_calls_json = excluded.staged_tool_calls_json,
                        stdout = excluded.stdout,
                        result_summary_json = excluded.result_summary_json,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at,
                        finished_at = excluded.finished_at
                    """,
                    (
                        payload["program_run_id"],
                        payload["thread_id"],
                        payload["turn_id"],
                        payload["status"],
                        payload["source_program"],
                        payload["normalized_ir_json"],
                        payload["pc"],
                        payload["locals_json"],
                        payload["staged_tool_calls_json"],
                        payload["stdout"],
                        payload["result_summary_json"],
                        payload["metadata_json"],
                        payload["created_at"],
                        payload["updated_at"],
                        payload["finished_at"],
                    ),
                )
                conn.commit()
        row = self.get_program_run(program_run_id)
        assert row is not None
        return row

    def get_program_run(self, program_run_id: str) -> ToolProgramRunRecord | None:
        with self.store._lock:
            with self.store._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM tool_program_runs WHERE program_run_id = ?",
                    ((program_run_id or "").strip(),),
                ).fetchone()
        return dict(row) if row else None

    def list_program_runs(
        self,
        *,
        thread_id: str = "",
        statuses: tuple[str, ...] | None = None,
        limit: int = 50,
    ) -> list[ToolProgramRunRecord]:
        where_parts: list[str] = []
        params: list[object] = []
        if thread_id:
            where_parts.append("thread_id = ?")
            params.append(thread_id)
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
                    FROM tool_program_runs
                    {where_sql}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    tuple(params),
                ).fetchall()
        return [dict(row) for row in rows]
