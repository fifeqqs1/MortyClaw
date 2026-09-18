from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..storage.store import RuntimeStore, get_runtime_store


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class HarnessStore:
    def __init__(self, store: RuntimeStore | None = None):
        self.store = store or get_runtime_store()

    def current_generation(self, thread_id: str) -> int:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT MAX(generation) AS generation FROM harness_session_bindings WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return int(row["generation"] or 0) if row else 0

    def session_id(self, thread_id: str, generation: int | None = None) -> str:
        generation = self.current_generation(thread_id) if generation is None else generation
        with self.store._lock, self.store._connect() as conn:
            row = conn.execute(
                "SELECT harness_session_id FROM harness_session_bindings WHERE thread_id = ? AND generation = ?",
                (thread_id, generation),
            ).fetchone()
            if row:
                return str(row["harness_session_id"])
            digest = hashlib.sha256(f"mortyclaw:{thread_id}:{generation}:{secrets.token_hex(16)}".encode()).hexdigest()
            session_id = f"mc-{digest[:40]}"
            now = _iso(_now())
            conn.execute(
                "INSERT INTO harness_session_bindings VALUES (?, ?, ?, ?, ?)",
                (thread_id, generation, session_id, now, now),
            )
            conn.commit()
            return session_id

    def reset(self, thread_id: str) -> str:
        return self.session_id(thread_id, self.current_generation(thread_id) + 1)

    def acquire_lease(self, session_id: str, owner_id: str, ttl_seconds: int) -> bool:
        now = _now()
        expires = now + timedelta(seconds=ttl_seconds)
        with self.store._lock, self.store._connect() as conn:
            conn.execute("DELETE FROM harness_session_leases WHERE expires_at <= ?", (_iso(now),))
            row = conn.execute(
                "SELECT owner_id FROM harness_session_leases WHERE harness_session_id = ?", (session_id,)
            ).fetchone()
            if row and row["owner_id"] != owner_id:
                conn.commit()
                return False
            conn.execute(
                "INSERT OR REPLACE INTO harness_session_leases VALUES (?, ?, ?, ?)",
                (session_id, owner_id, _iso(now), _iso(expires)),
            )
            conn.commit()
        return True

    def release_lease(self, session_id: str, owner_id: str) -> None:
        with self.store._lock, self.store._connect() as conn:
            conn.execute(
                "DELETE FROM harness_session_leases WHERE harness_session_id = ? AND owner_id = ?",
                (session_id, owner_id),
            )
            conn.commit()

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def issue_context_token(
        self, *, thread_id: str, turn_id: str, source: str, workspace: str, ttl_seconds: int = 7200
    ) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self.store._lock, self.store._connect() as conn:
            conn.execute(
                "INSERT INTO harness_context_tokens VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (self.token_hash(token), thread_id, turn_id, source, str(Path(workspace).resolve()),
                 _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
            )
            conn.commit()
        return token

    def validate_context_token(self, token: str, *, workspace: str | None = None) -> dict[str, str] | None:
        if not token:
            return None
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM harness_context_tokens WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?",
                (self.token_hash(token), _iso(_now())),
            ).fetchone()
        if not row:
            return None
        value = dict(row)
        if workspace and Path(value["workspace"]).resolve() != Path(workspace).resolve():
            return None
        return value

    def stage_approval(
        self, *, context: dict[str, str], tool_name: str, arguments: dict[str, Any], risk_reason: str,
        ttl_seconds: int = 900,
    ) -> tuple[str, str, bool]:
        fingerprint = hashlib.sha256(f"{tool_name}\n{_json(arguments)}".encode()).hexdigest()
        now = _now()
        with self.store._lock, self.store._connect() as conn:
            row = conn.execute(
                "SELECT batch_id FROM approval_batches WHERE thread_id=? AND turn_id=? AND status='pending' AND expires_at>?",
                (context["thread_id"], context["turn_id"], _iso(now)),
            ).fetchone()
            if row:
                batch_id = str(row["batch_id"])
            else:
                batch_id = f"ap-{uuid.uuid4().hex[:16]}"
                conn.execute(
                    "INSERT INTO approval_batches VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, NULL, '{}')",
                    (batch_id, context["thread_id"], context["turn_id"], context["source"],
                     context["workspace"], _iso(now), _iso(now + timedelta(seconds=ttl_seconds))),
                )
            existing = conn.execute(
                "SELECT operation_id FROM approval_operations WHERE batch_id=? AND arguments_fingerprint=?",
                (batch_id, fingerprint),
            ).fetchone()
            if existing:
                conn.commit()
                return batch_id, str(existing["operation_id"]), False
            ordinal = conn.execute(
                "SELECT COUNT(*) AS n FROM approval_operations WHERE batch_id=?", (batch_id,)
            ).fetchone()["n"]
            operation_id = f"op-{uuid.uuid4().hex[:16]}"
            conn.execute(
                "INSERT INTO approval_operations VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', '{}', '', ?, NULL)",
                (operation_id, batch_id, int(ordinal), tool_name, _json(arguments), fingerprint, risk_reason, _iso(now)),
            )
            conn.commit()
        return batch_id, operation_id, True

    def pending_batch(self, thread_id: str, turn_id: str | None = None) -> dict[str, Any] | None:
        params: list[Any] = [thread_id, _iso(_now())]
        clause = "thread_id=? AND status='pending' AND expires_at>?"
        if turn_id:
            clause += " AND turn_id=?"
            params.append(turn_id)
        with self.store._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM approval_batches WHERE {clause} ORDER BY created_at DESC LIMIT 1", params
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["operations"] = [dict(item) for item in conn.execute(
                "SELECT * FROM approval_operations WHERE batch_id=? ORDER BY ordinal", (result["batch_id"],)
            ).fetchall()]
            return result

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.store._connect() as conn:
            row = conn.execute("SELECT * FROM approval_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["operations"] = [dict(item) for item in conn.execute(
                "SELECT * FROM approval_operations WHERE batch_id=? ORDER BY ordinal", (batch_id,)
            ).fetchall()]
        return result

    def resolve_batch(self, batch_id: str, status: str) -> None:
        with self.store._lock, self.store._connect() as conn:
            conn.execute(
                "UPDATE approval_batches SET status=?, resolved_at=? WHERE batch_id=? AND status='pending'",
                (status, _iso(_now()), batch_id),
            )
            if status in {"rejected", "cancelled", "expired"}:
                conn.execute(
                    "UPDATE approval_operations SET status=? WHERE batch_id=? AND status='pending'",
                    (status, batch_id),
                )
            conn.commit()

    def update_operation(self, operation_id: str, status: str, *, result: Any = None, error_type: str = "") -> None:
        with self.store._lock, self.store._connect() as conn:
            conn.execute(
                "UPDATE approval_operations SET status=?, result_json=?, error_type=?, executed_at=? WHERE operation_id=?",
                (status, _json(result or {}), error_type, _iso(_now()), operation_id),
            )
            conn.commit()

    def list_batches(self, status: str = "pending", limit: int = 20) -> list[dict[str, Any]]:
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approval_batches WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        return [dict(row) for row in rows]

    def cancel_pending_for_thread(self, thread_id: str) -> None:
        pending = self.pending_batch(thread_id)
        if pending:
            self.resolve_batch(pending["batch_id"], "cancelled")


def process_owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
