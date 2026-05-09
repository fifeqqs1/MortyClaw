from __future__ import annotations

import os
import threading
from contextlib import contextmanager


def _normalize_scope_entries(project_root: str, paths: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    normalized_root = os.path.realpath(os.path.expanduser(project_root or "")).strip()
    entries: list[str] = []
    if not normalized_root:
        return tuple(entries)

    for raw_path in paths or []:
        candidate = str(raw_path or "").strip()
        if not candidate:
            continue
        if os.path.isabs(candidate):
            resolved = os.path.realpath(candidate)
        else:
            resolved = os.path.realpath(os.path.join(normalized_root, candidate))
        if not resolved.startswith(normalized_root):
            continue
        entries.append(os.path.relpath(resolved, normalized_root).replace(os.sep, "/"))
    return tuple(sorted(set(entries)))


class ProjectPathLockManager:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._holders: dict[str, str] = {}
        self._scope_holders: dict[str, tuple[str, tuple[str, ...]]] = {}

    def _lock_keys(self, project_root: str, paths: tuple[str, ...]) -> list[str]:
        normalized_root = os.path.realpath(os.path.expanduser(project_root or "")).strip()
        if not normalized_root:
            return []
        return [f"{normalized_root}:{path_value}" for path_value in paths]

    def _scope_conflicts(
        self,
        *,
        holder_id: str,
        project_root: str,
        write_scope: tuple[str, ...],
    ) -> bool:
        normalized_root = os.path.realpath(os.path.expanduser(project_root or "")).strip()
        if not normalized_root or not write_scope:
            return False
        for other_holder_id, (other_root, other_scope) in self._scope_holders.items():
            if other_holder_id == holder_id or other_root != normalized_root:
                continue
            if set(write_scope) & set(other_scope):
                return True
        return False

    @contextmanager
    def acquire(
        self,
        *,
        holder_id: str,
        project_root: str,
        paths: list[str] | tuple[str, ...] | None = None,
        write_scope: list[str] | tuple[str, ...] | None = None,
    ):
        normalized_paths = _normalize_scope_entries(project_root, list(paths or []))
        normalized_scope = _normalize_scope_entries(project_root, list(write_scope or normalized_paths))
        lock_keys = self._lock_keys(project_root, normalized_paths)
        with self._condition:
            while True:
                path_conflict = any(self._holders.get(lock_key) not in {None, holder_id} for lock_key in lock_keys)
                scope_conflict = self._scope_conflicts(
                    holder_id=holder_id,
                    project_root=project_root,
                    write_scope=normalized_scope,
                )
                if not path_conflict and not scope_conflict:
                    break
                self._condition.wait(timeout=0.1)

            for lock_key in lock_keys:
                self._holders[lock_key] = holder_id
            if normalized_scope:
                self._scope_holders[holder_id] = (
                    os.path.realpath(os.path.expanduser(project_root or "")).strip(),
                    normalized_scope,
                )

        try:
            yield {
                "holder_id": holder_id,
                "project_root": os.path.realpath(os.path.expanduser(project_root or "")).strip(),
                "paths": list(normalized_paths),
                "write_scope": list(normalized_scope),
            }
        finally:
            with self._condition:
                for lock_key in lock_keys:
                    if self._holders.get(lock_key) == holder_id:
                        self._holders.pop(lock_key, None)
                if self._scope_holders.get(holder_id):
                    self._scope_holders.pop(holder_id, None)
                self._condition.notify_all()


_default_lock_manager = ProjectPathLockManager()


def get_project_path_lock_manager() -> ProjectPathLockManager:
    return _default_lock_manager


__all__ = [
    "ProjectPathLockManager",
    "get_project_path_lock_manager",
]
