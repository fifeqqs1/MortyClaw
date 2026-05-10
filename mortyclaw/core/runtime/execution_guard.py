from __future__ import annotations

import hashlib
import json
import os

from ..tools.project.common import (
    _file_hash,
    _relative_path,
    _resolve_project_path,
    _resolve_project_root,
)
from ..storage.runtime import get_tool_program_run_repository
from ..tools.project.patch import _extract_patch_paths, _run_git_apply


_PATH_ARG_NAMES_BY_TOOL = {
    "write_project_file": ("path",),
    "edit_project_file": ("path",),
    "read_project_file": ("filepath",),
}


def _normalize_tool_calls(tool_calls: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        raw_args = tool_call.get("args")
        if raw_args is None:
            raw_args = tool_call.get("tool_args")
        normalized.append({
            "name": str(tool_call.get("name") or tool_call.get("tool_name") or "").strip(),
            "args": dict(raw_args or {}),
        })
    return normalized


def build_approval_context_hash(state: dict, pending_tool_calls: list[dict] | None = None) -> str:
    plan = state.get("plan", []) or []
    current_step_index = int(state.get("current_step_index", 0) or 0)
    current_step = plan[current_step_index] if 0 <= current_step_index < len(plan) else {}
    payload = {
        "goal": str(state.get("goal", "") or "").strip(),
        "current_project_path": str(state.get("current_project_path", "") or "").strip(),
        "current_step_index": current_step_index,
        "todo_revision": int(state.get("todo_revision", 0) or 0),
        "current_step": {
            "description": str(current_step.get("description", "") or "").strip(),
            "risk_level": str(current_step.get("risk_level", "") or "").strip(),
            "intent": str(current_step.get("intent", "") or "").strip(),
        },
        "pending_tool_calls": _normalize_tool_calls(
            pending_tool_calls if pending_tool_calls is not None else state.get("pending_tool_calls", [])
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_tool_program_approval_hash(
    *,
    program_run_id: str,
    project_root: str,
    goal: str,
    pc: int,
    locals_json: str,
    staged_tool_calls: list[dict] | None,
    write_scope: list[str] | None = None,
) -> str:
    payload = {
        "program_run_id": str(program_run_id or "").strip(),
        "project_root": str(project_root or "").strip(),
        "goal": str(goal or "").strip(),
        "pc": max(0, int(pc or 0)),
        "locals_json": str(locals_json or ""),
        "staged_tool_calls": _normalize_tool_calls(staged_tool_calls),
        "write_scope": sorted(str(item or "").strip() for item in (write_scope or []) if str(item or "").strip()),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _collect_target_paths(project_root: str, pending_tool_calls: list[dict]) -> list[str]:
    paths: list[str] = []
    for tool_call in pending_tool_calls:
        tool_name = str(tool_call.get("name") or "").strip()
        args = dict(tool_call.get("args") or {})
        for arg_name in _PATH_ARG_NAMES_BY_TOOL.get(tool_name, ()):
            raw_path = str(args.get(arg_name, "") or "").strip()
            if not raw_path:
                continue
            target = _resolve_project_path(project_root, raw_path, allow_missing=True, allow_sensitive=False)
            paths.append(_relative_path(project_root, target))
        if tool_name == "apply_project_patch":
            patch_text = str(args.get("patch", "") or "")
            for rel_path in _extract_patch_paths(patch_text):
                target = _resolve_project_path(project_root, rel_path, allow_missing=True, allow_sensitive=False)
                paths.append(_relative_path(project_root, target))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def build_pending_execution_snapshot(state: dict, pending_tool_calls: list[dict] | None) -> dict:
    normalized_tool_calls = _normalize_tool_calls(pending_tool_calls)
    project_root = str(state.get("current_project_path", "") or "").strip()
    resolved_project_root = ""
    path_snapshots: list[dict] = []
    patch_checks: list[dict] = []

    if project_root:
        try:
            resolved_project_root = _resolve_project_root(project_root)
        except Exception:
            resolved_project_root = os.path.realpath(os.path.expanduser(project_root))

    if resolved_project_root and os.path.isdir(resolved_project_root):
        for rel_path in _collect_target_paths(resolved_project_root, normalized_tool_calls):
            target = _resolve_project_path(resolved_project_root, rel_path, allow_missing=True, allow_sensitive=False)
            exists = os.path.exists(target)
            path_snapshots.append({
                "path": rel_path,
                "exists": bool(exists),
                "sha256": _file_hash(target) if exists else "",
            })

        for tool_call in normalized_tool_calls:
            if tool_call.get("name") != "apply_project_patch":
                continue
            patch_text = str((tool_call.get("args") or {}).get("patch", "") or "")
            if patch_text.strip():
                patch_checks.append({
                    "patch": patch_text,
                    "changed_paths": _extract_patch_paths(patch_text),
                })

    return {
        "approval_context_hash": build_approval_context_hash(state, normalized_tool_calls),
        "project_root": resolved_project_root,
        "path_snapshots": path_snapshots,
        "patch_checks": patch_checks,
    }


def _validate_project_snapshot(snapshot: dict) -> dict:
    project_root = str(snapshot.get("project_root", "") or "").strip()
    if project_root and not os.path.isdir(project_root):
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": f"恢复执行前检测到项目目录已不存在：{project_root}",
        }

    for item in snapshot.get("path_snapshots", []) or []:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path", "") or "").strip()
        if not rel_path or not project_root:
            continue
        target = _resolve_project_path(project_root, rel_path, allow_missing=True, allow_sensitive=False)
        exists = os.path.exists(target)
        if exists != bool(item.get("exists", False)):
            return {
                "ok": False,
                "status": "replan_requested",
                "reason": f"恢复执行前检测到目标文件存在性已变化：{rel_path}",
            }
        if exists:
            current_file_hash = _file_hash(target)
            if current_file_hash != str(item.get("sha256", "") or ""):
                return {
                    "ok": False,
                    "status": "replan_requested",
                    "reason": f"恢复执行前检测到目标文件内容已变化：{rel_path}",
                }

    for patch_item in snapshot.get("patch_checks", []) or []:
        if not isinstance(patch_item, dict):
            continue
        patch_text = str(patch_item.get("patch", "") or "")
        if not patch_text.strip() or not project_root:
            continue
        result = _run_git_apply(project_root, patch_text, check_only=True)
        if result.returncode != 0:
            changed_paths = ", ".join(patch_item.get("changed_paths", []) or []) or "patch target files"
            return {
                "ok": False,
                "status": "replan_requested",
                "reason": f"恢复执行前检测到 patch 上下文已失效：{changed_paths}",
            }

    return {"ok": True, "status": "passed", "reason": ""}


def _validate_tool_program_snapshot(state: dict, snapshot: dict) -> dict:
    program_run_id = str(snapshot.get("program_run_id", "") or "").strip()
    if not program_run_id:
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": "恢复执行前缺少 program_run_id。",
        }
    run = get_tool_program_run_repository().get_program_run(program_run_id)
    if run is None:
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": f"恢复执行前未找到程序化执行记录：{program_run_id}",
        }
    if str(run.get("status", "") or "") != "awaiting_approval":
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": "恢复执行前检测到程序化执行状态已变化，需要重新规划。",
        }

    metadata_json = str(run.get("metadata_json", "") or "")
    metadata = json.loads(metadata_json) if metadata_json.strip().startswith("{") else {}
    project_root = str((metadata or {}).get("project_root") or snapshot.get("project_root", "") or "").strip()
    goal = str((metadata or {}).get("goal") or state.get("goal", "") or "").strip()
    locals_json = str(run.get("locals_json", "") or "")
    staged_tool_calls = json.loads(str(run.get("staged_tool_calls_json", "[]") or "[]"))
    write_scope = (metadata or {}).get("write_scope") or snapshot.get("write_scope", []) or []
    current_hash = build_tool_program_approval_hash(
        program_run_id=program_run_id,
        project_root=project_root,
        goal=goal,
        pc=int(run.get("pc", 0) or 0),
        locals_json=locals_json,
        staged_tool_calls=staged_tool_calls,
        write_scope=write_scope,
    )
    expected_hash = str(snapshot.get("approval_context_hash", "") or "").strip()
    if expected_hash and current_hash != expected_hash:
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": "恢复执行前检测到程序化执行审批上下文已变化。",
        }
    locals_hash = hashlib.sha256(locals_json.encode("utf-8")).hexdigest()
    if str(snapshot.get("locals_hash", "") or "").strip() and locals_hash != str(snapshot.get("locals_hash", "") or "").strip():
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": "恢复执行前检测到程序局部状态已变化。",
        }

    project_validation = _validate_project_snapshot(dict(snapshot.get("base_snapshot", {}) or {}))
    if not project_validation.get("ok", False):
        return project_validation
    return {"ok": True, "status": "passed", "reason": ""}


def validate_pending_execution_snapshot(state: dict) -> dict:
    try:
        snapshot = dict(state.get("pending_execution_snapshot", {}) or {})
        if not snapshot:
            return {"ok": True, "status": "skipped", "reason": "no pending execution snapshot"}

        if str(snapshot.get("kind", "") or "").strip() == "tool_program":
            return _validate_tool_program_snapshot(state, snapshot)

        current_hash = build_approval_context_hash(state)
        expected_hash = str(snapshot.get("approval_context_hash", "") or "").strip()
        if expected_hash and current_hash != expected_hash:
            return {
                "ok": False,
                "status": "replan_requested",
                "reason": "恢复执行前检测到审批上下文已变化，原批准上下文失效。",
            }

        return _validate_project_snapshot(snapshot)
    except Exception as exc:
        return {
            "ok": False,
            "status": "replan_requested",
            "reason": f"恢复执行前环境校验失败：{exc}",
        }
