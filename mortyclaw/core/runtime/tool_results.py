from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from typing import Any

from langchain_core.messages import ToolMessage

from ..config import RUNTIME_ARTIFACTS_DIR


PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
DEFAULT_PREVIEW_CHARS = 2400
DEFAULT_RESULT_THRESHOLD = 9000
MAX_TURN_BUDGET_CHARS = 24000
CONTEXT_ARTIFACT_DIR_NAME = "context"
CONTEXT_ARTIFACT_MANIFEST = "manifest.jsonl"

TOOL_RESULT_THRESHOLDS = {
    "run_project_command": 7000,
    "run_project_tests": 7000,
    "show_git_diff": 7000,
    "search_project_code": 8000,
    "summarize_content": 8000,
    "tavily_web_search": 8000,
    "edit_project_file": 12000,
    "read_project_file": 14000,
    "read_office_file": 14000,
    "write_project_file": 14000,
}


def _safe_component(value: str, *, default: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return normalized or default


def _artifact_dir(thread_id: str, turn_id: str) -> str:
    return os.path.join(
        RUNTIME_ARTIFACTS_DIR,
        _safe_component(thread_id, default="thread"),
        _safe_component(turn_id, default="turn"),
    )


def _artifact_path(thread_id: str, turn_id: str, tool_call_id: str) -> str:
    return os.path.join(
        _artifact_dir(thread_id, turn_id),
        f"{_safe_component(tool_call_id, default='tool_call')}.txt",
    )


def _context_artifact_dir(thread_id: str, turn_id: str) -> str:
    return os.path.join(
        RUNTIME_ARTIFACTS_DIR,
        CONTEXT_ARTIFACT_DIR_NAME,
        _safe_component(thread_id, default="thread"),
        _safe_component(turn_id, default="turn"),
    )


def _context_artifact_path(thread_id: str, turn_id: str, ref_id: str) -> str:
    return os.path.join(
        _context_artifact_dir(thread_id, turn_id),
        f"{_safe_component(ref_id, default='context_artifact')}.txt",
    )


def _context_manifest_path(thread_id: str, turn_id: str) -> str:
    return os.path.join(_context_artifact_dir(thread_id, turn_id), CONTEXT_ARTIFACT_MANIFEST)


def _sha256_text(content: str) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8", errors="replace")).hexdigest()


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_metadata(message: ToolMessage) -> dict[str, Any]:
    metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    return dict(metadata.get("mortyclaw_artifact") or {})


def _generate_preview(content: str, *, limit: int = DEFAULT_PREVIEW_CHARS) -> tuple[str, bool]:
    if len(content) <= limit:
        return content, False
    truncated = content[:limit]
    last_break = truncated.rfind("\n")
    if last_break > limit // 2:
        truncated = truncated[: last_break + 1]
    return truncated, True


def _threshold_for_tool(tool_name: str) -> int:
    return int(TOOL_RESULT_THRESHOLDS.get(tool_name, DEFAULT_RESULT_THRESHOLD))


def build_artifact_message(
    *,
    preview: str,
    has_more: bool,
    artifact_path: str,
    original_size: int,
) -> str:
    size_kb = original_size / 1024
    size_text = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
    lines = [
        PERSISTED_OUTPUT_TAG,
        f"该工具结果较大（{original_size:,} 字符，{size_text}）。",
        f"完整输出已保存到：{artifact_path}",
        "如需查看完整内容，请使用 read_office_file 读取该文件。",
        "",
        f"预览（前 {len(preview)} 字符）：",
        preview,
    ]
    if has_more:
        lines.append("...")
    lines.append(PERSISTED_OUTPUT_CLOSING_TAG)
    return "\n".join(lines)


def persist_context_artifact(
    *,
    content: str,
    thread_id: str,
    turn_id: str,
    tool_name: str = "",
    args_summary: str = "",
    restore_hint: str = "",
    ref_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist original context content before replacing it with a prompt stub."""

    normalized_content = str(content or "")
    normalized_ref_id = _safe_component(ref_id, default="")
    if not normalized_ref_id:
        normalized_ref_id = f"ctx_{uuid.uuid4().hex}"
    artifact_path = _context_artifact_path(thread_id, turn_id, normalized_ref_id)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(normalized_content)

    content_hash = f"sha256:{_sha256_text(normalized_content)}"
    record = {
        "ref_id": normalized_ref_id,
        "artifact_path": artifact_path,
        "content_hash": content_hash,
        "original_chars": len(normalized_content),
        "tool_name": str(tool_name or ""),
        "args_summary": str(args_summary or ""),
        "restore_hint": str(restore_hint or ""),
        "metadata": dict(metadata or {}),
    }
    manifest_path = _context_manifest_path(thread_id, turn_id)
    with open(manifest_path, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(_safe_json_dumps(record) + "\n")
    return record


def _iter_context_artifact_records() -> list[dict[str, Any]]:
    root = os.path.join(RUNTIME_ARTIFACTS_DIR, CONTEXT_ARTIFACT_DIR_NAME)
    if not os.path.isdir(root):
        return []
    records: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if CONTEXT_ARTIFACT_MANIFEST not in filenames:
            continue
        manifest_path = os.path.join(dirpath, CONTEXT_ARTIFACT_MANIFEST)
        try:
            with open(manifest_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        records.append(payload)
        except OSError:
            continue
    return records


def _is_path_within_root(path: str, root: str) -> bool:
    try:
        resolved_path = os.path.realpath(path)
        resolved_root = os.path.realpath(root)
        return os.path.commonpath([resolved_path, resolved_root]) == resolved_root
    except Exception:
        return False


def restore_context_artifact(ref_id: str, *, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    normalized_ref_id = _safe_component(str(ref_id or ""), default="")
    if not normalized_ref_id:
        return json.dumps({"ok": False, "message": "ref_id is required"}, ensure_ascii=False)

    matching_record = None
    for record in _iter_context_artifact_records():
        if str(record.get("ref_id", "") or "") == normalized_ref_id:
            matching_record = record
            break
    if not matching_record:
        return json.dumps({"ok": False, "message": f"context artifact not found: {normalized_ref_id}"}, ensure_ascii=False)

    artifact_path = str(matching_record.get("artifact_path", "") or "")
    if not _is_path_within_root(artifact_path, RUNTIME_ARTIFACTS_DIR):
        return json.dumps({"ok": False, "message": "artifact path is outside runtime artifact root"}, ensure_ascii=False)
    if not os.path.exists(artifact_path):
        return json.dumps({"ok": False, "message": "artifact file is missing", "ref_id": normalized_ref_id}, ensure_ascii=False)

    with open(artifact_path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    actual_hash = f"sha256:{_sha256_text(content)}"
    expected_hash = str(matching_record.get("content_hash", "") or "")
    preview, has_more = _generate_preview(content, limit=max(200, int(preview_chars or DEFAULT_PREVIEW_CHARS)))
    return json.dumps(
        {
            "ok": True,
            "ref_id": normalized_ref_id,
            "artifact_path": artifact_path,
            "content_hash": actual_hash,
            "hash_matches": not expected_hash or actual_hash == expected_hash,
            "original_chars": len(content),
            "tool_name": matching_record.get("tool_name", ""),
            "args_summary": matching_record.get("args_summary", ""),
            "restore_hint": matching_record.get("restore_hint", ""),
            "has_more": has_more,
            "preview": preview,
        },
        ensure_ascii=False,
    )


def maybe_persist_tool_result(
    *,
    content: str,
    tool_name: str,
    thread_id: str,
    turn_id: str,
    tool_call_id: str,
    threshold: int | None = None,
) -> tuple[str, dict[str, Any]]:
    normalized_content = str(content or "")
    effective_threshold = int(threshold if threshold is not None else _threshold_for_tool(tool_name))
    if len(normalized_content) <= effective_threshold:
        return normalized_content, {
            "artifact_persisted": False,
            "artifact_path": "",
            "artifact_size": len(normalized_content),
            "preview_chars": min(len(normalized_content), DEFAULT_PREVIEW_CHARS),
        }

    artifact_path = _artifact_path(thread_id, turn_id, tool_call_id)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(normalized_content)

    preview, has_more = _generate_preview(normalized_content)
    return build_artifact_message(
        preview=preview,
        has_more=has_more,
        artifact_path=artifact_path,
        original_size=len(normalized_content),
    ), {
        "artifact_persisted": True,
        "artifact_path": artifact_path,
        "artifact_size": len(normalized_content),
        "preview_chars": len(preview),
    }


def _clone_tool_message(message: ToolMessage, *, content: str, metadata: dict[str, Any]) -> ToolMessage:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs["mortyclaw_artifact"] = metadata
    return ToolMessage(
        content=content,
        tool_call_id=getattr(message, "tool_call_id", None),
        name=getattr(message, "name", None),
        id=getattr(message, "id", None),
        additional_kwargs=additional_kwargs,
    )


def prepare_tool_messages_for_budget(
    messages: list[Any],
    *,
    thread_id: str,
    turn_id: str,
) -> list[Any]:
    if not messages:
        return messages

    prepared = list(messages)
    tool_candidates: list[tuple[int, ToolMessage, dict[str, Any]]] = []
    total_size = 0

    for index, message in enumerate(prepared):
        if not isinstance(message, ToolMessage):
            continue
        content = str(getattr(message, "content", "") or "")
        total_size += len(content)
        metadata = _extract_metadata(message)
        if PERSISTED_OUTPUT_TAG in content or metadata.get("artifact_persisted"):
            continue
        tool_candidates.append((index, message, metadata))

    if not tool_candidates:
        return prepared

    for index, message, _metadata in tool_candidates:
        tool_name = str(getattr(message, "name", "") or "tool")
        content = str(getattr(message, "content", "") or "")
        if len(content) <= _threshold_for_tool(tool_name):
            continue
        tool_name = str(getattr(message, "name", "") or "tool")
        tool_call_id = str(getattr(message, "tool_call_id", "") or getattr(message, "id", "") or f"tool-{index}")
        updated_content, updated_metadata = maybe_persist_tool_result(
            content=content,
            tool_name=tool_name,
            thread_id=thread_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
        )
        prepared[index] = _clone_tool_message(message, content=updated_content, metadata=updated_metadata)

    total_size = sum(len(str(getattr(message, "content", "") or "")) for message in prepared if isinstance(message, ToolMessage))
    if total_size <= MAX_TURN_BUDGET_CHARS:
        return prepared

    oversized = sorted(
        [
            (index, len(str(getattr(message, "content", "") or "")), message)
            for index, message in enumerate(prepared)
            if isinstance(message, ToolMessage) and PERSISTED_OUTPUT_TAG not in str(getattr(message, "content", "") or "")
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    for index, _size, message in oversized:
        if total_size <= MAX_TURN_BUDGET_CHARS:
            break
        tool_name = str(getattr(message, "name", "") or "tool")
        tool_call_id = str(getattr(message, "tool_call_id", "") or getattr(message, "id", "") or f"budget-{index}")
        original_content = str(getattr(message, "content", "") or "")
        updated_content, updated_metadata = maybe_persist_tool_result(
            content=original_content,
            tool_name=tool_name,
            thread_id=thread_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            threshold=0,
        )
        if updated_content != original_content:
            total_size -= len(original_content)
            total_size += len(updated_content)
            prepared[index] = _clone_tool_message(message, content=updated_content, metadata=updated_metadata)

    return prepared
