from __future__ import annotations

from pathlib import Path
from typing import Any

from .dynamic import build_context_file_blocks


_MAX_ANCESTOR_WALK = 5
_PATH_TOOLS = frozenset(
    {
        "read_project_file",
        "edit_project_file",
        "write_project_file",
        "search_project_code",
        "run_project_command",
        "show_git_diff",
    }
)


def _safe_resolve(project_root: str, candidate: str) -> Path | None:
    if not candidate:
        return None
    try:
        root = Path(project_root).resolve()
        target = (root / candidate).resolve()
    except Exception:
        return None
    if not str(target).startswith(str(root)):
        return None
    return target


def _extract_glob_prefix(file_glob: str) -> str:
    if not file_glob:
        return ""
    parts = []
    for piece in str(file_glob).replace("\\", "/").split("/"):
        if any(ch in piece for ch in "*?[]"):
            break
        parts.append(piece)
    return "/".join(part for part in parts if part)


def _extract_candidate_path(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name not in _PATH_TOOLS:
        return ""
    for key in ("filepath", "path", "pathspec"):
        value = str(args.get(key, "") or "").strip()
        if value:
            return value
    if tool_name == "search_project_code":
        return _extract_glob_prefix(str(args.get("file_glob", "") or ""))
    return ""


def _discover_hints(
    *,
    project_root: str,
    directory: Path,
    loaded_dirs: set[str],
    char_budget: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    root = Path(project_root).resolve()
    current = directory.resolve()
    walked = 0
    discovered: list[dict[str, Any]] = []
    updated_dirs = set(loaded_dirs)

    while walked < _MAX_ANCESTOR_WALK and str(current).startswith(str(root)):
        dir_key = str(current)
        if dir_key in updated_dirs:
            break
        updated_dirs.add(dir_key)
        if current != root:
            rel_dir = current.relative_to(root).as_posix()
            blocks = build_context_file_blocks(
                str(current),
                char_budget=char_budget,
                source_prefix=f"subdir:{rel_dir}",
                include_cursor_rules=True,
            )
            discovered.extend(blocks)
        if current == root:
            break
        current = current.parent
        walked += 1
    return discovered, updated_dirs


def update_subdirectory_context_from_messages(
    *,
    state: dict,
    messages: list[Any],
    char_budget: int,
) -> dict[str, Any]:
    project_root = str(state.get("current_project_path", "") or "").strip()
    if not project_root:
        return {}

    root_path = Path(project_root)
    if not root_path.is_dir():
        return {}

    tool_call_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if getattr(message, "type", "") != "ai":
            continue
        for tool_call in list(getattr(message, "tool_calls", []) or []):
            if not isinstance(tool_call, dict):
                continue
            tool_id = str(tool_call.get("id", "") or "").strip()
            tool_name = str(tool_call.get("name", "") or "").strip()
            if not tool_id or tool_name not in _PATH_TOOLS:
                continue
            tool_call_map[tool_id] = (tool_name, dict(tool_call.get("args", {}) or {}))

    loaded_dirs = {
        str(item).strip()
        for item in (state.get("loaded_subdirectory_contexts", []) or [])
        if str(item or "").strip()
    }
    if not loaded_dirs:
        loaded_dirs.add(str(root_path.resolve()))

    existing_blocks = [
        dict(item)
        for item in (state.get("subdirectory_context_hints", []) or [])
        if isinstance(item, dict)
    ]
    existing_sources = {str(item.get("source", "") or "").strip() for item in existing_blocks}
    updated_dirs = set(loaded_dirs)
    new_blocks: list[dict[str, Any]] = []

    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        if not tool_call_id or tool_call_id not in tool_call_map:
            continue
        tool_name, tool_args = tool_call_map[tool_call_id]
        candidate_path = _extract_candidate_path(tool_name, tool_args)
        if not candidate_path:
            continue
        target = _safe_resolve(project_root, candidate_path)
        if target is None:
            continue
        target_dir = target if target.is_dir() else target.parent
        if not target_dir.exists() or not target_dir.is_dir():
            continue
        discovered, updated_dirs = _discover_hints(
            project_root=project_root,
            directory=target_dir,
            loaded_dirs=updated_dirs,
            char_budget=char_budget,
        )
        for block in discovered:
            source = str(block.get("source", "") or "").strip()
            if source and source not in existing_sources:
                existing_sources.add(source)
                new_blocks.append(block)

    if not new_blocks and updated_dirs == loaded_dirs:
        return {}

    merged_blocks = (existing_blocks + new_blocks)[-12:]
    return {
        "loaded_subdirectory_contexts": sorted(updated_dirs),
        "subdirectory_context_hints": merged_blocks,
    }
