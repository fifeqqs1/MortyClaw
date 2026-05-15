from dataclasses import dataclass, field
from typing import Iterable, Literal

from langchain_core.tools import BaseTool


ToolRiskLevel = Literal["low", "medium", "high"]
ToolRoute = Literal["fast", "slow"]

TOOL_META_ATTR = "mortyclaw_meta"
TOOL_META_REGISTRY: dict[str, "ToolMeta"] = {}


@dataclass(frozen=True)
class ToolMeta:
    name: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    risk_level: ToolRiskLevel = "medium"
    allowed_routes: frozenset[ToolRoute] = field(default_factory=lambda: frozenset({"slow"}))
    requires_approval: bool = False

    @classmethod
    def build(
        cls,
        *,
        name: str,
        capabilities: Iterable[str] = (),
        risk_level: ToolRiskLevel = "medium",
        allowed_routes: Iterable[ToolRoute] = ("slow",),
        requires_approval: bool = False,
    ) -> "ToolMeta":
        return cls(
            name=name,
            capabilities=frozenset(str(item).strip() for item in capabilities if str(item).strip()),
            risk_level=risk_level,
            allowed_routes=frozenset(allowed_routes),
            requires_approval=requires_approval,
        )


UNKNOWN_TOOL_META = ToolMeta.build(name="unknown")


def attach_tool_meta(tool: BaseTool, meta: ToolMeta) -> BaseTool:
    object.__setattr__(tool, TOOL_META_ATTR, meta)
    if meta.name:
        TOOL_META_REGISTRY[meta.name] = meta
    return tool


def get_tool_meta(tool: BaseTool | None) -> ToolMeta:
    meta = getattr(tool, TOOL_META_ATTR, None)
    if isinstance(meta, ToolMeta):
        return meta
    name = str(getattr(tool, "name", "") or "").strip() if tool is not None else "unknown"
    return ToolMeta.build(name=name or "unknown")


def get_registered_tool_meta(name: str) -> ToolMeta | None:
    return TOOL_META_REGISTRY.get(str(name or "").strip())


def is_fast_route_safe(meta: ToolMeta) -> bool:
    blocked_capabilities = {
        "file_write",
        "file_delete",
        "shell_exec",
        "program_exec",
        "task_write",
        "memory_write",
        "subagent_write",
    }
    return (
        meta.risk_level == "low"
        and "fast" in meta.allowed_routes
        and not meta.requires_approval
        and meta.capabilities.isdisjoint(blocked_capabilities)
    )
