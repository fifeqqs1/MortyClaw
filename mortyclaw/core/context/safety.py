from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"ignore.{0,40}(previous|above|earlier).{0,30}(instruction|prompt|message)",
        "prompt-override",
    ),
    (
        r"(system prompt|developer message|hidden instruction|tool call)",
        "system-reference",
    ),
    (
        r"(reveal|print|dump|exfiltrate|leak).{0,40}(secret|token|api key|credential|system prompt)",
        "secret-exfiltration",
    ),
    (
        r"(jailbreak|bypass|override|disable).{0,40}(guard|policy|safety|rule)",
        "guard-bypass",
    ),
    (
        r"<\s*/?\s*(system|developer|assistant|tool)\s*>",
        "role-spoofing",
    ),
)

_BIDI_OVERRIDE_NAMES = {
    "LEFT-TO-RIGHT EMBEDDING",
    "RIGHT-TO-LEFT EMBEDDING",
    "LEFT-TO-RIGHT OVERRIDE",
    "RIGHT-TO-LEFT OVERRIDE",
    "LEFT-TO-RIGHT ISOLATE",
    "RIGHT-TO-LEFT ISOLATE",
    "FIRST STRONG ISOLATE",
    "POP DIRECTIONAL ISOLATE",
    "POP DIRECTIONAL FORMATTING",
}


@dataclass(frozen=True)
class SanitizedContext:
    source: str
    text: str
    issues: tuple[str, ...] = ()
    removed_hidden_chars: int = 0
    truncated: bool = False
    blocked: bool = False


def _is_hidden_char(ch: str) -> bool:
    if ch in {"\n", "\r", "\t"}:
        return False
    if ch == "\ufeff":
        return True
    category = unicodedata.category(ch)
    if category in {"Cf", "Cc", "Cs"}:
        return True
    return unicodedata.name(ch, "") in _BIDI_OVERRIDE_NAMES


def _strip_hidden_chars(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    for ch in text:
        if _is_hidden_char(ch):
            removed += 1
            continue
        kept.append(ch)
    return "".join(kept), removed


def scan_context_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    issues: list[str] = []
    for pattern, label in _PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            issues.append(label)
    return issues


def sanitize_context_text(
    text: str,
    *,
    source: str,
    max_chars: int | None = None,
    block_on_threats: bool = True,
) -> SanitizedContext:
    normalized = str(text or "").strip()
    if not normalized:
        return SanitizedContext(source=source, text="")

    stripped, removed_hidden_chars = _strip_hidden_chars(normalized)
    truncated = False
    if max_chars is not None and max_chars > 0 and len(stripped) > max_chars:
        stripped = stripped[:max_chars].rstrip()
        truncated = True

    issues = scan_context_text(stripped)
    if removed_hidden_chars:
        issues.append("hidden-unicode")
    if truncated:
        issues.append("truncated")

    deduped = tuple(dict.fromkeys(issues))
    blocked = bool(block_on_threats and any(issue not in {"hidden-unicode", "truncated"} for issue in deduped))
    return SanitizedContext(
        source=source,
        text=stripped,
        issues=deduped,
        removed_hidden_chars=removed_hidden_chars,
        truncated=truncated,
        blocked=blocked,
    )
