from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySafetyResult:
    ok: bool
    reason: str = ""
    rule_id: str = ""


_THREAT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"system\s+prompt\s+override", "system_prompt_override"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)", "bypass_restrictions"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"authorized_keys", "ssh_backdoor"),
    (r"\$HOME/\.ssh|~/\.ssh", "ssh_access"),
)

_INVISIBLE_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
}


def scan_memory_content(content: str) -> MemorySafetyResult:
    text = str(content or "")
    for char in _INVISIBLE_CHARS:
        if char in text:
            return MemorySafetyResult(
                ok=False,
                rule_id="invisible_unicode",
                reason=f"记忆内容包含不可见 Unicode 字符 U+{ord(char):04X}，疑似 prompt injection。",
            )

    for pattern, rule_id in _THREAT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return MemorySafetyResult(
                ok=False,
                rule_id=rule_id,
                reason=f"记忆内容命中高危规则 `{rule_id}`，已拒绝写入。",
            )

    return MemorySafetyResult(ok=True)
