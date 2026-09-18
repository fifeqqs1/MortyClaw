from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from ..config import PROJECT_ROOT


def _integer(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)) or default))
    except ValueError:
        return default


@dataclass(frozen=True)
class HarnessSettings:
    api_key: str
    base_url: str | None
    model: str
    profile: str
    home: Path
    max_tokens: int
    timeout_seconds: int
    max_concurrency: int
    approval_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "HarnessSettings":
        load_dotenv()
        base = (os.getenv("DEEPSEEK_BASE_URL") or "").strip()
        if base.rstrip("/") == "https://api.deepseek.com":
            base = ""
        old_base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").strip()
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key and "deepseek" in old_base.lower():
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not base and old_base and "deepseek" in old_base.lower() and old_base.rstrip("/") != "https://api.deepseek.com":
            base = old_base
        model = (os.getenv("MORTYCLAW_HARNESS_MODEL") or "deepseek-v4-flash").strip()
        if model == "deepseek-flash":
            model = "deepseek-v4-flash"
        home = Path(os.getenv("MORTYCLAW_HARNESS_HOME", "workspace/harness-home")).expanduser()
        if not home.is_absolute():
            home = Path(PROJECT_ROOT) / home
        return cls(
            api_key=api_key,
            base_url=base or None,
            model=model,
            profile=(os.getenv("MORTYCLAW_HARNESS_PROFILE") or "mortyclaw-sdk").strip(),
            home=home.resolve(),
            max_tokens=_integer("MORTYCLAW_HARNESS_MAX_TOKENS", 32768, 1024),
            timeout_seconds=_integer("MORTYCLAW_HARNESS_TURN_TIMEOUT_SECONDS", 600, 10),
            max_concurrency=_integer("MORTYCLAW_HARNESS_MAX_CONCURRENCY", 4),
            approval_ttl_seconds=_integer("MORTYCLAW_APPROVAL_TTL_SECONDS", 900, 60),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请执行 mortyclaw harness configure。")

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "deepseek-official", "model": self.model, "profile": self.profile,
            "home": str(self.home), "base_url": "custom" if self.base_url else "official",
            "api_key_configured": bool(self.api_key), "timeout_seconds": self.timeout_seconds,
            "max_concurrency": self.max_concurrency,
        }
