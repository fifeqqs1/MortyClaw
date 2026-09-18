from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

from ..config import PROJECT_ROOT
from ..harness import AgentTurnRequest, HarnessRuntime
from ..storage.runtime import get_session_repository


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    fallback = "1" if default else "0"
    return os.getenv(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FeishuBotSettings:
    app_id: str
    app_secret: str
    domain: str = "https://open.feishu.cn"
    dm_enabled: bool = True
    group_enabled: bool = True
    require_mention: bool = True
    max_reply_chars: int = 3500

    @classmethod
    def from_env(cls) -> "FeishuBotSettings":
        try:
            max_reply_chars = int(os.getenv("FEISHU_BOT_MAX_REPLY_CHARS", "3500") or 3500)
        except ValueError:
            max_reply_chars = 3500
        return cls(
            app_id=os.getenv("FEISHU_APP_ID", "").strip(),
            app_secret=os.getenv("FEISHU_APP_SECRET", "").strip(),
            domain=os.getenv("FEISHU_MCP_DOMAIN", "https://open.feishu.cn").strip()
            or "https://open.feishu.cn",
            dm_enabled=_env_flag("FEISHU_BOT_DM_ENABLED", True),
            group_enabled=_env_flag("FEISHU_BOT_GROUP_ENABLED", True),
            require_mention=_env_flag("FEISHU_BOT_REQUIRE_MENTION", True),
            max_reply_chars=max(500, min(max_reply_chars, 10000)),
        )

    def validate(self) -> None:
        if not self.app_id or not self.app_secret:
            raise RuntimeError(
                "缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET，请先执行 mortyclaw feishu-config。"
            )


def feishu_thread_id(chat_id: str) -> str:
    """Build a stable local thread id without persisting the raw Feishu chat id."""
    digest = hashlib.sha256(str(chat_id or "").encode("utf-8")).hexdigest()[:20]
    return f"feishu-{digest}"


def split_feishu_reply(text: str, max_chars: int = 3500) -> list[str]:
    """Split a reply on paragraph boundaries while respecting Feishu limits."""
    normalized = str(text or "").strip()
    if not normalized:
        return []
    limit = max(1, int(max_chars))
    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


class MortyClawFeishuRuntime:
    """Translate Feishu messages into DeepSeek Harness turns."""

    def __init__(self) -> None:
        self._runtime = HarnessRuntime()
        self._started = False
        self._chat_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        if self._started:
            return
        await self._runtime.start()
        self._started = True

    async def close(self) -> None:
        await self._runtime.close()
        self._started = False

    async def handle_text(self, *, chat_id: str, text: str) -> str:
        await self.start()
        normalized = str(text or "").strip()
        thread_id = feishu_thread_id(chat_id)
        lock = self._chat_locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            notices = self._drain_inbox(thread_id)
            if normalized.lower() in {"/help", "帮助"}:
                reply = (
                    "我是 MortyClaw。直接发送问题即可继续对话；发送 /reset 可清空当前飞书会话的上下文。"
                )
                return "\n\n".join([*notices, reply])
            if normalized.lower() in {"/reset", "/new"}:
                await self._runtime.reset(thread_id)
                return "\n\n".join([*notices, "当前飞书会话的上下文已清空，我们可以重新开始。"])
            if not normalized:
                return "\n\n".join([*notices, "我在。请直接告诉我你需要处理什么。"])
            reply = await self._run_turn(thread_id=thread_id, user_input=normalized)
            return "\n\n".join([*notices, reply])

    @staticmethod
    def _drain_inbox(thread_id: str) -> list[str]:
        repository = get_session_repository()
        notices: list[str] = []
        for event in repository.list_pending_inbox_events(thread_id, limit=20):
            try:
                payload = json.loads(event.get("payload") or "{}")
            except (TypeError, json.JSONDecodeError):
                payload = {}
            content = str(payload.get("content") or "").strip()
            if content:
                notices.append(f"【定时任务通知】\n{content}")
            repository.mark_inbox_event_delivered(event["event_id"])
        return notices

    async def _run_turn(self, *, thread_id: str, user_input: str) -> str:
        turn_id = str(uuid.uuid4())
        result = await self._runtime.run_turn(AgentTurnRequest(
            thread_id=thread_id, turn_id=turn_id, text=user_input,
            source="feishu", workspace=PROJECT_ROOT,
        ))
        return result.final_response


def build_feishu_channel(settings: FeishuBotSettings):
    settings.validate()
    try:
        from lark_channel import (
            FeishuChannel,
            InboundConfig,
            LogLevel,
            PolicyConfig,
            SecurityConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少 lark-channel-sdk，请在项目虚拟环境中重新执行 pip install -e .。"
        ) from exc

    channel = FeishuChannel(
        app_id=settings.app_id,
        app_secret=settings.app_secret,
        domain=settings.domain,
        log_level=LogLevel.WARNING,
        policy=PolicyConfig(
            dm_policy="open" if settings.dm_enabled else "disabled",
            group_policy="open" if settings.group_enabled else "disabled",
            require_mention=settings.require_mention,
            respond_to_mention_all=False,
        ),
        inbound=InboundConfig(emit_raw_events=True),
        security=SecurityConfig(mode="strict"),
    )
    _install_ws_thread_loop_compat(channel)
    return channel


def _install_ws_thread_loop_compat(channel: Any) -> None:
    """Give the SDK's blocking WS worker its own loop.

    lark-channel-sdk 1.4.0 captures an event loop when its WS module is
    imported. ``start_background`` then runs the blocking client in another
    thread, which fails on Python 3.12 if the captured loop is already running.
    Keep this workaround local to the channel's start worker.
    """
    original_start = channel.start

    def start_with_thread_loop() -> None:
        from lark_channel.ws import client as ws_client_module

        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        ws_client_module.loop = thread_loop
        try:
            original_start()
        finally:
            if not thread_loop.is_running() and not thread_loop.is_closed():
                pending = asyncio.all_tasks(thread_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    thread_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                thread_loop.close()

    channel.start = start_with_thread_loop


async def serve_feishu_bot(
    *,
    settings: FeishuBotSettings | None = None,
    ready_callback=None,
) -> None:
    resolved = settings or FeishuBotSettings.from_env()
    resolved.validate()
    channel = build_feishu_channel(resolved)
    runtime = MortyClawFeishuRuntime()

    async def on_message(message) -> None:
        chat_id = str(getattr(message, "chat_id", "") or "").strip()
        message_id = str(getattr(message, "message_id", "") or "").strip()
        body_text = str(getattr(message, "body_text", "") or "").strip()
        content_text = str(getattr(message, "content_text", "") or "").strip()
        if not chat_id or not message_id:
            return
        print(f"[Feishu Bot] 收到消息 message_id={message_id}，开始交给 MortyClaw 处理。", flush=True)
        try:
            reply = await runtime.handle_text(chat_id=chat_id, text=body_text or content_text)
            chunks = split_feishu_reply(reply, resolved.max_reply_chars)
            for index, chunk in enumerate(chunks):
                opts = {"reply_to": message_id} if index == 0 else None
                result = await channel.send(chat_id, {"text": chunk}, opts)
                if not getattr(result, "success", True):
                    logger.error("飞书消息发送失败：%s", getattr(result, "error", "unknown error"))
                    break
            print(f"[Feishu Bot] 已完成回复 message_id={message_id}。", flush=True)
        except Exception:
            logger.exception("处理飞书消息失败")
            try:
                await channel.send(
                    chat_id,
                    {"text": "MortyClaw 处理这条消息时遇到了异常，请稍后重试。"},
                    {"reply_to": message_id},
                )
            except Exception:
                logger.exception("发送飞书错误提示失败")

    async def on_error(error) -> None:
        logger.error("飞书通道异常：%s", error)

    async def on_raw_event(event) -> None:
        header = event.get("header", {}) if isinstance(event, dict) else {}
        event_type = str(header.get("event_type", "") or "unknown")
        event_id = str(header.get("event_id", "") or "unknown")
        print(f"[Feishu Event] 收到原始事件 type={event_type} event_id={event_id}。", flush=True)

    async def on_reject(event) -> None:
        print(
            "[Feishu Event] 消息被策略过滤 "
            f"message_id={getattr(event, 'message_id', 'unknown')} "
            f"reason={getattr(event, 'reason', 'unknown')}。",
            flush=True,
        )

    channel.on("message", on_message)
    channel.on("error", on_error)
    channel.on("raw", on_raw_event)
    channel.on("reject", on_reject)

    await channel.start_background(timeout=30)
    runtime_start = channel.schedule(runtime.start())
    await asyncio.wrap_future(runtime_start)
    if ready_callback is not None:
        ready_callback()

    try:
        await asyncio.Event().wait()
    finally:
        try:
            runtime_close = channel.schedule(runtime.close())
            await asyncio.wrap_future(runtime_close)
        finally:
            await channel.disconnect()


__all__ = [
    "FeishuBotSettings",
    "MortyClawFeishuRuntime",
    "build_feishu_channel",
    "feishu_thread_id",
    "serve_feishu_bot",
    "split_feishu_reply",
]
