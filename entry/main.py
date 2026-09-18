from __future__ import annotations

import asyncio
import json
import os
import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from mortyclaw.core.config import PROJECT_ROOT
from mortyclaw.core.harness import AgentTurnRequest, HarnessRuntime
from mortyclaw.core.storage.runtime import get_session_repository, get_task_repository

console = Console()


def _sessions() -> None:
    rows = get_session_repository().list_sessions(limit=20)
    console.print("\n".join(f"- {row['thread_id']} | {row['status']} | {row['last_active_at']}" for row in rows) or "暂无会话。")


def _tasks() -> None:
    rows = get_task_repository().list_tasks(limit=20)
    console.print("\n".join(f"- {row['task_id']} | {row['status']} | {row['description']}" for row in rows) or "暂无任务。")


def _show_pending_inbox(thread_id: str) -> None:
    repository = get_session_repository()
    for event in repository.list_pending_inbox_events(thread_id, limit=20):
        try:
            payload = json.loads(event.get("payload") or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        content = str(payload.get("content") or "").strip()
        if content:
            console.print(f"[bold yellow]定时任务通知[/bold yellow]\n{content}")
        repository.mark_inbox_event_delivered(event["event_id"])


async def async_main(thread_id: str | None = None) -> None:
    runtime = HarnessRuntime()
    current = thread_id or "session-1"
    prompt = PromptSession()
    console.print("[bold cyan]MortyClaw / DeepSeek Harness[/bold cyan]  输入 /help 查看命令")
    try:
        await runtime.start()
        while True:
            _show_pending_inbox(current)
            with patch_stdout():
                text = (await prompt.prompt_async(f"{current}> ")).strip()
            if not text:
                continue
            command = text.lower()
            if command in {"/exit", "/quit"}:
                break
            if command == "/help":
                console.print("/sessions /tasks /new /reset /clear /exit")
                continue
            if command == "/sessions":
                _sessions()
                continue
            if command == "/tasks":
                _tasks()
                continue
            if command == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if command in {"/new", "/reset"}:
                if command == "/new":
                    current = f"session-{uuid.uuid4().hex[:8]}"
                await runtime.reset(current)
                console.print("[dim]已开始全新 Harness 会话；长期记忆与历史检索仍保留。[/dim]")
                continue
            try:
                result = await runtime.run_turn(AgentTurnRequest(
                    thread_id=current, turn_id=str(uuid.uuid4()), text=text,
                    source="cli", workspace=PROJECT_ROOT,
                ))
                console.print(result.final_response)
            except Exception as exc:
                console.print(f"[bold red]处理失败：[/bold red] {type(exc).__name__}: {exc}")
    finally:
        await runtime.close()


def main(thread_id: str | None = None) -> None:
    asyncio.run(async_main(thread_id=thread_id))


if __name__ == "__main__":
    main()
