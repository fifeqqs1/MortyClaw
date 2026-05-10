from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

from langchain_core.messages import HumanMessage


MAX_SUMMARY_SESSIONS = 3
MAX_SESSION_MATERIAL_CHARS = 20_000


def aggregate_session_results(results: list[dict], *, limit: int = MAX_SUMMARY_SESSIONS) -> list[dict]:
    grouped: dict[str, dict] = {}
    for result in results:
        key = str(result.get("lineage_root_thread_id") or result.get("thread_id") or "")
        if not key:
            continue
        if key not in grouped:
            clone = dict(result)
            clone["raw_hits"] = list(result.get("hits") or [])
            grouped[key] = clone
        else:
            grouped[key]["raw_hits"].extend(result.get("hits") or [])
            grouped[key]["hits"] = grouped[key]["raw_hits"][:3]
    return list(grouped.values())[: max(1, min(int(limit or MAX_SUMMARY_SESSIONS), 5))]


def build_session_summary_material(result: dict, *, max_chars: int = MAX_SESSION_MATERIAL_CHARS) -> str:
    parts = [
        f"thread_id: {result.get('thread_id', '')}",
        f"title: {result.get('title', '')}",
        f"when: {result.get('when', '')}",
        f"model: {result.get('model', '')}",
    ]
    for index, hit in enumerate(result.get("raw_hits") or result.get("hits") or [], start=1):
        parts.append(f"\n[Hit {index}] role={hit.get('role', '')} tool={hit.get('tool_name', '')}")
        snippet = str(hit.get("snippet") or hit.get("content_preview") or "").strip()
        if snippet:
            parts.append(f"snippet: {snippet}")
        tool_result = str(hit.get("tool_result_preview") or "").strip()
        if tool_result:
            parts.append(f"tool_result: {tool_result}")
        window = hit.get("window") or []
        if window:
            parts.append("window:")
            for item in window:
                role = item.get("role", "")
                tool_name = item.get("tool_name", "")
                content = item.get("content_preview", "")
                label = f"{role}:{tool_name}" if tool_name else role
                parts.append(f"- {label}: {content}")

    material = "\n".join(parts).strip()
    if len(material) <= max_chars:
        return material
    return material[: max_chars - 80] + "\n...[session recall material truncated]..."


def _invoke_summary_llm(llm, prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)], config={"callbacks": []})
    return str(getattr(response, "content", "") or "").strip()


def summarize_session_result(
    result: dict,
    *,
    query: str,
    llm_factory: Callable[[], Any] | None,
    timeout_seconds: int = 45,
) -> dict:
    updated = dict(result)
    updated.setdefault("raw_hits", list(result.get("hits") or []))
    if llm_factory is None:
        updated["summary"] = _fallback_summary(updated)
        updated["summary_status"] = "fallback_raw"
        return updated

    material = build_session_summary_material(updated)
    prompt = (
        "你正在帮助 MortyClaw 回忆历史会话。请根据下面的命中材料，用中文给出聚焦摘要。\n"
        "要求：只陈述材料中能支持的事实；保留关键文件、命令、错误、结论和未解决事项；不要编造。\n\n"
        f"检索主题：{query}\n\n"
        f"历史会话材料：\n{material}\n\n"
        "请输出 4-8 条要点。"
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: _invoke_summary_llm(llm_factory(), prompt))
    try:
        summary = future.result(timeout=max(1, int(timeout_seconds or 45)))
    except concurrent.futures.TimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        updated["summary"] = _fallback_summary(updated)
        updated["summary_status"] = "timeout"
        return updated
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        updated["summary"] = _fallback_summary(updated)
        updated["summary_status"] = "error"
        updated["summary_error"] = str(exc)
        return updated
    finally:
        if future.done():
            executor.shutdown(wait=False, cancel_futures=True)

    if not summary:
        updated["summary"] = _fallback_summary(updated)
        updated["summary_status"] = "fallback_raw"
        return updated
    updated["summary"] = summary
    updated["summary_status"] = "generated"
    return updated


def summarize_session_results(
    results: list[dict],
    *,
    query: str,
    llm_factory: Callable[[], Any] | None,
    timeout_seconds: int = 45,
    limit: int = MAX_SUMMARY_SESSIONS,
) -> list[dict]:
    aggregated = aggregate_session_results(results, limit=limit)
    return [
        summarize_session_result(
            result,
            query=query,
            llm_factory=llm_factory,
            timeout_seconds=timeout_seconds,
        )
        for result in aggregated
    ]


def _fallback_summary(result: dict) -> str:
    hits = result.get("raw_hits") or result.get("hits") or []
    previews = []
    for hit in hits[:3]:
        text = str(hit.get("snippet") or hit.get("content_preview") or "").strip()
        if text:
            previews.append(text)
    if not previews:
        return "未能生成摘要；没有可用命中预览。"
    return "未能生成模型摘要，以下是原始命中预览：\n" + "\n".join(f"- {item}" for item in previews)
