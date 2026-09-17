import json
import os
import re
from urllib import error, request

from .base import mortyclaw_tool


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MORTYCLAW_PASSTHROUGH_FLAG = "_mortyclaw_passthrough"


def _compact_text(value: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@mortyclaw_tool
def tavily_web_search(
    query: str,
    topic: str = "general",
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = True,
) -> str:
    """
    使用 Tavily 联网搜索最新网页信息。
    适合处理以下场景：
    1. 用户明确要求联网、搜索、查资料、找来源。
    2. 问题涉及新闻、实时信息、最新动态、当前版本、外部网页内容。
    3. 回答时需要附带来源链接。

    参数说明：
    - query: 搜索关键词或完整问题。
    - topic: 搜索主题，推荐使用 "general"；如果是新闻/时事，使用 "news"。
    - search_depth: "basic" 或 "advanced"。普通查询用 basic，需要更深入检索时用 advanced。
    - max_results: 返回结果数量，建议 3 到 8。
    - include_answer: 是否让 Tavily 返回一个搜索摘要。
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "Tavily 搜索不可用：未配置 TAVILY_API_KEY 环境变量。"

    query = (query or "").strip()
    if not query:
        return "Tavily 搜索参数错误：query 不能为空。"

    topic = (topic or "general").strip().lower()
    if topic not in {"general", "news"}:
        return "Tavily 搜索参数错误：topic 只能是 'general' 或 'news'。"

    search_depth = (search_depth or "basic").strip().lower()
    if search_depth not in {"basic", "advanced"}:
        return "Tavily 搜索参数错误：search_depth 只能是 'basic' 或 'advanced'。"

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return "Tavily 搜索参数错误：max_results 必须是整数。"

    max_results = max(1, min(max_results, 10))

    payload = {
        "query": query,
        "topic": topic,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": bool(include_answer),
        "include_raw_content": False,
        "include_images": False,
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
            parsed = json.loads(detail) if detail else {}
            message = parsed.get("detail") or parsed.get("error") or detail
        except Exception:
            message = str(exc)
        return f"Tavily 搜索失败：HTTP {exc.code}，{_compact_text(message, 200)}"
    except error.URLError as exc:
        return f"Tavily 搜索失败：网络错误，{exc.reason}"
    except Exception as exc:
        return f"Tavily 搜索失败：{exc}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "Tavily 搜索失败：返回结果不是合法 JSON。"

    answer = _compact_text(data.get("answer", ""), 500)
    results = data.get("results") or []

    if not results and not answer:
        return "Tavily 未返回任何搜索结果。"

    lines = [
        f"Tavily 搜索完成：query={query}",
        f"topic={topic}, search_depth={search_depth}, max_results={max_results}",
    ]

    if answer:
        lines.append(f"搜索摘要：{answer}")

    if not results:
        lines.append("来源链接：无")
        return "\n".join(lines)

    lines.append("来源结果：")
    for index, item in enumerate(results, start=1):
        title = _compact_text(item.get("title", "(无标题)"), 160)
        url = item.get("url", "")
        content = _compact_text(item.get("content", ""), 300)
        score = item.get("score")

        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if content:
            lines.append(f"   摘要: {content}")
        if isinstance(score, (int, float)):
            lines.append(f"   相关度: {score:.3f}")

    return "\n".join(lines)
