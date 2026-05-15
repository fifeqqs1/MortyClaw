import json
import os

from langchain_core.messages import HumanMessage


ROUTE_CLASSIFIER_MODEL = "qwen3.5-flash"
ROUTE_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7


def get_route_classifier_model() -> str:
    return (
        os.getenv("ROUTE_CLASSIFIER_MODEL", "").strip()
        or os.getenv("DEFAULT_MODEL", "").strip()
        or ROUTE_CLASSIFIER_MODEL
    )


def _extract_json_object(text: str) -> dict | None:
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start:end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def classify_route_with_llm(query: str, llm) -> dict | None:
    if not query or llm is None:
        return None

    classifier_prompt = f"""
你是 MortyClaw 的任务路由器。请判断下面的请求应该进入哪种执行模式。

可选模式：
1. fast
适合：
- 简单问答
- 代码片段讲解
- 不需要访问项目文件、不需要工具链执行的问题
- 能力确认、权限确认、流程确认、边界确认这类元问题，例如“你可以写代码吗”“你支持运行测试吗”“现在是不是只读模式”

2. slow + autonomous
适合：
- 目标明确
- 可以边读文件边执行
- 可以边修改边验证
- 用户已经给出明确项目路径和明确要求
- 即使有多个步骤，但步骤之间线性清楚
- 即使需要写文件、运行命令、执行测试，只要任务目标明确、范围收敛、执行路径清楚，仍优先 autonomous
- 即使涉及多个文件或多个模块，只要不是全项目级架构重构，也应优先选择 autonomous

默认优先选择 autonomous。

3. slow + structured
适合：
- 用户目标模糊，需要先拆解
- 任务范围非常大，涉及全项目架构分析、全局重构、模块体系重新设计
- 多模块、多文件、大重构
- 需要先形成结构化计划再执行
- 代理必须先决定“要做什么”，而不是已经清楚“怎么开始做”

只有任务确实十分复杂、范围大、目标不清晰，才选择 structured。

只输出一个 JSON 对象，不要输出 Markdown，不要补充解释：
{{
  "route": "fast" 或 "slow",
  "slow_execution_mode": "autonomous" 或 "structured"，如果 route=fast 则填空字符串 "",
  "reason": "一句简短原因"
}}

用户请求：
{query}
""".strip()

    try:
        response = llm.invoke([HumanMessage(content=classifier_prompt)], config={"callbacks": []})
    except TypeError:
        try:
            response = llm.invoke([HumanMessage(content=classifier_prompt)])
        except Exception:
            return None
    except Exception:
        return None

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    payload = _extract_json_object(content if isinstance(content, str) else str(content))
    if payload is None:
        return None

    route = str(payload.get("route", "")).strip().lower()
    if route not in {"fast", "slow"}:
        return None
    slow_execution_mode = str(payload.get("slow_execution_mode", "")).strip().lower()
    if route == "fast":
        slow_execution_mode = ""
    elif slow_execution_mode not in {"autonomous", "structured"}:
        slow_execution_mode = ""

    reason = str(payload.get("reason", "")).strip()
    return {
        "route": route,
        "slow_execution_mode": slow_execution_mode,
        "reason": reason,
    }
