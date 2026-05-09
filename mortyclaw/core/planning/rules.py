from __future__ import annotations

import re

from ..routing import SLOW_PATH_HIGH_RISK_HINTS, contains_query_hint


FILE_WRITE_ACTION_HINTS = (
    "生成",
    "创建",
    "新建",
    "实现",
    "开发",
    "落地",
    "搭建",
    "写",
    "编写",
    "改",
    "修改",
    "修复",
    "保存",
    "输出",
    "放到",
    "放在",
    "patch",
    "edit",
    "fix",
    "generate",
    "create",
    "write",
    "save",
)

FILE_WRITE_TARGET_PATTERN = re.compile(
    r"\b[\w./-]+\.(?:py|js|ts|tsx|jsx|md|txt|json|ya?ml|sh|csv|html|css)\b",
    re.IGNORECASE,
)

SCRIPT_EXECUTION_PATTERN = re.compile(
    r"\b(?:python(?:3(?:\.\d+)?)?|uv run python|node|bash|sh)\s+[A-Za-z0-9_./-]+",
    re.IGNORECASE,
)

SHELL_STEP_HINTS = (
    "运行",
    "执行",
    "shell",
    "命令",
    "python ",
    "bash ",
    "powershell",
    "command",
    "run ",
    "execute",
)

TEST_STEP_HINTS = (
    "测试",
    "验证",
    "检查",
    "回归",
    "pytest",
    "unittest",
    "ruff",
    "mypy",
    "test",
    "tests",
    "verify",
    "check",
)

TASK_CREATE_HINTS = (
    "提醒",
    "闹钟",
    "定时",
    "任务",
    "schedule",
    "alarm",
    "remind",
)

TASK_MODIFY_HINTS = (
    "修改任务",
    "调整任务",
    "改一下任务",
    "reschedule",
    "modify task",
)

TASK_DELETE_HINTS = (
    "删除任务",
    "取消任务",
    "移除任务",
    "delete task",
    "cancel task",
)

SKILL_STEP_HINTS = (
    "技能",
    "skill",
    "插件",
    "plugin",
)

ANALYZE_STEP_HINTS = (
    "分析",
    "解读",
    "review",
    "审查",
    "梳理",
    "定位",
    "排查",
    "对比",
    "比较",
)

READ_STEP_HINTS = (
    "读取",
    "查看",
    "检查",
    "打开",
    "搜索",
    "read",
    "inspect",
    "search",
    "look at",
)

SUMMARY_STEP_HINTS = (
    "总结",
    "归纳",
    "提炼",
    "摘要",
    "summarize",
    "summary",
)

PAPER_RESEARCH_STEP_HINTS = (
    "论文",
    "paper",
    "papers",
    "arxiv",
    "文献",
    "研究方法",
    "method",
)

REPORT_STEP_HINTS = (
    "建议",
    "结论",
    "输出报告",
    "报告",
    "给出建议",
    "改进意见",
    "优化建议",
    "report",
    "recommend",
    "recommendation",
)

RUNTIME_EXECUTION_PHRASES = (
    "运行脚本",
    "执行脚本",
    "运行这个脚本",
    "执行这个脚本",
    "运行该脚本",
    "执行该脚本",
    "运行命令",
    "执行命令",
    "run script",
    "execute script",
)

VERIFICATION_RESULT_PHRASES = (
    "验证输出",
    "检查输出",
    "验证结果",
    "检查结果",
    "测试结果",
    "回归结果",
    "确认通过或失败原因",
    "verify output",
    "check output",
    "validate result",
)

VALID_STEP_INTENTS = {
    "analyze",
    "read",
    "paper_research",
    "code_edit",
    "file_write",
    "shell_execute",
    "test_verify",
    "summarize",
    "report",
}
VALID_EXECUTION_MODES = {"structured", "programmatic", "delegated"}
PROGRAMMATIC_EXECUTION_HINTS = (
    "批量",
    "遍历",
    "循环",
    "逐个",
    "多个文件",
    "读改测",
    "筛选",
    "收集结果",
    "batch",
    "loop",
    "iterate",
    "multiple files",
)
DELEGATED_EXECUTION_HINTS = (
    "并行",
    "子任务",
    "分头",
    "分别处理",
    "worker",
    "delegate",
    "parallel",
)
DELEGATED_INDEPENDENCE_HINTS = (
    "互不冲突",
    "独立",
    "各自",
    "分别负责",
    "拆成两部分",
    "拆成两个",
    "两路",
    "双线",
)
DELEGABLE_TARGET_HINTS = (
    "模块",
    "子系统",
    "服务",
    "目录",
    "仓库",
    "repo",
    "repository",
    "前端",
    "后端",
    "frontend",
    "backend",
    "api",
    "ui",
    "组件",
    "页面",
    "包",
    "测试",
    "日志",
    "文件",
)
DELEGABLE_ACTION_HINTS = (
    "分析",
    "检查",
    "审查",
    "review",
    "排查",
    "定位",
    "修复",
    "实现",
    "重构",
    "验证",
    "对比",
    "比较",
    "总结",
)
MULTI_TARGET_CONNECTOR_PATTERN = re.compile(
    r"(?:前端|后端|frontend|backend|api|ui|客户端|服务端|server|client|模块|目录|服务|组件|页面|包|测试|日志|文件|repo|repository|仓库)"
    r"[^。；;,\n]{0,24}(?:和|与|以及|及|/|、)[^。；;,\n]{0,24}"
    r"(?:前端|后端|frontend|backend|api|ui|客户端|服务端|server|client|模块|目录|服务|组件|页面|包|测试|日志|文件|repo|repository|仓库)",
    re.IGNORECASE,
)
MULTI_TARGET_QUANTITY_PATTERN = re.compile(
    r"(?:两|两个|两类|两部分|两组|多个|多处|多组|multi(?:ple)?)"
    r"[^。；;,\n]{0,12}"
    r"(?:模块|子系统|服务|目录|仓库|repo|repository|前端|后端|frontend|backend|api|ui|组件|页面|包|测试|日志|文件)",
    re.IGNORECASE,
)


def looks_like_file_write_request(step_text: str) -> bool:
    lowered = step_text.lower()
    has_action = any(hint in step_text or hint in lowered for hint in FILE_WRITE_ACTION_HINTS)
    has_target = bool(FILE_WRITE_TARGET_PATTERN.search(step_text)) or any(
        marker in step_text for marker in ("文件", "脚本", "代码", "函数", "模块", "bug", "BUG")
    )
    return has_action and has_target


def step_matches_shell_action(step_text: str) -> bool:
    lowered = step_text.lower()
    if any(marker in step_text or marker in lowered for marker in ("命令行工具", "cli 工具", "cli tool", "command line tool")):
        return False
    return contains_query_hint(step_text, lowered, SHELL_STEP_HINTS)


def step_matches_test_action(step_text: str) -> bool:
    lowered = step_text.lower()
    return contains_query_hint(step_text, lowered, TEST_STEP_HINTS)


def step_matches_task_creation(step_text: str) -> bool:
    lowered = step_text.lower()
    return contains_query_hint(step_text, lowered, TASK_CREATE_HINTS)


def step_matches_task_modification(step_text: str) -> bool:
    lowered = step_text.lower()
    return contains_query_hint(step_text, lowered, TASK_MODIFY_HINTS)


def step_matches_task_deletion(step_text: str) -> bool:
    lowered = step_text.lower()
    return contains_query_hint(step_text, lowered, TASK_DELETE_HINTS)


def step_matches_skill_usage(step_text: str) -> bool:
    lowered = step_text.lower()
    return contains_query_hint(step_text, lowered, SKILL_STEP_HINTS)


def looks_like_runtime_execution_step(step_text: str) -> bool:
    lowered = (step_text or "").lower()
    if not lowered:
        return False
    if SCRIPT_EXECUTION_PATTERN.search(step_text):
        return True
    if contains_query_hint(step_text, lowered, RUNTIME_EXECUTION_PHRASES):
        return True
    has_run_action = any(marker in step_text or marker in lowered for marker in ("运行", "执行", "run ", "execute"))
    has_script_target = any(
        marker in step_text or marker in lowered
        for marker in ("脚本", "命令", "command", ".py", "python ", "python3", "bash ", "node ")
    )
    return has_run_action and has_script_target


def looks_like_runtime_verification_step(step_text: str) -> bool:
    lowered = (step_text or "").lower()
    if not lowered:
        return False
    if contains_query_hint(step_text, lowered, VERIFICATION_RESULT_PHRASES):
        return True
    has_test_action = any(
        marker in step_text or marker in lowered
        for marker in ("测试", "验证", "回归", "pytest", "unittest", "ruff", "mypy", "test", "verify")
    )
    has_result_target = any(
        marker in step_text or marker in lowered
        for marker in ("输出", "结果", "通过", "失败", "报错", "日志", "output", "result")
    )
    return has_test_action and has_result_target


def infer_step_intent(step_text: str) -> str:
    lowered = (step_text or "").lower()
    if step_matches_test_action(step_text):
        return "test_verify"
    if step_matches_shell_action(step_text):
        return "shell_execute"
    if contains_query_hint(step_text, lowered, PAPER_RESEARCH_STEP_HINTS):
        return "paper_research"
    if looks_like_file_write_request(step_text):
        return _derive_write_intent(step_text)
    if contains_query_hint(step_text, lowered, READ_STEP_HINTS):
        return "read"
    if contains_query_hint(step_text, lowered, ANALYZE_STEP_HINTS):
        return "analyze"
    if contains_query_hint(step_text, lowered, REPORT_STEP_HINTS):
        return "report"
    if contains_query_hint(step_text, lowered, SUMMARY_STEP_HINTS):
        return "summarize"
    return "analyze"


def _derive_write_intent(step_text: str) -> str:
    lowered = (step_text or "").lower()
    if any(marker in step_text or marker in lowered for marker in ("代码", "函数", "模块", "patch", "edit", "fix", ".py", ".js", ".ts", ".tsx", ".jsx")):
        return "code_edit"
    return "file_write"


def normalize_intent(raw_intent: str | None, step_text: str) -> str:
    normalized = str(raw_intent or "").strip().lower()
    if normalized not in VALID_STEP_INTENTS:
        return infer_step_intent(step_text)

    # LLM-provided intent wins by default. We only correct clearly conflicting
    # labels that would otherwise hide required write/execute/test tools.
    if normalized in {"analyze", "read", "summarize", "report"}:
        if looks_like_runtime_execution_step(step_text):
            return "shell_execute"
        if looks_like_runtime_verification_step(step_text):
            return "test_verify"
        if looks_like_file_write_request(step_text):
            return _derive_write_intent(step_text)

    return normalized


def _looks_like_programmatic_execution(step_text: str, intent: str) -> bool:
    lowered = (step_text or "").lower()
    if any(marker in step_text or marker in lowered for marker in PROGRAMMATIC_EXECUTION_HINTS):
        return True
    if intent in {"code_edit", "file_write"} and ("验证" in step_text or "test" in lowered or "diff" in lowered):
        return True
    if intent in {"shell_execute", "test_verify"} and ("然后" in step_text or "并" in step_text or "collect" in lowered):
        return True
    return False


def _looks_like_structural_delegation(step_text: str, intent: str) -> bool:
    lowered = (step_text or "").lower()
    if not lowered:
        return False
    has_action = any(marker in step_text or marker in lowered for marker in DELEGABLE_ACTION_HINTS)
    if not has_action:
        return False

    has_independence = any(marker in step_text or marker in lowered for marker in DELEGATED_INDEPENDENCE_HINTS)
    has_multi_target_shape = bool(
        MULTI_TARGET_CONNECTOR_PATTERN.search(step_text)
        or MULTI_TARGET_QUANTITY_PATTERN.search(step_text)
    )
    has_delegable_target = any(marker in step_text or marker in lowered for marker in DELEGABLE_TARGET_HINTS)

    if not ((has_independence and has_delegable_target) or has_multi_target_shape):
        return False

    # Keep obviously mechanical batch work on the programmatic path unless
    # the step also carries a strong independence signal.
    if _looks_like_programmatic_execution(step_text, intent) and not has_independence:
        return False

    return True


def _has_explicit_delegation_signal(step_text: str) -> bool:
    lowered = (step_text or "").lower()
    return any(marker in step_text or marker in lowered for marker in DELEGATED_EXECUTION_HINTS)


def infer_execution_mode(step_text: str, intent: str) -> str:
    if _has_explicit_delegation_signal(step_text):
        return "delegated"
    if _looks_like_structural_delegation(step_text, intent):
        return "delegated"
    if _looks_like_programmatic_execution(step_text, intent):
        return "programmatic"
    return "structured"


def normalize_execution_mode(raw_execution_mode: str | None, step_text: str, intent: str) -> str:
    normalized = str(raw_execution_mode or "").strip().lower()
    if normalized not in VALID_EXECUTION_MODES:
        return infer_execution_mode(step_text, intent)

    # LLM-provided execution_mode wins by default. We only apply narrow
    # guardrails when the chosen mode is obviously unsafe or conflicts with
    # a clearly mechanical batch/loop workflow.
    if normalized == "delegated":
        has_delegation_signal = _has_explicit_delegation_signal(step_text) or _looks_like_structural_delegation(step_text, intent)
        if _looks_like_programmatic_execution(step_text, intent) and not has_delegation_signal:
            return "programmatic"

    return normalized


def classify_step_risk(step_text: str, default_risk_level: str, *, total_steps: int) -> str:
    step_lowered = step_text.lower()
    if contains_query_hint(step_text, step_lowered, SLOW_PATH_HIGH_RISK_HINTS) or looks_like_file_write_request(step_text):
        return "high"

    if default_risk_level == "high":
        return "high" if total_steps <= 1 else "medium"

    return default_risk_level


def _default_success_criteria(step_text: str, intent: str) -> str:
    if intent == "test_verify":
        return "得到明确的验证结果，并确认通过或失败原因。"
    if intent == "shell_execute":
        return "完成命令执行，并拿到可用于下一步判断的输出。"
    if intent == "paper_research":
        return "提炼出与当前任务相关的论文方法、结论和可引用依据。"
    if intent in {"code_edit", "file_write"}:
        return "完成目标修改，并保留可验证的变更结果。"
    if intent == "report":
        return "形成清晰的结论和建议。"
    if intent == "summarize":
        return "提炼出关键信息并总结给用户。"
    if intent == "read":
        return "读取并提取与目标最相关的信息。"
    return f"完成当前步骤：{step_text}"


def _default_verification_hint(step_text: str, intent: str) -> str:
    if intent == "test_verify":
        return "记录测试命令、通过/失败状态和关键报错。"
    if intent == "paper_research":
        return "记录 arxiv_rag_ask 返回的关键方法点、证据片段或论文结论。"
    if intent in {"code_edit", "file_write"}:
        return "查看 diff，并在可能时运行相关验证命令。"
    if intent == "shell_execute":
        return "记录命令、退出状态和关键输出。"
    if intent in {"read", "analyze"}:
        return "引用关键文件、函数、模块或搜索结果作为依据。"
    if intent in {"summarize", "report"}:
        return "确保输出与前面步骤结果一致，不夸大完成度。"
    return "给出可核对的结果依据。"


def normalize_plan_steps(
    steps: list[dict] | None,
    *,
    fallback_risk_level: str,
) -> list[dict]:
    normalized_steps: list[dict] = []
    for index, raw_step in enumerate(steps or [], start=1):
        if not isinstance(raw_step, dict):
            continue
        description = str(raw_step.get("description") or "").strip()
        if not description:
            continue
        intent = normalize_intent(raw_step.get("intent"), description)
        risk_level = str(raw_step.get("risk_level") or "").strip().lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = classify_step_risk(
                description,
                fallback_risk_level,
                total_steps=max(1, len(steps or [])),
            )
        normalized_steps.append({
            "step": index,
            "description": description,
            "status": "pending",
            "risk_level": risk_level,
            "intent": intent,
            "execution_mode": normalize_execution_mode(raw_step.get("execution_mode"), description, intent),
            "success_criteria": str(raw_step.get("success_criteria") or _default_success_criteria(description, intent)).strip(),
            "verification_hint": str(raw_step.get("verification_hint") or _default_verification_hint(description, intent)).strip(),
            "needs_tools": bool(raw_step.get("needs_tools", True)),
        })
    return normalized_steps


def build_rule_execution_plan(query: str | None, risk_level: str) -> list[dict]:
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    split_pattern = r"(?:然后|接着|最后|再|并且|step by step|after that|finally|then)"
    raw_parts = re.split(split_pattern, normalized_query, flags=re.IGNORECASE)

    plan_parts = []
    for raw_part in raw_parts:
        cleaned = raw_part.strip(" ，。；;,.!?！？")
        if cleaned.startswith("先"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            plan_parts.append(cleaned)

    if not plan_parts:
        plan_parts = [normalized_query]

    plan = []
    total_steps = len(plan_parts)
    for index, part in enumerate(plan_parts, start=1):
        step_risk = classify_step_risk(part, risk_level, total_steps=total_steps)
        intent = infer_step_intent(part)
        plan.append({
            "step": index,
            "description": part,
            "status": "pending",
            "risk_level": step_risk,
            "intent": intent,
            "execution_mode": infer_execution_mode(part, intent),
            "success_criteria": _default_success_criteria(part, intent),
            "verification_hint": _default_verification_hint(part, intent),
            "needs_tools": True,
        })
    return plan


def build_execution_plan(query: str | None, risk_level: str) -> list[dict]:
    return build_rule_execution_plan(query, risk_level)
