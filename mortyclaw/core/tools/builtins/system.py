from __future__ import annotations

from datetime import datetime


def get_system_model_info_impl() -> str:
    from ...harness.settings import HarnessSettings

    settings = HarnessSettings.from_env()
    return (
        "当前对话内核是 DeepSeek Harness，"
        f"提供商是 deepseek-official，模型是 {settings.model}。"
    )


def get_current_time_impl(*, now_fn=datetime.now) -> str:
    now = now_fn()
    return f"当前本地系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculator_impl(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"表达式 '{expression}' 的计算结果是: {result}"
    except Exception as exc:
        return f"计算出错，请检查表达式格式。错误信息: {str(exc)}"
