import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
'''
多模型适配(Factory)
'''
load_dotenv()

# 各大厂商官方的 OpenAI 兼容接口地址 (当用户未配置 BASE_URL 时作为兜底)
COMPATIBLE_BASE_URLS = {
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "z.ai": "https://open.bigmodel.cn/api/paas/v4",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1"
}

COMPATIBLE_API_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "other": ("OPENAI_API_KEY",),
    "aliyun": ("ALIYUN_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    "dashscope": ("DASHSCOPE_API_KEY", "ALIYUN_API_KEY", "OPENAI_API_KEY"),
    "z.ai": ("ZAI_API_KEY", "OPENAI_API_KEY"),
    "tencent": ("TENCENT_API_KEY", "OPENAI_API_KEY"),
}

COMPATIBLE_BASE_URL_ENV_VARS = {
    "openai": ("OPENAI_API_BASE",),
    "other": ("OPENAI_API_BASE",),
    "aliyun": ("ALIYUN_BASE_URL", "DASHSCOPE_BASE_URL", "OPENAI_API_BASE"),
    "dashscope": ("DASHSCOPE_BASE_URL", "ALIYUN_BASE_URL", "OPENAI_API_BASE"),
    "z.ai": ("ZAI_BASE_URL", "OPENAI_API_BASE"),
    "tencent": ("TENCENT_BASE_URL", "OPENAI_API_BASE"),
}


def get_compatible_provider_api_key_env_vars(provider_name: str) -> tuple[str, ...]:
    normalized = str(provider_name or "").strip().lower()
    return COMPATIBLE_API_KEY_ENV_VARS.get(normalized, ("OPENAI_API_KEY",))


def get_compatible_provider_base_url_env_vars(provider_name: str) -> tuple[str, ...]:
    normalized = str(provider_name or "").strip().lower()
    return COMPATIBLE_BASE_URL_ENV_VARS.get(normalized, ("OPENAI_API_BASE",))


def resolve_compatible_provider_api_key(provider_name: str, api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    for env_key in get_compatible_provider_api_key_env_vars(provider_name):
        value = os.environ.get(env_key)
        if value:
            return value
    return None


def resolve_compatible_provider_base_url(provider_name: str, base_url: str | None = None) -> str | None:
    if base_url:
        return base_url
    for env_key in get_compatible_provider_base_url_env_vars(provider_name):
        value = os.environ.get(env_key)
        if value:
            return value
    return COMPATIBLE_BASE_URLS.get(str(provider_name or "").strip().lower())

def get_provider(
    provider_name: str = "openai", 
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.0,
    base_url: str | None = None,  # 允许外部传入
    api_key: str | None = None,   # 允许外部传入
    **kwargs: Any
) -> BaseChatModel:
    """
    模型适配器工厂
    """
    provider_name = provider_name.lower()
    
    if provider_name in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
        from langchain_openai import ChatOpenAI

        current_api_key = resolve_compatible_provider_api_key(provider_name, api_key)
        if not current_api_key:
            supported_keys = ", ".join(get_compatible_provider_api_key_env_vars(provider_name))
            raise ValueError(f"未找到 API Key！请确保 .env 中配置了以下环境变量之一：{supported_keys}")

        final_base_url = resolve_compatible_provider_base_url(provider_name, base_url)

        return ChatOpenAI(
            model=model_name, 
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        
        current_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量！")
            
        final_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        return ChatAnthropic(
            model_name=model_name, 
            temperature=temperature, 
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )
        
    elif provider_name == "ollama":
        from langchain_community.chat_models import ChatOllama
        
        final_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name, 
            temperature=temperature, 
            base_url=final_base_url,
            **kwargs
        )
        
    else:
        raise ValueError(f"不支持的模型提供商: {provider_name}")

# 测试模型调用    
# LLM = get_provider(provider_name='aliyun', model_name='glm-5')
# res = LLM.invoke('你是谁')
# print(type(res))
# print(res)

