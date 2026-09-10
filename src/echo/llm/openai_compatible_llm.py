"""
OpenAICompatibleLLM —— 任何 OpenAI 兼容的 /chat/completions 服务
可用于 DeepSeek / 通义千问 / vLLM / OpenAI 等（2026-09-10 检索）

API Key 从环境变量读取，避免把密钥写进配置文件：
    set ECHO_LLM_API_KEY=sk-xxxx
"""
import os

import requests

from .llm_interface import LLMInterface
from .llm_factory import register_llm


@register_llm("openai_compatible")
class OpenAICompatibleLLM(LLMInterface):
    """OpenAI 兼容 HTTP 后端"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key_env: str = "ECHO_LLM_API_KEY",
        temperature: float = 0.7,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"未找到环境变量 {api_key_env}，请先设置 API Key：\n"
                f"  PowerShell: $env:{api_key_env}='sk-xxxx'"
            )
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"LLM 请求失败（{self.base_url}）：{e}") from e
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
