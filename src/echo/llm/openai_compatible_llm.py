"""
OpenAICompatibleLLM —— 任何 OpenAI 兼容的 /chat/completions 服务
可用于 DeepSeek / 通义千问 / vLLM / OpenAI 等（2026-09-10 检索）

API Key 从环境变量读取，避免把密钥写进版本库：
    方式一（推荐）：在项目根目录建 .env 文件，写 ECHO_LLM_API_KEY=sk-xxxx
    方式二：PowerShell 执行 $env:ECHO_LLM_API_KEY='sk-xxxx'
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
        max_tokens: int = 256,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"未找到环境变量 {api_key_env}，请先设置 API Key：\n"
                f"  方式一（推荐）：在项目根目录 .env 里写 {api_key_env}=sk-xxxx\n"
                f"  方式二：PowerShell 执行 $env:{api_key_env}='sk-xxxx'"
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.Timeout as e:
            raise RuntimeError(f"LLM 请求超时（{self.timeout}s）：{url}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"LLM 网络错误（{url}）：{e}") from e

        # 常见错误给出可执行的排查提示，而不是把状态码直接甩给用户
        if resp.status_code == 401:
            raise RuntimeError(
                f"API Key 无效或已过期（环境变量 {self.api_key_env}），请检查密钥是否正确"
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"接口不存在（404）：{url}\n"
                f"请检查 base_url 是否写全，例如 https://api.deepseek.com/v1"
            )
        if resp.status_code == 429:
            raise RuntimeError("触发限流（429）：请降低请求频率或检查账户配额")
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM 返回错误 {resp.status_code}：{resp.text[:200]}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"无法解析 LLM 响应：{str(data)[:200]}") from e
