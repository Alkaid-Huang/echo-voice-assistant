"""
OllamaLLM —— 本地 Ollama 后端
对照 Ollama 官方 API 文档 POST /api/chat（2026-09-10 检索）

前置条件：
1. 安装 Ollama（https://ollama.com）
2. 拉模型：ollama pull qwen2.5:3b
3. 确认服务在跑：ollama list
"""
import requests

from .llm_interface import LLMInterface
from .llm_factory import register_llm


@register_llm("ollama")
class OllamaLLM(LLMInterface):
    """本地 Ollama 对话后端（默认非流式，演示用足够）"""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        num_predict: int = 256,
        timeout: float = 60.0,
        **kwargs,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama 请求失败（{self.host}）：{e}\n"
                f"请确认已安装 Ollama、服务已启动，并执行过：ollama pull {self.model}"
            ) from e
        data = resp.json()
        return data["message"]["content"].strip()
