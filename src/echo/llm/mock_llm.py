"""MockLLM 假实现 —— 不联网、不加载模型，用于测试和离线开发"""
from .llm_interface import LLMInterface
from .llm_factory import register_llm


@register_llm("mock_llm")
class MockLLM(LLMInterface):
    """假实现：返回固定回复"""

    def __init__(self, reply: str = "（模拟回复）我在听，你继续说。", **kwargs):
        self.reply = reply

    def chat(self, messages: list[dict]) -> str:
        return self.reply
