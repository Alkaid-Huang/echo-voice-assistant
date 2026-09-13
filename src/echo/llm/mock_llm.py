"""MockLLM 假实现 —— 不联网、不加载模型，用于测试和离线开发"""
from .llm_interface import LLMInterface
from .llm_interface import LLMResponse, ToolCall
from .llm_factory import register_llm


@register_llm("mock_llm")
class MockLLM(LLMInterface):
    """假实现：返回固定回复"""

    def __init__(self, reply: str = "（模拟回复）我在听，你继续说。", **kwargs):
        self.reply = reply

    def chat(self, messages: list[dict]) -> str:
        return self.reply


@register_llm("mock_tool_llm")
class MockToolLLM(LLMInterface):
    """
    按脚本返回结果的假模型，用于测试 Agent 的工具调用循环。

    script 是一串 LLMResponse：先返回 tool_calls，再返回最终文本。
    """

    def __init__(self, script: list[LLMResponse] | None = None, **kwargs):
        self.script = list(script or [])
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.seen_messages.append(messages)
        if self.script:
            response = self.script.pop(0)
            return response.content
        return "（脚本已用完）"

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.seen_messages.append(messages)
        if self.script:
            return self.script.pop(0)
        return LLMResponse(content="（脚本已用完）")
