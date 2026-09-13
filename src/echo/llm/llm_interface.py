"""
LLM 抽象接口 —— 所有大模型后端的统一契约

模式与 ASRInterface 完全一致：同步抽象方法 + 接口提供的默认异步包装。
这样每个后端只写同步版，异步调用不会阻塞事件循环。
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ToolCall:
    """模型请求调用某个工具"""

    id: str
    name: str
    arguments: str = "{}"  # JSON 字符串（OpenAI 兼容格式）


@dataclass
class LLMResponse:
    """一次模型调用的结构化结果：文本 + 可能的工具调用"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class LLMInterface(ABC):
    """大模型抽象基类"""

    @abstractmethod
    def chat(self, messages: list[dict]) -> str:
        """
        同步对话。

        参数:
            messages: OpenAI 风格消息列表，
                      [{"role": "system"|"user"|"assistant", "content": str}, ...]
        返回:
            模型回复的文本（去掉首尾空白）
        """
        raise NotImplementedError

    async def async_chat(self, messages: list[dict]) -> str:
        """异步包装：默认把同步实现扔进线程池，后端一般不用重写"""
        return await asyncio.to_thread(self.chat, messages)

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """
        带工具调用的对话。

        默认实现：忽略工具，退化成普通对话（不支持 function calling 的后端自动兼容）。
        支持工具的后端（如 OpenAI 兼容 API）应覆写本方法。
        """
        return LLMResponse(content=self.chat(messages), tool_calls=[])

    async def async_chat_with_tools(
        self, messages: list[dict], tools: list[dict]
    ) -> LLMResponse:
        """异步包装：默认扔进线程池"""
        return await asyncio.to_thread(self.chat_with_tools, messages, tools)
