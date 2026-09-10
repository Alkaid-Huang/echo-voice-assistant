"""
LLM 抽象接口 —— 所有大模型后端的统一契约

模式与 ASRInterface 完全一致：同步抽象方法 + 接口提供的默认异步包装。
这样每个后端只写同步版，异步调用不会阻塞事件循环。
"""
import asyncio
from abc import ABC, abstractmethod


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
