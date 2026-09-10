"""
TTS 抽象接口 —— 所有语音合成后端的统一契约

与 ASR/LLM 接口同构：同步抽象方法 + 默认异步包装。
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class TTSInterface(ABC):
    """语音合成抽象基类"""

    @abstractmethod
    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        同步合成：把文本合成为音频文件。

        参数:
            text: 要合成的文本
            output_path: 输出路径；为 None 时由后端自动生成
        返回:
            生成的音频文件路径
        """
        raise NotImplementedError

    async def async_synthesize(
        self, text: str, output_path: Optional[str] = None
    ) -> str:
        """异步包装：默认扔进线程池"""
        return await asyncio.to_thread(self.synthesize, text, output_path)
