"""
ASR 抽象接口 —— 所有 ASR 后端的统一契约
对照 Open-LLM-VTuber src/open_llm_vtuber/asr/asr_interface.py（2026-08-13 检索）

对齐真实源码的 3 个细节：
1. SAMPLE_RATE 等常量定义在接口上——整条 VAD→ASR 管线统一 16000Hz
2. transcribe_np(audio) 只收音频、不收采样率——"统一约定"代替"每次传参"
3. 异步默认实现里统一做 float32 归一化（真实源码也这么做）
"""
import asyncio
from abc import ABC, abstractmethod

import numpy as np


class ASRInterface(ABC):
    """ASR 抽象基类：定义所有 ASR 后端必须实现的契约"""

    # 整条 VAD→ASR 管线的统一约定：16kHz 单声道 16bit
    SAMPLE_RATE = 16000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH = 2

    @abstractmethod
    def transcribe_np(self, audio_np: np.ndarray) -> str:
        """
        同步转写：把一段音频（numpy 数组）转成文本。

        参数:
            audio_np: float32 单声道音频，采样率必须是 SAMPLE_RATE（16000Hz）
        返回:
            识别出的文本字符串（去掉首尾空白）
        """
        raise NotImplementedError

    async def async_transcribe_np(self, audio_np: np.ndarray) -> str:
        """
        异步转写：默认实现先统一转成 float32，
        再用 asyncio.to_thread 包装同步版，不阻塞事件循环。
        具体后端一般不需要重写本方法。
        """
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        return await asyncio.to_thread(self.transcribe_np, audio_np)
