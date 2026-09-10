"""
VAD 抽象接口 —— 所有 VAD 后端的统一契约
对照 Open-LLM-VTuber 的 vad_interface.py（2026-08-03 检索）
"""
from abc import ABC, abstractmethod
from typing import Generator


class VADInterface(ABC):
    """VAD 抽象基类：定义所有 VAD 后端必须实现的契约"""

    @abstractmethod
    def detect_speech(self, audio_chunks) -> Generator[bytes, None, None]:
        """
        流式检测语音段。

        参数:
            audio_chunks: 可迭代的音频块，每块是 (audio_np, chunk_bytes)
                          audio_np: numpy 数组 (用于算分贝和概率)
                          chunk_bytes: 该块的原始字节 (用于输出)

        返回:
            生成器，每次 yield 一个完整的语音段 (bytes)
            没有完整语音段时不 yield（等状态机累积到输出条件）
            """
        pass

    def process_block(self, audio_np, chunk_bytes) -> "bytes | None":
        """
        处理单个音频块，返回完整语音段（bytes）或 None。

        与 detect_speech 的关系：detect_speech 就是"对一批块循环调用本方法"
        的便捷包装；需要逐块控制（例如播放回复时做打断检测）时直接调用本方法。
        """
        raise NotImplementedError

    def is_speaking(self) -> bool:
        """
        当前是否处于"正在说话"状态。

        打断检测用：播放回复期间本方法变为 True，说明用户开始说话，
        应停止播放。默认实现恒为 False（不支持的后端无打断能力）。
        """
        return False
