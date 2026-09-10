"""
Silero VAD 引擎 —— VADInterface 的具体实现
对照 Open-LLM-VTuber silero.py 的 VADEngine（2026-08-03 检索）

用 @register_vad 装饰器注册到工厂，使 create_vad 能按配置创建它
"""
import asyncio
from typing import Generator

from .vad_interface import VADInterface
from .vad_factory import register_vad  # 注册装饰器
from .state_machine import State, StateMachine
from silero_vad import load_silero_vad


@register_vad("silero_vad")  # 注册到工厂
class SileroVADEngine(VADInterface):
    """Silero VAD 流式引擎"""

    def __init__(
        self,
        prob_threshold: float = 0.5,
        db_threshold: float = -20.0,
        required_hits: int = 3,
        required_misses: int = 24,
        smoothing_window: int = 5,
        pre_buffer_size: int = 20,
        window_size_samples: int = 512,
    ):
        # 加载 silero 模型（只加载一次）
        self.model = load_silero_vad()
        self.window_size_samples = window_size_samples

        # 创建状态机
        self.state_machine = StateMachine(
            prob_threshold=prob_threshold,
            db_threshold=db_threshold,
            required_hits=required_hits,
            required_misses=required_misses,
            smoothing_window=smoothing_window,
            pre_buffer_size=pre_buffer_size,
        )

    def detect_speech(self, audio_chunks) -> Generator[bytes, None, None]:
        """流式检测语音段（同步生成器）"""
        for audio_chunk in audio_chunks:
            result = self.process_block(audio_chunk[0], audio_chunk[1])
            if result is not None:
                yield result

    def process_block(self, audio_np, chunk_bytes):
        """处理单块：算语音概率 → 交给状态机 → 返回完整语音段或 None"""
        prob = self.model(audio_np, self.window_size_samples).item()
        return self.state_machine.process(
            prob=prob, audio_np=audio_np, chunk_bytes=chunk_bytes
        )

    def is_speaking(self) -> bool:
        """状态机处于 ACTIVE 表示正在说话（供打断检测使用）"""
        return self.state_machine.state == State.ACTIVE

    async def async_detect_speech(self, audio_chunks):
        """异步版检测（用 asyncio.to_thread 包装，不阻塞事件循环）"""
        sync_gen = self.detect_speech(audio_chunks)
        while True:
            try:
                chunk = await asyncio.to_thread(next, sync_gen)
            except StopIteration:
                break
            yield chunk
