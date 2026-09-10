"""
Silero VAD 引擎 —— VADInterface 的具体实现
对照 Open-LLM-VTuber silero.py 的 VADEngine（2026-08-03 检索）

用 @register_vad 装饰器注册到工厂，使 create_vad 能按配置创建它
"""
import asyncio
from typing import Generator

import numpy as np
import torch

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
        sample_rate: int = 16000,
        db_margin: float = 6.0,
        noise_floor_alpha: float = 0.02,
        db_adapt_limit: float = 10.0,
    ):
        # silero-vad 只支持 8k / 16k，且对每帧采样点数有硬性要求：
        # 16k → 512 点，8k → 256 点。配错时模型会在推理时报晦涩错误，
        # 所以这里在启动阶段就把配置错误暴露出来。
        if sample_rate not in (8000, 16000):
            raise ValueError(f"silero-vad 只支持 8000/16000 Hz，当前配置: {sample_rate}")
        expected_window = 256 if sample_rate == 8000 else 512
        if window_size_samples != expected_window:
            raise ValueError(
                f"silero-vad 在 {sample_rate} Hz 下要求每帧 {expected_window} 个采样点，"
                f"当前配置为 {window_size_samples}"
            )
        # 加载 silero 模型（只加载一次）
        self.model = load_silero_vad()
        self.window_size_samples = window_size_samples
        self.sample_rate = sample_rate

        # 创建状态机
        self.state_machine = StateMachine(
            prob_threshold=prob_threshold,
            db_threshold=db_threshold,
            required_hits=required_hits,
            required_misses=required_misses,
            smoothing_window=smoothing_window,
            pre_buffer_size=pre_buffer_size,
            db_margin=db_margin,
            noise_floor_alpha=noise_floor_alpha,
            db_adapt_limit=db_adapt_limit,
        )

    def detect_speech(self, audio_chunks) -> Generator[bytes, None, None]:
        """流式检测语音段（同步生成器）"""
        for audio_chunk in audio_chunks:
            result = self.process_block(audio_chunk[0], audio_chunk[1])
            if result is not None:
                yield result

    def process_block(self, audio_np, chunk_bytes):
        """
        处理单块：算语音概率 → 交给状态机 → 返回完整语音段或 None

        注意（真实 bug 修复）：silero-vad 的模型要求 torch.Tensor，
        传 numpy 数组会报 "Expected a value of type 'Tensor' ... found 'ndarray'"。
        因此这里统一转换一次；对已经是 Tensor 的输入也兼容。
        """
        prob = self.speech_probability(audio_np)
        return self.state_machine.process(
            prob=prob, audio_np=audio_np, chunk_bytes=chunk_bytes
        )

    def speech_probability(self, audio_np) -> float:
        """
        单帧语音概率（状态机与诊断模式共用）。

        注意：模型第二个参数是采样率，不是帧长（真实 bug：
        早期代码把 window_size_samples 传进去，报 "Supported sampling rates: [8000, 16000]"）。
        """
        if isinstance(audio_np, np.ndarray):
            audio_tensor = torch.from_numpy(
                np.ascontiguousarray(audio_np, dtype=np.float32)
            )
        else:
            audio_tensor = audio_np
        return float(self.model(audio_tensor, self.sample_rate).item())

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
