"""
Silero VAD 真实推理测试（不依赖麦克风，但需要 silero-vad + torch）

存在意义：mock 测试覆盖不到真实模型调用，
这条用例专门防住"模型输入类型不对"这类只在真机才暴露的 bug。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

silero_vad = pytest.importorskip("silero_vad")

from echo.vad.silero_engine import SileroVADEngine


def test_silero_process_block_accepts_numpy():
    """单帧 numpy 音频不应抛异常（回归：曾因传 ndarray 而报 Tensor 类型错误）"""
    engine = SileroVADEngine(required_hits=2, required_misses=2)
    chunk = np.zeros(512, dtype=np.float32)  # 1 帧静音
    result = engine.process_block(chunk, chunk.tobytes())
    assert result is None  # 静音不该产生语音段
    assert engine.is_speaking() is False


def test_silero_detect_speech_generator_runs():
    """流式接口跑通：给几帧静音，不应抛异常，也不应产出语音段"""
    engine = SileroVADEngine(required_hits=2, required_misses=2)
    chunks = [
        (np.zeros(512, dtype=np.float32), np.zeros(512, dtype=np.float32).tobytes())
        for _ in range(5)
    ]
    assert list(engine.detect_speech(chunks)) == []
