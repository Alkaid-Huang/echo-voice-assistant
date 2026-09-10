"""
MockTTS 假实现 —— 生成一段极短的静音 WAV，用于测试和离线开发
（真实播放会用到声卡，自动化测试里不适合真发声）
"""
import os
import uuid
import wave

import numpy as np

from .tts_interface import TTSInterface
from .tts_factory import register_tts


@register_tts("mock_tts")
class MockTTS(TTSInterface):
    """假实现：写一个短静音 WAV，返回路径"""

    def __init__(self, output_dir: str = "outputs", duration: float = 0.2, **kwargs):
        self.output_dir = output_dir
        self.duration = duration

    def synthesize(self, text: str, output_path=None) -> str:
        path = output_path or os.path.join(
            self.output_dir, f"mock_tts_{uuid.uuid4().hex[:8]}.wav"
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        samples = np.zeros(int(16000 * self.duration), dtype=np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())
        return path
